#!/usr/bin/env python3
"""
Enhanced Experiment Runner per esperimenti di Federated Learning.

Versione migliorata che include:
- Sistema di configurazione centralizzato
- Validazione robusta dei parametri
- Sistema di recovery e checkpoint avanzato
- Monitoraggio delle risorse di sistema
- Gestione degli errori più specifica
- Supporto per retry intelligente
- Parallelizzazione controllata
- Caching e persistence migliorate
"""

import pandas as pd
import numpy as np
import subprocess
import time
import json
import os
import sys
import re
import logging
import threading
import socket
import signal
import psutil
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import pickle
import hashlib
from enum import Enum
import queue
from contextlib import contextmanager

# Import existing modules
from experiment_runner import ExperimentConfig as BaseExperimentConfig
from checkpoint_manager import CheckpointManager
from retry_manager import RetryManager, RetryConfig, CONSERVATIVE_RETRY
from config_manager import get_config_manager

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_experiment_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Configurazione del sistema."""
    max_retries: int = 3
    retry_delay: int = 5
    process_timeout: int = 600
    port: int = 8080
    log_level: str = "INFO"
    max_parallel_experiments: int = 1
    resource_monitoring: bool = True
    checkpoint_interval: int = 10  # Save checkpoint every N experiments
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if self.process_timeout <= 0:
            raise ValueError("process_timeout must be positive")
        if not (1024 <= self.port <= 65535):
            raise ValueError("port must be between 1024 and 65535")
        if self.max_parallel_experiments <= 0:
            raise ValueError("max_parallel_experiments must be positive")


@dataclass
class ExperimentDefaults:
    """Parametri di default per gli esperimenti."""
    num_rounds: int = 10
    num_clients: int = 10
    learning_rate: float = 0.01
    batch_size: int = 32
    
    def __post_init__(self):
        """Validate defaults after initialization."""
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class ExperimentMetrics:
    """Metriche di sistema per un esperimento."""
    experiment_id: str
    start_time: float
    end_time: Optional[float] = None
    cpu_usage: List[float] = None
    memory_usage: List[float] = None
    success: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        if self.cpu_usage is None:
            self.cpu_usage = []
        if self.memory_usage is None:
            self.memory_usage = []
    
    @property
    def duration(self) -> Optional[float]:
        """Durata dell'esperimento in secondi."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def avg_cpu_usage(self) -> Optional[float]:
        """Utilizzo medio della CPU."""
        return np.mean(self.cpu_usage) if self.cpu_usage else None
    
    @property
    def avg_memory_usage(self) -> Optional[float]:
        """Utilizzo medio della memoria."""
        return np.mean(self.memory_usage) if self.memory_usage else None


class ExperimentStatus(Enum):
    """Stato di un esperimento."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class EnhancedConfigManager:
    """Gestore di configurazione centralizzato."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("enhanced_config.yaml")
        self.system = SystemConfig()
        self.defaults = ExperimentDefaults()
        self.load_config()
    
    def load_config(self):
        """Carica configurazione da file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                if 'system' in config_data:
                    system_data = config_data['system']
                    self.system = SystemConfig(**system_data)
                
                if 'defaults' in config_data:
                    defaults_data = config_data['defaults']
                    self.defaults = ExperimentDefaults(**defaults_data)
                    
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
                logger.info("Using default configuration")
    
    def save_config(self):
        """Salva configurazione su file."""
        config_data = {
            'system': asdict(self.system),
            'defaults': asdict(self.defaults)
        }
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_file}: {e}")


class MetricsCollector:
    """Raccoglitore avanzato di metriche di sistema."""
    
    def __init__(self):
        self.metrics: Dict[str, ExperimentMetrics] = {}
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.stop_monitoring: Dict[str, threading.Event] = {}
    
    def start_monitoring(self, experiment_id: str):
        """Inizia il monitoraggio delle risorse per un esperimento."""
        if experiment_id in self.monitoring_threads:
            self.stop_monitoring_for_experiment(experiment_id)
        
        self.metrics[experiment_id] = ExperimentMetrics(
            experiment_id=experiment_id,
            start_time=time.time()
        )
        
        stop_event = threading.Event()
        self.stop_monitoring[experiment_id] = stop_event
        
        thread = threading.Thread(
            target=self._monitor_resources,
            args=(experiment_id, stop_event),
            daemon=True
        )
        self.monitoring_threads[experiment_id] = thread
        thread.start()
        
        logger.debug(f"Started resource monitoring for {experiment_id}")
    
    def _monitor_resources(self, experiment_id: str, stop_event: threading.Event):
        """Loop di monitoraggio delle risorse."""
        while not stop_event.is_set():
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                if experiment_id in self.metrics:
                    self.metrics[experiment_id].cpu_usage.append(cpu_percent)
                    self.metrics[experiment_id].memory_usage.append(memory_percent)
                
                time.sleep(5)  # Sample every 5 seconds
            except Exception as e:
                logger.warning(f"Error monitoring resources for {experiment_id}: {e}")
                break
    
    def stop_monitoring_for_experiment(self, experiment_id: str):
        """Ferma il monitoraggio per un esperimento specifico."""
        if experiment_id in self.stop_monitoring:
            self.stop_monitoring[experiment_id].set()
        
        if experiment_id in self.monitoring_threads:
            thread = self.monitoring_threads[experiment_id]
            thread.join(timeout=5)
            del self.monitoring_threads[experiment_id]
        
        if experiment_id in self.stop_monitoring:
            del self.stop_monitoring[experiment_id]
    
    def finish_experiment(self, experiment_id: str, success: bool, error_msg: Optional[str] = None, retry_count: int = 0):
        """Completa il tracciamento di un esperimento."""
        self.stop_monitoring_for_experiment(experiment_id)
        
        if experiment_id in self.metrics:
            metrics = self.metrics[experiment_id]
            metrics.end_time = time.time()
            metrics.success = success
            metrics.error_message = error_msg
            metrics.retry_count = retry_count
    
    def get_summary(self) -> Dict[str, Any]:
        """Restituisce un riassunto delle metriche."""
        total_experiments = len(self.metrics)
        successful = sum(1 for m in self.metrics.values() if m.success)
        
        durations = [m.duration for m in self.metrics.values() if m.duration is not None]
        avg_duration = np.mean(durations) if durations else 0
        
        cpu_usages = [m.avg_cpu_usage for m in self.metrics.values() if m.avg_cpu_usage is not None]
        avg_cpu = np.mean(cpu_usages) if cpu_usages else 0
        
        memory_usages = [m.avg_memory_usage for m in self.metrics.values() if m.avg_memory_usage is not None]
        avg_memory = np.mean(memory_usages) if memory_usages else 0
        
        return {
            "total_experiments": total_experiments,
            "successful": successful,
            "failed": total_experiments - successful,
            "success_rate": successful / total_experiments if total_experiments > 0 else 0,
            "average_duration": avg_duration,
            "average_cpu_usage": avg_cpu,
            "average_memory_usage": avg_memory
        }
    
    def cleanup(self):
        """Pulisce tutte le risorse di monitoraggio."""
        for experiment_id in list(self.monitoring_threads.keys()):
            self.stop_monitoring_for_experiment(experiment_id)


class PortManager:
    """Gestore delle porte per esperimenti paralleli."""
    
    def __init__(self, base_port: int = 8080, num_ports: int = 4):
        self.base_port = base_port
        self.available_ports = list(range(base_port, base_port + num_ports))
        self.used_ports = set()
        self.lock = threading.Lock()
        logger.info(f"Port manager initialized with ports {base_port}-{base_port + num_ports - 1}")
    
    def acquire_port(self) -> int:
        """Acquisisce una porta disponibile."""
        with self.lock:
            if not self.available_ports:
                raise RuntimeError("No available ports")
            port = self.available_ports.pop(0)
            self.used_ports.add(port)
            logger.debug(f"Acquired port {port}")
            return port
    
    def release_port(self, port: int):
        """Rilascia una porta."""
        with self.lock:
            if port in self.used_ports:
                self.used_ports.remove(port)
                self.available_ports.append(port)
                logger.debug(f"Released port {port}")
    
    def is_port_free(self, port: int) -> bool:
        """Verifica se una porta è libera."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    @contextmanager
    def get_port(self):
        """Context manager per acquisire e rilasciare automaticamente una porta."""
        port = self.acquire_port()
        try:
            yield port
        finally:
            self.release_port(port)


class EnhancedExperimentConfig(BaseExperimentConfig):
    """Configurazione di esperimento migliorata con validazione."""
    
    def __init__(self, *args, **kwargs):
        # Initialize with validation
        super().__init__(*args, **kwargs)
        self._validate_all()
    
    def _validate_all(self):
        """Valida tutti i parametri della configurazione."""
        # Validation is already done in parent class
        pass
    
    def get_config_hash(self) -> str:
        """Genera un hash della configurazione per rilevare cambiamenti."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def validate_consistency(self) -> List[str]:
        """Valida la coerenza dei parametri e restituisce eventuali warnings."""
        warnings = []
        
        # Check attack-specific warnings
        if self.attack == "noise" and self.attack_params.get("noise_std", 0) > 1.0:
            warnings.append("High noise standard deviation may severely impact learning")
        
        if self.attack == "labelflip" and self.attack_params.get("labelflip_fraction", 0) > 0.5:
            warnings.append("High label flip fraction may prevent convergence")
        
        # Check strategy-specific warnings
        if self.strategy == "fedprox" and self.strategy_params.get("proximal_mu", 0) > 1.0:
            warnings.append("High proximal_mu may overly constrain client updates")
        
        return warnings


class EnhancedExperimentRunner:
    """Runner di esperimenti con funzionalità avanzate."""
    
    def __init__(self, 
                 base_dir: str = ".",
                 results_dir: str = "enhanced_experiment_results",
                 config_manager: Optional[EnhancedConfigManager] = None):
        
        self.base_dir = Path(base_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.config_manager = config_manager or EnhancedConfigManager()
        self.system_config = self.config_manager.system
        
        # Managers
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.results_dir / "checkpoints"
        )
        self.retry_manager = RetryManager(CONSERVATIVE_RETRY)
        self.metrics_collector = MetricsCollector()
        self.port_manager = PortManager(
            base_port=self.system_config.port,
            num_ports=self.system_config.max_parallel_experiments
        )
        
        # Results storage
        self.results_df = pd.DataFrame(columns=[
            "algorithm", "attack", "dataset", "run", "client_id", 
            "round", "metric", "value"
        ])
        
        # State tracking
        self.current_round = 0
        self.experiment_queue = queue.Queue()
        self.completed_experiments = 0
        self.failed_experiments = 0
        
        logger.info(f"Enhanced Experiment Runner initialized")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Max parallel experiments: {self.system_config.max_parallel_experiments}")
    
    def validate_prerequisites(self) -> bool:
        """Valida i prerequisiti del sistema."""
        logger.info("Validating system prerequisites...")
        
        required_files = [
            self.base_dir / "run_with_attacks.py",
            self.base_dir / "server.py",
            self.base_dir / "client.py"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                return False
        
        # Test run_with_attacks.py
        try:
            result = subprocess.run(
                [sys.executable, "run_with_attacks.py", "--help"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.error("run_with_attacks.py is not responding correctly")
                return False
        except Exception as e:
            logger.error(f"Error testing run_with_attacks.py: {e}")
            return False
        
        # Check port availability
        if not self.port_manager.is_port_free(self.system_config.port):
            logger.warning(f"Port {self.system_config.port} is not free, will attempt cleanup")
        
        logger.info("Prerequisites validation passed")
        return True
    
    def kill_flower_processes(self):
        """Termina tutti i processi Flower in esecuzione."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if any(keyword in cmdline.lower() for keyword in 
                               ['flower', 'server.py', 'client.py', 'run_with_attacks.py']):
                            logger.debug(f"Killing process {proc.info['pid']}: {cmdline}")
                            proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            time.sleep(2)  # Grace period
        except Exception as e:
            logger.warning(f"Error killing processes: {e}")
    
    def wait_for_port(self, port: int, timeout: int = 60):
        """Attende che una porta diventi libera."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.port_manager.is_port_free(port):
                return
            time.sleep(1)
        raise TimeoutError(f"Port {port} did not become free within {timeout} seconds")
    
    def build_attack_command(self, config: EnhancedExperimentConfig, port: int) -> List[str]:
        """Costruisce il comando per eseguire un esperimento."""
        cmd = [
            sys.executable, "run_with_attacks.py",
            "--strategy", config.strategy,
            "--attack", config.attack,
            "--dataset", config.dataset,
            "--num-rounds", str(config.num_rounds),
            "--num-clients", str(config.num_clients),
            "--port", str(port)
        ]
        
        # Add attack parameters
        for param, value in config.attack_params.items():
            cmd.extend([f"--{param.replace('_', '-')}", str(value)])
        
        # Add strategy parameters with proper mapping
        strategy_param_mapping = {
            "proximal_mu": "proximal-mu",
            "server_learning_rate": "server-learning-rate",
            "server_momentum": "server-momentum",
            "num_byzantine": "num-byzantine",
            "beta": "beta",
            "step_size": "step-size",
            "compressor_coords": "compressor-coords",
            "alpha": "alpha",
            "tau": "tau",
            "to_keep": "to-keep",
            "fedopt_tau": "fedopt-tau",
            "fedopt_beta1": "fedopt-beta1",
            "fedopt_beta2": "fedopt-beta2",
            "fedopt_eta": "fedopt-eta",
            "fedopt_eta_l": "fedopt-eta-l",
        }
        
        for param, value in config.strategy_params.items():
            mapped_param = strategy_param_mapping.get(param, param.replace('_', '-'))
            cmd.extend([f"--{mapped_param}", str(value)])
        
        return cmd
    
    def run_single_experiment_with_port(self, 
                                      config: EnhancedExperimentConfig, 
                                      run_id: int, 
                                      port: int) -> Tuple[bool, str]:
        """Esegue un singolo esperimento con una porta specifica."""
        experiment_id = config.get_experiment_id()
        full_experiment_id = f"{experiment_id}_run_{run_id}"
        
        # Start monitoring
        if self.system_config.resource_monitoring:
            self.metrics_collector.start_monitoring(full_experiment_id)
        
        try:
            logger.info(f"Starting experiment {full_experiment_id} on port {port}")
            
            # Cleanup and wait for port
            self.kill_flower_processes()
            self.wait_for_port(port, timeout=30)
            
            # Build and execute command
            cmd = self.build_attack_command(config, port)
            logger.info(f"Running command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.base_dir,
                bufsize=1,
                universal_newlines=True,
            )
            
            output_lines: List[str] = []
            try:
                while True:
                    if process.stdout is None:
                        break
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        stripped = line.rstrip()
                        output_lines.append(stripped)
                        self.parse_and_store_metrics(stripped, config, run_id)
                
                return_code = process.wait(timeout=self.system_config.process_timeout)
                success = return_code == 0
                
            except subprocess.TimeoutExpired:
                process.kill()
                output_lines.append("Process timed out")
                success = False
            finally:
                self.kill_flower_processes()
                if process.stdout:
                    process.stdout.close()
            
            output = "\n".join(output_lines)
            
            if success:
                logger.info(f"Experiment {full_experiment_id} completed successfully")
            else:
                logger.error(f"Experiment {full_experiment_id} failed")
                
            return success, output
            
        except Exception as e:
            error_msg = f"Unexpected error in experiment {full_experiment_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, error_msg
        finally:
            # Finish monitoring
            if self.system_config.resource_monitoring:
                self.metrics_collector.finish_experiment(
                    full_experiment_id, 
                    success=False,  # Will be updated by caller
                    error_msg=None
                )
    
    def run_single_experiment(self, config: EnhancedExperimentConfig, run_id: int) -> Tuple[bool, str]:
        """Esegue un singolo esperimento con retry automatico."""
        experiment_id = config.get_experiment_id()
        full_experiment_id = f"{experiment_id}_run_{run_id}"
        
        # Use retry manager
        return self.retry_manager.execute_with_retry(
            experiment_id=full_experiment_id,
            experiment_func=self._run_single_experiment_internal,
            config=config,
            run_id=run_id
        )
    
    def _run_single_experiment_internal(self, config: EnhancedExperimentConfig, run_id: int) -> Tuple[bool, str]:
        """Esecuzione interna dell'esperimento (senza retry)."""
        with self.port_manager.get_port() as port:
            return self.run_single_experiment_with_port(config, run_id, port)
    
    def parse_and_store_metrics(self, log_line: str, config: EnhancedExperimentConfig, run_id: int):
        """Analizza e memorizza le metriche dai log."""
        # Use attack name with parameters
        attack_name = config.get_attack_name_with_params()
        
        # Extract round number
        round_patterns = [
            r'\[ROUND (\d+)\]',
            r'Round (\d+)',
        ]
        
        for pattern in round_patterns:
            round_match = re.search(pattern, log_line, re.IGNORECASE)
            if round_match:
                self.current_round = int(round_match.group(1))
                break
        
        # Parse metrics
        metric_patterns = [
            (r'accuracy[:\s=]+([0-9.]+)', 'accuracy'),
            (r'loss[:\s=]+([0-9.]+)', 'loss'),
            (r'eval_accuracy[:\s=]+([0-9.]+)', 'eval_accuracy'),
            (r'eval_loss[:\s=]+([0-9.]+)', 'eval_loss'),
        ]
        
        for pattern, metric_name in metric_patterns:
            match = re.search(pattern, log_line, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    
                    # Add to results DataFrame
                    new_row = pd.DataFrame([{
                        'algorithm': config.strategy,
                        'attack': attack_name,
                        'dataset': config.dataset,
                        'run': run_id,
                        'client_id': -1,  # Server metrics
                        'round': self.current_round,
                        'metric': metric_name,
                        'value': value
                    }])
                    
                    self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
                    
                except ValueError:
                    logger.warning(f"Could not parse metric value: {match.group(1)}")
    
    def run_experiments_parallel(self, 
                                configs: List[EnhancedExperimentConfig], 
                                num_runs: int = 1) -> pd.DataFrame:
        """Esegue esperimenti in parallelo con gestione delle risorse."""
        total_experiments = len(configs) * num_runs
        logger.info(f"Starting {total_experiments} experiments in parallel mode")
        
        if not self.validate_prerequisites():
            logger.error("Prerequisites validation failed")
            return self.results_df
        
        # Create experiment queue
        experiment_tasks = []
        for config in configs:
            for run_id in range(num_runs):
                experiment_tasks.append((config, run_id))
        
        completed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.system_config.max_parallel_experiments) as executor:
            # Submit initial batch
            future_to_experiment = {}
            
            for config, run_id in experiment_tasks:
                future = executor.submit(self.run_single_experiment, config, run_id)
                future_to_experiment[future] = (config, run_id)
            
            # Process completed experiments
            for future in as_completed(future_to_experiment):
                config, run_id = future_to_experiment[future]
                experiment_id = config.get_experiment_id()
                
                try:
                    success, output = future.result()
                    
                    if success:
                        completed_count += 1
                        logger.info(f"Experiment {experiment_id} run {run_id} completed successfully")
                    else:
                        failed_count += 1
                        logger.error(f"Experiment {experiment_id} run {run_id} failed")
                    
                    # Update checkpoint
                    self.checkpoint_manager.mark_run_completed(
                        experiment_id, run_id, success, 
                        None if success else output[:500]
                    )
                    
                    # Periodic save
                    if (completed_count + failed_count) % self.system_config.checkpoint_interval == 0:
                        self.save_results(intermediate=True)
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Exception in experiment {experiment_id} run {run_id}: {e}")
                
                # Progress report
                progress = completed_count + failed_count
                logger.info(f"Progress: {progress}/{total_experiments} ({progress/total_experiments*100:.1f}%)")
        
        self.completed_experiments = completed_count
        self.failed_experiments = failed_count
        
        return self.results_df
    
    def run_experiments_sequential(self, 
                                 configs: List[EnhancedExperimentConfig], 
                                 num_runs: int = 1,
                                 resume: bool = False) -> pd.DataFrame:
        """Esegue esperimenti sequenzialmente con checkpoint support."""
        total_experiments = len(configs) * num_runs
        logger.info(f"Starting {total_experiments} experiments sequentially")
        
        if not self.validate_prerequisites():
            logger.error("Prerequisites validation failed")
            return self.results_df
        
        # Setup checkpoint manager
        config_dicts = [cfg.to_dict() for cfg in configs]
        
        if resume:
            logger.info("Resuming from checkpoint...")
            self.checkpoint_manager.load_state()
            progress = self.checkpoint_manager.get_progress_summary()
            logger.info(f"Resuming: {progress['completed_runs']}/{progress['total_runs']} completed")
        else:
            self.checkpoint_manager.register_experiments(config_dicts, num_runs)
        
        # Get experiments to run
        if resume:
            pending_experiments = self.checkpoint_manager.get_pending_experiments()
            experiments_to_run = []
            for pending in pending_experiments:
                for run_id in pending['remaining_runs']:
                    original_config = next((cfg for cfg in configs 
                                          if cfg.get_experiment_id() == pending['experiment_id']), None)
                    if original_config:
                        experiments_to_run.append((original_config, run_id))
        else:
            experiments_to_run = [(cfg, run_id) for cfg in configs for run_id in range(num_runs)]
        
        completed_count = 0
        failed_count = 0
        
        for i, (config, run_id) in enumerate(experiments_to_run):
            experiment_id = config.get_experiment_id()
            
            try:
                success, output = self.run_single_experiment(config, run_id)
                
                # Update checkpoint
                self.checkpoint_manager.mark_run_completed(
                    experiment_id, run_id, success,
                    None if success else output[:500]
                )
                
                if success:
                    completed_count += 1
                else:
                    failed_count += 1
                
                # Periodic save
                if (completed_count + failed_count) % self.system_config.checkpoint_interval == 0:
                    self.save_results(intermediate=True)
                    logger.info(f"Checkpoint saved at {completed_count + failed_count} experiments")
                
                # Progress report
                progress = i + 1
                logger.info(f"Progress: {progress}/{len(experiments_to_run)} ({progress/len(experiments_to_run)*100:.1f}%)")
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, saving progress...")
                self.save_results(intermediate=True)
                break
            except Exception as e:
                failed_count += 1
                logger.error(f"Unexpected error in experiment {experiment_id} run {run_id}: {e}")
                self.checkpoint_manager.mark_run_completed(experiment_id, run_id, False, str(e))
        
        self.completed_experiments = completed_count
        self.failed_experiments = failed_count
        
        return self.results_df
    
    def save_results(self, intermediate: bool = False):
        """Salva i risultati con backup multipli."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if intermediate:
            filename = f"intermediate_results_{timestamp}.csv"
        else:
            filename = f"final_results_{timestamp}.csv"
        
        filepath = self.results_dir / filename
        
        try:
            self.results_df.to_csv(filepath, index=False)
            logger.info(f"Results saved to {filepath}")
            
            # JSON backup
            json_filepath = filepath.with_suffix('.json')
            self.results_df.to_json(json_filepath, orient='records', indent=2)
            
            # Compressed backup for large files
            if self.results_df.shape[0] > 10000:
                compressed_filepath = filepath.with_suffix('.csv.gz')
                self.results_df.to_csv(compressed_filepath, index=False, compression='gzip')
                logger.info(f"Compressed backup saved to {compressed_filepath}")
                
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Genera un report finale completo."""
        system_metrics = self.metrics_collector.get_summary()
        retry_metrics = self.retry_manager.get_failure_summary()
        checkpoint_progress = self.checkpoint_manager.get_progress_summary()
        
        report = {
            "experiment_summary": {
                "total_configurations": len(self.results_df['algorithm'].unique()) if not self.results_df.empty else 0,
                "total_runs": len(self.results_df['run'].unique()) if not self.results_df.empty else 0,
                "completed_experiments": self.completed_experiments,
                "failed_experiments": self.failed_experiments,
                "success_rate": self.completed_experiments / (self.completed_experiments + self.failed_experiments) if (self.completed_experiments + self.failed_experiments) > 0 else 0
            },
            "system_metrics": system_metrics,
            "retry_metrics": retry_metrics,
            "checkpoint_progress": checkpoint_progress,
            "results_shape": self.results_df.shape,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report
        report_file = self.results_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Final report saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save final report: {e}")
        
        return report
    
    def cleanup(self):
        """Pulisce tutte le risorse."""
        logger.info("Cleaning up resources...")
        self.metrics_collector.cleanup()
        self.kill_flower_processes()
        logger.info("Cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def create_enhanced_configurations() -> List[EnhancedExperimentConfig]:
    """Crea configurazioni migrate con validazione."""
    config_mgr = get_config_manager()
    
    strategies = config_mgr.get_valid_strategies()
    datasets = config_mgr.get_valid_datasets()
    
    configs: List[EnhancedExperimentConfig] = []
    
    for strategy in strategies:
        for attack in config_mgr.get_valid_attacks():
            param_list = config_mgr.get_attack_params(attack)
            for params in param_list:
                for dataset in datasets:
                    try:
                        cfg = EnhancedExperimentConfig(
                            strategy=strategy,
                            attack=attack,
                            dataset=dataset,
                            attack_params=params,
                            strategy_params=config_mgr.get_strategy_params(strategy),
                            num_rounds=config_mgr.defaults.num_rounds,
                            num_clients=config_mgr.defaults.num_clients,
                        )
                        
                        # Check for warnings
                        warnings = cfg.validate_consistency()
                        if warnings:
                            logger.warning(f"Configuration {cfg.get_experiment_id()} has warnings: {warnings}")
                        
                        configs.append(cfg)
                        
                    except ValueError as e:
                        logger.error(f"Invalid configuration for {strategy}-{attack}-{dataset}: {e}")
                        continue
    
    logger.info(f"Created {len(configs)} valid configurations")
    return configs


def main():
    """Funzione principale con interfaccia CLI migliorata."""
    parser = argparse.ArgumentParser(
        description="Enhanced Experiment Runner for Federated Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments sequentially with checkpoint support
  python enhanced_experiment_runner.py --num-runs 10 --sequential
  
  # Resume from checkpoint
  python enhanced_experiment_runner.py --resume
  
  # Run in parallel mode
  python enhanced_experiment_runner.py --num-runs 5 --parallel --max-parallel 2
  
  # Test mode with minimal configurations
  python enhanced_experiment_runner.py --test-mode --num-runs 2
        """
    )
    
    parser.add_argument("--num-runs", type=int, default=1,
                       help="Number of runs per configuration")
    parser.add_argument("--results-dir", type=str, default="enhanced_results",
                       help="Directory for results")
    parser.add_argument("--config-file", type=str, 
                       help="Configuration file path")
    parser.add_argument("--sequential", action="store_true",
                       help="Run experiments sequentially (default)")
    parser.add_argument("--parallel", action="store_true",
                       help="Run experiments in parallel")
    parser.add_argument("--max-parallel", type=int, default=1,
                       help="Maximum parallel experiments")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--test-mode", action="store_true",
                       help="Test mode with reduced configurations")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    config_file = Path(args.config_file) if args.config_file else None
    config_manager = EnhancedConfigManager(config_file)
    
    # Override parallel settings from command line
    if args.max_parallel:
        config_manager.system.max_parallel_experiments = args.max_parallel
    
    # Create runner
    with EnhancedExperimentRunner(
        results_dir=args.results_dir,
        config_manager=config_manager
    ) as runner:
        
        # Create configurations
        if args.test_mode:
            configs = [
                EnhancedExperimentConfig("fedavg", "none", "MNIST", num_rounds=3, num_clients=5),
                EnhancedExperimentConfig("fedprox", "noise", "MNIST", 
                                       attack_params={"noise_std": 0.1, "noise_fraction": 0.3},
                                       strategy_params={"proximal_mu": 0.01},
                                       num_rounds=3, num_clients=5)
            ]
            logger.info("Running in test mode with minimal configurations")
        else:
            configs = create_enhanced_configurations()
        
        logger.info(f"Created {len(configs)} experiment configurations")
        
        try:
            # Run experiments
            if args.parallel:
                logger.info("Running experiments in parallel mode")
                results = runner.run_experiments_parallel(configs, args.num_runs)
            else:
                logger.info("Running experiments sequentially")
                results = runner.run_experiments_sequential(configs, args.num_runs, resume=args.resume)
            
            # Save final results
            runner.save_results(intermediate=False)
            
            # Generate final report
            final_report = runner.generate_final_report()
            
            # Print summary
            logger.info("=== EXPERIMENT COMPLETION SUMMARY ===")
            summary = final_report["experiment_summary"]
            logger.info(f"Total configurations: {summary['total_configurations']}")
            logger.info(f"Completed experiments: {summary['completed_experiments']}")
            logger.info(f"Failed experiments: {summary['failed_experiments']}")
            logger.info(f"Success rate: {summary['success_rate']*100:.1f}%")
            
            if final_report["system_metrics"]["total_experiments"] > 0:
                sys_metrics = final_report["system_metrics"]
                logger.info(f"Average duration: {sys_metrics['average_duration']:.1f}s")
                logger.info(f"Average CPU usage: {sys_metrics['average_cpu_usage']:.1f}%")
                logger.info(f"Average memory usage: {sys_metrics['average_memory_usage']:.1f}%")
            
            logger.info(f"Results saved to: {runner.results_dir}")
            logger.info("=====================================")
            
        except KeyboardInterrupt:
            logger.info("Experiment run interrupted by user")
            runner.save_results(intermediate=True)
        except Exception as e:
            logger.error(f"Unexpected error during experiment run: {e}")
            logger.error(traceback.format_exc())
            runner.save_results(intermediate=True)


if __name__ == "__main__":
    main()
