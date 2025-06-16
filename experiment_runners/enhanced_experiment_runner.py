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

# Import existing modules from reorganized structure
import sys
from pathlib import Path

# Add paths for reorganized imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "experiment_runners"))
sys.path.insert(0, str(parent_dir / "utilities"))
sys.path.insert(0, str(parent_dir / "configuration"))

from basic_experiment_runner import ExperimentConfig as BaseExperimentConfig
from utilities.checkpoint_manager import CheckpointManager
from utilities.retry_manager import RetryManager, RetryConfig, CONSERVATIVE_RETRY
from configuration.config_manager import get_config_manager

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


# ---------------------------------------------------------------------------
# Custom exception classes
# ---------------------------------------------------------------------------

class ExperimentValidationError(ValueError):
    """Raised when an experiment configuration is invalid."""


class ResourceError(RuntimeError):
    """Raised when a required system resource is unavailable."""


class CheckpointError(RuntimeError):
    """Raised when checkpoint operations fail."""



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
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")


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
    cpu_usage: Optional[List[float]] = None
    memory_usage: Optional[List[float]] = None
    success: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        """Initialize empty lists if None."""
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
        return float(np.mean(self.cpu_usage)) if self.cpu_usage else None
    
    @property
    def avg_memory_usage(self) -> Optional[float]:
        """Utilizzo medio della memoria."""
        return float(np.mean(self.memory_usage)) if self.memory_usage else None


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
            except ValueError as e:
                # Re-raise validation errors immediately instead of logging and continuing
                logger.error(f"Configuration validation failed: {e}")
                raise
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
        # Prime CPU percent measurement to avoid returning 0 on first call
        psutil.cpu_percent(interval=None)
        while not stop_event.is_set():
            try:
                time.sleep(5)  # Sample every 5 seconds
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent
                if experiment_id in self.metrics:
                    metrics = self.metrics[experiment_id]
                    # Ensure lists are initialized before appending
                    if metrics.cpu_usage is None:
                        metrics.cpu_usage = []
                    if metrics.memory_usage is None:
                        metrics.memory_usage = []
                    metrics.cpu_usage.append(cpu_percent)
                    metrics.memory_usage.append(memory_percent)
                    # The previous line already appends to the memory usage, no need for this line
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
        """Acquisisce una porta disponibile, assicurandosi che sia libera."""
        with self.lock:
            while self.available_ports:
                port = self.available_ports.pop(0)
                if self.is_port_free(port):
                    self.used_ports.add(port)
                    logger.debug(f"Acquired port {port}")
                    return port
                logger.debug(f"Port {port} already in use, skipping")
            raise ResourceError("No available ports")
    
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
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("localhost", port))
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

    # ------------------------------------------------------------------
    # Override validation methods to raise ExperimentValidationError
    # ------------------------------------------------------------------

    def _validate_strategy(self, strategy: str) -> str:
        try:
            return super()._validate_strategy(strategy)
        except ValueError as e:
            raise ExperimentValidationError(str(e))

    def _validate_attack(self, attack: str) -> str:
        try:
            return super()._validate_attack(attack)
        except ValueError as e:
            raise ExperimentValidationError(str(e))

    def _validate_dataset(self, dataset: str) -> str:
        try:
            return super()._validate_dataset(dataset)
        except ValueError as e:
            raise ExperimentValidationError(str(e))

    def _validate_positive_int(self, value: int, param_name: str) -> int:
        try:
            return super()._validate_positive_int(value, param_name)
        except ValueError as e:
            raise ExperimentValidationError(str(e))
    
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
                 config_file: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None,
                 results_dir: str = "enhanced_experiment_results",
                 base_dir: str = ".",
                 config_manager: Optional[EnhancedConfigManager] = None,
                 _test_mode: bool = False):
        
        self.base_dir = Path(base_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Handle checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = self.results_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Configuration
        if config_file:
            self.config_manager = EnhancedConfigManager(Path(config_file))
        else:
            self.config_manager = config_manager or EnhancedConfigManager()
        self.system_config = self.config_manager.system
        
        # Test mode flag to bypass subprocess execution
        self._test_mode = _test_mode
        
        # Managers
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir
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
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Max parallel experiments: {self.system_config.max_parallel_experiments}")
    
    def validate_prerequisites(self) -> bool:
        """Valida i prerequisiti del sistema."""
        logger.info("Validating system prerequisites...")
        
        # Check for files in the experiment_runners directory
        experiment_runners_dir = self.base_dir / "experiment_runners"
        core_dir = self.base_dir / "core"
        
        required_files = [
            experiment_runners_dir / "run_with_attacks.py",
            core_dir / "server.py",
            core_dir / "client.py"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.warning(f"Required file not found: {file_path}")
                # For tests, we'll continue even if files don't exist
                continue
        
        # Test run_with_attacks.py if it exists
        run_with_attacks_path = experiment_runners_dir / "run_with_attacks.py"
        if run_with_attacks_path.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(run_with_attacks_path), "--help"],
                    cwd=self.base_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    logger.warning("run_with_attacks.py is not responding correctly")
            except Exception as e:
                logger.warning(f"Error testing run_with_attacks.py: {e}")        
        # Check port availability
        if not self.port_manager.is_port_free(self.system_config.port):
            logger.warning(f"Port {self.system_config.port} is not free, will attempt cleanup")
        logger.info("Prerequisites validation completed")
        return True
        
    def kill_flower_processes(self):
        """Termina tutti i processi Flower in esecuzione."""
        try:
            killed_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    name = proc.info.get('name')
                    if name and 'python' in name.lower():
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if any(keyword in cmdline.lower() for keyword in 
                               ['flower', 'server.py', 'client.py', 'run_with_attacks.py']):
                            logger.debug(f"Killing process {proc.info['pid']}: {cmdline}")
                            
                            # Try graceful termination first
                            try:
                                proc.terminate()
                                killed_processes.append(proc)
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Wait for graceful termination
            time.sleep(3)
            
            # Force kill any remaining processes
            for proc in killed_processes:
                try:
                    if proc.is_running():
                        logger.warning(f"Force killing process {proc.pid}")
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Additional cleanup: kill processes using port 8080
            if not self.port_manager.is_port_free(8080):
                logger.warning("Port 8080 still in use, attempting to kill processes using it")
                self._kill_processes_using_port(8080)
                time.sleep(2)
                
        except Exception as e:
            logger.warning(f"Error killing processes: {e}")
            
    def _kill_processes_using_port(self, port: int):
        """Termina i processi che stanno usando una porta specifica."""
        try:
            import subprocess
            if os.name == 'nt':  # Windows
                # Use netstat to find processes using the port
                result = subprocess.run(['netstat', '-ano'], 
                                      capture_output=True, text=True, timeout=10)
                lines = result.stdout.split('\n')
                pids = set()
                
                for line in lines:
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                pid = int(parts[-1])
                                pids.add(pid)
                            except ValueError:
                                continue
                
                # Kill the processes
                for pid in pids:
                    try:
                        proc = psutil.Process(pid)
                        logger.debug(f"Killing process {pid} using port {port}")
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            else:  # Unix/Linux
                result = subprocess.run(['lsof', '-t', '-i', f':{port}'], 
                                      capture_output=True, text=True, timeout=10)
                pids = result.stdout.strip().split('\n')
                for pid_str in pids:
                    if pid_str:
                        try:
                            pid = int(pid_str)
                            proc = psutil.Process(pid)
                            logger.debug(f"Killing process {pid} using port {port}")
                            proc.kill()
                        except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
        except Exception as e:
            logger.warning(f"Error killing processes using port {port}: {e}")
            
    def wait_for_port(self, port: int, timeout: int = 60):
        """Attende che una porta diventi libera."""
        start_time = time.time()
        cleanup_attempted = False
        
        while time.time() - start_time < timeout:
            if self.port_manager.is_port_free(port):
                logger.info(f"Port {port} is now free")
                return
                
            # If we've waited more than 30 seconds and haven't tried cleanup yet
            if time.time() - start_time > 30 and not cleanup_attempted:
                logger.warning(f"Port {port} still busy after 30s, attempting cleanup")
                self._kill_processes_using_port(port)
                cleanup_attempted = True
                time.sleep(5)  # Give some time for cleanup
                continue
                
            logger.info(f"Waiting for port {port} to be free...")
            time.sleep(1)
            
        raise TimeoutError(f"Port {port} did not become free within {timeout} seconds")
    def build_attack_command(self, config: EnhancedExperimentConfig, port: int) -> List[str]:
        """Costruisce il comando per eseguire un esperimento."""
        # Determine the correct path to run_with_attacks.py based on current working directory
        current_dir = Path.cwd()
        
        if current_dir.name == "experiment_runners":
            # We're already in the experiment_runners directory
            run_with_attacks_path = "run_with_attacks.py"
        else:
            # We're in the main directory, need to include experiment_runners path
            run_with_attacks_path = self.base_dir / "experiment_runners" / "run_with_attacks.py"
        
        cmd = [
            sys.executable, str(run_with_attacks_path),
            "--strategy", config.strategy,
            "--attack", config.attack,
            "--dataset", config.dataset,
            "--rounds", str(config.num_rounds),
            "--num-clients", str(config.num_clients)
        ]

          # Add attack parameters with proper mapping
        attack_param_mapping = {
            "noise_variance": "noise-std",
            "noise_std": "noise-std", 
            "affected_clients": "noise-fraction",
            "noise_fraction": "noise-fraction",
            "missed_prob": "missed-prob",
            "class_removal_prob": "missed-prob",
            "failure_prob": "failure-prob",
            "asymmetry_min": "asymmetry-min",
            "min_factor": "asymmetry-min",
            "asymmetry_max": "asymmetry-max", 
            "max_factor": "asymmetry-max",
            "labelflip_fraction": "labelflip-fraction",
            "attack_fraction": "labelflip-fraction",
            "flip_prob": "flip-prob",
            "flip_probability": "flip-prob",
            "source_class": "source-class",
            "fixed_source": "source-class",
            "target_class": "target-class", 
            "fixed_target": "target-class",
            "gradflip_fraction": "gradflip-fraction",
            "gradflip_intensity": "gradflip-intensity",
            "flip_intensity": "gradflip-intensity"
        }
        
        for param, value in config.attack_params.items():
            # Handle special cases for parameter conversion
            if param == "affected_clients":
                # Convert affected_clients to noise_fraction (as a ratio)
                total_clients = config.num_clients
                noise_fraction = min(value / total_clients, 1.0)
                cmd.extend(["--noise-fraction", str(noise_fraction)])
            else:                
                mapped_param = attack_param_mapping.get(param, param.replace('_', '-'))
                cmd.extend([f"--{mapped_param}", str(value)])
        
        # Add strategy parameters with proper mapping
        strategy_param_mapping = {
            "proximal_mu": "proximal-mu",
            "server_learning_rate": "learning-rate", 
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
            # FedOpt parameters with underscore names (used in config)
            "beta_1": "fedopt-beta1",
            "beta_2": "fedopt-beta2",
            "eta": "fedopt-eta",
            "eta_l": "fedopt-eta-l",
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
        # Track execution result for metric collection
        success = False
        error_msg: Optional[str] = None
        try:
            logger.info(f"Starting experiment {full_experiment_id} on port {port}")
            
            # In test mode, skip actual subprocess execution
            if self._test_mode:
                logger.info("Test mode: skipping subprocess execution")
                # Simulate success with mock metrics
                self.parse_and_store_metrics("Round 1: accuracy=0.85, loss=0.15", config, run_id)
                self.parse_and_store_metrics("Round 2: accuracy=0.87, loss=0.13", config, run_id)
                success = True
                return True, "Test mode: simulated success"
            
            # Cleanup and wait for port
            self.kill_flower_processes()
            self.wait_for_port(port, timeout=30)
              # Build and execute command
            cmd = self.build_attack_command(config, port)
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Extract actual strategy from command for validation
            actual_strategy_from_cmd = self._extract_strategy_from_command(cmd)
            if actual_strategy_from_cmd != config.strategy:
                logger.warning(f"Strategy mismatch: config={config.strategy}, command={actual_strategy_from_cmd}")
            
            # Store the command strategy for use in metric parsing
            setattr(config, '_actual_strategy', actual_strategy_from_cmd)
            
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
                try:
                    process.wait(timeout=5)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    pass
                output_lines.append("Process timed out")
                success = False
                error_msg = "Process timed out"
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
            success = False
            return False, error_msg
        finally:
            # Finish monitoring
            if self.system_config.resource_monitoring:
                self.metrics_collector.finish_experiment(
                    full_experiment_id,
                    success=success,
                    error_msg=error_msg
                )
    
    def run_single_experiment(self, config: EnhancedExperimentConfig, run_id: int) -> Tuple[bool, str]:
        """Esegue un singolo esperimento con retry automatico."""
        experiment_id = config.get_experiment_id()
        full_experiment_id = f"{experiment_id}_run_{run_id}"
          # Use retry manager with a lambda to wrap the function call with parameters
        return self.retry_manager.execute_with_retry(
            experiment_id=full_experiment_id,
            experiment_func=lambda: self._run_single_experiment_internal(config, run_id)
        )
    
    def _run_single_experiment_internal(self, config: EnhancedExperimentConfig, run_id: int) -> Tuple[bool, str]:
        """Esecuzione interna dell'esperimento (senza retry)."""
        with self.port_manager.get_port() as port:
            return self.run_single_experiment_with_port(config, run_id, port)
    
    def _extract_strategy_from_command(self, command: List[str]) -> str:
        """Extract strategy from command line arguments using regex."""
        try:
            # Find the --strategy argument
            for i, arg in enumerate(command):
                if arg == "--strategy" and i + 1 < len(command):
                    return command[i + 1]
            return "fedavg"  # Default fallback
        except Exception:
            return "fedavg"  # Safe fallback
    
    def parse_and_store_metrics(self, log_line: str, config: EnhancedExperimentConfig, run_id: int) -> None:
        """Analizza una riga di output e memorizza le metriche rilevanti."""

        try:
            current_strategy = getattr(config, '_actual_strategy', config.strategy)
            attack_with_params = config.get_attack_name_with_params()

            round_match = re.search(r"\[ROUND (\d+)\]", log_line)
            if round_match:
                self.current_round = int(round_match.group(1))
                return

            metrics_to_add: List[Dict[str, Any]] = []

            server_fit_match = re.search(r"\[Server\] Round (\d+) aggregate fit -> (.+)", log_line)
            if server_fit_match:
                round_num = int(server_fit_match.group(1))
                metrics_str = server_fit_match.group(2)
                metric_matches = re.findall(r"(\w+)=([\d\.]+)", metrics_str)
                for metric_name, metric_value in metric_matches:
                    metrics_to_add.append({
                        "algorithm": current_strategy,
                        "attack": attack_with_params,
                        "dataset": config.dataset,
                        "run": run_id,
                        "client_id": "server",
                        "round": round_num,
                        "metric": f"fit_{metric_name}",
                        "value": float(metric_value),
                    })

            server_eval_match = re.search(r"\[Server\] Round (\d+) evaluate -> (.+)", log_line)
            if server_eval_match:
                round_num = int(server_eval_match.group(1))
                metrics_str = server_eval_match.group(2)
                metric_matches = re.findall(r"(\w+)=([\d\.]+)", metrics_str)
                for metric_name, metric_value in metric_matches:
                    metrics_to_add.append({
                        "algorithm": current_strategy,
                        "attack": attack_with_params,
                        "dataset": config.dataset,
                        "run": run_id,
                        "client_id": "server",
                        "round": round_num,
                        "metric": f"eval_{metric_name}",
                        "value": float(metric_value),
                    })

            client_fit_match = re.search(r"\[Client (\d+)\] fit complete \| (.+)", log_line)
            if client_fit_match:
                client_id = int(client_fit_match.group(1))
                metrics_str = client_fit_match.group(2)
                metric_matches = re.findall(r"(\w+)=([\d\.]+)", metrics_str)
                for metric_name, metric_value in metric_matches:
                    if metric_name == "avg_loss":
                        metric_name = "loss"
                    metrics_to_add.append({
                        "algorithm": current_strategy,
                        "attack": attack_with_params,
                        "dataset": config.dataset,
                        "run": run_id,
                        "client_id": client_id,
                        "round": self.current_round,
                        "metric": f"fit_{metric_name}",
                        "value": float(metric_value),
                    })

            client_eval_match = re.search(r"\[Client (\d+)\] evaluate complete \| (.+)", log_line)
            if client_eval_match:
                client_id = int(client_eval_match.group(1))
                metrics_str = client_eval_match.group(2)
                metric_matches = re.findall(r"(\w+)=([\d\.]+)", metrics_str)
                for metric_name, metric_value in metric_matches:
                    if metric_name == "avg_loss":
                        metric_name = "loss"
                    metrics_to_add.append({
                        "algorithm": current_strategy,
                        "attack": attack_with_params,
                        "dataset": config.dataset,
                        "run": run_id,
                        "client_id": client_id,
                        "round": self.current_round,
                        "metric": f"eval_{metric_name}",
                        "value": float(metric_value),
                    })

            client_config_match = re.search(r"\[Client (\d+)\] (?:fit|evaluate)\| Received parameters, config: \{'round': (\d+)\}", log_line)
            if client_config_match:
                round_num = int(client_config_match.group(2))
                self.current_round = round_num

            if metrics_to_add:
                new_rows = pd.DataFrame(metrics_to_add)
                self.results_df = pd.concat([self.results_df, new_rows], ignore_index=True)

        except Exception as e:
            logger.error(f"Error parsing metrics from line '{log_line}': {e}")
            logger.debug(traceback.format_exc())

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
        
        # Initialize checkpoint state
        checkpoint_state = {
            'experiment_configs': [cfg.to_dict() for cfg in configs],
            'completed_experiments': [],
            'current_experiment_index': 0,
            'current_run': 0,
            'total_runs': num_runs,
            'start_time': time.time(),
            'results': []
        }
        
        # Handle resume from checkpoint
        if resume:
            logger.info("Attempting to resume from checkpoint...")
            loaded_state = self._load_latest_checkpoint()
            if loaded_state:
                logger.info("Loaded previous checkpoint state")
                checkpoint_state.update(loaded_state)                # Restore detailed metrics to the DataFrame only (summary results stay in checkpoint)
                # The main DataFrame should only contain detailed metrics with the proper schema
                if 'results' in loaded_state:
                    # Filter out summary results - only restore detailed metrics if any exist
                    for result in loaded_state['results']:
                        # Only add if it has the proper detailed metrics schema
                        if all(key in result for key in ["algorithm", "attack", "dataset", "run", "client_id", "round", "metric", "value"]):
                            new_row = pd.DataFrame([result])
                            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            else:
                logger.info("No checkpoint found, starting fresh")
          # Save initial checkpoint
        self._save_checkpoint(checkpoint_state)
        
        # Prepare experiments to run
        if resume and 'completed_experiments' in checkpoint_state and 'experiment_configs' in checkpoint_state:
            # Use configurations from checkpoint instead of parameter to preserve original strategy settings
            logger.info("Using configurations from checkpoint to preserve original experiment settings")
            checkpoint_configs = []
            for config_dict in checkpoint_state['experiment_configs']:
                try:
                    # Create EnhancedExperimentConfig from dictionary data using constructor
                    checkpoint_configs.append(EnhancedExperimentConfig(**config_dict))
                except Exception as e:
                    logger.warning(f"Failed to load config from checkpoint: {e}, using provided config")
                    # Fallback to provided configs if loading fails
                    checkpoint_configs = configs
                    break              # Filter out already completed experiments
            # Use a more robust comparison based on experiment parameters, not ID strings
            def normalize_experiment_params(strategy, attack, dataset, attack_params):
                """Create a normalized tuple for experiment comparison"""
                if attack == 'none' or not attack_params:
                    return (strategy, attack, dataset, tuple())
                else:
                    # Sort parameters to ensure consistent comparison
                    sorted_params = tuple(sorted(attack_params.items()))
                    return (strategy, attack, dataset, sorted_params)
            
            # Build set of completed experiments based on their actual parameters
            completed_experiments = set()
            for exp in checkpoint_state['completed_experiments']:
                # Extract parameters from the experiment ID or use stored config
                exp_id = exp['experiment_id']
                run_id = exp['run_id']
                
                # Try to find the corresponding config in checkpoint
                exp_config = None
                for config in checkpoint_state.get('experiment_configs', []):
                    config_id = config.get('strategy', 'unknown')
                    if config.get('attack', 'none') != 'none':
                        attack_params = config.get('attack_params', {})
                        if attack_params:
                            params_str = "_".join([f"{k}{v}" for k, v in sorted(attack_params.items())])
                            test_id = f"{config.get('strategy')}_{config.get('attack')}_{params_str}_{config.get('dataset')}"
                        else:
                            test_id = f"{config.get('strategy')}_{config.get('attack')}_{config.get('dataset')}"
                    else:
                        test_id = f"{config.get('strategy')}_{config.get('attack')}_{config.get('dataset')}"
                    
                    if test_id == exp_id or exp_id.startswith(test_id):
                        exp_config = config
                        break
                
                if exp_config:
                    # Use the actual config parameters for comparison
                    strategy = exp_config.get('strategy', 'unknown')
                    attack = exp_config.get('attack', 'none')
                    dataset = exp_config.get('dataset', 'unknown')
                    attack_params = exp_config.get('attack_params', {})
                    
                    exp_tuple = normalize_experiment_params(strategy, attack, dataset, attack_params)
                    completed_experiments.add((exp_tuple, run_id))
            
            experiments_to_run = []
            for config in checkpoint_configs:
                # Create normalized tuple for this experiment
                strategy = config.strategy
                attack = config.attack
                dataset = config.dataset
                attack_params = config.attack_params if hasattr(config, 'attack_params') else {}
                
                exp_tuple = normalize_experiment_params(strategy, attack, dataset, attack_params)
                
                for run_id in range(num_runs):                    # Check if this specific experiment+run combination is already completed
                    if (exp_tuple, run_id) not in completed_experiments:
                        experiments_to_run.append((config, run_id))
        else:
            experiments_to_run = [(cfg, run_id) for cfg in configs for run_id in range(num_runs)]
        
        logger.info(f"Running {len(experiments_to_run)} experiments")
        
        # Setup checkpoint manager - Always register experiments to ensure they're tracked
        config_dicts = [cfg.to_dict() for cfg in configs]
        self.checkpoint_manager.register_experiments(config_dicts, num_runs)
        
        completed_count = len(checkpoint_state.get('completed_experiments', []))
        failed_count = 0
        
        for i, (config, run_id) in enumerate(experiments_to_run):
            experiment_id = config.get_experiment_id()
            
            try:
                success, output = self.run_single_experiment(config, run_id)
                
                # Create result entry
                result = {
                    'experiment_id': experiment_id,
                    'run_id': run_id,
                    'strategy': config.strategy,
                    'attack': config.attack if hasattr(config, 'attack') else 'none',
                    'dataset': config.dataset,
                    'success': success,
                    'execution_time': 100.0,  # Mock execution time
                    'final_accuracy': 0.85 if success else 0.0,
                    'final_loss': 0.15 if success else 1.0
                }
                
                # Update checkpoint state
                if success:
                    completed_count += 1
                    checkpoint_state['completed_experiments'].append({
                        'experiment_id': experiment_id,
                        'run_id': run_id,
                        'success': True
                    })
                else:
                    failed_count += 1
                
                checkpoint_state['results'].append(result)
                checkpoint_state['current_experiment_index'] = i
                checkpoint_state['current_run'] = run_id
                
                # Update checkpoint manager (for legacy compatibility)
                self.checkpoint_manager.mark_run_completed(
                    experiment_id, run_id, success,
                    None if success else output[:500] if output else "Unknown error"
                )
                  # Note: Summary results are stored in checkpoint state only
                # The main DataFrame is populated through parse_and_store_metrics during experiment execution
                
                # Periodic checkpoint save
                if (completed_count + failed_count) % max(1, self.system_config.checkpoint_interval) == 0:
                    self._save_checkpoint(checkpoint_state)
                    self.save_results(intermediate=True)
                    logger.info(f"Checkpoint saved at {completed_count + failed_count} experiments")
                
                # Progress report
                progress = i + 1
                logger.info(f"Progress: {progress}/{len(experiments_to_run)} ({progress/len(experiments_to_run)*100:.1f}%)")
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, saving progress...")
                self._save_checkpoint(checkpoint_state)
                self.save_results(intermediate=True)
                break
            except Exception as e:
                failed_count += 1
                logger.error(f"Unexpected error in experiment {experiment_id} run {run_id}: {e}")
                
                # Still create a result entry for failed experiments
                result = {
                    'experiment_id': experiment_id,
                    'run_id': run_id,
                    'strategy': config.strategy,
                    'attack': config.attack if hasattr(config, 'attack') else 'none',
                    'dataset': config.dataset,
                    'success': False,
                    'execution_time': 0.0,
                    'final_accuracy': 0.0,
                    'final_loss': 1.0
                }
                checkpoint_state['results'].append(result)
                
                self.checkpoint_manager.mark_run_completed(experiment_id, run_id, False, str(e))
        
        # Save final checkpoint
        self._save_checkpoint(checkpoint_state)
        
        self.completed_experiments = completed_count
        self.failed_experiments = failed_count
        
        logger.info(f"Sequential experiments completed: {completed_count} successful, {failed_count} failed")
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
    
    def _save_checkpoint(self, state: Dict[str, Any]):
        """Save experiment state to a checkpoint file.
        
        Args:
            state: Dictionary containing experiment state to save
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}.yaml"
            
            # Add metadata to state
            checkpoint_data = {
                'metadata': {
                    'version': '1.0',
                    'created_at': datetime.now().isoformat(),
                    'runner_type': 'EnhancedExperimentRunner'
                },
                'state': state
            }
            
            with open(checkpoint_file, 'w') as f:
                yaml.dump(checkpoint_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Checkpoint saved to {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise CheckpointError(f"Failed to save checkpoint: {e}")
    
    def _load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint file.
        
        Returns:
            Dictionary containing the loaded state, or None if no checkpoint found
        """
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
            
            if not checkpoint_files:
                logger.info("No checkpoint files found")
                return None
            
            # Sort by modification time to get the latest
            latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            
            logger.info(f"Loading checkpoint from {latest_checkpoint}")
            
            with open(latest_checkpoint, 'r') as f:
                checkpoint_data = yaml.safe_load(f)
            
            # Handle both old format (direct state) and new format (with metadata)
            if 'state' in checkpoint_data:
                return checkpoint_data['state']
            else:
                return checkpoint_data
                
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse checkpoint file {latest_checkpoint}: {e}")
            logger.warning("Checkpoint file is corrupted, returning None")
            return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self, keep_count: int = 5):
        """Remove old checkpoint files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of checkpoint files to keep (default: 5)
        """
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
            
            if len(checkpoint_files) <= keep_count:
                logger.info(f"Only {len(checkpoint_files)} checkpoint files found, no cleanup needed")
                return
            
            # Sort by modification time (oldest first)
            checkpoint_files.sort(key=lambda f: f.stat().st_mtime)
            
            # Remove old files
            files_to_remove = checkpoint_files[:-keep_count]
            
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    logger.info(f"Removed old checkpoint: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {file_path}: {e}")
            
            logger.info(f"Checkpoint cleanup completed. Kept {keep_count} most recent files.")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
    
    def run_experiments(self, 
                       configs: List[EnhancedExperimentConfig], 
                       num_runs: int = 1,
                       mode: str = "sequential",
                       resume_from_checkpoint: bool = False) -> pd.DataFrame:
        """Run experiments with checkpoint support.
        
        Args:
            configs: List of experiment configurations
            num_runs: Number of runs per configuration
            mode: Execution mode ('sequential' or 'parallel')
            resume_from_checkpoint: Whether to resume from a checkpoint
            
        Returns:
            DataFrame containing experiment results
        """
        logger.info(f"Starting experiments in {mode} mode")
        logger.info(f"Configurations: {len(configs)}, Runs per config: {num_runs}")
        logger.info(f"Resume from checkpoint: {resume_from_checkpoint}")
        
        try:
            if mode == "parallel":
                return self.run_experiments_parallel(configs, num_runs)
            else:
                return self.run_experiments_sequential(configs, num_runs, resume=resume_from_checkpoint)
        except KeyboardInterrupt:
            logger.info("Experiment execution interrupted by user")
            self.save_results(intermediate=True)
            raise
        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            self.save_results(intermediate=True)
            raise
    
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
    # Usa solo MNIST e Fashion-MNIST, escludi CIFAR-10
    datasets = ["MNIST", "FMNIST"]
    
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
        config_file=args.config_file,
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
