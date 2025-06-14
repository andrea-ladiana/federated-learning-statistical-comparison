"""
Sistema per l'esecuzione sistematica di esperimenti di Federated Learning.

Questo modulo consente di eseguire esperimenti multipli combinando diverse:
- Strategie di aggregazione
- Attacchi con parametri fissi
- Dataset

Raccoglie i risultati in formato long-form pandas per l'analisi successiva.
"""

import pandas as pd
import numpy as np
import subprocess
from collections import deque
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
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExperimentConfig:
    """Configurazione per un singolo esperimento."""
    def __init__(self, 
                 strategy: str,
                 attack: str,
                 dataset: str,
                 attack_params: Optional[Dict[str, Any]] = None,
                 strategy_params: Optional[Dict[str, Any]] = None,
                 num_rounds: int = 10,
                 num_clients: int = 10):
        self.strategy = self._validate_strategy(strategy)
        self.attack = self._validate_attack(attack)
        self.dataset = self._validate_dataset(dataset)
        self.attack_params = attack_params or {}
        self.strategy_params = strategy_params or {}
        self.num_rounds = self._validate_positive_int(num_rounds, "num_rounds")
        self.num_clients = self._validate_positive_int(num_clients, "num_clients")
    
    def _validate_strategy(self, strategy: str) -> str:
        """Valida la strategia di aggregazione."""
        valid_strategies = [
            "fedavg", "fedavgm", "fedprox", "fednova", "scaffold", "fedadam",
            "krum", "trimmedmean", "bulyan", "dasha", "depthfl", "heterofl",
            "fedmeta", "fedper", "fjord", "flanders", "fedopt"
        ]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Valid options: {valid_strategies}")
        return strategy
    
    def _validate_attack(self, attack: str) -> str:
        """Valida il tipo di attacco."""
        valid_attacks = ["none", "noise", "missed", "failure", "asymmetry", "labelflip", "gradflip"]
        if attack not in valid_attacks:
            raise ValueError(f"Invalid attack: {attack}. Valid options: {valid_attacks}")
        return attack
    
    def _validate_dataset(self, dataset: str) -> str:
        """Valida il dataset."""
        valid_datasets = ["MNIST", "FMNIST", "CIFAR10"]
        if dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {dataset}. Valid options: {valid_datasets}")
        return dataset
    
    def _validate_positive_int(self, value: int, param_name: str) -> int:
        """Valida che un valore sia un intero positivo."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{param_name} must be a positive integer, got: {value}")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte la configurazione in dizionario."""
        return {
            'strategy': self.strategy,
            'attack': self.attack,
            'dataset': self.dataset,
            'attack_params': self.attack_params,
            'strategy_params': self.strategy_params,
            'num_rounds': self.num_rounds,
            'num_clients': self.num_clients        }
    
    def get_experiment_id(self) -> str:
        """Genera un ID univoco per l'esperimento."""
        attack_str = f"{self.attack}"
        if self.attack_params:
            params_str = "_".join([f"{k}{v}" for k, v in self.attack_params.items()])
            attack_str += f"_{params_str}"
        
        return f"{self.strategy}_{attack_str}_{self.dataset}"
    
    def get_attack_name_with_params(self) -> str:
        """Genera il nome dell'attacco con i parametri inclusi."""
        attack_str = f"{self.attack}"
        if self.attack_params:
            params_str = "_".join([f"{k}={v}" for k, v in sorted(self.attack_params.items())])
            attack_str += f"_({params_str})"
        return attack_str

class MetricsCollector:
    """Raccoglie metriche dai log dell'esperimento."""
    
    def __init__(self):
        self.client_metrics = []
        self.server_metrics = []
    
    def parse_client_log(self, log_line: str, client_id: int, run_id: int) -> Optional[Dict]:
        """Estrae metriche dai log del client."""
        match = re.search(r'(fit|evaluate) complete\s*\|\s*(.*)', log_line)
        if match:
            metric_type = 'training' if match.group(1) == 'fit' else 'evaluation'
            metrics_str = match.group(2)
            metrics = {}
            for part in metrics_str.split(','):
                if '=' in part:
                    k, v = part.strip().split('=', 1)
                    metrics[k.strip()] = float(v)

            round_match = re.search(r'Round (\d+)', log_line)
            round_num = int(round_match.group(1)) if round_match else 0

            metrics.update({
                'client_id': client_id,
                'run': run_id,
                'round': round_num,
                'metric_type': metric_type,
            })
            return metrics

        return None
    
    def parse_server_log(self, log_line: str, run_id: int) -> Optional[Dict]:
        """Estrae metriche dai log del server."""
        # Pattern per metriche aggregate del server
        match = re.search(r'Round (\d+).*-> (.*)', log_line)
        if match:
            round_num = int(match.group(1))
            metrics_str = match.group(2)
            metrics = {}
            for part in metrics_str.split(','):
                if '=' in part:
                    k, v = part.strip().split('=', 1)
                    metrics[k.strip()] = float(v)
            metrics.update({'run': run_id, 'round': round_num})
            return metrics

        return None

class ExperimentRunner:
    """Gestore principale degli esperimenti."""
    
    def __init__(self, base_dir: str = ".", results_dir: str = "experiment_results", process_timeout: int = 600):
        self.base_dir = Path(base_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.process_timeout = process_timeout
        
        # DataFrame per raccogliere tutti i risultati
        self.results_df = pd.DataFrame(columns=[
            "algorithm", "attack", "dataset", "run", "client_id", 
            "round", "metric", "value"
        ])
        self.collector = MetricsCollector()
        
        # Track current round across log lines
        self.current_round = 0
    
    def is_port_free(self, port: int) -> bool:
        """Verifica se una porta è libera."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False
    
    def wait_for_port(self, port: int, timeout: int = 60):
        """Attende che una porta diventi libera."""
        start_time = time.time()
        while not self.is_port_free(port):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Port {port} still busy after {timeout} seconds")
            time.sleep(1)
    
    def kill_flower_processes(self):
        """Termina tutti i processi Flower in esecuzione."""
        logger.info("Scanning for existing Flower processes...")
        killed_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                cwd = proc.info.get('cwd')
                in_base_dir = cwd and Path(cwd).resolve() == self.base_dir.resolve()
                if (
                    in_base_dir and 'python' in proc.info['name'].lower() and
                    ('server.py' in cmdline or 'client.py' in cmdline or 'run_with_attacks.py' in cmdline)
                ):
                    logger.info(f"Terminating process {proc.info['pid']}: {cmdline}")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                        killed_count += 1
                    except psutil.TimeoutExpired:
                        logger.warning(f"Process {proc.info['pid']} did not terminate, killing...")
                        proc.kill()
                        killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
        
        if killed_count > 0:
            logger.info(f"Terminated {killed_count} processes")
        else:
            logger.info("No existing Flower processes found")
    
    def build_attack_command(self, config: ExperimentConfig) -> List[str]:
        """Costruisce il comando run_with_attacks.py."""
        cmd = [
            sys.executable, "run_with_attacks.py",
            "--num-clients", str(config.num_clients),
            "--rounds", str(config.num_rounds),
            "--dataset", config.dataset,
            "--strategy", config.strategy,
            "--attack", config.attack
        ]
        
        # Aggiungi parametri di attacco
        for param, value in config.attack_params.items():
            cmd.extend([f"--{param.replace('_', '-')}", str(value)])
        
        # Mappatura specifica per i parametri di strategia
        strategy_param_mapping = {
            # FedProx
            "proximal_mu": "proximal-mu",
            # FedAdam
            "learning_rate": "learning-rate",
            # FedAvgM
            "server_momentum": "server-momentum",
            # Krum e Bulyan
            "num_byzantine": "num-byzantine",
            # TrimmedMean
            "beta": "beta",
            # DASHA
            "step_size": "step-size",
            "compressor_coords": "compressor-coords",
            # DepthFL
            "alpha": "alpha",
            "tau": "tau",
            # FLANDERS
            "to_keep": "to-keep",
            # FedOpt - mantenere i nomi specifici
            "fedopt_tau": "fedopt-tau",
            "fedopt_beta1": "fedopt-beta1",
            "fedopt_beta2": "fedopt-beta2",
            "fedopt_eta": "fedopt-eta",
            "fedopt_eta_l": "fedopt-eta-l",
        }
        
        # Aggiungi parametri di strategia con mappatura corretta
        for param, value in config.strategy_params.items():
            mapped_param = strategy_param_mapping.get(param, param.replace('_', '-'))
            cmd.extend([f"--{mapped_param}", str(value)])
        
        return cmd
    
    def run_single_experiment(self, config: ExperimentConfig, run_id: int) -> bool:
        """Esegue un singolo esperimento con logging migliorato."""
        experiment_id = config.get_experiment_id()
        logger.info(f"Starting experiment {experiment_id}, run {run_id}")
        
        # Reset round tracking for new experiment
        self.current_round = 0
        
        try:
            # Assicurati che la porta sia libera
            logger.info("Killing existing Flower processes...")
            self.kill_flower_processes()
            logger.info("Waiting for port 8080 to be free...")
            self.wait_for_port(8080, timeout=30)
            
            # Costruisci e esegui il comando
            cmd = self.build_attack_command(config)
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # CRITICAL FIX: Extract actual strategy from command and store it
            actual_strategy = self._extract_strategy_from_command(cmd)
            setattr(config, '_actual_strategy', actual_strategy)
            
            if actual_strategy != config.strategy:
                logger.warning(f"Strategy mismatch: config={config.strategy}, command={actual_strategy}")
            
            # Esegui l'esperimento con timeout
            logger.info("Starting subprocess...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.base_dir,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            logger.info(f"Process started with PID: {process.pid}")
              # Raccogli l'output in tempo reale (manteniamo solo le ultime 1000 righe)
            output_lines = deque(maxlen=1000)
            line_count = 0
            last_log_time = time.time()
            last_progress_report = time.time()
            
            while True:
                if process.stdout is not None:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        line_stripped = line.strip()
                        output_lines.append(line_stripped)
                        line_count += 1
                        
                        current_time = time.time()
                        
                        # Report progresso ogni 30 secondi
                        if current_time - last_progress_report > 30:
                            logger.info(f"Experiment still running... Processed {line_count} lines")
                            last_progress_report = current_time
                        
                        # Log righe importanti immediatamente
                        if any(keyword in line_stripped.lower() for keyword in 
                              ['round', 'client', 'server', 'accuracy', 'loss', 'error', 'exception', 'failed', 'starting']):
                            logger.info(f"OUTPUT: {line_stripped}")
                          # Log ogni 50 righe con sample dell'output
                        elif line_count % 50 == 0:
                            logger.info(f"Processing line {line_count}: {line_stripped[:100]}...")
                        
                        # Parse metrics in tempo reale
                        self.parse_and_store_metrics(line_stripped, config, run_id)
                else:
                    break
            
            # Attendi che il processo finisca
            logger.info("Waiting for process to complete...")
            return_code = process.wait(timeout=self.process_timeout)
            
            logger.info(f"Process completed with return code: {return_code}")
            logger.info(f"Total output lines processed: {line_count}")
            
            if return_code == 0:
                logger.info(f"Experiment {experiment_id}, run {run_id} completed successfully")
                return True
            else:
                logger.error(f"Experiment {experiment_id}, run {run_id} failed with return code {return_code}")
                # Log ultimi 10 righe di output per debug
                if output_lines:
                    logger.error("Last 10 lines of output:")
                    for line in list(output_lines)[-10:]:
                        logger.error(f"  {line}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(
                f"Experiment {experiment_id}, run {run_id} timed out after {self.process_timeout} seconds"
            )
            process.kill()            # Log ultimi 10 righe di output per debug
            if output_lines:
                logger.error("Last 10 lines before timeout:")
                for line in list(output_lines)[-10:]:
                    logger.error(f"  {line}")
            return False
        except Exception as e:
            logger.error(f"Error in experiment {experiment_id}, run {run_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            # Cleanup
            logger.info("Cleaning up processes...")
            self.kill_flower_processes()
            time.sleep(2)  # Grace period
            # The method must return a bool, but since we're in finally block
            # the actual return value comes from the try/except blocks above

    def _extract_strategy_from_command(self, command: List[str]) -> str:
        """Extract strategy from command line arguments."""
        try:
            # Find the --strategy argument
            for i, arg in enumerate(command):
                if arg == "--strategy" and i + 1 < len(command):
                    return command[i + 1]
            return "fedavg"  # Default fallback
        except Exception:
            return "fedavg"  # Safe fallback
    
    def parse_and_store_metrics(self, log_line: str, config: ExperimentConfig, run_id: int):
        """Analizza e memorizza le metriche dai log con pattern migliorati."""
        # CRITICAL FIX: Use the strategy extracted from command, not from config
        # This ensures we store the actual strategy that was executed
        if hasattr(config, '_actual_strategy'):
            actual_strategy = getattr(config, '_actual_strategy')
        else:
            # Fallback to config strategy if command extraction failed
            actual_strategy = config.strategy
        
        # Use attack name with parameters
        attack_name = config.get_attack_name_with_params()
        
        # Buffer per raccogliere metriche prima di aggiungerle al DataFrame
        metrics_to_add = []
        
        # Extract round number and update instance state
        round_patterns = [
            r'\[ROUND (\d+)\]',  # [ROUND 1]
            r'Round (\d+)',      # Round 1
        ]
        
        for pattern in round_patterns:
            round_match = re.search(pattern, log_line, re.IGNORECASE)
            if round_match:
                self.current_round = int(round_match.group(1))
                logger.debug(f"Updated current round to: {self.current_round}")
                break
        
        client_match = re.search(r'\[Client (\d+)\]\s+(fit|evaluate) complete\s*\|\s*(.*)', log_line)
        if client_match:
            client_id = int(client_match.group(1))
            phase = client_match.group(2)
            metrics_str = client_match.group(3)
            metric_prefix = "" if phase == "fit" else "eval_"
            pairs = [m.strip() for m in metrics_str.split(',') if '=' in m]
            for pair in pairs:
                key, val = pair.split('=', 1)
                metrics_to_add.append({
                    "algorithm": actual_strategy,
                    "attack": attack_name,
                    "dataset": config.dataset,
                    "run": run_id,
                    "client_id": client_id,
                    "round": self.current_round,
                    "metric": f"{metric_prefix}{key.strip()}",
                    "value": float(val)
                })

        server_line_match = re.search(r'\[Server\] Round (\d+) .*-> (.*)', log_line)
        if server_line_match:
            round_num = int(server_line_match.group(1))
            metrics_str = server_line_match.group(2)
            pairs = [m.strip() for m in metrics_str.split(',') if '=' in m]
            for pair in pairs:
                key, val = pair.split('=', 1)
                metrics_to_add.append({
                    "algorithm": actual_strategy,
                    "attack": attack_name,
                    "dataset": config.dataset,
                    "run": run_id,
                    "client_id": -1,
                    "round": round_num,
                    "metric": f"server_{key.strip()}",
                    "value": float(val)
                })
        
        # Pattern 3: Server aggregate accuracy - {'accuracy': [(1, 0.334), (2, 0.7272), (3, 0.8848)]
        server_accuracy_match = re.search(r"'accuracy':\s*\[([^\]]+)\]", log_line)
        if server_accuracy_match:
            # Parse the tuples in the list
            tuples_str = server_accuracy_match.group(1)
            tuple_matches = re.findall(r'\((\d+),\s*([0-9.]+)\)', tuples_str)
            for round_num, accuracy in tuple_matches:                metrics_to_add.append({
                    "algorithm": actual_strategy,
                    "attack": attack_name,
                    "dataset": config.dataset,
                    "run": run_id,
                    "client_id": -1,  # -1 indicates server aggregate
                    "round": int(round_num),
                    "metric": "server_accuracy",
                    "value": float(accuracy)
                })
        
        # Pattern 4: Server aggregate loss - {'loss': [(1, 1.8629), (2, 0.9087), (3, 0.4353)]
        server_loss_match = re.search(r"'loss':\s*\[([^\]]+)\]", log_line)
        if server_loss_match:
            # Parse the tuples in the list
            tuples_str = server_loss_match.group(1)
            tuple_matches = re.findall(r'\((\d+),\s*([0-9.]+)\)', tuples_str)
            for round_num, loss in tuple_matches:                metrics_to_add.append({
                    "algorithm": actual_strategy,
                    "attack": attack_name,
                    "dataset": config.dataset,
                    "run": run_id,
                    "client_id": -1,  # -1 indicates server aggregate
                    "round": int(round_num),
                    "metric": "server_loss",
                    "value": float(loss)                })
        
        # Add all collected metrics to DataFrame
        if metrics_to_add:
            try:
                new_rows = pd.DataFrame(metrics_to_add)
                # Check if results_df is empty and handle accordingly
                if self.results_df.empty:
                    self.results_df = new_rows
                else:
                    self.results_df = pd.concat([self.results_df, new_rows], ignore_index=True)
                
                # Log debug info for first few metrics
                if len(self.results_df) <= 20:
                    logger.info(f"Added {len(metrics_to_add)} metrics from line: {log_line[:100]}...")
                    
            except Exception as e:
                logger.warning(f"Failed to add metrics: {e}")
                # Fallback: add one metric at a time
                for metric in metrics_to_add:
                    try:
                        if self.results_df.empty:
                            self.results_df = pd.DataFrame([metric])
                        else:
                            self.results_df = pd.concat([self.results_df, pd.DataFrame([metric])], ignore_index=True)
                    except Exception as e2:
                        logger.warning(f"Failed to add single metric: {e2}")
    
    def run_experiments(self, 
                       configs: List[ExperimentConfig], 
                       num_runs: int = 10,
                       max_parallel: int = 1) -> pd.DataFrame:
        """Esegue tutti gli esperimenti configurati."""
        total_experiments = len(configs) * num_runs
        completed = 0
        failed = 0
        
        logger.info(f"Starting {total_experiments} experiments ({len(configs)} configs × {num_runs} runs)")
        
        # Verifica prerequisiti
        logger.info("Checking prerequisites...")
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed. Aborting experiments.")
            return self.results_df
        logger.info("Prerequisites check passed.")
        
        # Esegui sequenzialmente per evitare conflitti di porta
        for config in configs:
            experiment_id = config.get_experiment_id()
            logger.info(f"Running configuration: {experiment_id}")
            
            for run in range(num_runs):
                success = self.run_single_experiment(config, run)
                if success:
                    completed += 1
                    logger.info(f"Progress: {completed}/{total_experiments} completed")
                else:
                    failed += 1
                    logger.error(f"Progress: {completed}/{total_experiments} completed, {failed} failed")
                
                # Salva risultati intermedi
                if completed % 5 == 0:  # Salva ogni 5 esperimenti
                    self.save_results(intermediate=True)
        
        logger.info(f"All experiments completed: {completed} successful, {failed} failed")
        return self.results_df
    
    def save_results(self, intermediate: bool = False):
        """Salva i risultati su file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if intermediate:
            filename = f"intermediate_results_{timestamp}.csv"
        else:
            filename = f"final_results_{timestamp}.csv"
        
        filepath = self.results_dir / filename
        self.results_df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
        
        # Salva anche in formato JSON per backup
        json_filepath = filepath.with_suffix('.json')
        self.results_df.to_json(json_filepath, orient='records', indent=2)
    
    def check_prerequisites(self) -> bool:
        """Verifica che tutti i file necessari esistano."""
        required_files = ["run_with_attacks.py", "server.py", "client.py"]
        
        for file in required_files:
            file_path = self.base_dir / file
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                return False
            logger.info(f"Found required file: {file_path}")
        
        # Test aggiuntivo per run_with_attacks.py
        logger.info("Testing run_with_attacks.py functionality...")
        if not self.test_run_with_attacks():
            logger.error("run_with_attacks.py functionality test failed")
            return False
        
        return True

    def test_run_with_attacks(self) -> bool:
        """Testa se run_with_attacks.py è eseguibile e risponde ai parametri help."""
        try:
            cmd = [sys.executable, "run_with_attacks.py", "--help"]
            logger.info(f"Testing run_with_attacks.py with: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.base_dir
            )
            
            if result.returncode == 0:
                logger.info("run_with_attacks.py test successful")
                return True
            else:
                logger.error(f"run_with_attacks.py test failed with return code: {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("run_with_attacks.py test timed out")
            return False
        except Exception as e:
            logger.error(f"Error testing run_with_attacks.py: {e}")
            return False

def create_experiment_configurations() -> List[ExperimentConfig]:
    """Crea tutte le configurazioni di esperimenti."""
    
    # Strategie disponibili (espanse da strategies.py e run_with_attacks.py)
    strategies = [
        # Strategie principali
        "fedavg", "fedavgm", "fedprox", "fednova", "scaffold", "fedadam",
        # Strategie robuste
        "krum", "trimmedmean", "bulyan",
        # Baselines da repository Flower
        "dasha", "depthfl", "heterofl", "fedmeta", "fedper", 
        "fjord", "flanders", "fedopt"
    ]
    
    # Attacchi con parametri fissi
    attacks = {
        "none": {},
        "noise": {"noise_std": 0.1, "noise_fraction": 0.3},
        "missed": {"missed_prob": 0.3},
        "failure": {"failure_prob": 0.2},
        "asymmetry": {"asymmetry_min": 0.5, "asymmetry_max": 3.0},
        "labelflip": {"labelflip_fraction": 0.2, "flip_prob": 0.8},
        "gradflip": {"gradflip_fraction": 0.2, "gradflip_intensity": 1.0}
    }
      # Dataset disponibili (escluso CIFAR-10)
    datasets = ["MNIST", "FMNIST"]
    
    # Parametri specifici per strategie (espansi)
    strategy_params = {
        # Strategie principali
        "fedprox": {"proximal_mu": 0.01},
        "fedavgm": {"server_momentum": 0.9},
        "fedadam": {"learning_rate": 0.1},
        
        # Strategie robuste
        "krum": {"num_byzantine": 2},
        "trimmedmean": {"beta": 0.1},
        "bulyan": {"num_byzantine": 2},
        
        # Baselines con parametri specifici
        "dasha": {"step_size": 0.5, "compressor_coords": 10},
        "depthfl": {"alpha": 0.75, "tau": 0.6},
        "flanders": {"to_keep": 0.6},
        "fedopt": {
            "fedopt_tau": 1e-3, "fedopt_beta1": 0.9, "fedopt_beta2": 0.99,
            "fedopt_eta": 1e-3, "fedopt_eta_l": 1e-3
        }
    }
    
    configurations = []
    
    for strategy in strategies:
        for attack_name, attack_params in attacks.items():
            for dataset in datasets:
                config = ExperimentConfig(
                    strategy=strategy,
                    attack=attack_name,
                    dataset=dataset,
                    attack_params=attack_params,
                    strategy_params=strategy_params.get(strategy, {}),
                    num_rounds=10,
                    num_clients=10
                )
                configurations.append(config)
    
    return configurations

def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(description="Esegue esperimenti sistematici di FL")
    parser.add_argument("--num-runs", type=int, default=10, 
                       help="Numero di run per ogni configurazione")
    parser.add_argument("--results-dir", type=str, default="experiment_results",
                       help="Directory per salvare i risultati")
    parser.add_argument("--max-parallel", type=int, default=1,
                       help="Numero massimo di esperimenti paralleli")
    parser.add_argument("--test-mode", action="store_true",
                       help="Modalità test con configurazioni ridotte")
    
    args = parser.parse_args()
    
    # Crea il runner
    runner = ExperimentRunner(results_dir=args.results_dir)
    
    if args.test_mode:
        # Configurazioni ridotte per test
        configs = [
            ExperimentConfig("fedavg", "none", "MNIST", num_rounds=3, num_clients=5),
            ExperimentConfig("fedavg", "noise", "MNIST", 
                           attack_params={"noise_std": 0.1, "noise_fraction": 0.3},
                           num_rounds=3, num_clients=5)
        ]
        num_runs = 2
    else:
        # Configurazioni complete
        configs = create_experiment_configurations()
        num_runs = args.num_runs
    
    logger.info(f"Created {len(configs)} experiment configurations")
    
    # Esegui gli esperimenti
    results = runner.run_experiments(
        configs=configs,
        num_runs=num_runs,
        max_parallel=args.max_parallel
    )
    
    # Salva i risultati finali
    runner.save_results(intermediate=False)
    
    # Stampa statistiche
    logger.info(f"Final results shape: {results.shape}")
    logger.info(f"Unique experiments: {results[['algorithm', 'attack', 'dataset']].drop_duplicates().shape[0]}")
    logger.info(f"Total runs: {results['run'].nunique()}")
    
    print("\nEsperimenti completati!")
    print(f"Risultati salvati in: {runner.results_dir}")
    print(f"Forma finale del DataFrame: {results.shape}")

if __name__ == "__main__":
    main()
