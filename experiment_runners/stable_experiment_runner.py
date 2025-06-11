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
        self.strategy = strategy
        self.attack = attack
        self.dataset = dataset
        self.attack_params = attack_params or {}
        self.strategy_params = strategy_params or {}
        self.num_rounds = num_rounds
        self.num_clients = num_clients
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte la configurazione in dizionario."""
        return {
            'strategy': self.strategy,
            'attack': self.attack,
            'dataset': self.dataset,
            'attack_params': self.attack_params,
            'strategy_params': self.strategy_params,
            'num_rounds': self.num_rounds,
            'num_clients': self.num_clients
        }
    
    def get_experiment_id(self) -> str:
        """Genera un ID univoco per l'esperimento."""
        attack_str = f"{self.attack}"
        if self.attack_params:
            params_str = "_".join([f"{k}{v}" for k, v in self.attack_params.items()])
            attack_str += f"_{params_str}"
        
        return f"{self.strategy}_{attack_str}_{self.dataset}"

class MetricsCollector:
    """Raccoglie metriche dai log dell'esperimento."""
    
    def __init__(self):
        self.client_metrics = []
        self.server_metrics = []
    
    def parse_client_log(self, log_line: str, client_id: int, run_id: int) -> Optional[Dict]:
        """Estrae metriche dai log del client."""
        patterns = [
            # Pattern per fit metrics
            r'fit complete.*avg_loss=([0-9.]+).*accuracy=([0-9.]+)',
            # Pattern per evaluate metrics  
            r'evaluate complete.*avg_loss=([0-9.]+).*accuracy=([0-9.]+)',
            # Pattern per training metrics durante il fit
            r'Training batch.*loss=([0-9.]+).*acc=([0-9.]+)',
            # Pattern per evaluation metrics durante evaluate
            r'Eval batch.*loss=([0-9.]+).*acc=([0-9.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, log_line)
            if match:
                loss = float(match.group(1))
                accuracy = float(match.group(2))
                
                # Determina il tipo di metrica e il round
                if 'fit complete' in log_line:
                    metric_type = 'training'
                elif 'evaluate complete' in log_line:
                    metric_type = 'evaluation'
                else:
                    continue  # Skip intermediate metrics
                
                # Estrai il round se presente
                round_match = re.search(r'Round (\d+)', log_line)
                round_num = int(round_match.group(1)) if round_match else 0
                
                return {
                    'client_id': client_id,
                    'run': run_id,
                    'round': round_num,
                    'metric_type': metric_type,
                    'loss': loss,
                    'accuracy': accuracy
                }
        
        return None
    
    def parse_server_log(self, log_line: str, run_id: int) -> Optional[Dict]:
        """Estrae metriche dai log del server."""
        # Pattern per metriche aggregate del server
        patterns = [
            r'Round (\d+).*aggr.*accuracy.*([0-9.]+)',
            r'Round (\d+).*aggregated.*loss.*([0-9.]+)',
            r'evaluate.*Round (\d+).*accuracy.*([0-9.]+)',
            r'fit.*Round (\d+).*loss.*([0-9.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, log_line)
            if match:
                round_num = int(match.group(1))
                value = float(match.group(2))
                
                metric_name = 'accuracy' if 'accuracy' in log_line else 'loss'
                phase = 'evaluation' if 'evaluate' in log_line else 'training'
                
                return {
                    'run': run_id,
                    'round': round_num,
                    'metric': metric_name,
                    'value': value,
                    'phase': phase
                }
        
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
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if ('python' in proc.info['name'].lower() and 
                    ('server.py' in cmdline or 'client.py' in cmdline or 'run_with_attacks.py' in cmdline)):
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
        
        try:
            # Assicurati che la porta sia libera
            logger.info("Killing existing Flower processes...")
            self.kill_flower_processes()
            logger.info("Waiting for port 8080 to be free...")
            self.wait_for_port(8080, timeout=30)
            
            # Costruisci e esegui il comando
            cmd = self.build_attack_command(config)
            logger.info(f"Running command: {' '.join(cmd)}")
            
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
            
            # Raccogli l'output in tempo reale
            output_lines = []
            line_count = 0
            last_log_time = time.time()
            last_progress_report = time.time()
            
            try:
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
                        for line in output_lines[-10:]:
                            logger.error(f"  {line}")
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.error(
                    f"Experiment {experiment_id}, run {run_id} timed out after {self.process_timeout} seconds"
                )
                process.kill()
                # Log ultimi 10 righe di output per debug
                if output_lines:
                    logger.error("Last 10 lines before timeout:")
                    for line in output_lines[-10:]:
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
    
    def parse_and_store_metrics(self, log_line: str, config: ExperimentConfig, run_id: int):
        """Analizza e memorizza le metriche dai log con pattern migliorati."""
        # Pattern migliorati per estrarre metriche dai log
        patterns = {
            # Pattern per metriche del client durante il fit
            'client_fit': r'Client (\d+).*(?:fit.*complete|training.*completed).*(?:loss|avg_loss)[=:]?\s*([0-9.]+).*(?:acc|accuracy)[=:]?\s*([0-9.]+)',
            
            # Pattern per metriche del client durante l'evaluation
            'client_eval': r'Client (\d+).*(?:eval|evaluate).*(?:complete|finished).*(?:loss|avg_loss)[=:]?\s*([0-9.]+).*(?:acc|accuracy)[=:]?\s*([0-9.]+)',
            
            # Pattern per metriche aggregate del server
            'server_aggregate': r'Round (\d+).*(?:aggregat|aggr).*(?:accuracy|acc)[=:]?\s*([0-9.]+)',
            'server_loss': r'Round (\d+).*(?:aggregat|aggr).*(?:loss)[=:]?\s*([0-9.]+)',
            
            # Pattern per metriche di fine round del server
            'server_round_end': r'Round (\d+).*(?:complete|finished).*(?:accuracy|acc)[=:]?\s*([0-9.]+)',
            
            # Pattern alternativi per diversi formati di log
            'client_training': r'Client (\d+).*Round (\d+).*(?:training|fit).*loss[=:]?\s*([0-9.]+).*accuracy[=:]?\s*([0-9.]+)',
            'client_validation': r'Client (\d+).*Round (\d+).*(?:validation|eval).*loss[=:]?\s*([0-9.]+).*accuracy[=:]?\s*([0-9.]+)',
        }
        
        current_round = 0
        
        # Estrai il numero del round con pattern multipli
        round_patterns = [
            r'Round[:\s]+(\d+)',
            r'round[:\s]+(\d+)',
            r'Round (\d+)',
            r'fit_round (\d+)',
            r'evaluate_round (\d+)'
        ]
        
        for pattern in round_patterns:
            round_match = re.search(pattern, log_line, re.IGNORECASE)
            if round_match:
                current_round = int(round_match.group(1))
                break
        
        # Buffer per raccogliere metriche prima di aggiungerle al DataFrame
        metrics_to_add = []
        
        # Controlla tutti i pattern
        for pattern_name, pattern in patterns.items():
            match = re.search(pattern, log_line, re.IGNORECASE)
            if match and len(match.groups()) >= 2:
                try:
                    if pattern_name in ['client_training', 'client_validation'] and len(match.groups()) >= 4:
                        # Pattern con round esplicito
                        client_id = int(match.group(1))
                        round_num = int(match.group(2))
                        loss = float(match.group(3))
                        accuracy = float(match.group(4))
                        
                        metric_prefix = "eval_" if pattern_name == 'client_validation' else ""
                        
                        metrics_to_add.extend([
                            {
                                "algorithm": config.strategy,
                                "attack": config.attack,
                                "dataset": config.dataset,
                                "run": run_id,
                                "client_id": client_id,
                                "round": round_num,
                                "metric": f"{metric_prefix}loss",
                                "value": loss
                            },
                            {
                                "algorithm": config.strategy,
                                "attack": config.attack,
                                "dataset": config.dataset,
                                "run": run_id,
                                "client_id": client_id,
                                "round": round_num,
                                "metric": f"{metric_prefix}accuracy",
                                "value": accuracy
                            }
                        ])
                    
                    elif pattern_name in ['client_fit', 'client_eval'] and len(match.groups()) >= 3:
                        # Pattern client senza round esplicito
                        client_id = int(match.group(1))
                        loss = float(match.group(2))
                        accuracy = float(match.group(3))
                        
                        metric_prefix = "eval_" if pattern_name == 'client_eval' else ""
                        
                        metrics_to_add.extend([
                            {
                                "algorithm": config.strategy,
                                "attack": config.attack,
                                "dataset": config.dataset,
                                "run": run_id,
                                "client_id": client_id,
                                "round": current_round,
                                "metric": f"{metric_prefix}loss",
                                "value": loss
                            },
                            {
                                "algorithm": config.strategy,
                                "attack": config.attack,
                                "dataset": config.dataset,
                                "run": run_id,
                                "client_id": client_id,
                                "round": current_round,
                                "metric": f"{metric_prefix}accuracy",
                                "value": accuracy
                            }
                        ])
                    
                    elif pattern_name in ['server_aggregate', 'server_loss', 'server_round_end']:
                        # Pattern server
                        round_num = int(match.group(1))
                        value = float(match.group(2))
                        
                        if pattern_name == 'server_loss':
                            metric_name = "server_loss"
                        else:
                            metric_name = "server_accuracy"
                        
                        metrics_to_add.append({
                            "algorithm": config.strategy,
                            "attack": config.attack,
                            "dataset": config.dataset,
                            "run": run_id,
                            "client_id": -1,  # -1 indica metriche aggregate del server
                            "round": round_num,
                            "metric": metric_name,
                            "value": value
                        })
                
                except (ValueError, IndexError) as e:
                    # Ignora errori di parsing
                    continue
        
        # Aggiungi tutte le metriche raccolte al DataFrame in una volta
        if metrics_to_add:
            try:
                new_rows = pd.DataFrame(metrics_to_add)
                self.results_df = pd.concat([self.results_df, new_rows], ignore_index=True)
            except Exception as e:
                logger.warning(f"Failed to add metrics: {e}")
                # Fallback: aggiungi una riga alla volta
                for metric in metrics_to_add:
                    try:
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
