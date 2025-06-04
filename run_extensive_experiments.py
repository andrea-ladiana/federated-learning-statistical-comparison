#!/usr/bin/env python3
"""Run extensive federated learning experiments.

This script generates all combinations of aggregation strategy, attack type and
 dataset. For attacks that support parameters, two different configurations are
 tried. Results are collected using :class:`experiment_runner.ExperimentRunner`
 and saved in the usual long-form format. Whenever an experiment fails, the
 failing configuration and the captured output are appended to a log file.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from experiment_runner import ExperimentRunner, ExperimentConfig
from checkpoint_manager import CheckpointManager
from retry_manager import RetryManager, RetryConfig, CONSERVATIVE_RETRY
from config_manager import get_config_manager
from enhanced_experiment_runner import (
    EnhancedExperimentRunner, 
    EnhancedExperimentConfig, 
    EnhancedConfigManager,
    create_enhanced_configurations
)

import logging
import subprocess
import time
import psutil

# Enhanced logging with more detailed format
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s',
                    handlers=[
                        logging.FileHandler('run_extensive_experiments.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


class ExtensiveExperimentRunner(EnhancedExperimentRunner):
    """Extension of :class:`EnhancedExperimentRunner` that returns process output and handles extensive experiments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_manager = RetryManager(CONSERVATIVE_RETRY)

    def run_single_experiment(self, config: ExperimentConfig, run_id: int) -> Tuple[bool, str]:
        """Run a single experiment and capture the output.

        Returns a tuple ``(success, output)`` where ``output`` contains the
        entire stdout of the process.
        """
        experiment_id = config.get_experiment_id()
        
        # Usa il retry manager per eseguire l'esperimento
        return self.retry_manager.execute_with_retry(
            experiment_id=f"{experiment_id}_run_{run_id}",
            experiment_func=self._run_single_experiment_internal,
            config=config,
            run_id=run_id
        )
    
    def _run_single_experiment_internal(self, config: ExperimentConfig, run_id: int) -> Tuple[bool, str]:
        """Esecuzione interna dell'esperimento (senza retry)."""
        experiment_id = config.get_experiment_id()
        logger.info(f"Starting experiment {experiment_id}, run {run_id}")
        self.current_round = 0

        try:
            logger.info("Killing existing Flower processes...")
            self.kill_flower_processes()
            logger.info("Waiting for port 8080 to be free...")
            self.wait_for_port(8080, timeout=30)

            cmd = self.build_attack_command(config)
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

                return_code = process.wait(timeout=self.process_timeout)
                success = return_code == 0
            except subprocess.TimeoutExpired:
                process.kill()
                output_lines.append("Process timed out")
                success = False
            finally:
                self.kill_flower_processes()
                if process.stdout:
                    process.stdout.close()

            return success, "\n".join(output_lines)
        except subprocess.TimeoutExpired:
            error_msg = f"Experiment {experiment_id} timed out after {self.process_timeout}s"
            logger.error(error_msg)
            return False, error_msg
        except subprocess.CalledProcessError as e:
            error_msg = f"Experiment {experiment_id} failed with return code {e.returncode}: {e.stderr}"
            logger.error(error_msg)
            return False, error_msg
        except FileNotFoundError as e:
            error_msg = f"Required file not found for experiment {experiment_id}: {e.filename}"
            logger.error(error_msg)
            return False, error_msg
        except PermissionError as e:
            error_msg = f"Permission denied for experiment {experiment_id}: {e.filename}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as exc:  # catch broad exceptions to log them
            error_msg = f"Unexpected error in experiment {experiment_id}: {exc}"
            logger.error(error_msg)
            return False, error_msg


def create_extensive_configurations() -> List[EnhancedExperimentConfig]:
    """Create configurations for all combinations with parameter variations."""
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
                        
                        # Validate configuration and log warnings
                        warnings = cfg.validate_consistency()
                        if warnings:
                            logger.warning(f"Configuration {cfg.get_experiment_id()} warnings: {warnings}")
                        
                        configs.append(cfg)
                        
                    except ValueError as e:
                        logger.error(f"Invalid configuration for {strategy}-{attack}-{dataset}: {e}")
                        continue
    
    logger.info(f"Created {len(configs)} valid configurations")
    return configs


def run_extensive(configs: List[EnhancedExperimentConfig], num_runs: int, results_dir: Path, log_file: Path, 
                 resume: bool = False, parallel: bool = False, max_parallel: int = 1) -> None:
    """Esegue esperimenti estensivi con supporto per checkpoint, resume e parallelizzazione."""
    
    # Crea configurazione migliorata
    enhanced_config = EnhancedConfigManager()
    enhanced_config.system.max_parallel_experiments = max_parallel
    
    # Usa il runner migliorato
    with EnhancedExperimentRunner(
        results_dir=str(results_dir),
        config_manager=enhanced_config
    ) as runner:
        
        results_dir.mkdir(exist_ok=True)
        
        try:
            if parallel and max_parallel > 1:
                logger.info(f"Running experiments in parallel mode with {max_parallel} workers")
                results = runner.run_experiments_parallel(configs, num_runs)
            else:
                logger.info("Running experiments sequentially with enhanced checkpoint support")
                results = runner.run_experiments_sequential(configs, num_runs, resume=resume)
            
            # Salva risultati finali
            runner.save_results(intermediate=False)
            
            # Genera e salva report finale
            final_report = runner.generate_final_report()
            
            # Mostra statistiche finali
            summary = final_report["experiment_summary"]
            system_metrics = final_report["system_metrics"]
            retry_metrics = final_report["retry_metrics"]
            
            logger.info("=== EXPERIMENT COMPLETION SUMMARY ===")
            logger.info(f"Total configurations: {summary['total_configurations']}")
            logger.info(f"Completed experiments: {summary['completed_experiments']}")
            logger.info(f"Failed experiments: {summary['failed_experiments']}")
            logger.info(f"Success rate: {summary['success_rate']*100:.1f}%")
            
            if system_metrics["total_experiments"] > 0:
                logger.info(f"Average duration: {system_metrics['average_duration']:.1f}s")
                if enhanced_config.system.resource_monitoring:
                    logger.info(f"Average CPU usage: {system_metrics['average_cpu_usage']:.1f}%")
                    logger.info(f"Average memory usage: {system_metrics['average_memory_usage']:.1f}%")
            
            if retry_metrics['total_experiments_with_failures'] > 0:
                logger.info(f"Experiments requiring retries: {retry_metrics['total_experiments_with_failures']}")
                logger.info(f"Total retry attempts: {retry_metrics['total_failures']}")
                logger.info("Failure types: " + ", ".join([f"{k}: {v}" for k, v in retry_metrics['failure_type_counts'].items()]))
            
            logger.info("Results and checkpoints saved to: " + str(results_dir))
            logger.info("=====================================")
            
        except KeyboardInterrupt:
            logger.info("Experiment run interrupted by user")
            runner.save_results(intermediate=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during experiment run: {e}")
            runner.save_results(intermediate=True)
            raise

def run_extensive_original(configs: List[ExperimentConfig], num_runs: int, results_dir: Path, log_file: Path) -> None:
    """Versione originale senza checkpoint (mantenuta per compatibilità)."""
    runner = ExtensiveExperimentRunner(results_dir=str(results_dir))
    results_dir.mkdir(exist_ok=True)

    with log_file.open("w") as lf:
        for cfg in configs:
            for run_id in range(num_runs):
                success, output = runner.run_single_experiment(cfg, run_id)
                if not success:
                    lf.write(f"FAILED: {cfg.get_experiment_id()} run {run_id}\n")
                    lf.write(json.dumps(cfg.to_dict()) + "\n")
                    lf.write(output + "\n\n")

    runner.save_results(intermediate=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extensive experiment grid")
    parser.add_argument("--num-runs", type=int, default=1, help="Repetitions per configuration")
    parser.add_argument("--results-dir", type=str, default="extensive_results", help="Directory for results")
    parser.add_argument("--log-file", type=str, default="extensive_failures.log", help="File to log failures")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoint")
    parser.add_argument("--reset-failed", action="store_true", help="Reset failed experiments for retry")
    args = parser.parse_args()

    configs = create_extensive_configurations()
    logger.info(f"Generated {len(configs)} configurations")

    # Gestione dei reset se richiesto
    if args.reset_failed:
        checkpoint_manager = CheckpointManager(checkpoint_dir=Path(args.results_dir) / "checkpoints")
        checkpoint_manager.reset_failed_experiments()
        logger.info("Failed experiments have been reset for retry")

    run_extensive(configs, args.num_runs, Path(args.results_dir), Path(args.log_file), resume=args.resume)
    logger.info("Experiments completed")


if __name__ == "__main__":
    main()
