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

import logging
import subprocess

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('run_extensive_experiments.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


class ExtensiveExperimentRunner(ExperimentRunner):
    """Extension of :class:`ExperimentRunner` that returns process output."""

    def run_single_experiment(self, config: ExperimentConfig, run_id: int) -> Tuple[bool, str]:
        """Run a single experiment and capture the output.

        Returns a tuple ``(success, output)`` where ``output`` contains the
        entire stdout of the process.
        """
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
                process.stdout.close() if process.stdout else None

            return success, "\n".join(output_lines)
        except Exception as exc:  # catch broad exceptions to log them
            logger.error(f"Experiment {experiment_id} raised an error: {exc}")
            return False, str(exc)


def create_extensive_configurations() -> List[ExperimentConfig]:
    """Create configurations for all combinations with parameter variations."""
    strategies = [
        "fedavg", "fedavgm", "fedprox", "fednova", "scaffold", "fedadam",
        "krum", "trimmedmean", "bulyan",
        "dasha", "depthfl", "heterofl", "fedmeta", "fedper",
        "fjord", "flanders", "fedopt",
    ]

    attack_params: Dict[str, List[Dict[str, Any]]] = {
        "none": [{}],
        "noise": [
            {"noise_std": 0.1, "noise_fraction": 0.3},
            {"noise_std": 0.5, "noise_fraction": 0.8},
        ],
        "missed": [
            {"missed_prob": 0.3},
            {"missed_prob": 0.8},
        ],
        "failure": [
            {"failure_prob": 0.3},
            {"failure_prob": 0.8},
        ],
        "asymmetry": [
            {"asymmetry_min": 0.5, "asymmetry_max": 1.5},
            {"asymmetry_min": 0.1, "asymmetry_max": 3.0},
        ],
        "labelflip": [
            {"labelflip_fraction": 0.2, "flip_prob": 0.8},
            {"labelflip_fraction": 0.4, "flip_prob": 0.8},
        ],
        "gradflip": [
            {"gradflip_fraction": 0.2, "gradflip_intensity": 1.0},
            {"gradflip_fraction": 0.4, "gradflip_intensity": 0.5},
        ],
    }

    datasets = ["MNIST", "FMNIST", "CIFAR10"]

    strategy_params = {
        "fedprox": {"proximal_mu": 0.01},
        "fedavgm": {"server_momentum": 0.9},
        "fedadam": {"learning_rate": 0.1},
        "krum": {"num_byzantine": 2},
        "trimmedmean": {"beta": 0.1},
        "bulyan": {"num_byzantine": 2},
        "dasha": {"step_size": 0.5, "compressor_coords": 10},
        "depthfl": {"alpha": 0.75, "tau": 0.6},
        "flanders": {"to_keep": 0.6},
        "fedopt": {
            "fedopt_tau": 1e-3,
            "fedopt_beta1": 0.9,
            "fedopt_beta2": 0.99,
            "fedopt_eta": 1e-3,
            "fedopt_eta_l": 1e-3,
        },
    }

    configs: List[ExperimentConfig] = []
    for strategy in strategies:
        for attack, param_list in attack_params.items():
            for params in param_list:
                for dataset in datasets:
                    cfg = ExperimentConfig(
                        strategy=strategy,
                        attack=attack,
                        dataset=dataset,
                        attack_params=params,
                        strategy_params=strategy_params.get(strategy, {}),
                        num_rounds=10,
                        num_clients=10,
                    )
                    configs.append(cfg)
    return configs


def run_extensive(configs: List[ExperimentConfig], num_runs: int, results_dir: Path, log_file: Path) -> None:
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
    args = parser.parse_args()

    configs = create_extensive_configurations()
    logger.info(f"Generated {len(configs)} configurations")

    run_extensive(configs, args.num_runs, Path(args.results_dir), Path(args.log_file))
    logger.info("Experiments completed")


if __name__ == "__main__":
    main()
