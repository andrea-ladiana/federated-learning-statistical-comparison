#!/usr/bin/env python3
"""
Test script per l'experiment runner.
Testa un piccolo subset di configurazioni per verificare che tutto funzioni.
"""

import sys
import os
from pathlib import Path

# Add paths for reorganized imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir / "experiment_runners"))

from experiment_runners.basic_experiment_runner import ExperimentRunner, ExperimentConfig, create_experiment_configurations
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration_creation():
    """Testa la creazione delle configurazioni."""
    print("Testing configuration creation...")
    
    configs = create_experiment_configurations()
    print(f"Total configurations created: {len(configs)}")
    
    # Verifica che abbiamo tutte le strategie
    strategies = set([config.strategy for config in configs])
    expected_strategies = {
        "fedavg", "fedavgm", "fedprox", "fednova", "scaffold", "fedadam",
        "krum", "trimmedmean", "bulyan", "dasha", "depthfl", "heterofl", 
        "fedmeta", "fedper", "fjord", "flanders", "fedopt"
    }
    
    print(f"Strategies found: {sorted(strategies)}")
    print(f"Expected strategies: {sorted(expected_strategies)}")
    print(f"Missing strategies: {expected_strategies - strategies}")
    print(f"Extra strategies: {strategies - expected_strategies}")
    
    # Verifica che abbiamo tutti gli attacchi
    attacks = set([config.attack for config in configs])
    expected_attacks = {"none", "noise", "missed", "failure", "asymmetry", "labelflip", "gradflip"}
    
    print(f"Attacks found: {sorted(attacks)}")
    print(f"Expected attacks: {sorted(expected_attacks)}")
    
    # Verifica che abbiamo tutti i dataset
    datasets = set([config.dataset for config in configs])
    expected_datasets = {"MNIST", "FMNIST", "CIFAR10"}
    
    print(f"Datasets found: {sorted(datasets)}")
    print(f"Expected datasets: {sorted(expected_datasets)}")
    
    # Calcola il numero totale atteso
    expected_total = len(expected_strategies) * len(expected_attacks) * len(expected_datasets)
    print(f"Expected total configurations: {expected_total}")
    print(f"Actual total configurations: {len(configs)}")
    
    return configs

def test_command_building():
    """Testa la costruzione dei comandi."""
    print("\nTesting command building...")
    
    runner = ExperimentRunner()
    
    # Test con alcune configurazioni specifiche
    test_configs = [
        ExperimentConfig("fedavg", "none", "MNIST"),
        ExperimentConfig("fedprox", "noise", "CIFAR10", 
                        attack_params={"noise_std": 0.1, "noise_fraction": 0.3},
                        strategy_params={"proximal_mu": 0.01}),
        ExperimentConfig("dasha", "labelflip", "FMNIST",
                        attack_params={"labelflip_fraction": 0.2, "flip_prob": 0.8},
                        strategy_params={"step_size": 0.5, "compressor_coords": 10}),
    ]
    
    for config in test_configs:
        cmd = runner.build_attack_command(config)
        print(f"\nStrategy: {config.strategy}, Attack: {config.attack}, Dataset: {config.dataset}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Experiment ID: {config.get_experiment_id()}")

def test_small_experiment():
    """Testa l'esecuzione di un piccolo esperimento (solo configurazione, senza esecuzione reale)."""
    print("\nTesting small experiment setup...")
    
    runner = ExperimentRunner(results_dir="test_results")
    
    # Crea alcune configurazioni di test
    test_configs = [
        ExperimentConfig("fedavg", "none", "MNIST", num_rounds=2, num_clients=3),
        ExperimentConfig("fedprox", "noise", "MNIST", 
                        attack_params={"noise_std": 0.1, "noise_fraction": 0.3},
                        strategy_params={"proximal_mu": 0.01},
                        num_rounds=2, num_clients=3),
    ]
    
    print(f"Test configurations: {len(test_configs)}")
    for i, config in enumerate(test_configs):
        print(f"{i+1}. {config.get_experiment_id()}")
        
    # Verifica che il DataFrame sia inizializzato correttamente
    print(f"Initial DataFrame shape: {runner.results_df.shape}")
    print(f"DataFrame columns: {list(runner.results_df.columns)}")
    
    return test_configs

def main():
    """Funzione principale del test."""
    print("=" * 60)
    print("EXPERIMENT RUNNER TEST")
    print("=" * 60)
    
    try:
        # Test 1: Creazione configurazioni
        configs = test_configuration_creation()
        
        # Test 2: Costruzione comandi
        test_command_building()
        
        # Test 3: Setup esperimento piccolo
        test_configs = test_small_experiment()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nSummary:")
        print(f"- Total configurations available: {len(configs)}")
        print(f"- Test configurations created: {len(test_configs)}")
        print("- Ready for full experiment execution")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
