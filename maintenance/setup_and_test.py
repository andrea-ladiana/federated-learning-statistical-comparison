#!/usr/bin/env python3
"""
Script di setup e test per il sistema di esperimenti migliorato.

Verifica prerequisiti, crea configurazioni di default e testa i componenti principali.
"""

import sys
import logging
from pathlib import Path
import subprocess
import time

# Aggiungi la directory corrente al path per importare dalla nuova struttura
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir / "configuration"))
sys.path.insert(0, str(parent_dir / "utilities"))
sys.path.insert(0, str(parent_dir / "experiment_runners"))

from config_manager import get_config_manager, create_default_config_file
from checkpoint_manager import CheckpointManager
from retry_manager import RetryManager, CONSERVATIVE_RETRY
from basic_experiment_runner import ExperimentConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Verifica che tutte le dipendenze necessarie siano installate."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('psutil', 'psutil'),
        ('yaml', 'pyyaml'),
    ]
    
    missing_packages = []
    
    for package_name, pip_name in required_packages:
        try:
            __import__(package_name)
            logger.info(f"✓ {package_name} is available")
        except ImportError:
            logger.error(f"✗ {package_name} is missing")
            missing_packages.append(pip_name)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("All dependencies are available")
    return True

def check_required_files():
    """Verifica che tutti i file richiesti esistano."""
    logger.info("Checking required files...")
    
    required_files = [
        "run_with_attacks.py",
        "server.py", 
        "client.py",
        "experiment_runner.py"
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            logger.info(f"✓ {file} exists")
        else:
            logger.error(f"✗ {file} is missing")
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing files: {', '.join(missing_files)}")
        return False
    
    logger.info("All required files are present")
    return True

def test_config_manager():
    """Testa il sistema di configurazione."""
    logger.info("Testing configuration manager...")
    
    try:
        # Crea file di configurazione di default se non esiste
        create_default_config_file()
        
        # Testa caricamento configurazione
        config_mgr = get_config_manager()
        
        # Testa validazione
        config_mgr.validate_all()
        
        # Testa accesso ai dati
        strategies = config_mgr.get_valid_strategies()
        attacks = config_mgr.get_valid_attacks()
        datasets = config_mgr.get_valid_datasets()
        
        logger.info(f"✓ Found {len(strategies)} strategies, {len(attacks)} attacks, {len(datasets)} datasets")
        
        # Testa controllo risorse
        resources = config_mgr.check_system_resources()
        logger.info(f"✓ System resources check: {resources}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration manager test failed: {e}")
        return False

def test_checkpoint_manager():
    """Testa il sistema di checkpoint."""
    logger.info("Testing checkpoint manager...")
    
    try:
        # Crea directory di test
        test_dir = Path("test_checkpoints")
        test_dir.mkdir(exist_ok=True)
        
        # Crea checkpoint manager
        checkpoint_mgr = CheckpointManager(checkpoint_dir=test_dir)
        
        # Testa registrazione esperimenti
        test_configs = [
            {
                'strategy': 'fedavg',
                'attack': 'none',
                'dataset': 'MNIST',
                'attack_params': {},
                'strategy_params': {},
                'num_rounds': 5,
                'num_clients': 5
            }
        ]
        
        checkpoint_mgr.register_experiments(test_configs, num_runs=2)
        
        # Testa aggiornamento stato
        checkpoint_mgr.mark_run_completed("fedavg_none_MNIST", 0, success=True)
        checkpoint_mgr.mark_run_completed("fedavg_none_MNIST", 1, success=False, error_message="Test error")
        
        # Testa ottenimento progresso
        progress = checkpoint_mgr.get_progress_summary()
        logger.info(f"✓ Progress: {progress}")
        
        # Testa ottenimento esperimenti pendenti
        pending = checkpoint_mgr.get_pending_experiments()
        logger.info(f"✓ Pending experiments: {len(pending)}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Checkpoint manager test failed: {e}")
        return False

def test_retry_manager():
    """Testa il sistema di retry."""
    logger.info("Testing retry manager...")
    
    try:
        retry_mgr = RetryManager(CONSERVATIVE_RETRY)
        
        # Testa classificazione errori
        timeout_type = retry_mgr.classify_failure("Process timed out after 600s")
        assert timeout_type.value == "timeout"
        
        permission_type = retry_mgr.classify_failure("Permission denied: access to file")
        assert permission_type.value == "permission_error"
        
        # Testa calcolo delay
        delay1 = retry_mgr.get_delay(0)
        delay2 = retry_mgr.get_delay(1)
        assert delay2 > delay1  # Exponential backoff
        
        logger.info("✓ Retry manager tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Retry manager test failed: {e}")
        return False

def test_experiment_config():
    """Testa la validazione delle configurazioni esperimenti."""
    logger.info("Testing experiment configuration validation...")
    
    try:
        # Testa configurazione valida
        valid_config = ExperimentConfig(
            strategy="fedavg",
            attack="none", 
            dataset="MNIST",
            num_rounds=10,
            num_clients=5
        )
        logger.info("✓ Valid configuration accepted")
        
        # Testa configurazioni invalide
        try:
            invalid_strategy = ExperimentConfig(
                strategy="invalid_strategy",
                attack="none",
                dataset="MNIST"
            )
            logger.error("✗ Invalid strategy should have been rejected")
            return False
        except ValueError:
            logger.info("✓ Invalid strategy correctly rejected")
        
        try:
            invalid_attack = ExperimentConfig(
                strategy="fedavg",
                attack="invalid_attack",
                dataset="MNIST"
            )
            logger.error("✗ Invalid attack should have been rejected")
            return False
        except ValueError:
            logger.info("✓ Invalid attack correctly rejected")
        
        try:
            invalid_dataset = ExperimentConfig(
                strategy="fedavg",
                attack="none",
                dataset="INVALID_DATASET"
            )
            logger.error("✗ Invalid dataset should have been rejected")
            return False
        except ValueError:
            logger.info("✓ Invalid dataset correctly rejected")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Experiment config test failed: {e}")
        return False

def run_all_tests():
    """Esegue tutti i test di sistema."""
    logger.info("Starting system tests...")
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Required Files", check_required_files), 
        ("Configuration Manager", test_config_manager),
        ("Checkpoint Manager", test_checkpoint_manager),
        ("Retry Manager", test_retry_manager),
        ("Experiment Config", test_experiment_config),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            results[test_name] = False
    
    # Riassunto
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for use.")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix issues before proceeding.")
        return False

def main():
    """Funzione principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup and test experiment system")
    parser.add_argument("--setup-only", action="store_true", help="Only create config files, don't run tests")
    parser.add_argument("--test-only", action="store_true", help="Only run tests, don't create configs")
    
    args = parser.parse_args()
    
    if args.setup_only:
        logger.info("Creating default configuration files...")
        create_default_config_file()
        logger.info("Setup completed")
        return
    
    if args.test_only:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    
    # Default: setup e test
    logger.info("Setting up and testing experiment system...")
    
    # Setup
    create_default_config_file()
    
    # Test
    success = run_all_tests()
    
    if success:
        logger.info("\n🚀 System is ready! You can now run:")
        logger.info("python run_extensive_experiments.py --help")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
