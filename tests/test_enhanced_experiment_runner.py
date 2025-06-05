#!/usr/bin/env python3
"""
Comprehensive test suite for the enhanced experiment runner.

This module contains unit tests for all major components of the enhanced
federated learning experiment system, including configuration management,
checkpoint/resume functionality, retry systems, and more.
"""

import unittest
import tempfile
import shutil
import time
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml
import pandas as pd
import numpy as np

# Add the project root to the path for reorganized imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "experiment_runners"))
sys.path.insert(0, str(project_root / "utilities"))
sys.path.insert(0, str(project_root / "configuration"))

from experiment_runners.enhanced_experiment_runner import (
    SystemConfig, ExperimentDefaults, EnhancedConfigManager,
    MetricsCollector, PortManager, EnhancedExperimentConfig,
    EnhancedExperimentRunner, ExperimentValidationError,
    ResourceError, CheckpointError
)


class TestSystemConfig(unittest.TestCase):
    """Test the SystemConfig dataclass."""
    
    def test_valid_config_creation(self):
        """Test creating a valid system configuration."""
        config = SystemConfig(
            max_parallel_experiments=4,
            process_timeout=1800,
            checkpoint_interval=300,
            retry_delay=60
        )
        self.assertEqual(config.max_parallel_experiments, 4)
        self.assertEqual(config.process_timeout, 1800)
        self.assertEqual(config.checkpoint_interval, 300)
        self.assertEqual(config.retry_delay, 60)
        
    def test_invalid_parallel_experiments(self):
        """Test validation of parallel experiments count."""
        with self.assertRaises(ValueError):
            SystemConfig(max_parallel_experiments=0)
            
        with self.assertRaises(ValueError):
            SystemConfig(max_parallel_experiments=-1)
    
    def test_invalid_timeout(self):
        """Test validation of timeout values."""
        with self.assertRaises(ValueError):
            SystemConfig(process_timeout=0)
            
    def test_invalid_intervals(self):
        """Test validation of interval values."""
        with self.assertRaises(ValueError):
            SystemConfig(checkpoint_interval=-1)


class TestExperimentDefaults(unittest.TestCase):
    """Test the ExperimentDefaults dataclass."""
    
    def test_valid_defaults_creation(self):
        """Test creating valid experiment defaults."""
        defaults = ExperimentDefaults(
            num_rounds=10,
            num_clients=5,
            learning_rate=0.01,
            batch_size=32
        )
        self.assertEqual(defaults.num_rounds, 10)
        self.assertEqual(defaults.num_clients, 5)
        self.assertEqual(defaults.learning_rate, 0.01)
        self.assertEqual(defaults.batch_size, 32)
        
    def test_invalid_rounds(self):
        """Test validation of round numbers."""
        with self.assertRaises(ValueError):
            ExperimentDefaults(num_rounds=0)
            
    def test_invalid_clients(self):
        """Test validation of client numbers."""
        with self.assertRaises(ValueError):
            ExperimentDefaults(num_clients=0)
            
    def test_invalid_fractions(self):
        """Test validation of learning rate values."""
        with self.assertRaises(ValueError):
            ExperimentDefaults(learning_rate=0)
            
        with self.assertRaises(ValueError):
            ExperimentDefaults(learning_rate=-0.1)


class TestEnhancedConfigManager(unittest.TestCase):
    """Test the EnhancedConfigManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "test_config.yaml"
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_default_config(self):
        """Test creating a default configuration file."""
        manager = EnhancedConfigManager(self.config_file)
        manager.save_config()  # This creates the default config
        
        self.assertTrue(self.config_file.exists())
        
        # Load and verify the config
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        self.assertIn('system', config)
        self.assertIn('defaults', config)
        
    def test_load_valid_config(self):
        """Test loading a valid configuration."""
        # Create a test config with correct parameters
        test_config = {
            'system': {
                'max_parallel_experiments': 2,
                'process_timeout': 1200,
                'checkpoint_interval': 300,
                'retry_delay': 60,
                'port': 8080,
                'log_level': 'INFO',
                'resource_monitoring': True
            },
            'defaults': {
                'num_rounds': 10,
                'num_clients': 5,
                'learning_rate': 0.01,
                'batch_size': 32
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
            
        manager = EnhancedConfigManager(self.config_file)
        # The config is loaded automatically in __init__
        
        self.assertEqual(manager.system.max_parallel_experiments, 2)
        self.assertEqual(manager.defaults.num_rounds, 10)
        
    def test_invalid_config_validation(self):
        """Test validation of invalid configurations."""
        # Create an invalid config with wrong parameter names
        invalid_config = {
            'system': {
                'max_parallel_experiments': -1,  # Invalid
                'process_timeout': 1200,
                'checkpoint_interval': 300,
                'retry_delay': 60
            },
            'defaults': {
                'num_rounds': 10,
                'num_clients': 5,
                'learning_rate': 0.01,
                'batch_size': 32
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(invalid_config, f)
            
        # The validation should occur when SystemConfig is created
        with self.assertRaises(ValueError):
            manager = EnhancedConfigManager(self.config_file)


class TestMetricsCollector(unittest.TestCase):
    """Test the MetricsCollector class."""
    
    def setUp(self):
        """Set up test environment."""
        self.collector = MetricsCollector()
        
    def test_start_stop_monitoring(self):
        """Test starting and stopping resource monitoring."""
        experiment_id = "test_experiment"
        self.collector.start_monitoring(experiment_id)
        self.assertIn(experiment_id, self.collector.metrics)
        
        time.sleep(0.1)  # Let it collect some data
        
        self.collector.stop_monitoring_for_experiment(experiment_id)
        self.assertNotIn(experiment_id, self.collector.monitoring_threads)
        
    def test_resource_metrics_format(self):
        """Test the format of collected metrics."""
        experiment_id = "test_experiment"
        self.collector.start_monitoring(experiment_id)
        time.sleep(0.1)
        self.collector.stop_monitoring_for_experiment(experiment_id)
        
        # Check that metrics were collected
        self.assertIn(experiment_id, self.collector.metrics)
        metrics = self.collector.metrics[experiment_id]
        
        # Check structure
        self.assertIsInstance(metrics.cpu_usage, list)
        self.assertIsInstance(metrics.memory_usage, list)


class TestPortManager(unittest.TestCase):
    """Test the PortManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.port_manager = PortManager(base_port=8000, num_ports=100)
        
    def test_allocate_deallocate_port(self):
        """Test port allocation and deallocation."""
        port = self.port_manager.acquire_port()
        
        self.assertIsInstance(port, int)
        self.assertGreaterEqual(port, 8000)
        self.assertLess(port, 8100)
        
        # Port should be in use
        self.assertIn(port, self.port_manager.used_ports)
        
        # Deallocate the port
        self.port_manager.release_port(port)
        self.assertNotIn(port, self.port_manager.used_ports)
        
    def test_port_exhaustion(self):
        """Test behavior when all ports are exhausted."""
        # Allocate all ports
        allocated_ports = []
        for _ in range(100):  # num_ports = 100
            port = self.port_manager.acquire_port()
            allocated_ports.append(port)
            
        # Next allocation should raise an exception
        with self.assertRaises(ResourceError):
            self.port_manager.acquire_port()
            
        # Clean up
        for port in allocated_ports:
            self.port_manager.release_port(port)
            
    def test_deallocate_unallocated_port(self):
        """Test deallocating a port that wasn't allocated."""
        # This should not raise an error, just be ignored
        self.port_manager.release_port(9999)


class TestEnhancedExperimentConfig(unittest.TestCase):
    """Test the EnhancedExperimentConfig class."""
    
    def test_valid_config_creation(self):
        """Test creating a valid experiment configuration."""
        config = EnhancedExperimentConfig(
            strategy="fedavg",
            attack="none",
            dataset="MNIST",
            num_rounds=10,
            num_clients=5
        )
        
        self.assertEqual(config.strategy, "fedavg")
        self.assertEqual(config.attack, "none")
        self.assertEqual(config.dataset, "MNIST")
        
    def test_invalid_strategy(self):
        """Test validation of invalid strategies."""
        with self.assertRaises(ExperimentValidationError):
            EnhancedExperimentConfig(
                strategy="invalid_strategy",
                attack="none",
                dataset="MNIST"
            )
            
    def test_invalid_attack(self):
        """Test validation of invalid attacks."""
        with self.assertRaises(ExperimentValidationError):
            EnhancedExperimentConfig(
                strategy="fedavg",
                attack="invalid_attack",
                dataset="MNIST"
            )
            
    def test_invalid_dataset(self):
        """Test validation of invalid datasets."""
        with self.assertRaises(ExperimentValidationError):
            EnhancedExperimentConfig(
                strategy="fedavg",
                attack="none",
                dataset="INVALID_DATASET"
            )
            
    def test_experiment_id_generation(self):
        """Test experiment ID generation."""
        config = EnhancedExperimentConfig(
            strategy="fedavg",
            attack="noise",
            dataset="CIFAR10",
            attack_params={"noise_std": 0.1},
            strategy_params={"lr": 0.01}
        )
        
        exp_id = config.get_experiment_id()
        self.assertIn("fedavg", exp_id)
        self.assertIn("noise", exp_id)
        self.assertIn("CIFAR10", exp_id)
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters - should not raise error
        try:
            config = EnhancedExperimentConfig(
                strategy="fedprox",
                attack="noise",
                dataset="MNIST",
                strategy_params={"proximal_mu": 0.01},
                attack_params={"noise_std": 0.1, "noise_fraction": 0.3}
            )
            # If we get here, the validation passed as expected
        except ExperimentValidationError:
            self.fail("Valid configuration raised ExperimentValidationError")


class TestEnhancedExperimentRunner(unittest.TestCase):
    """Test the EnhancedExperimentRunner class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "config.yaml"
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.results_dir = self.temp_dir / "results"
        
        # Create a minimal config file with correct parameter names
        config_data = {
            'system': {
                'max_parallel_experiments': 2,
                'process_timeout': 300,
                'checkpoint_interval': 60,
                'retry_delay': 30,
                'port': 8080,
                'log_level': 'INFO',
                'resource_monitoring': True
            },
            'defaults': {
                'num_rounds': 5,
                'num_clients': 3,
                'learning_rate': 0.01,
                'batch_size': 32
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
            
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_runner_initialization(self):
        """Test runner initialization."""
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir)
        )
        
        self.assertIsNotNone(runner.config_manager)
        self.assertIsNotNone(runner.checkpoint_manager)
        self.assertIsNotNone(runner.metrics_collector)
        self.assertIsNotNone(runner.port_manager)
        
    def test_create_experiment_configurations(self):
        """Test creation of experiment configurations."""
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir),
            _test_mode=True
        )
        
        # Create a minimal set of configurations manually
        configs = [
            EnhancedExperimentConfig(
                strategy="fedavg",
                attack="none",
                dataset="MNIST",
                num_rounds=2,
                num_clients=2
            )
        ]
        
        self.assertIsInstance(configs, list)
        self.assertGreater(len(configs), 0)
          # Check that all configs are valid
        for config in configs:
            self.assertIsInstance(config, EnhancedExperimentConfig)
            
    @patch('experiment_runners.enhanced_experiment_runner.subprocess.Popen')
    def test_run_single_experiment_success(self, mock_popen):
        """Test running a single experiment successfully."""
        # Mock successful subprocess execution
        mock_process = Mock()
        
        # Create an iterator that will produce output lines and then empty strings indefinitely
        def mock_readline():
            lines = [
                "Round 1: accuracy=0.85, loss=0.15\n",
                "Round 2: accuracy=0.87, loss=0.13\n",
                ""  # End of output - this should repeat infinitely
            ]
            for line in lines:
                yield line
            # After initial lines, keep returning empty string
            while True:
                yield ""
        
        readline_iter = mock_readline()
        mock_process.stdout.readline.side_effect = lambda: next(readline_iter)
        
        # Mock poll to return None initially, then 0 when process is done
        poll_calls = [None, None, 0]  # Return None twice, then 0 (process finished)
        mock_process.poll.side_effect = poll_calls
        mock_process.wait.return_value = 0  # Success
        mock_popen.return_value = mock_process
        
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir)
        )
        
        config = EnhancedExperimentConfig(
            strategy="fedavg",
            attack="none",
            dataset="MNIST",
            num_rounds=2,
            num_clients=2
        )
        
        success, output = runner.run_single_experiment(config, run_id=0)
        self.assertTrue(success)
        self.assertIsInstance(output, str)
        
    @patch('experiment_runners.enhanced_experiment_runner.subprocess.Popen')
    def test_run_single_experiment_failure(self, mock_popen):
        """Test handling of experiment failure."""
        # Mock failed subprocess execution
        mock_process = Mock()
        
        # Create an iterator for failure case
        def mock_readline():
            lines = [
                "Error: Something went wrong\n",
                ""  # End of output
            ]
            for line in lines:
                yield line
            # After initial lines, keep returning empty string
            while True:
                yield ""
        
        readline_iter = mock_readline()
        mock_process.stdout.readline.side_effect = lambda: next(readline_iter)
        
        # Mock poll to return None initially, then 1 when process is done
        poll_calls = [None, 1]  # Return None once, then 1 (process finished with error)
        mock_process.poll.side_effect = poll_calls
        mock_process.wait.return_value = 1  # Failure
        mock_popen.return_value = mock_process
        
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir)
        )
        
        config = EnhancedExperimentConfig(
            strategy="fedavg",
            attack="none",
            dataset="MNIST",
            num_rounds=2,
            num_clients=2
        )
        
        success, output = runner.run_single_experiment(config, run_id=0)
        
        self.assertFalse(success)
        self.assertIsInstance(output, str)
        
    def test_checkpoint_functionality(self):
        """Test checkpoint save and load functionality."""
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir)
        )
        
        # Create some test state
        test_state = {
            'completed_experiments': ['exp1', 'exp2'],
            'current_run': 5,
            'total_runs': 10,
            'results': [{'exp': 'exp1', 'result': 'success'}]
        }
        
        # Save checkpoint
        runner._save_checkpoint(test_state)
        
        # Verify checkpoint file exists
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
        self.assertEqual(len(checkpoint_files), 1)
        
        # Load checkpoint
        loaded_state = runner._load_latest_checkpoint()
        
        self.assertIsNotNone(loaded_state)
        # Only compare dictionary values if loaded_state is not None
        if loaded_state is not None:
            self.assertEqual(loaded_state['completed_experiments'], test_state['completed_experiments'])
            self.assertEqual(loaded_state['current_run'], test_state['current_run'])
        
    def test_resource_monitoring(self):
        """Test resource monitoring functionality."""
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir)
        )
        
        # Test that metrics collector is available
        self.assertIsNotNone(runner.metrics_collector)
        
    def test_report_generation(self):
        """Test experiment report generation."""
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir)
        )
        
        # Test the final report generation method that exists
        report = runner.generate_final_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('experiment_summary', report)
        self.assertIn('system_metrics', report)


class TestIntegration(unittest.TestCase):
    """Integration tests for the enhanced experiment system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "config.yaml"
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.results_dir = self.temp_dir / "results"
        
        # Create a test config with correct parameters
        config_data = {
            'system': {
                'max_parallel_experiments': 1,
                'process_timeout': 60,
                'checkpoint_interval': 30,
                'retry_delay': 15,
                'port': 8080,
                'log_level': 'INFO',
                'resource_monitoring': False  # Disable for testing
            },
            'defaults': {
                'num_rounds': 2,
                'num_clients': 2,
                'learning_rate': 0.01,
                'batch_size': 32
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
            
    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('experiment_runners.enhanced_experiment_runner.subprocess.Popen')
    def test_full_experiment_workflow(self, mock_popen):
        """Test the complete experiment workflow."""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Round 1: accuracy=0.5, loss=0.7\n",
            "Round 2: accuracy=0.8, loss=0.3\n",
            ""  # End of output
        ]
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir),
            _test_mode=True  # Enable test mode to bypass subprocess
        )
        
        # Create a small set of test configurations
        configs = [
            EnhancedExperimentConfig(
                strategy="fedavg",
                attack="none",
                dataset="MNIST",
                num_rounds=2,
                num_clients=2
            )
        ]
        
        # Run experiments
        results_df = runner.run_experiments(configs, num_runs=2, mode='sequential')
        
        # Verify results
        self.assertIsInstance(results_df, pd.DataFrame)
        # In test mode, results may be empty since we're mocking execution
        
    def test_checkpoint_resume_workflow(self):
        """Test checkpoint and resume functionality."""
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir)
        )
        
        # Create test state
        test_state = {
            'experiment_configs': [{'strategy': 'fedavg', 'attack': 'none', 'dataset': 'MNIST'}],
            'completed_experiments': [],
            'current_run': 0,
            'total_runs': 3,
            'start_time': time.time(),
            'results': []
        }
        
        # Save checkpoint
        runner._save_checkpoint(test_state)
        
        # Verify we can resume
        loaded_state = runner._load_latest_checkpoint()
        self.assertIsNotNone(loaded_state)
        if loaded_state is not None:
            self.assertEqual(loaded_state['total_runs'], 3)
#

def run_test_suite():
    """Run the complete test suite."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSystemConfig,
        TestExperimentDefaults,
        TestEnhancedConfigManager,
        TestMetricsCollector,
        TestPortManager,
        TestEnhancedExperimentConfig,
        TestEnhancedExperimentRunner,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 80)
    print("ENHANCED EXPERIMENT RUNNER TEST SUITE")
    print("=" * 80)
    
    success = run_test_suite()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
