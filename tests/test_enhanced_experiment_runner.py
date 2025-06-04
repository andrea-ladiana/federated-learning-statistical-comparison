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

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from enhanced_experiment_runner import (
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
            default_timeout=1800,
            checkpoint_interval=300,
            resource_check_interval=60
        )
        self.assertEqual(config.max_parallel_experiments, 4)
        self.assertEqual(config.default_timeout, 1800)
        
    def test_invalid_parallel_experiments(self):
        """Test validation of parallel experiments count."""
        with self.assertRaises(ValueError):
            SystemConfig(max_parallel_experiments=0)
            
        with self.assertRaises(ValueError):
            SystemConfig(max_parallel_experiments=-1)
    
    def test_invalid_timeout(self):
        """Test validation of timeout values."""
        with self.assertRaises(ValueError):
            SystemConfig(default_timeout=0)
            
    def test_invalid_intervals(self):
        """Test validation of interval values."""
        with self.assertRaises(ValueError):
            SystemConfig(checkpoint_interval=0)
            
        with self.assertRaises(ValueError):
            SystemConfig(resource_check_interval=0)


class TestExperimentDefaults(unittest.TestCase):
    """Test the ExperimentDefaults dataclass."""
    
    def test_valid_defaults_creation(self):
        """Test creating valid experiment defaults."""
        defaults = ExperimentDefaults(
            num_rounds=10,
            num_clients=5,
            fraction_fit=0.8,
            fraction_evaluate=0.6
        )
        self.assertEqual(defaults.num_rounds, 10)
        self.assertEqual(defaults.num_clients, 5)
        
    def test_invalid_rounds(self):
        """Test validation of round numbers."""
        with self.assertRaises(ValueError):
            ExperimentDefaults(num_rounds=0)
            
    def test_invalid_clients(self):
        """Test validation of client numbers."""
        with self.assertRaises(ValueError):
            ExperimentDefaults(num_clients=0)
            
    def test_invalid_fractions(self):
        """Test validation of fraction values."""
        with self.assertRaises(ValueError):
            ExperimentDefaults(fraction_fit=1.5)
            
        with self.assertRaises(ValueError):
            ExperimentDefaults(fraction_evaluate=-0.1)


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
        manager.create_default_config()
        
        self.assertTrue(self.config_file.exists())
        
        # Load and verify the config
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        self.assertIn('system', config)
        self.assertIn('experiment_defaults', config)
        
    def test_load_valid_config(self):
        """Test loading a valid configuration."""
        # Create a test config
        test_config = {
            'system': {
                'max_parallel_experiments': 2,
                'default_timeout': 1200,
                'checkpoint_interval': 300,
                'resource_check_interval': 60,
                'memory_limit_gb': 8.0,
                'cpu_limit_percent': 80.0
            },
            'experiment_defaults': {
                'num_rounds': 10,
                'num_clients': 5,
                'fraction_fit': 0.8,
                'fraction_evaluate': 0.6,
                'min_fit_clients': 4,
                'min_evaluate_clients': 3,
                'min_available_clients': 5
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
            
        manager = EnhancedConfigManager(self.config_file)
        loaded_config = manager.load_config()
        
        self.assertEqual(loaded_config.system.max_parallel_experiments, 2)
        self.assertEqual(loaded_config.experiment_defaults.num_rounds, 10)
        
    def test_invalid_config_validation(self):
        """Test validation of invalid configurations."""
        # Create an invalid config
        invalid_config = {
            'system': {
                'max_parallel_experiments': -1,  # Invalid
                'default_timeout': 1200,
                'checkpoint_interval': 300,
                'resource_check_interval': 60,
                'memory_limit_gb': 8.0,
                'cpu_limit_percent': 80.0
            },
            'experiment_defaults': {
                'num_rounds': 10,
                'num_clients': 5,
                'fraction_fit': 0.8,
                'fraction_evaluate': 0.6,
                'min_fit_clients': 4,
                'min_evaluate_clients': 3,
                'min_available_clients': 5
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(invalid_config, f)
            
        manager = EnhancedConfigManager(self.config_file)
        
        with self.assertRaises(ValueError):
            manager.load_config()


class TestMetricsCollector(unittest.TestCase):
    """Test the MetricsCollector class."""
    
    def setUp(self):
        """Set up test environment."""
        self.collector = MetricsCollector()
        
    def test_start_stop_monitoring(self):
        """Test starting and stopping resource monitoring."""
        self.collector.start_monitoring()
        self.assertTrue(self.collector.monitoring)
        
        time.sleep(0.1)  # Let it collect some data
        
        self.collector.stop_monitoring()
        self.assertFalse(self.collector.monitoring)
        
        # Check that some metrics were collected
        metrics = self.collector.get_metrics()
        self.assertIn('cpu_usage', metrics)
        self.assertIn('memory_usage', metrics)
        
    def test_get_current_usage(self):
        """Test getting current system usage."""
        cpu, memory = self.collector.get_current_usage()
        
        self.assertIsInstance(cpu, float)
        self.assertIsInstance(memory, float)
        self.assertGreaterEqual(cpu, 0.0)
        self.assertGreaterEqual(memory, 0.0)
        
    def test_resource_metrics_format(self):
        """Test the format of collected metrics."""
        self.collector.start_monitoring()
        time.sleep(0.1)
        self.collector.stop_monitoring()
        
        metrics = self.collector.get_metrics()
        
        # Check structure
        self.assertIsInstance(metrics['cpu_usage'], list)
        self.assertIsInstance(metrics['memory_usage'], list)
        self.assertIsInstance(metrics['timestamps'], list)
        
        # Check that lists have same length
        self.assertEqual(len(metrics['cpu_usage']), len(metrics['timestamps']))
        self.assertEqual(len(metrics['memory_usage']), len(metrics['timestamps']))


class TestPortManager(unittest.TestCase):
    """Test the PortManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.port_manager = PortManager(start_port=8000, port_range=100)
        
    def test_allocate_deallocate_port(self):
        """Test port allocation and deallocation."""
        port = self.port_manager.allocate_port()
        
        self.assertIsInstance(port, int)
        self.assertGreaterEqual(port, 8000)
        self.assertLess(port, 8100)
        
        # Port should be in use
        self.assertIn(port, self.port_manager.used_ports)
        
        # Deallocate the port
        self.port_manager.deallocate_port(port)
        self.assertNotIn(port, self.port_manager.used_ports)
        
    def test_port_exhaustion(self):
        """Test behavior when all ports are exhausted."""
        # Allocate all ports
        allocated_ports = []
        for _ in range(100):  # port_range = 100
            port = self.port_manager.allocate_port()
            allocated_ports.append(port)
            
        # Next allocation should raise an exception
        with self.assertRaises(ResourceError):
            self.port_manager.allocate_port()
            
        # Clean up
        for port in allocated_ports:
            self.port_manager.deallocate_port(port)
            
    def test_deallocate_unallocated_port(self):
        """Test deallocating a port that wasn't allocated."""
        # This should not raise an error, just be ignored
        self.port_manager.deallocate_port(9999)


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
        # Valid parameters
        config = EnhancedExperimentConfig(
            strategy="fedprox",
            attack="noise",
            dataset="MNIST",
            strategy_params={"proximal_mu": 0.01},
            attack_params={"noise_std": 0.1, "noise_fraction": 0.3}
        )
        
        # Invalid parameters should raise validation error
        with self.assertRaises(ExperimentValidationError):
            EnhancedExperimentConfig(
                strategy="fedprox",
                attack="noise",
                dataset="MNIST",
                strategy_params={"proximal_mu": -1.0}  # Invalid negative value
            )


class TestEnhancedExperimentRunner(unittest.TestCase):
    """Test the EnhancedExperimentRunner class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "config.yaml"
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.results_dir = self.temp_dir / "results"
        
        # Create a minimal config file
        config_data = {
            'system': {
                'max_parallel_experiments': 2,
                'default_timeout': 300,
                'checkpoint_interval': 60,
                'resource_check_interval': 30,
                'memory_limit_gb': 4.0,
                'cpu_limit_percent': 70.0
            },
            'experiment_defaults': {
                'num_rounds': 5,
                'num_clients': 3,
                'fraction_fit': 0.8,
                'fraction_evaluate': 0.6,
                'min_fit_clients': 2,
                'min_evaluate_clients': 2,
                'min_available_clients': 3
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
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        self.assertIsNotNone(runner.config_manager)
        self.assertIsNotNone(runner.checkpoint_manager)
        self.assertIsNotNone(runner.metrics_collector)
        self.assertIsNotNone(runner.port_manager)
        
    def test_create_experiment_configurations(self):
        """Test creation of experiment configurations."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        configs = runner.create_experiment_configurations(test_mode=True)
        
        self.assertIsInstance(configs, list)
        self.assertGreater(len(configs), 0)
        
        # Check that all configs are valid
        for config in configs:
            self.assertIsInstance(config, EnhancedExperimentConfig)
            
    @patch('enhanced_experiment_runner.subprocess.run')
    def test_run_single_experiment_success(self, mock_subprocess):
        """Test running a single experiment successfully."""
        # Mock successful subprocess execution
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Experiment completed successfully",
            stderr=""
        )
        
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        config = EnhancedExperimentConfig(
            strategy="fedavg",
            attack="none",
            dataset="MNIST",
            num_rounds=2,
            num_clients=2
        )
        
        result = runner._run_single_experiment(config, run_id=0)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('execution_time', result)
        self.assertIn('metrics', result)
        
    @patch('enhanced_experiment_runner.subprocess.run')
    def test_run_single_experiment_failure(self, mock_subprocess):
        """Test handling of experiment failure."""
        # Mock failed subprocess execution
        mock_subprocess.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Error: Something went wrong"
        )
        
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        config = EnhancedExperimentConfig(
            strategy="fedavg",
            attack="none",
            dataset="MNIST",
            num_rounds=2,
            num_clients=2
        )
        
        result = runner._run_single_experiment(config, run_id=0)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertFalse(result['success'])
        self.assertIn('error_message', result)
        
    def test_checkpoint_functionality(self):
        """Test checkpoint save and load functionality."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
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
        
        self.assertEqual(loaded_state['completed_experiments'], test_state['completed_experiments'])
        self.assertEqual(loaded_state['current_run'], test_state['current_run'])
        
    def test_resource_monitoring(self):
        """Test resource monitoring functionality."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Test resource check
        cpu, memory = runner._check_resources()
        
        self.assertIsInstance(cpu, float)
        self.assertIsInstance(memory, float)
        self.assertGreaterEqual(cpu, 0.0)
        self.assertGreaterEqual(memory, 0.0)
        
    def test_report_generation(self):
        """Test experiment report generation."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Create some mock results
        mock_results = [
            {
                'experiment_id': 'fedavg_none_MNIST',
                'run_id': 0,
                'success': True,
                'execution_time': 120.5,
                'final_accuracy': 0.95,
                'final_loss': 0.05
            },
            {
                'experiment_id': 'fedavg_none_MNIST',
                'run_id': 1,
                'success': True,
                'execution_time': 115.2,
                'final_accuracy': 0.94,
                'final_loss': 0.06
            }
        ]
        
        df = pd.DataFrame(mock_results)
        report = runner._generate_experiment_report(df)
        
        self.assertIsInstance(report, dict)
        self.assertIn('summary', report)
        self.assertIn('statistics', report)
        self.assertIn('performance_metrics', report)


class TestIntegration(unittest.TestCase):
    """Integration tests for the enhanced experiment system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "config.yaml"
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.results_dir = self.temp_dir / "results"
        
        # Create a test config
        config_data = {
            'system': {
                'max_parallel_experiments': 1,
                'default_timeout': 60,
                'checkpoint_interval': 30,
                'resource_check_interval': 15,
                'memory_limit_gb': 2.0,
                'cpu_limit_percent': 50.0
            },
            'experiment_defaults': {
                'num_rounds': 2,
                'num_clients': 2,
                'fraction_fit': 1.0,
                'fraction_evaluate': 1.0,
                'min_fit_clients': 2,
                'min_evaluate_clients': 2,
                'min_available_clients': 2
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
            
    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('enhanced_experiment_runner.subprocess.run')
    def test_full_experiment_workflow(self, mock_subprocess):
        """Test the complete experiment workflow."""
        # Mock successful subprocess execution
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Round 1: accuracy=0.5, loss=0.7\nRound 2: accuracy=0.8, loss=0.3",
            stderr=""
        )
        
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
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
        self.assertGreater(len(results_df), 0)
        
        # Check that results were saved
        result_files = list(self.results_dir.glob("*.csv"))
        self.assertGreater(len(result_files), 0)
        
    def test_checkpoint_resume_workflow(self):
        """Test checkpoint and resume functionality."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
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
        self.assertEqual(loaded_state['total_runs'], 3)


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
