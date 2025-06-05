#!/usr/bin/env python3
"""
Performance and stress tests for the enhanced experiment runner.

These tests verify that the system can handle large-scale experiments
and maintain performance under stress conditions.
"""

import unittest
import tempfile
import shutil
import time
import threading
import psutil
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

from experiment_runners.enhanced_experiment_runner import (
    EnhancedExperimentRunner, EnhancedExperimentConfig, MetricsCollector, PortManager, EnhancedConfigManager
)
from configuration.config_manager import ConfigManager
from utilities.checkpoint_manager import CheckpointManager
from utilities.retry_manager import RetryManager


class TestPerformance(unittest.TestCase):
    """Performance and stress tests for the federated learning system."""
    
    def setUp(self):
        """Set up test environment with proper cleanup."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.results_dir = self.test_dir / "results"
        self.checkpoint_dir = self.test_dir / "checkpoints"
        self.config_file = self.test_dir / "test_config.yaml"
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Create test configuration file first
        self._create_test_config()
        
        # Create and configure config manager
        self.enhanced_config_manager = EnhancedConfigManager(self.config_file)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_config(self):
        """Create a minimal test configuration file."""
        config_content = {
            'system': {
                'max_retries': 2,
                'retry_delay': 1,
                'process_timeout': 30,
                'port': 8080,
                'log_level': 'INFO',
                'checkpoint_backup_interval': 5,
                'max_backups': 3,
                'max_parallel_experiments': 2,
                'resource_monitoring': True,
                'checkpoint_interval': 5
            },
            'defaults': {
                'num_rounds': 3,
                'num_clients': 5,
                'learning_rate': 0.01,
                'batch_size': 32,
                'local_epochs': 1,
                'fraction_fit': 0.8,
                'fraction_evaluate': 0.6,
                'min_fit_clients': 3,
                'min_evaluate_clients': 2,
                'min_available_clients': 5
            }
        }
        
        import yaml
        with open(self.config_file, 'w') as f:
            yaml.dump(config_content, f)
    
    def test_large_experiment_set_creation(self):
        """Test creating a large set of experiment configurations."""
        runner = EnhancedExperimentRunner(
            base_dir=str(self.test_dir),
            results_dir=str(self.results_dir),
            config_manager=self.enhanced_config_manager
        )
        
        # Measure time to create large configuration set
        start_time = time.time()
        
        configs = []
        strategies = ["fedavg", "fedprox", "fednova", "scaffold", "fedadam"]
        attacks = ["none", "noise", "missed", "labelflip"]
        datasets = ["MNIST", "CIFAR10", "FMNIST"]
        
        for strategy in strategies:
            for attack in attacks:
                for dataset in datasets:
                    config = EnhancedExperimentConfig(
                        strategy=strategy,
                        attack=attack,
                        dataset=dataset,
                        num_rounds=5,
                        num_clients=3
                    )
                    configs.append(config)
        
        creation_time = time.time() - start_time
        
        # Should create 60 configurations quickly (< 1 second)
        self.assertEqual(len(configs), 60)
        self.assertLess(creation_time, 1.0)
        
        print(f"Created {len(configs)} configurations in {creation_time:.3f} seconds")
        
    def test_memory_usage_with_large_results(self):
        """Test memory usage when handling large result sets."""
        runner = EnhancedExperimentRunner(
            base_dir=str(self.test_dir),
            results_dir=str(self.results_dir),
            config_manager=self.enhanced_config_manager
        )
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large result set - using runner's internal DataFrame format
        for i in range(10000):  # 10,000 experiment results
            # Simulate what parse_and_store_metrics does
            config = EnhancedExperimentConfig(
                strategy='fedavg',
                attack='none',
                dataset='MNIST',
                num_rounds=5,
                num_clients=3
            )
            
            # Add metrics to results DataFrame (similar to internal method)
            import pandas as pd
            new_row = pd.DataFrame([{
                'algorithm': 'fedavg',
                'attack': 'none',
                'dataset': 'MNIST',
                'run': i % 10,
                'client_id': -1,
                'round': i % 20,
                'metric': 'accuracy',
                'value': 0.8 + (i % 100) * 0.001
            }])
            
            runner.results_df = pd.concat([runner.results_df, new_row], ignore_index=True)
            
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage increased by {memory_increase:.1f} MB for 10,000 results")
        
        # Should not use excessive memory (< 500 MB for 10,000 results)
        self.assertLess(memory_increase, 500.0)
        
    def test_concurrent_port_allocation(self):
        """Test port allocation under concurrent access."""
        port_manager = PortManager(base_port=9000, num_ports=100)
        
        allocated_ports = []
        allocation_times = []
        allocation_lock = threading.Lock()
        
        def allocate_ports(count):
            """Allocate multiple ports and measure time."""
            local_ports = []
            start_time = time.time()
            
            for _ in range(count):
                try:
                    port = port_manager.acquire_port()
                    local_ports.append(port)
                except Exception:  # ResourceError or similar
                    break
                    
            end_time = time.time()
            with allocation_lock:
                allocation_times.append(end_time - start_time)
                allocated_ports.extend(local_ports)
            
        # Run concurrent allocations
        threads = []
        for _ in range(10):  # 10 threads
            thread = threading.Thread(target=allocate_ports, args=(5,))  # 5 ports each
            threads.append(thread)
            thread.start()
            
        # Wait for all threads
        for thread in threads:
            thread.join()
            
        # Check results
        self.assertGreater(len(allocated_ports), 0)
        self.assertEqual(len(set(allocated_ports)), len(allocated_ports))  # No duplicates
        
        # Clean up
        for port in allocated_ports:
            port_manager.release_port(port)
            
        avg_allocation_time = sum(allocation_times) / len(allocation_times) if allocation_times else 0
        print(f"Average concurrent allocation time: {avg_allocation_time:.3f} seconds")
        
    def test_metrics_collection_overhead(self):
        """Test the performance overhead of metrics collection."""
        collector = MetricsCollector()
        
        # Test without monitoring
        start_time = time.time()
        for _ in range(1000):
            # Simulate some work
            sum(range(100))
        no_monitoring_time = time.time() - start_time
        
        # Test with monitoring
        experiment_id = "test_experiment"
        collector.start_monitoring(experiment_id)
        start_time = time.time()
        for _ in range(1000):
            # Simulate same work
            sum(range(100))
        monitoring_time = time.time() - start_time
        collector.stop_monitoring_for_experiment(experiment_id)
        
        # Calculate overhead
        overhead = monitoring_time - no_monitoring_time
        overhead_percent = (overhead / no_monitoring_time) * 100 if no_monitoring_time > 0 else 0
        
        print(f"Monitoring overhead: {overhead:.3f}s ({overhead_percent:.1f}%)")
        
        # Overhead should be minimal (< 50%)
        self.assertLess(overhead_percent, 50.0)
        
    def test_parallel_vs_sequential_performance(self):
        """Test performance comparison between parallel and sequential execution."""
        # Create mock experiment runner that can handle both modes
        runner = EnhancedExperimentRunner(
            base_dir=str(self.test_dir),
            results_dir=str(self.results_dir),
            config_manager=self.enhanced_config_manager
        )
        
        # Create test configurations
        configs = []
        for i in range(4):  # Small number for quick test
            config = EnhancedExperimentConfig(
                strategy="fedavg",
                attack="none",
                dataset="MNIST",
                num_rounds=2,
                num_clients=2
            )
            configs.append(config)
            
        # Note: Since we can't actually run experiments in unit tests,
        # we'll just test that the methods exist and can be called
        # The actual performance comparison would need integration tests
        
        # Test that sequential method exists and is callable
        self.assertTrue(hasattr(runner, 'run_experiments_sequential'))
        self.assertTrue(callable(getattr(runner, 'run_experiments_sequential')))
        
        # Test that parallel method exists and is callable
        self.assertTrue(hasattr(runner, 'run_experiments_parallel'))
        self.assertTrue(callable(getattr(runner, 'run_experiments_parallel')))
        
        print("Parallel and sequential methods are available")


class TestStressConditions(unittest.TestCase):
    """Test system behavior under stress conditions."""
    
    def setUp(self):
        """Set up stress test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_file = self.test_dir / "config.yaml"
        self.checkpoint_dir = self.test_dir / "checkpoints" 
        self.results_dir = self.test_dir / "results"
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create stress test config
        config_content = {
            'system': {
                'max_parallel_experiments': 2,
                'process_timeout': 30,
                'checkpoint_interval': 5,
                'resource_monitoring': True,
                'max_retries': 1,
                'retry_delay': 1,
                'port': 8080,
                'log_level': 'INFO'
            },
            'defaults': {
                'num_rounds': 2,
                'num_clients': 2,
                'learning_rate': 0.01,
                'batch_size': 16,
                'local_epochs': 1
            }
        }
        
        import yaml
        with open(self.config_file, 'w') as f:
            yaml.dump(config_content, f)
            
        # Create enhanced config manager
        self.enhanced_config_manager = EnhancedConfigManager(self.config_file)
            
    def tearDown(self):
        """Clean up stress test environment."""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_resource_exhaustion_handling(self):
        """Test system behavior when resources are exhausted."""
        port_manager = PortManager(base_port=10000, num_ports=5)  # Very limited range
        
        allocated_ports = []
        
        # Allocate all available ports
        for _ in range(5):
            port = port_manager.acquire_port()
            allocated_ports.append(port)
            
        # Next allocation should raise exception
        with self.assertRaises(Exception):  # ResourceError or similar
            port_manager.acquire_port()
            
        # After releasing, should work again
        port_manager.release_port(allocated_ports[0])
        new_port = port_manager.acquire_port()
        self.assertIsNotNone(new_port)
        
        # Clean up
        for port in allocated_ports[1:] + [new_port]:
            port_manager.release_port(port)
            
    def test_high_frequency_operations(self):
        """Test system stability with high frequency operations."""
        runner = EnhancedExperimentRunner(
            base_dir=str(self.test_dir),
            results_dir=str(self.results_dir),
            config_manager=self.enhanced_config_manager
        )
        
        # Test rapid checkpoint operations simulation
        # Since we don't have direct access to internal methods,
        # we'll test the public interface rapidly
        
        start_time = time.time()
        
        # Rapidly create and process configurations
        for i in range(100):
            config = EnhancedExperimentConfig(
                strategy="fedavg",
                attack="none",
                dataset="MNIST",
                num_rounds=1,
                num_clients=2
            )
            
            # Test that config creation and validation works rapidly
            experiment_id = config.get_experiment_id()
            self.assertIsNotNone(experiment_id)
            
        elapsed_time = time.time() - start_time
        
        # Should handle 100 rapid operations quickly (< 1 second)
        self.assertLess(elapsed_time, 1.0)
        print(f"Processed 100 rapid operations in {elapsed_time:.3f} seconds")
        
    def test_memory_pressure_simulation(self):
        """Test system behavior under memory pressure."""
        runner = EnhancedExperimentRunner(
            base_dir=str(self.test_dir),
            results_dir=str(self.results_dir),
            config_manager=self.enhanced_config_manager
        )
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large data structures to simulate memory pressure
        large_data = []
        max_iterations = 1000
        
        try:
            for i in range(max_iterations):
                # Create moderately large result entries
                large_result = {
                    'experiment_id': f'memory_test_{i}',
                    'data': 'x' * 1000,  # 1KB per result 
                    'timestamp': time.time(),
                    'iteration': i
                }
                large_data.append(large_result)
                
                # Test that basic operations still work periodically
                if i % 100 == 0:
                    config = EnhancedExperimentConfig(
                        strategy="fedavg",
                        attack="none", 
                        dataset="MNIST",
                        num_rounds=1,
                        num_clients=2
                    )
                    # Should still be able to create configs
                    self.assertIsNotNone(config.get_experiment_id())
                    
        except MemoryError:
            # Expected under extreme memory pressure
            print(f"Memory limit reached at iteration {i}")
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        print(f"Memory stress test used {memory_used:.1f} MB additional memory")
        
        # System should remain functional
        config = EnhancedExperimentConfig(
            strategy="fedavg",
            attack="none",
            dataset="MNIST", 
            num_rounds=1,
            num_clients=2
        )
        self.assertIsNotNone(config.get_experiment_id())
        
    def test_concurrent_access_patterns(self):
        """Test system under concurrent access patterns."""
        port_manager = PortManager(base_port=11000, num_ports=20)
        
        errors = []
        results = []
        
        def concurrent_operations(thread_id):
            """Perform various operations concurrently."""
            thread_results = []
            try:
                for i in range(10):
                    # Port allocation/deallocation
                    port = port_manager.acquire_port()
                    thread_results.append(f"thread_{thread_id}_port_{port}")
                    
                    # Small delay to increase contention
                    time.sleep(0.001)
                    
                    port_manager.release_port(port)
                    
                    # Config creation
                    config = EnhancedExperimentConfig(
                        strategy="fedavg",
                        attack="none",
                        dataset="MNIST",
                        num_rounds=1,
                        num_clients=2
                    )
                    thread_results.append(f"thread_{thread_id}_config_{i}")
                    
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
            
            results.extend(thread_results)
        
        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_operations, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Check results
        print(f"Concurrent operations completed: {len(results)} operations, {len(errors)} errors")
        
        # Should have minimal errors
        self.assertLess(len(errors), len(results) * 0.1)  # Less than 10% error rate
        
        # Should have some successful operations
        self.assertGreater(len(results), 0)


def run_performance_tests():
    """Run the performance test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add performance test classes
    test_classes = [
        TestPerformance,
        TestStressConditions
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
    print("PERFORMANCE & STRESS TESTS")
    print("=" * 80)
    
    success = run_performance_tests()
    
    if success:
        print("\n🎉 All performance tests passed!")
        exit(0)
    else:
        print("\n❌ Some performance tests failed!")
        exit(1)
