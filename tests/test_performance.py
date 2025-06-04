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
import sys
import threading
import concurrent.futures
from pathlib import Path
from unittest.mock import patch, Mock
import yaml
import pandas as pd
import psutil

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from enhanced_experiment_runner import (
    EnhancedExperimentRunner, EnhancedExperimentConfig,
    MetricsCollector, PortManager, ResourceError
)


class TestPerformance(unittest.TestCase):
    """Test system performance with realistic workloads."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "config.yaml"
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.results_dir = self.temp_dir / "results"
        
        # Create performance-optimized config
        config_data = {
            'system': {
                'max_parallel_experiments': 4,
                'default_timeout': 120,
                'checkpoint_interval': 30,
                'resource_check_interval': 10,
                'memory_limit_gb': 8.0,
                'cpu_limit_percent': 80.0
            },
            'experiment_defaults': {
                'num_rounds': 5,
                'num_clients': 3,
                'fraction_fit': 1.0,
                'fraction_evaluate': 1.0,
                'min_fit_clients': 3,
                'min_evaluate_clients': 3,
                'min_available_clients': 3
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
            
    def tearDown(self):
        """Clean up performance test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_large_experiment_set_creation(self):
        """Test creating a large set of experiment configurations."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
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
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large result set
        large_results = []
        for i in range(10000):  # 10,000 experiment results
            result = {
                'experiment_id': f'exp_{i}',
                'run_id': i % 10,
                'strategy': 'fedavg',
                'attack': 'none',
                'dataset': 'MNIST',
                'success': True,
                'execution_time': 100.0 + i * 0.1,
                'final_accuracy': 0.8 + (i % 100) * 0.001,
                'final_loss': 0.2 - (i % 100) * 0.001,
                'round_accuracies': [0.5 + j * 0.01 for j in range(20)],
                'round_losses': [0.5 - j * 0.005 for j in range(20)]
            }
            large_results.append(result)
            
        # Convert to DataFrame (simulating real usage)
        df = pd.DataFrame(large_results)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage increased by {memory_increase:.1f} MB for 10,000 results")
        
        # Should not use excessive memory (< 500 MB for 10,000 results)
        self.assertLess(memory_increase, 500.0)
        
    def test_concurrent_port_allocation(self):
        """Test port allocation under concurrent access."""
        port_manager = PortManager(start_port=9000, port_range=100)
        
        allocated_ports = []
        allocation_times = []
        
        def allocate_ports(count):
            """Allocate multiple ports and measure time."""
            local_ports = []
            start_time = time.time()
            
            for _ in range(count):
                try:
                    port = port_manager.allocate_port()
                    local_ports.append(port)
                except ResourceError:
                    break
                    
            end_time = time.time()
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
            port_manager.deallocate_port(port)
            
        avg_allocation_time = sum(allocation_times) / len(allocation_times)
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
        collector.start_monitoring()
        start_time = time.time()
        for _ in range(1000):
            # Simulate same work
            sum(range(100))
        monitoring_time = time.time() - start_time
        collector.stop_monitoring()
        
        # Calculate overhead
        overhead = monitoring_time - no_monitoring_time
        overhead_percent = (overhead / no_monitoring_time) * 100
        
        print(f"Monitoring overhead: {overhead:.3f}s ({overhead_percent:.1f}%)")
        
        # Overhead should be minimal (< 50%)
        self.assertLess(overhead_percent, 50.0)
        
    @patch('enhanced_experiment_runner.subprocess.run')
    def test_parallel_experiment_performance(self, mock_subprocess):
        """Test performance of parallel experiment execution."""
        # Mock subprocess with varying execution times
        def mock_run(*args, **kwargs):
            # Simulate varying execution times
            time.sleep(0.1)  # Simulate quick experiment
            return Mock(
                returncode=0,
                stdout="Round 1: accuracy=0.8, loss=0.2",
                stderr=""
            )
            
        mock_subprocess.side_effect = mock_run
        
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Create moderate-sized experiment set
        configs = []
        for i in range(8):  # 8 experiments
            config = EnhancedExperimentConfig(
                strategy="fedavg",
                attack="none",
                dataset="MNIST",
                num_rounds=2,
                num_clients=2
            )
            configs.append(config)
            
        # Test sequential execution
        start_time = time.time()
        results_seq = runner.run_experiments(configs, num_runs=1, mode='sequential')
        sequential_time = time.time() - start_time
        
        # Test parallel execution
        start_time = time.time()
        results_par = runner.run_experiments(configs, num_runs=1, mode='parallel')
        parallel_time = time.time() - start_time
        
        # Parallel should be faster (allowing for overhead)
        speedup = sequential_time / parallel_time
        print(f"Parallel speedup: {speedup:.2f}x")
        
        # Should achieve some speedup (> 1.5x for 4 parallel experiments)
        self.assertGreater(speedup, 1.5)
        
        # Results should be equivalent
        self.assertEqual(len(results_seq), len(results_par))


class TestStressConditions(unittest.TestCase):
    """Test system behavior under stress conditions."""
    
    def setUp(self):
        """Set up stress test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "config.yaml"
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.results_dir = self.temp_dir / "results"
        
        # Create stress test config
        config_data = {
            'system': {
                'max_parallel_experiments': 2,
                'default_timeout': 30,
                'checkpoint_interval': 5,
                'resource_check_interval': 2,
                'memory_limit_gb': 1.0,  # Low limit for stress testing
                'cpu_limit_percent': 60.0
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
        """Clean up stress test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_resource_exhaustion_handling(self):
        """Test system behavior when resources are exhausted."""
        port_manager = PortManager(start_port=10000, port_range=5)  # Very limited range
        
        allocated_ports = []
        
        # Allocate all available ports
        for _ in range(5):
            port = port_manager.allocate_port()
            allocated_ports.append(port)
            
        # Next allocation should raise ResourceError
        with self.assertRaises(ResourceError):
            port_manager.allocate_port()
            
        # After deallocating, should work again
        port_manager.deallocate_port(allocated_ports[0])
        new_port = port_manager.allocate_port()
        self.assertIsNotNone(new_port)
        
        # Clean up
        for port in allocated_ports[1:] + [new_port]:
            port_manager.deallocate_port(port)
            
    def test_high_frequency_checkpointing(self):
        """Test system stability with very frequent checkpointing."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Create rapid checkpoint saves
        start_time = time.time()
        for i in range(50):  # 50 rapid checkpoints
            state = {
                'experiment_configs': [],
                'completed_experiments': [],
                'current_run': i,
                'total_runs': 50,
                'start_time': start_time,
                'results': [{'id': j, 'data': f'result_{j}'} for j in range(i)]
            }
            runner._save_checkpoint(state)
            
        # System should remain stable
        final_state = runner._load_latest_checkpoint()
        self.assertEqual(final_state['current_run'], 49)
        
        # Check that cleanup works with many files
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
        initial_count = len(checkpoint_files)
        
        runner._cleanup_old_checkpoints(keep_count=10)
        
        remaining_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
        self.assertLessEqual(len(remaining_files), 10)
        
    def test_memory_pressure_simulation(self):
        """Test system behavior under memory pressure."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Create large data structures to simulate memory pressure
        large_data = []
        try:
            for i in range(1000):  # Create large result sets
                large_result = {
                    'experiment_id': f'memory_test_{i}',
                    'large_data': ['x' * 1000] * 100,  # 100KB per result
                    'timestamp': time.time()
                }
                large_data.append(large_result)
                
                # Test that basic operations still work
                if i % 100 == 0:
                    state = {
                        'experiment_configs': [],
                        'completed_experiments': [],
                        'current_run': i,
                        'total_runs': 1000,
                        'start_time': time.time(),
                        'results': large_data[-10:]  # Keep only recent results
                    }
                    runner._save_checkpoint(state)
                    
        except MemoryError:
            # Expected under extreme memory pressure
            pass
            
        # System should still be functional
        final_state = runner._load_latest_checkpoint()
        self.assertIsNotNone(final_state)
        
    @patch('enhanced_experiment_runner.subprocess.run')
    def test_experiment_failure_cascade(self, mock_subprocess):
        """Test handling of cascading experiment failures."""
        # Mock subprocess to always fail
        mock_subprocess.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Simulated failure"
        )
        
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        configs = []
        for i in range(5):
            config = EnhancedExperimentConfig(
                strategy="fedavg",
                attack="none",
                dataset="MNIST",
                num_rounds=2,
                num_clients=2
            )
            configs.append(config)
            
        # Run experiments (all will fail)
        start_time = time.time()
        results_df = runner.run_experiments(configs, num_runs=1, mode='sequential')
        execution_time = time.time() - start_time
        
        # System should handle all failures gracefully
        self.assertIsInstance(results_df, pd.DataFrame)
        
        # Should complete in reasonable time despite failures
        self.assertLess(execution_time, 60.0)  # Should not hang
        
        # Check that failures were recorded
        failed_results = results_df[results_df['success'] == False]
        self.assertGreater(len(failed_results), 0)
        
    def test_concurrent_checkpoint_access(self):
        """Test checkpoint system under concurrent access."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        def concurrent_checkpoint_operations(thread_id):
            """Perform checkpoint operations concurrently."""
            for i in range(10):
                state = {
                    'experiment_configs': [],
                    'completed_experiments': [],
                    'current_run': i,
                    'total_runs': 10,
                    'thread_id': thread_id,
                    'start_time': time.time(),
                    'results': []
                }
                
                # Save checkpoint
                runner._save_checkpoint(state)
                
                # Small delay
                time.sleep(0.01)
                
                # Try to load checkpoint
                loaded_state = runner._load_latest_checkpoint()
                self.assertIsNotNone(loaded_state)
                
        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_checkpoint_operations, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # System should remain stable
        final_state = runner._load_latest_checkpoint()
        self.assertIsNotNone(final_state)


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
        sys.exit(0)
    else:
        print("\n❌ Some performance tests failed!")
        sys.exit(1)
