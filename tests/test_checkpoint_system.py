#!/usr/bin/env python3
"""
Integration tests for checkpoint and resume functionality.

These tests specifically focus on the checkpoint/resume system which is
critical for long-running experiments that may span multiple days.
"""

import unittest
import tempfile
import shutil
import time
import sys
import json
import yaml
from pathlib import Path
from unittest.mock import patch, Mock
import pandas as pd

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from enhanced_experiment_runner import (
    EnhancedExperimentRunner, EnhancedExperimentConfig, 
    EnhancedConfigManager, CheckpointError
)


class TestCheckpointResumeSystem(unittest.TestCase):
    """Test the complete checkpoint and resume system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "config.yaml"
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.results_dir = self.temp_dir / "results"
        
        # Create minimal config
        config_data = {
            'system': {
                'max_parallel_experiments': 2,
                'default_timeout': 300,
                'checkpoint_interval': 10,  # Short interval for testing
                'resource_check_interval': 5,
                'memory_limit_gb': 4.0,
                'cpu_limit_percent': 70.0
            },
            'experiment_defaults': {
                'num_rounds': 3,
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
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_checkpoint_creation_and_loading(self):
        """Test that checkpoints are created and can be loaded correctly."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Create a test state to checkpoint
        test_state = {
            'experiment_configs': [
                {
                    'strategy': 'fedavg',
                    'attack': 'none',
                    'dataset': 'MNIST',
                    'num_rounds': 3,
                    'num_clients': 2
                }
            ],
            'completed_experiments': [
                {'experiment_id': 'fedavg_none_MNIST', 'run_id': 0, 'success': True}
            ],
            'current_run': 1,
            'total_runs': 5,
            'start_time': time.time(),
            'results': [
                {
                    'experiment_id': 'fedavg_none_MNIST',
                    'run_id': 0,
                    'strategy': 'fedavg',
                    'attack': 'none',
                    'dataset': 'MNIST',
                    'success': True,
                    'execution_time': 120.5,
                    'final_accuracy': 0.85,
                    'final_loss': 0.15
                }
            ]
        }
        
        # Save checkpoint
        runner._save_checkpoint(test_state)
        
        # Verify checkpoint file was created
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
        self.assertEqual(len(checkpoint_files), 1)
        
        # Load the checkpoint
        loaded_state = runner._load_latest_checkpoint()
        
        # Verify loaded state matches original
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state['current_run'], test_state['current_run'])
        self.assertEqual(loaded_state['total_runs'], test_state['total_runs'])
        self.assertEqual(len(loaded_state['completed_experiments']), 1)
        self.assertEqual(len(loaded_state['results']), 1)
        self.assertEqual(loaded_state['results'][0]['experiment_id'], 'fedavg_none_MNIST')
        
    def test_checkpoint_with_multiple_saves(self):
        """Test that multiple checkpoints are handled correctly."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Create multiple checkpoints with different states
        states = []
        for i in range(3):
            state = {
                'experiment_configs': [{'strategy': 'fedavg', 'attack': 'none', 'dataset': 'MNIST'}],
                'completed_experiments': [],
                'current_run': i,
                'total_runs': 5,
                'start_time': time.time(),
                'results': []
            }
            states.append(state)
            runner._save_checkpoint(state)
            time.sleep(0.1)  # Ensure different timestamps
            
        # Verify multiple checkpoint files exist
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
        self.assertEqual(len(checkpoint_files), 3)
        
        # Load latest checkpoint
        loaded_state = runner._load_latest_checkpoint()
        
        # Should be the last one saved (current_run = 2)
        self.assertEqual(loaded_state['current_run'], 2)
        
    def test_checkpoint_cleanup(self):
        """Test that old checkpoints are cleaned up."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Create many checkpoints
        for i in range(10):
            state = {
                'experiment_configs': [],
                'completed_experiments': [],
                'current_run': i,
                'total_runs': 10,
                'start_time': time.time(),
                'results': []
            }
            runner._save_checkpoint(state)
            time.sleep(0.01)
            
        # Cleanup should keep only the last 5 (default)
        runner._cleanup_old_checkpoints(keep_count=5)
        
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
        self.assertLessEqual(len(checkpoint_files), 5)
        
    @patch('enhanced_experiment_runner.subprocess.run')
    def test_resume_from_checkpoint(self, mock_subprocess):
        """Test resuming an experiment run from a checkpoint."""
        # Mock successful subprocess execution
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Round 1: accuracy=0.6, loss=0.4\nRound 2: accuracy=0.8, loss=0.2",
            stderr=""
        )
        
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Create initial checkpoint state (simulating partial completion)
        configs = [
            EnhancedExperimentConfig(
                strategy="fedavg",
                attack="none", 
                dataset="MNIST",
                num_rounds=3,
                num_clients=2
            ),
            EnhancedExperimentConfig(
                strategy="fedprox",
                attack="none",
                dataset="MNIST", 
                num_rounds=3,
                num_clients=2
            )
        ]
        
        initial_state = {
            'experiment_configs': [config.to_dict() for config in configs],
            'completed_experiments': [
                {'experiment_id': 'fedavg_none_MNIST', 'run_id': 0, 'success': True}
            ],
            'current_experiment_index': 0,
            'current_run': 1,
            'total_runs': 3,
            'start_time': time.time() - 3600,  # Started 1 hour ago
            'results': [
                {
                    'experiment_id': 'fedavg_none_MNIST',
                    'run_id': 0,
                    'strategy': 'fedavg',
                    'attack': 'none',
                    'dataset': 'MNIST',
                    'success': True,
                    'execution_time': 100.0,
                    'final_accuracy': 0.85,
                    'final_loss': 0.15
                }
            ]
        }
        
        # Save initial checkpoint
        runner._save_checkpoint(initial_state)
        
        # Resume and run remaining experiments
        results_df = runner.run_experiments(
            configs, 
            num_runs=3, 
            mode='sequential',
            resume_from_checkpoint=True
        )
        
        # Verify that results include both completed and newly run experiments
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertGreater(len(results_df), 1)  # Should have more than the initial result
        
        # Verify that the initial result is preserved
        fedavg_results = results_df[
            (results_df['strategy'] == 'fedavg') & 
            (results_df['attack'] == 'none') & 
            (results_df['dataset'] == 'MNIST')
        ]
        self.assertGreaterEqual(len(fedavg_results), 1)
        
    def test_checkpoint_corruption_handling(self):
        """Test handling of corrupted checkpoint files."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Create a corrupted checkpoint file
        self.checkpoint_dir.mkdir(exist_ok=True)
        corrupted_file = self.checkpoint_dir / "checkpoint_corrupted.yaml"
        with open(corrupted_file, 'w') as f:
            f.write("invalid: yaml: content: [")
            
        # Should handle corruption gracefully
        loaded_state = runner._load_latest_checkpoint()
        self.assertIsNone(loaded_state)
        
    def test_experiment_state_consistency(self):
        """Test that experiment state remains consistent across checkpoints."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        configs = [
            EnhancedExperimentConfig(
                strategy="fedavg",
                attack="none",
                dataset="MNIST",
                num_rounds=2,
                num_clients=2
            )
        ]
        
        # Create state with specific experiment progress
        state = {
            'experiment_configs': [config.to_dict() for config in configs],
            'completed_experiments': [],
            'current_experiment_index': 0,
            'current_run': 2,
            'total_runs': 5,
            'start_time': time.time(),
            'results': [
                {
                    'experiment_id': 'fedavg_none_MNIST',
                    'run_id': 0,
                    'strategy': 'fedavg',
                    'attack': 'none',
                    'dataset': 'MNIST',
                    'success': True,
                    'execution_time': 95.5,
                    'final_accuracy': 0.82,
                    'final_loss': 0.18
                },
                {
                    'experiment_id': 'fedavg_none_MNIST',
                    'run_id': 1,
                    'strategy': 'fedavg',
                    'attack': 'none',
                    'dataset': 'MNIST',
                    'success': True,
                    'execution_time': 98.2,
                    'final_accuracy': 0.84,
                    'final_loss': 0.16
                }
            ]
        }
        
        # Save and load
        runner._save_checkpoint(state)
        loaded_state = runner._load_latest_checkpoint()
        
        # Verify exact consistency
        self.assertEqual(loaded_state['current_run'], state['current_run'])
        self.assertEqual(len(loaded_state['results']), len(state['results']))
        
        # Verify results are identical
        for original, loaded in zip(state['results'], loaded_state['results']):
            self.assertEqual(original['experiment_id'], loaded['experiment_id'])
            self.assertEqual(original['run_id'], loaded['run_id'])
            self.assertEqual(original['success'], loaded['success'])
            self.assertAlmostEqual(original['execution_time'], loaded['execution_time'], places=1)
            
    @patch('enhanced_experiment_runner.subprocess.run')
    def test_experiment_failure_recovery(self, mock_subprocess):
        """Test recovery from experiment failures with checkpoints."""
        # Mock subprocess to simulate failure then success
        def side_effect(*args, **kwargs):
            # First call fails, subsequent calls succeed
            if not hasattr(side_effect, 'call_count'):
                side_effect.call_count = 0
            side_effect.call_count += 1
            
            if side_effect.call_count == 1:
                return Mock(returncode=1, stdout="", stderr="Simulated failure")
            else:
                return Mock(
                    returncode=0, 
                    stdout="Round 1: accuracy=0.7, loss=0.3",
                    stderr=""
                )
                
        mock_subprocess.side_effect = side_effect
        
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        configs = [
            EnhancedExperimentConfig(
                strategy="fedavg",
                attack="none",
                dataset="MNIST",
                num_rounds=2,
                num_clients=2
            )
        ]
        
        # Run experiment (first attempt will fail, retry should succeed)
        results_df = runner.run_experiments(configs, num_runs=2, mode='sequential')
        
        # Should have results despite the initial failure
        self.assertIsInstance(results_df, pd.DataFrame)
        
        # Check that checkpoint captured the failure and recovery
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
        self.assertGreater(len(checkpoint_files), 0)
        
    def test_long_running_experiment_simulation(self):
        """Simulate a long-running experiment with multiple checkpoints."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Simulate progress through a large experiment
        total_experiments = 5
        total_runs = 3
        
        configs = []
        for i in range(total_experiments):
            config = EnhancedExperimentConfig(
                strategy="fedavg",
                attack="none",
                dataset="MNIST",
                num_rounds=2,
                num_clients=2
            )
            configs.append(config)
            
        # Simulate incremental progress with checkpoints
        current_state = {
            'experiment_configs': [config.to_dict() for config in configs],
            'completed_experiments': [],
            'current_experiment_index': 0,
            'current_run': 0,
            'total_runs': total_runs,
            'start_time': time.time(),
            'results': []
        }
        
        # Simulate progress through experiments
        for exp_idx in range(total_experiments):
            for run_id in range(total_runs):
                # Update state
                current_state['current_experiment_index'] = exp_idx
                current_state['current_run'] = run_id
                
                # Add mock result
                result = {
                    'experiment_id': f'experiment_{exp_idx}',
                    'run_id': run_id,
                    'strategy': 'fedavg',
                    'attack': 'none',
                    'dataset': 'MNIST',
                    'success': True,
                    'execution_time': 100.0 + exp_idx * 10 + run_id * 5,
                    'final_accuracy': 0.8 + 0.01 * exp_idx,
                    'final_loss': 0.2 - 0.01 * exp_idx
                }
                current_state['results'].append(result)
                
                # Save checkpoint
                runner._save_checkpoint(current_state)
                
        # Verify final state
        final_state = runner._load_latest_checkpoint()
        self.assertEqual(len(final_state['results']), total_experiments * total_runs)
        self.assertEqual(final_state['current_experiment_index'], total_experiments - 1)
        self.assertEqual(final_state['current_run'], total_runs - 1)


class TestCheckpointPerformance(unittest.TestCase):
    """Test checkpoint system performance with large experiments."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "config.yaml"
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.results_dir = self.temp_dir / "results"
        
        # Create config for performance testing
        config_data = {
            'system': {
                'max_parallel_experiments': 1,
                'default_timeout': 60,
                'checkpoint_interval': 5,
                'resource_check_interval': 5,
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
        """Clean up performance test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_large_checkpoint_performance(self):
        """Test checkpoint performance with large result sets."""
        runner = EnhancedExperimentRunner(
            config_file=self.config_file,
            checkpoint_dir=self.checkpoint_dir,
            results_dir=self.results_dir
        )
        
        # Create a large state (simulating hundreds of completed experiments)
        large_results = []
        for i in range(1000):
            result = {
                'experiment_id': f'experiment_{i}',
                'run_id': i % 10,
                'strategy': 'fedavg',
                'attack': 'none',
                'dataset': 'MNIST',
                'success': True,
                'execution_time': 100.0 + i * 0.1,
                'final_accuracy': 0.8 + (i % 100) * 0.001,
                'final_loss': 0.2 - (i % 100) * 0.001,
                'detailed_metrics': {
                    'round_accuracies': [0.5 + j * 0.1 for j in range(10)],
                    'round_losses': [0.5 - j * 0.05 for j in range(10)]
                }
            }
            large_results.append(result)
            
        large_state = {
            'experiment_configs': [],
            'completed_experiments': [],
            'current_experiment_index': 999,
            'current_run': 9,
            'total_runs': 10,
            'start_time': time.time() - 86400,  # Started 24 hours ago
            'results': large_results
        }
        
        # Measure checkpoint save time
        start_time = time.time()
        runner._save_checkpoint(large_state)
        save_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds for 1000 results)
        self.assertLess(save_time, 5.0)
        
        # Measure checkpoint load time
        start_time = time.time()
        loaded_state = runner._load_latest_checkpoint()
        load_time = time.time() - start_time
        
        # Should load in reasonable time
        self.assertLess(load_time, 5.0)
        
        # Verify data integrity
        self.assertEqual(len(loaded_state['results']), len(large_results))
        self.assertEqual(loaded_state['current_experiment_index'], 999)


def run_checkpoint_tests():
    """Run the checkpoint-specific test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add checkpoint test classes
    test_classes = [
        TestCheckpointResumeSystem,
        TestCheckpointPerformance
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
    print("CHECKPOINT & RESUME SYSTEM TESTS")
    print("=" * 80)
    
    success = run_checkpoint_tests()
    
    if success:
        print("\n🎉 All checkpoint tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some checkpoint tests failed!")
        sys.exit(1)
