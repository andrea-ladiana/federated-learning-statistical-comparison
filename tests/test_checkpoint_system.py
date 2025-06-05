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

# Add the project root to the path for reorganized imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "experiment_runners"))
sys.path.insert(0, str(project_root / "utilities"))
sys.path.insert(0, str(project_root / "configuration"))

# Import from the correct module path
from experiment_runners.enhanced_experiment_runner import (
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
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir),
            _test_mode=True
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
        
        # Verify checkpoint file exists
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
        self.assertEqual(len(checkpoint_files), 1)
        
        # Load checkpoint
        loaded_state = runner._load_latest_checkpoint()
        
        # Verify loaded state matches original
        self.assertIsNotNone(loaded_state)
        # Only proceed with comparisons if loaded_state is not None
        if loaded_state is not None:
            self.assertEqual(loaded_state['current_run'], test_state['current_run'])
            self.assertEqual(loaded_state['total_runs'], test_state['total_runs'])
            self.assertEqual(len(loaded_state['completed_experiments']), 
                            len(test_state['completed_experiments']))
            self.assertEqual(len(loaded_state['results']), len(test_state['results']))
        
        # Verify specific result data
        if loaded_state is not None:
            for original, loaded in zip(test_state['results'], loaded_state['results']):
                self.assertEqual(original['experiment_id'], loaded['experiment_id'])
                self.assertEqual(original['run_id'], loaded['run_id'])
            
    def test_checkpoint_with_multiple_saves(self):
        """Test multiple checkpoint saves with incremental data."""
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir),
            _test_mode=True
        )
        
        # Save multiple checkpoints
        for i in range(3):
            state = {
                'experiment_configs': [],
                'completed_experiments': [],
                'current_run': i,
                'total_runs': 5,
                'start_time': time.time(),
                'results': []
            }
            runner._save_checkpoint(state)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Verify multiple checkpoint files exist
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
        self.assertEqual(len(checkpoint_files), 3)
        
        # Load latest checkpoint
        loaded_state = runner._load_latest_checkpoint()
        
        # Should be the last one saved (current_run = 2)
        self.assertIsNotNone(loaded_state)
        # Only proceed with the comparison if loaded_state is not None
        if loaded_state is not None:
            self.assertEqual(loaded_state['current_run'], 2)
        
    def test_checkpoint_cleanup(self):
        """Test that old checkpoints are cleaned up."""
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir),
            _test_mode=True
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
        
    def test_resume_from_checkpoint(self):
        """Test resuming an experiment run from a checkpoint."""
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir),
            _test_mode=True
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
            (results_df['algorithm'] == 'fedavg') & 
            (results_df['attack'] == 'none') & 
            (results_df['dataset'] == 'MNIST')
        ]
        self.assertGreaterEqual(len(fedavg_results), 1)
        
    def test_checkpoint_corruption_handling(self):
        """Test handling of corrupted checkpoint files."""
        runner = EnhancedExperimentRunner(
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir),
            _test_mode=True
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
            config_file=str(self.config_file),
            checkpoint_dir=str(self.checkpoint_dir),
            results_dir=str(self.results_dir),
            _test_mode=True
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
        
        # Start experiments and capture initial state
        results_df = runner.run_experiments(configs, num_runs=2, mode='sequential')
        
        # Load checkpoint after completion
        loaded_state = runner._load_latest_checkpoint()
          # Verify state consistency
        self.assertIsNotNone(loaded_state)
        if loaded_state is not None:
            # Compare experiment-level counts: 1 config × 2 runs = 2 experiments
            expected_experiments = len(configs) * 2  # num_runs = 2
            self.assertEqual(len(loaded_state['results']), expected_experiments)
            
            # Verify the DataFrame contains metric data (should have multiple metrics per experiment)
            self.assertGreater(len(results_df), 0)            # Verify all expected experiments are represented in results_df
            unique_experiment_runs = results_df[['algorithm', 'attack', 'dataset', 'run']].drop_duplicates()
            print(f"DEBUG: Expected experiments: {expected_experiments}")
            print(f"DEBUG: Unique experiment runs found: {len(unique_experiment_runs)}")
            print(f"DEBUG: DataFrame columns: {list(results_df.columns)}")
            print(f"DEBUG: DataFrame shape: {results_df.shape}")
            print(f"DEBUG: Unique runs:\n{unique_experiment_runs}")
            print(f"DEBUG: All runs in DataFrame:\n{results_df[['algorithm', 'attack', 'dataset', 'run', 'metric']].head(15)}")
            self.assertEqual(len(unique_experiment_runs), expected_experiments)


if __name__ == "__main__":
    unittest.main()
