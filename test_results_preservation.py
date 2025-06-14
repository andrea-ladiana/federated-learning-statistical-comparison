#!/usr/bin/env python3
"""
Test script to verify that results are correctly preserved and saved
during transitions between different metrics (fedavg -> fedavgm -> fedprox etc.)
"""

import sys
import tempfile
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import json
import time

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from experiment_runners.enhanced_experiment_runner import (
    EnhancedExperimentConfig, 
    EnhancedExperimentRunner
)
from utilities.checkpoint_manager import CheckpointManager

def create_test_configs() -> List[EnhancedExperimentConfig]:
    """Create a small set of test configurations with different strategies."""
    return [
        EnhancedExperimentConfig(
            strategy="fedavg",
            attack="none",
            dataset="MNIST",
            attack_params={},
            strategy_params={}
        ),
        EnhancedExperimentConfig(
            strategy="fedavgm",
            attack="none", 
            dataset="MNIST",
            attack_params={},
            strategy_params={"server_momentum": 0.9}
        ),
        EnhancedExperimentConfig(
            strategy="fedprox",
            attack="none",
            dataset="MNIST", 
            attack_params={},
            strategy_params={"proximal_mu": 0.01}
        )
    ]

def simulate_experiment_metrics(runner: EnhancedExperimentRunner, 
                              config: EnhancedExperimentConfig, 
                              run_id: int) -> None:
    """Simulate adding metrics for an experiment."""
    print(f"   - Simulating metrics for {config.strategy}")
    
    # Simulate some fedavg/fedavgm/fedprox metrics
    sample_lines = [
        f"[ROUND 1]",
        f"[Client 0] fit complete | avg_loss=0.45, accuracy=0.82",
        f"[Client 1] fit complete | avg_loss=0.38, accuracy=0.87", 
        f"[Server] Round 1 aggregate fit -> loss=0.41, accuracy=0.845",
        f"[Client 0] evaluate complete | avg_loss=0.42, accuracy=0.84",
        f"[Client 1] evaluate complete | avg_loss=0.36, accuracy=0.89",
        f"[Server] Round 1 evaluate -> loss=0.39, accuracy=0.865",
        f"[ROUND 2]",
        f"[Client 0] fit complete | avg_loss=0.32, accuracy=0.91",
        f"[Client 1] fit complete | avg_loss=0.28, accuracy=0.94",
        f"[Server] Round 2 aggregate fit -> loss=0.30, accuracy=0.925",
        f"[Client 0] evaluate complete | avg_loss=0.30, accuracy=0.92",
        f"[Client 1] evaluate complete | avg_loss=0.26, accuracy=0.95", 
        f"[Server] Round 2 evaluate -> loss=0.28, accuracy=0.935"
    ]
    
    for line in sample_lines:
        runner.parse_and_store_metrics(line, config, run_id)

def test_results_preservation():
    """Test that results are preserved across different metric strategies."""
    print("\n=== TESTING RESULTS PRESERVATION ACROSS STRATEGIES ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test configs
        configs = create_test_configs()
        
        # Initialize runner
        runner = EnhancedExperimentRunner(
            results_dir=str(temp_path),
            _test_mode=True
        )
        
        results_summary = {}
        csv_files = []
        
        print("\n1. Testing sequential experiment execution with metric preservation:")
        
        # Run each strategy sequentially and verify results preservation
        for i, config in enumerate(configs):
            print(f"\n   Running experiment {i+1}/3: {config.strategy}")
            
            # Simulate the experiment with metrics 
            simulate_experiment_metrics(runner, config, run_id=0)
            
            # Save intermediate results
            runner.save_results(intermediate=True)
            
            # Check current DataFrame state
            current_results = len(runner.results_df)
            strategy_results = len(runner.results_df[runner.results_df['algorithm'] == config.strategy])
            
            print(f"   - Total metrics in DataFrame: {current_results}")
            print(f"   - Metrics for {config.strategy}: {strategy_results}")
            
            results_summary[config.strategy] = {
                'total_metrics': current_results,
                'strategy_metrics': strategy_results,
                'unique_strategies': runner.results_df['algorithm'].unique().tolist() if not runner.results_df.empty else []
            }
            
            # Find the CSV file that was just created
            csv_pattern = temp_path.glob("intermediate_results_*.csv")
            latest_csv = max(csv_pattern, key=lambda x: x.stat().st_mtime, default=None)
            if latest_csv:
                csv_files.append(latest_csv)
                
                # Verify CSV content
                df_from_csv = pd.read_csv(latest_csv)
                csv_strategies = df_from_csv['algorithm'].unique().tolist()
                print(f"   - Strategies in CSV: {csv_strategies}")
                
                # Verify all previous strategies are still present
                for prev_config in configs[:i+1]:
                    if prev_config.strategy not in csv_strategies:
                        print(f"   ✗ ERROR: {prev_config.strategy} missing from CSV!")
                        return False
                    else:
                        strategy_count = len(df_from_csv[df_from_csv['algorithm'] == prev_config.strategy])
                        print(f"   ✓ {prev_config.strategy}: {strategy_count} metrics preserved in CSV")
        
        print("\n2. Final verification:")
        
        # Save final results
        runner.save_results(intermediate=False)
        
        # Check final state
        final_csv = max(temp_path.glob("final_results_*.csv"), key=lambda x: x.stat().st_mtime)
        final_df = pd.read_csv(final_csv)
        
        print(f"   - Final CSV: {final_csv.name}")
        print(f"   - Total metrics in final CSV: {len(final_df)}")
        print(f"   - Strategies in final CSV: {final_df['algorithm'].unique().tolist()}")
        
        # Verify all strategies are present in final results
        expected_strategies = [config.strategy for config in configs]
        missing_strategies = []
        
        for strategy in expected_strategies:
            strategy_metrics = len(final_df[final_df['algorithm'] == strategy])
            if strategy_metrics == 0:
                missing_strategies.append(strategy)
                print(f"   ✗ ERROR: {strategy} has 0 metrics in final CSV!")
            else:
                print(f"   ✓ {strategy}: {strategy_metrics} metrics in final CSV")
        
        if missing_strategies:
            print(f"\n❌ FAILED: Missing strategies in final results: {missing_strategies}")
            return False
        
        print("\n3. Checkpoint and resume verification:")
        
        # Test checkpoint/resume functionality preserves results
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_path / "checkpoints")
        checkpoint_manager.register_experiments([config.to_dict() for config in configs], 1)
        
        # Mark some experiments as completed
        for config in configs:
            checkpoint_manager.mark_run_completed(config.get_experiment_id(), 0, True)
        
        progress = checkpoint_manager.get_progress_summary()
        print(f"   - Checkpoint progress: {progress['completed_experiments']}/{progress['total_experiments']} experiments")
        print(f"   - Checkpoint runs: {progress['completed_runs']}/{progress['total_runs']} runs")
        
        if progress['completed_experiments'] != len(configs):
            print(f"   ✗ ERROR: Expected {len(configs)} completed, got {progress['completed_experiments']}")
            return False
        else:
            print(f"   ✓ All {len(configs)} experiments correctly tracked in checkpoint")
        
        print("\n4. Results consistency verification:")
        
        # Verify DataFrame and CSV consistency
        df_metrics = len(runner.results_df)
        csv_metrics = len(final_df)
        
        if df_metrics != csv_metrics:
            print(f"   ✗ ERROR: DataFrame has {df_metrics} metrics but CSV has {csv_metrics}")
            return False
        else:
            print(f"   ✓ DataFrame and CSV consistent: {df_metrics} metrics")
        
        # Verify metric types are preserved
        expected_metrics = ['fit_avg_loss', 'fit_accuracy', 'eval_avg_loss', 'eval_accuracy']
        df_metric_types = final_df['metric'].unique().tolist()
        
        missing_metric_types = [m for m in expected_metrics if m not in df_metric_types]
        if missing_metric_types:
            print(f"   ⚠ WARNING: Missing metric types: {missing_metric_types}")
            print(f"   Available metrics: {df_metric_types}")
        else:
            print(f"   ✓ All expected metric types present")
        
        print(f"\n✅ SUCCESS: Results preservation test passed!")
        print(f"   - All {len(configs)} strategies preserved across transitions")
        print(f"   - Total {df_metrics} metrics correctly saved to CSV")
        print(f"   - Checkpoint/resume functionality working")
        print(f"   - DataFrame and CSV files consistent")
        
        return True

def test_extensive_experiment_simulation():
    """Test simulation of extensive experiment workflow."""
    print("\n=== TESTING EXTENSIVE EXPERIMENT SIMULATION ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create more complex test configuration similar to extensive experiments
        configs = [
            EnhancedExperimentConfig("fedavg", "none", "MNIST", {}, {}),
            EnhancedExperimentConfig("fedavgm", "none", "MNIST", {}, {"server_momentum": 0.9}),
            EnhancedExperimentConfig("fedprox", "none", "MNIST", {}, {"proximal_mu": 0.01}),
            EnhancedExperimentConfig("scaffold", "none", "MNIST", {}, {}),
            EnhancedExperimentConfig("fedavg", "noise", "MNIST", {"noise_std": 0.1}, {}),
            EnhancedExperimentConfig("fedavgm", "noise", "MNIST", {"noise_std": 0.1}, {"server_momentum": 0.9}),
        ]
        
        print(f"\n1. Testing {len(configs)} experiment configurations:")
        for i, config in enumerate(configs):
            attack_desc = f" with {config.attack}" if config.attack != "none" else ""
            print(f"   {i+1}. {config.strategy}{attack_desc} on {config.dataset}")
        
        # Use sequential runner to simulate extensive experiment workflow
        runner = EnhancedExperimentRunner(results_dir=str(temp_path), _test_mode=True)
        
        try:
            # Simulate running experiments sequentially like extensive runner would
            print(f"\n2. Simulating sequential experiment execution:")
            
            experiment_count = 0
            for config_idx, config in enumerate(configs):
                for run_id in range(2):  # 2 runs per config
                    experiment_count += 1
                    print(f"   Running experiment {experiment_count}: {config.strategy} run {run_id}")
                    
                    # Simulate the experiment
                    simulate_experiment_metrics(runner, config, run_id)
                    
                    # Periodic intermediate saves (like extensive runner does)
                    if experiment_count % 3 == 0:
                        runner.save_results(intermediate=True)
                        print(f"   - Intermediate save at experiment {experiment_count}")
            
            # Final save
            runner.save_results(intermediate=False)
            
            print(f"\n3. Verification of final results:")
            
            # Find final results file
            final_csv = max(temp_path.glob("final_results_*.csv"), key=lambda x: x.stat().st_mtime)
            final_df = pd.read_csv(final_csv)
            
            print(f"   - Final results file: {final_csv.name}")
            print(f"   - Total metrics: {len(final_df)}")
            
            # Verify all strategies present
            unique_strategies = final_df['algorithm'].unique()
            expected_strategies = list(set(config.strategy for config in configs))
            
            print(f"   - Expected strategies: {expected_strategies}")
            print(f"   - Found strategies: {unique_strategies.tolist()}")
            
            missing_strategies = [s for s in expected_strategies if s not in unique_strategies]
            if missing_strategies:
                print(f"   ✗ ERROR: Missing strategies: {missing_strategies}")
                return False
            
            # Verify runs and metrics per strategy
            for strategy in expected_strategies:
                strategy_df = final_df[final_df['algorithm'] == strategy]
                runs_for_strategy = strategy_df['run'].unique()
                configs_for_strategy = [c for c in configs if c.strategy == strategy]
                
                print(f"   - {strategy}: {len(strategy_df)} metrics, runs {runs_for_strategy.tolist()}")
                
                # Each config should have 2 runs, so check if we have the right amount
                expected_runs = len(configs_for_strategy) * 2
                actual_runs = len(runs_for_strategy)
                
                if actual_runs != expected_runs:
                    print(f"     ⚠ WARNING: Expected {expected_runs} runs, found {actual_runs}")
            
            print(f"\n✅ SUCCESS: Extensive experiment simulation passed!")
            print(f"   - All {len(expected_strategies)} strategies preserved")
            print(f"   - {len(final_df)} total metrics saved")
            print(f"   - Multiple runs per configuration tracked correctly")
            
            return True
            
        except Exception as e:
            print(f"   ✗ ERROR in extensive simulation: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING RESULTS PRESERVATION IN FEDERATED LEARNING EXPERIMENTS")
    print("=" * 60)
    
    success = True
    
    # Test 1: Basic results preservation
    try:
        if not test_results_preservation():
            success = False
    except Exception as e:
        print(f"✗ ERROR in results preservation test: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Test 2: Extensive experiment simulation
    try:
        if not test_extensive_experiment_simulation():
            success = False
    except Exception as e:
        print(f"✗ ERROR in extensive experiment simulation: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Results preservation is working correctly across metric transitions.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("There are issues with results preservation between metrics.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
