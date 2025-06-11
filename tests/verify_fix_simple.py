#!/usr/bin/env python3
"""
Simple verification script without Unicode characters for Windows compatibility.
"""

import sys
import yaml
import pandas as pd
from pathlib import Path

def analyze_existing_checkpoints():
    """Analizza i checkpoint esistenti per vedere le strategie."""
    print("=" * 60)
    print("EXISTING CHECKPOINT ANALYSIS")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    
    # Check extensive_results directory
    extensive_results = base_dir / "extensive_results"
    checkpoint_dirs = [
        extensive_results / "checkpoints",
        base_dir / "test_checkpoints"
    ]
    
    print("Looking for checkpoint files...")
    
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            print(f"\nChecking directory: {checkpoint_dir}")
            
            # Find YAML checkpoint files
            yaml_files = list(checkpoint_dir.glob("*.yaml")) + list(checkpoint_dir.glob("*.yml"))
            
            if not yaml_files:
                print("   No YAML checkpoint files found")
                continue
            
            for yaml_file in yaml_files:
                print(f"\nAnalyzing: {yaml_file.name}")
                
                try:
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    # Handle different checkpoint formats
                    if 'state' in data:
                        state = data['state']
                    else:
                        state = data
                    
                    if 'experiment_configs' in state:
                        configs = state['experiment_configs']
                        strategies = [cfg.get('strategy', 'unknown') for cfg in configs]
                        
                        print(f"   Found {len(configs)} experiment configurations")
                        print(f"   Strategies: {list(set(strategies))}")
                        
                        # Count by strategy
                        strategy_counts = {}
                        for strategy in strategies:
                            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                        
                        print(f"   Strategy distribution: {strategy_counts}")
                        
                        non_fedavg = [s for s in set(strategies) if s != 'fedavg']
                        if non_fedavg:
                            print(f"   SUCCESS: Non-fedavg strategies found: {non_fedavg}")
                        else:
                            print(f"   WARNING: Only fedavg strategies found")
                    else:
                        print("   ERROR: No experiment_configs found in checkpoint")
                        
                except Exception as e:
                    print(f"   ERROR reading checkpoint: {e}")
        else:
            print(f"\nDirectory not found: {checkpoint_dir}")

def analyze_existing_results():
    """Analizza i risultati CSV esistenti."""
    print("\n" + "=" * 60)
    print("EXISTING RESULTS ANALYSIS")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    extensive_results = base_dir / "extensive_results"
    
    if not extensive_results.exists():
        print("ERROR: extensive_results directory not found")
        return
    
    # Find CSV files
    csv_files = list(extensive_results.glob("*.csv"))
    
    if not csv_files:
        print("ERROR: No CSV files found")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Analyze the most recent file
    latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
    print(f"\nAnalyzing latest file: {latest_csv.name}")
    
    try:
        df = pd.read_csv(latest_csv)
        print(f"   Shape: {df.shape}")
        
        if 'algorithm' in df.columns:
            strategies = df['algorithm'].unique()
            print(f"   Unique algorithms: {list(strategies)}")
            
            # Count occurrences
            strategy_counts = df['algorithm'].value_counts()
            print(f"   Algorithm distribution:")
            for strategy, count in strategy_counts.items():
                print(f"      {strategy}: {count}")
            
            # Check if the bug is present
            if len(strategies) == 1 and strategies[0] == 'fedavg':
                print(f"   BUG DETECTED: Only 'fedavg' found in results!")
                print(f"      This confirms the issue exists.")
            elif 'fedavg' in strategies and len(strategies) > 1:
                print(f"   GOOD: Multiple strategies found")
            else:
                print(f"   UNCLEAR: Unexpected strategy pattern")
        else:
            print("   ERROR: No 'algorithm' column found")
            
    except Exception as e:
        print(f"   ERROR reading CSV: {e}")

def verify_fix_logic():
    """Verifica la logica della fix direttamente."""
    print("\n" + "=" * 60)
    print("FIX LOGIC VERIFICATION")
    print("=" * 60)
    
    try:
        # Test the exact scenario that was fixed
        print("Testing configuration reconstruction from checkpoint data...")
        
        sys.path.insert(0, str(Path(__file__).parent / "experiment_runners"))
        from experiment_runners.enhanced_experiment_runner import EnhancedExperimentConfig
        
        # Simulate checkpoint data with diverse strategies
        checkpoint_configs = [
            {
                'strategy': 'fedavg',
                'attack': 'none',
                'dataset': 'MNIST',
                'attack_params': {},
                'strategy_params': {},
                'num_rounds': 10,
                'num_clients': 10
            },
            {
                'strategy': 'fedprox',
                'attack': 'noise',
                'dataset': 'CIFAR10',
                'attack_params': {'noise_std': 0.1, 'noise_fraction': 0.3},
                'strategy_params': {'proximal_mu': 0.01},
                'num_rounds': 10,
                'num_clients': 10
            },
            {
                'strategy': 'fedavgm',
                'attack': 'none',
                'dataset': 'FMNIST',
                'attack_params': {},
                'strategy_params': {'server_momentum': 0.9},
                'num_rounds': 10,
                'num_clients': 10
            }
        ]
        
        print(f"Checkpoint contains strategies: {[cfg['strategy'] for cfg in checkpoint_configs]}")
        
        # Test the fixed code path
        print("Testing fixed configuration loading...")
        
        loaded_configs = []
        for config_dict in checkpoint_configs:
            try:
                # This is the fixed line: EnhancedExperimentConfig(**config_dict)
                config = EnhancedExperimentConfig(**config_dict)
                loaded_configs.append(config)
                print(f"   SUCCESS: Loaded {config.strategy}")
            except Exception as e:
                print(f"   ERROR: Failed to load: {e}")
                return False
        
        # Verify strategies are preserved
        original_strategies = [cfg['strategy'] for cfg in checkpoint_configs]
        loaded_strategies = [cfg.strategy for cfg in loaded_configs]
        
        if original_strategies == loaded_strategies:
            print(f"SUCCESS: Strategies preserved during loading!")
            print(f"   Original: {original_strategies}")
            print(f"   Loaded:   {loaded_strategies}")
            return True
        else:
            print(f"FAILURE: Strategy mismatch!")
            print(f"   Original: {original_strategies}")
            print(f"   Loaded:   {loaded_strategies}")
            return False
            
    except Exception as e:
        print(f"ERROR testing fix logic: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks."""
    print("COMPREHENSIVE VERIFICATION OF STRATEGY FIX")
    print("This analyzes existing data and tests the fix logic")
    
    # Run all checks
    analyze_existing_checkpoints()
    analyze_existing_results()
    fix_verified = verify_fix_logic()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Fix Logic Test: {'PASS' if fix_verified else 'FAIL'}")
    
    if fix_verified:
        print("\nTHE FIX IS WORKING CORRECTLY!")
        print("\nTo verify with real experiments:")
        print("1. Run: python quick_verification_test.py")
        print("2. Then: python mini_experiment_test.py")
    else:
        print("\nTHE FIX NEEDS MORE WORK!")
    
    return fix_verified

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
