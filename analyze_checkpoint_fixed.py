#!/usr/bin/env python3
"""
Analyze checkpoint file to understand the issue
"""
import yaml

def analyze_checkpoint():
    checkpoint_file = 'extensive_results/checkpoints/checkpoint_20250611_161729_764.yaml'
    
    with open(checkpoint_file, 'r') as f:
        data = yaml.safe_load(f)
    
    state = data['state']
    
    print("=== CHECKPOINT ANALYSIS ===")
    print(f"Total completed experiments: {len(state['completed_experiments'])}")
    print(f"Total experiment configs: {len(state['experiment_configs'])}")
    print(f"Current experiment index: {state.get('current_experiment_index', 'N/A')}")
    print(f"Current run: {state.get('current_run', 'N/A')}")
    print(f"Total runs: {state.get('total_runs', 'N/A')}")
    print(f"Total results in checkpoint: {len(state['results'])}")
    
    # Analyze strategies
    strategies = [cfg.get('strategy', 'unknown') for cfg in state['experiment_configs']]
    print(f"Unique strategies in configs: {set(strategies)}")
    
    # Count strategies
    strategy_counts = {}
    for strategy in strategies:
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    print(f"Strategy distribution: {strategy_counts}")
    
    # Analysis by strategy
    completed_by_strategy = {}
    for exp in state['completed_experiments']:
        exp_id = exp['experiment_id']
        # Extract strategy from experiment ID
        strategy = exp_id.split('_')[0]
        if strategy not in completed_by_strategy:
            completed_by_strategy[strategy] = 0
        completed_by_strategy[strategy] += 1
    
    print(f"Completed experiments by strategy: {completed_by_strategy}")
    
    print("\nFirst 5 experiment IDs:")
    for i, exp in enumerate(state['completed_experiments'][:5]):
        print(f"  {exp['experiment_id']}")
    
    print("\nLast 5 experiment IDs:")
    for i, exp in enumerate(state['completed_experiments'][-5:]):
        print(f"  {exp['experiment_id']}")
    
    # Check what's missing - compare configs vs completed
    total_expected = len(state['experiment_configs']) * state['total_runs']
    print(f"\nExpected total experiments: {total_expected}")
    print(f"Actually completed: {len(state['completed_experiments'])}")
    print(f"Missing: {total_expected - len(state['completed_experiments'])}")
    
    # Find which experiments are missing - FIXED VERSION
    completed_set = set()
    for exp in state['completed_experiments']:
        exp_key = f"{exp['experiment_id']}_run_{exp['run_id']}"
        completed_set.add(exp_key)
    
    print(f"Completed experiment keys sample (first 10):")
    for key in list(completed_set)[:10]:
        print(f"  {key}")
    
    # Check missing experiments - use same logic as the runner
    missing_experiments = []
    expected_experiments = []
    
    for i, config in enumerate(state['experiment_configs']):
        # Create experiment ID using the SAME logic as EnhancedExperimentConfig.get_experiment_id()
        strategy = config.get('strategy', 'unknown')
        attack = config.get('attack', 'none')
        dataset = config.get('dataset', 'unknown')
        
        # This is the critical part - need to match the exact ID generation logic
        attack_params = config.get('attack_params', {})
        if attack == 'none' or not attack_params:
            exp_id = f"{strategy}_{attack}_{dataset}"
        else:
            # Build parameter string - this must match exactly (NO underscore between k and v!)
            params_str = "_".join([f"{k}{v}" for k, v in sorted(attack_params.items())])
            exp_id = f"{strategy}_{attack}_{params_str}_{dataset}"
        
        for run_id in range(state['total_runs']):
            exp_key = f"{exp_id}_run_{run_id}"
            expected_experiments.append((exp_id, run_id, exp_key))
            if exp_key not in completed_set:
                missing_experiments.append((exp_id, run_id, exp_key))
    
    print(f"\nExpected experiment keys sample (first 10):")
    for exp_id, run_id, exp_key in expected_experiments[:10]:
        print(f"  {exp_key}")
    
    print(f"\nMissing experiments: {len(missing_experiments)}/expected {len(expected_experiments)}")
    if missing_experiments:
        print("First 10 missing experiments:")
        for exp_id, run_id, exp_key in missing_experiments[:10]:
            print(f"  {exp_key}")
        
        # Show difference in key format
        print(f"\nKey format comparison:")
        if completed_set and missing_experiments:
            completed_example = list(completed_set)[0]
            missing_example = missing_experiments[0][2]
            print(f"  Completed format: {completed_example}")
            print(f"  Expected format:  {missing_example}")
            print(f"  Match? {completed_example == missing_example}")
    
    # Let's also check what the actual first experiment configs look like
    print(f"\nFirst 3 experiment configs in checkpoint:")
    for i, config in enumerate(state['experiment_configs'][:3]):
        print(f"  Config {i}: strategy={config.get('strategy')}, attack={config.get('attack')}, dataset={config.get('dataset')}")
        print(f"    attack_params: {config.get('attack_params', {})}")
        # Show what ID this would generate
        strategy = config.get('strategy', 'unknown')
        attack = config.get('attack', 'none')
        dataset = config.get('dataset', 'unknown')
        attack_params = config.get('attack_params', {})
        if attack == 'none' or not attack_params:
            exp_id = f"{strategy}_{attack}_{dataset}"
        else:
            params_str = "_".join([f"{k}{v}" for k, v in sorted(attack_params.items())])
            exp_id = f"{strategy}_{attack}_{params_str}_{dataset}"
        print(f"    Generated ID: {exp_id}")
        print()

if __name__ == "__main__":
    analyze_checkpoint()
