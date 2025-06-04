#!/usr/bin/env python3
"""
Convenience script to run federated learning experiments.

This script provides a simple interface to run the most common experiment scenarios
using the reorganized codebase structure.
"""

import sys
import argparse
from pathlib import Path

# Add paths to import from reorganized structure
sys.path.insert(0, str(Path(__file__).parent / "experiment_runners"))
sys.path.insert(0, str(Path(__file__).parent / "configuration"))
sys.path.insert(0, str(Path(__file__).parent / "utilities"))

def main():
    parser = argparse.ArgumentParser(
        description="Run federated learning experiments with the reorganized codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic experiments
  python run_experiments.py --runner basic --num-runs 5
  
  # Run enhanced experiments with parallel execution
  python run_experiments.py --runner enhanced --num-runs 10 --parallel
  
  # Run extensive experiments with checkpoint support
  python run_experiments.py --runner extensive --num-runs 20 --resume
  
Available runners:
  - basic: Basic experiment runner (experiment_runners/basic_experiment_runner.py)
  - stable: Stable experiment runner (experiment_runners/stable_experiment_runner.py) 
  - enhanced: Enhanced experiment runner with advanced features
  - extensive: Extensive experiments with full configuration matrix
        """
    )
    
    parser.add_argument("--runner", choices=["basic", "stable", "enhanced", "extensive"],
                        default="enhanced", help="Which experiment runner to use")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs per configuration")
    parser.add_argument("--parallel", action="store_true",
                        help="Run experiments in parallel (where supported)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (where supported)")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in test mode with reduced configurations")
    
    args = parser.parse_args()
    
    # Import and run the appropriate runner
    if args.runner == "basic":
        from basic_experiment_runner import main as run_basic
        sys.argv = ["basic_experiment_runner.py", "--num-runs", str(args.num_runs)]
        if args.test_mode:
            sys.argv.append("--test-mode")
        run_basic()
        
    elif args.runner == "stable":
        from stable_experiment_runner import main as run_stable
        sys.argv = ["stable_experiment_runner.py", "--num-runs", str(args.num_runs)]
        if args.test_mode:
            sys.argv.append("--test-mode")
        run_stable()
        
    elif args.runner == "enhanced":
        from enhanced_experiment_runner import main as run_enhanced
        sys.argv = ["enhanced_experiment_runner.py", "--num-runs", str(args.num_runs)]
        if args.parallel:
            sys.argv.append("--parallel")
        if args.resume:
            sys.argv.append("--resume")
        if args.test_mode:
            sys.argv.append("--test-mode")
        run_enhanced()
        
    elif args.runner == "extensive":
        from run_extensive_experiments import main as run_extensive
        sys.argv = ["run_extensive_experiments.py", "--num-runs", str(args.num_runs)]
        if args.resume:
            sys.argv.append("--resume")
        run_extensive()

if __name__ == "__main__":
    main()
