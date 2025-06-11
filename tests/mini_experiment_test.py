#!/usr/bin/env python3
"""
Test mini con esperimenti reali ma molto veloci.
Usa configurazioni minime per verificare rapidamente il funzionamento.
"""

import sys
import subprocess
import time
import pandas as pd
from pathlib import Path

def run_mini_experiment_test():
    """Esegue un test mini con esperimenti reali ma veloci."""
    print("=" * 60)
    print("MINI EXPERIMENT TEST")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    script_path = base_dir / "experiment_runners" / "run_extensive_experiments.py"
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    print("🚀 Running mini experiment (2 rounds, 2 clients, 1 run)...")
    print("This should take only 2-3 minutes")
    
    # Test 1: Run a very short experiment with multiple strategies
    cmd = [
        sys.executable, str(script_path),
        "--num-runs", "1",
        "--num-rounds", "2",  # Very short
        "--num-clients", "2",  # Minimal clients
        "--results-dir", "mini_test_results",
        "--strategies", "fedavg,fedprox,fedavgm",  # Multiple strategies
        "--attacks", "none",  # No attacks for speed
        "--datasets", "MNIST"  # Single dataset
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run experiment
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=base_dir
        )
        
        # Let it run for a bit, then interrupt to test resume
        time.sleep(60)  # Let at least one experiment complete
        
        print("\n⏸️ Interrupting to test resume functionality...")
        process.terminate()
        process.wait()
        
        # Test 2: Resume the experiment
        print("\n🔄 Testing resume...")
        resume_cmd = cmd + ["--resume"]
        
        resume_process = subprocess.run(
            resume_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=base_dir,
            timeout=120  # 2 minutes max
        )
        
        print("✅ Resume completed")
        
        # Test 3: Check results
        print("\n🔍 Checking results...")
        results_dir = base_dir / "mini_test_results"
        
        if results_dir.exists():
            csv_files = list(results_dir.glob("*.csv"))
            if csv_files:
                latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
                print(f"📊 Reading results from: {latest_csv}")
                
                try:
                    df = pd.read_csv(latest_csv)
                    
                    if 'algorithm' in df.columns:
                        strategies = df['algorithm'].unique()
                        print(f"✅ Strategies found: {list(strategies)}")
                        
                        non_fedavg = [s for s in strategies if s != 'fedavg']
                        if non_fedavg:
                            print(f"🎉 SUCCESS: Non-fedavg strategies found: {non_fedavg}")
                            return True
                        else:
                            print("⚠️ WARNING: Only fedavg found - this indicates the bug persists")
                            return False
                    else:
                        print("❌ No 'algorithm' column found in results")
                        return False
                        
                except Exception as e:
                    print(f"❌ Error reading CSV: {e}")
                    return False
            else:
                print("❌ No CSV files found in results directory")
                return False
        else:
            print("❌ Results directory not found")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏱️ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    """Run the mini experiment test."""
    print("🧪 MINI EXPERIMENT TEST")
    print("This runs actual (but very short) experiments to verify the fix")
    print("Expected duration: 2-3 minutes")
    
    success = run_mini_experiment_test()
    
    if success:
        print("\n🎉 MINI TEST PASSED!")
        print("The strategy preservation fix is working correctly.")
    else:
        print("\n❌ MINI TEST FAILED!")
        print("The fix may need additional work.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
