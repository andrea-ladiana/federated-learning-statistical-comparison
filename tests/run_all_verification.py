#!/usr/bin/env python3
"""
Script di verifica automatica completo che esegue tutti i test.
"""

import sys
import subprocess
from pathlib import Path

def run_test_script(script_name, description):
    """Esegue uno script di test e restituisce il risultato."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        print(f"\n{'✅ PASSED' if success else '❌ FAILED'}: {description}")
        return success
        
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all verification tests in sequence."""
    print("🚀 AUTOMATED VERIFICATION SUITE")
    print("This runs all verification tests to check if the strategy fix works")
    print("Expected duration: 2-5 minutes")
    
    tests = [
        ("verify_fix_comprehensive.py", "Comprehensive Fix Verification"),
        ("quick_verification_test.py", "Quick Strategy Preservation Test"),
        # Note: mini_experiment_test.py is commented out as it takes longer
        # Uncomment if you want to run actual mini experiments
        # ("mini_experiment_test.py", "Mini Experiment Test"),
    ]
    
    results = {}
    
    for script, description in tests:
        results[description] = run_test_script(script, description)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe strategy preservation fix is working correctly!")
        print("\nRECOMMENDED NEXT STEPS:")
        print("1. Run a small real experiment to double-check:")
        print("   python experiment_runners/run_extensive_experiments.py \\")
        print("     --num-runs 1 --num-rounds 3 --num-clients 3 \\")
        print("     --strategies fedavg,fedprox,fedavgm \\")
        print("     --attacks none --datasets MNIST \\")
        print("     --results-dir test_verification")
        print("2. Stop it with Ctrl+C after 2-3 minutes")
        print("3. Resume with: (same command) + --resume")
        print("4. Check results CSV contains multiple strategies")
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nThe fix may need additional investigation.")
        print("Check the output above for specific issues.")
    
    print("="*60)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
