#!/usr/bin/env python3
"""
Test runner for the federated learning experiment system.

This script runs all tests and provides comprehensive coverage reporting
for the enhanced experiment runner and checkpoint system.
"""

import sys
import unittest
import time
from pathlib import Path
import argparse
import importlib

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from tests.test_enhanced_experiment_runner import run_test_suite as run_main_tests
from tests.test_checkpoint_system import run_checkpoint_tests


def run_all_tests(verbose=True, include_integration=True):
    """
    Run all test suites.
    
    Args:
        verbose: If True, run tests with verbose output
        include_integration: If True, include integration tests
        
    Returns:
        bool: True if all tests passed, False otherwise
    """
    print("=" * 80)
    print("FEDERATED LEARNING EXPERIMENT SYSTEM - FULL TEST SUITE")
    print("=" * 80)
    
    test_results = {}
    start_time = time.time()
    
    # Run main test suite
    print("\n" + "=" * 50)
    print("RUNNING MAIN TEST SUITE")
    print("=" * 50)
    
    try:
        main_result = run_main_tests()
        test_results['main'] = main_result
        print(f"Main tests: {'PASSED' if main_result else 'FAILED'}")
    except Exception as e:
        print(f"Main tests: FAILED with exception: {e}")
        test_results['main'] = False
    
    # Run checkpoint-specific tests
    print("\n" + "=" * 50)
    print("RUNNING CHECKPOINT SYSTEM TESTS")
    print("=" * 50)
    
    try:
        checkpoint_result = run_checkpoint_tests()
        test_results['checkpoint'] = checkpoint_result
        print(f"Checkpoint tests: {'PASSED' if checkpoint_result else 'FAILED'}")
    except Exception as e:
        print(f"Checkpoint tests: FAILED with exception: {e}")
        test_results['checkpoint'] = False
    
    # Run integration tests if requested
    if include_integration:
        print("\n" + "=" * 50)
        print("RUNNING INTEGRATION TESTS")
        print("=" * 50)
        
        try:
            integration_result = run_integration_tests()
            test_results['integration'] = integration_result
            print(f"Integration tests: {'PASSED' if integration_result else 'FAILED'}")
        except Exception as e:
            print(f"Integration tests: FAILED with exception: {e}")
            test_results['integration'] = False
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed_count = sum(1 for result in test_results.values() if result)
    total_count = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name.upper():.<30} {status}")
    
    print(f"\nResults: {passed_count}/{total_count} test suites passed")
    print(f"Total time: {total_time:.2f} seconds")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        print("The enhanced experiment system is ready for production use.")
        return True
    else:
        print(f"\n❌ {total_count - passed_count} TEST SUITE(S) FAILED!")
        print("Please review the test output and fix any issues.")
        return False


def run_integration_tests():
    """Run integration tests that test the complete system end-to-end."""
    print("Running integration tests...")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    try:
        # Import and run integration tests
        from tests.test_enhanced_experiment_runner import TestIntegration
        tests = loader.loadTestsFromTestCase(TestIntegration)
        suite.addTests(tests)
        
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except ImportError as e:
        print(f"Could not import integration tests: {e}")
        return False


def run_specific_test(test_name):
    """
    Run a specific test class or method.
    
    Args:
        test_name: Name of the test class or method to run
        
    Returns:
        bool: True if test passed, False otherwise
    """
    print(f"Running specific test: {test_name}")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    try:
        if '.' in test_name:
            # Test method specified
            suite.addTest(loader.loadTestsFromName(test_name))
        else:
            # Test class specified - try to find it dynamically
            enhanced_module = importlib.import_module(
                'tests.test_enhanced_experiment_runner')
            checkpoint_module = importlib.import_module(
                'tests.test_checkpoint_system')

            test_class = getattr(enhanced_module, test_name, None)
            if test_class is None:
                test_class = getattr(checkpoint_module, test_name, None)

            if test_class:
                tests = loader.loadTestsFromTestCase(test_class)
                suite.addTests(tests)
            else:
                print(f"Test class '{test_name}' not found")
                return False
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except Exception as e:
        print(f"Error running test '{test_name}': {e}")
        return False


def check_test_dependencies():
    """Check that all required dependencies for testing are available."""
    print("Checking test dependencies...")
    
    required_modules = [
        'unittest',
        'tempfile', 
        'pathlib',
        'yaml',
        'pandas',
        'numpy',
        'psutil'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} (missing)")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\nMissing dependencies: {', '.join(missing_modules)}")
        print("Please install missing dependencies before running tests.")
        return False
    
    print("All test dependencies are available.")
    return True


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description='Run tests for the federated learning experiment system')
    parser.add_argument('--test', type=str, help='Run a specific test class or method')
    parser.add_argument('--no-integration', action='store_true', help='Skip integration tests')
    parser.add_argument('--check-deps', action='store_true', help='Check test dependencies only')
    parser.add_argument('--quick', action='store_true', help='Run only essential tests')
    
    args = parser.parse_args()
    
    if args.check_deps:
        success = check_test_dependencies()
        sys.exit(0 if success else 1)
    
    # Check dependencies first
    if not check_test_dependencies():
        sys.exit(1)
    
    if args.test:
        # Run specific test
        success = run_specific_test(args.test)
    elif args.quick:
        # Run only main tests
        print("Running quick test suite (main tests only)...")
        success = run_main_tests()
    else:
        # Run all tests
        include_integration = not args.no_integration
        success = run_all_tests(include_integration=include_integration)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
