#!/usr/bin/env python3
"""
Quick setup script for the federated learning environment.

This script helps set up the environment and runs basic checks to ensure
everything is working correctly after the reorganization.
"""

import sys
import subprocess
from pathlib import Path

# Add paths for reorganized structure
sys.path.insert(0, str(Path(__file__).parent / "maintenance"))
sys.path.insert(0, str(Path(__file__).parent / "configuration"))

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['flwr', 'torch', 'numpy', 'pandas', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✓ All required packages are installed")
    return True

def main():
    print("=" * 60)
    print("FEDERATED LEARNING FRAMEWORK SETUP")
    print("=" * 60)
    print()
    
    print("🔧 Setting up the reorganized federated learning framework...")
    print()
    
    # Import and run setup
    try:
        from maintenance.setup_and_test import main as setup_main
        setup_main()
    except ImportError as e:
        print(f"❌ Could not import setup script: {e}")
        print("Running basic checks instead...")
        
        # Basic Python environment check
        print(f"✓ Python version: {sys.version}")
        print(f"✓ Python executable: {sys.executable}")
        
        # Check key directories exist
        key_dirs = ["core", "experiment_runners", "configuration", "utilities", "scripts", "maintenance"]
        for dir_name in key_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                print(f"✓ Directory exists: {dir_name}/")
            else:
                print(f"❌ Missing directory: {dir_name}/")
        
        # Check for required packages
        print()
        check_dependencies()
        
        print()
        print("🚀 To run experiments, use one of these options:")
        print("   # Basic experiment runner")
        print("   python experiment_runners/basic_experiment_runner.py --help")
        print("   # Enhanced experiment runner (recommended)")
        print("   python experiment_runners/enhanced_experiment_runner.py --help")
        print("   # Extensive experiments")
        print("   python experiment_runners/run_extensive_experiments.py --help")
        print("   # Run with attacks")
        print("   python experiment_runners/run_with_attacks.py --help")
        print()
        print("📊 To monitor experiments, use:")
        print("   python scripts/monitor_experiments.py --watch")
        print()
        print("🔧 For additional setup and testing:")
        print("   python maintenance/setup_and_test.py")

if __name__ == "__main__":
    main()
