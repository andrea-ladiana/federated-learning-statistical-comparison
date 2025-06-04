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

def main():
    print("=" * 60)
    print("FEDERATED LEARNING FRAMEWORK SETUP")
    print("=" * 60)
    print()
    
    print("🔧 Setting up the reorganized federated learning framework...")
    print()
    
    # Import and run setup
    try:
        from setup_and_test import main as setup_main
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
        
        print()
        print("🚀 To run experiments, use:")
        print("   python run_experiments.py --help")
        print()
        print("📊 To monitor experiments, use:")
        print("   python scripts/monitor_experiments.py")

if __name__ == "__main__":
    main()
