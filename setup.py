#!/usr/bin/env python3
"""
Standalone setup script for the federated learning framework.

This script performs comprehensive environment checks and setup without
depending on external maintenance scripts.
"""

import sys
import subprocess
import socket
import time
from pathlib import Path
import importlib.util
import os

# Essential package mapping: (import_name, pip_name)
REQUIRED_PACKAGES = [
    ('flwr', 'flwr'),
    ('torch', 'torch'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('matplotlib', 'matplotlib'),
    ('yaml', 'pyyaml'),
    ('psutil', 'psutil'),
]

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    print(f"[INFO] Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] Python 3.8+ is required")
        return False
    elif version.major == 3 and version.minor >= 12:
        print("[WARNING] Python 3.12+ may have compatibility issues with some packages")
    
    print("[OK] Python version is compatible")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    missing_packages = []
    installed_packages = []
    
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                # Try to actually import to catch broken installations
                __import__(import_name)
                print(f"[OK] {import_name} is installed")
                installed_packages.append(import_name)
            else:
                raise ImportError
        except ImportError:
            print(f"[MISSING] {import_name} is missing")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        print("Or install manually: pip install " + " ".join(missing_packages))
        return False, missing_packages
    
    print(f"[OK] All {len(installed_packages)} required packages are installed")
    return True, []

def check_directory_structure():
    """Check that all required directories and files exist."""
    print("\nChecking directory structure...")
    
    # Required directories
    required_dirs = [
        "core",
        "experiment_runners", 
        "configuration",
        "utilities",
        "scripts",
        "attacks",
        "models"
    ]
    
    # Required files
    required_files = [
        "core/server.py",
        "core/client.py",
        "core/strategies.py",
        "experiment_runners/run_with_attacks.py",
        "experiment_runners/basic_experiment_runner.py",
        "experiment_runners/enhanced_experiment_runner.py",
        "experiment_runners/run_extensive_experiments.py",
        "scripts/monitor_experiments.py",
        "configuration/config_manager.py",
        "requirements.txt"
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"[OK] Directory exists: {dir_name}/")
        else:
            print(f"[MISSING] Directory missing: {dir_name}/")
            missing_dirs.append(dir_name)
    
    # Check files
    for file_path in required_files:
        file_obj = Path(file_path)
        if file_obj.exists() and file_obj.is_file():
            print(f"[OK] File exists: {file_path}")
        else:
            print(f"[MISSING] File missing: {file_path}")
            missing_files.append(file_path)
    
    success = len(missing_dirs) == 0 and len(missing_files) == 0
    
    if not success:
        print(f"\n[ERROR] Missing {len(missing_dirs)} directories and {len(missing_files)} files")
    else:
        print("\n[OK] All required directories and files are present")
    
    return success, missing_dirs, missing_files

def check_port_availability():
    """Check if default ports are available."""
    print("\nChecking port availability...")
    ports_to_check = [8080, 8081, 8082]
    
    for port in ports_to_check:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    print(f"[WARNING] Port {port} is in use")
                else:
                    print(f"[OK] Port {port} is available")
        except Exception as e:
            print(f"[WARNING] Could not check port {port}: {e}")
    
    return True

def test_run_with_attacks():
    """Test if run_with_attacks.py is functional."""
    print("\nTesting run_with_attacks.py...")
    
    run_with_attacks_path = Path("experiment_runners/run_with_attacks.py")
    
    if not run_with_attacks_path.exists():
        print("[ERROR] run_with_attacks.py not found in experiment_runners/")
        return False
    
    try:
        # Test help command
        result = subprocess.run(
            [sys.executable, str(run_with_attacks_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("[OK] run_with_attacks.py responds to --help")
            return True
        else:
            print(f"[ERROR] run_with_attacks.py failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[ERROR] run_with_attacks.py timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to test run_with_attacks.py: {e}")
        return False

def create_example_config():
    """Create an example configuration file if it doesn't exist."""
    config_file = Path("experiment_config.yaml")
    
    if config_file.exists():
        print(f"[OK] Configuration file already exists: {config_file}")
        return True
    
    try:
        example_config = """# Federated Learning Experiment Configuration
# This is an example configuration file

experiment:
  name: "example_experiment"
  description: "Example FL experiment configuration"

dataset:
  name: "MNIST"
  num_clients: 10
  data_distribution: "iid"

strategy:
  name: "fedavg"
  num_rounds: 10
  min_fit_clients: 5
  min_eval_clients: 5

attacks:
  enabled: false
  type: "none"

logging:
  level: "INFO"
  save_logs: true
"""
        
        with open(config_file, 'w') as f:
            f.write(example_config)
        
        print(f"[OK] Created example configuration: {config_file}")
        return True
        
    except Exception as e:
        print(f"[WARNING] Could not create example config: {e}")
        return False

def print_usage_instructions():
    """Print instructions on how to use the framework."""
    print("\n" + "=" * 60)
    print("FRAMEWORK USAGE INSTRUCTIONS")
    print("=" * 60)
    
    print("\n1. BASIC EXPERIMENTS:")
    print("   python experiment_runners/basic_experiment_runner.py --help")
    print("   python experiment_runners/basic_experiment_runner.py --test-mode")
    
    print("\n2. ENHANCED EXPERIMENTS (Recommended):")
    print("   python experiment_runners/enhanced_experiment_runner.py --help")
    print("   python experiment_runners/enhanced_experiment_runner.py --test-mode")
    
    print("\n3. EXTENSIVE EXPERIMENTS:")
    print("   python experiment_runners/run_extensive_experiments.py --help")
    print("   python experiment_runners/run_extensive_experiments.py --num-runs 1")
    
    print("\n4. ATTACKS AND SECURITY:")
    print("   python experiment_runners/run_with_attacks.py --help")
    print("   python experiment_runners/run_with_attacks.py --strategy fedavg --attack noise --dataset MNIST")
    
    print("\n5. MONITORING:")
    print("   python scripts/monitor_experiments.py --watch")
    
    print("\n6. RESULTS ANALYSIS:")
    print("   python scripts/results_analyzer.py")
    
    print("\nFor detailed documentation, see the examples/ directory and README.md")

def install_missing_packages(missing_packages):
    """Attempt to install missing packages."""
    if not missing_packages:
        return True
    
    print(f"\n[INFO] Attempting to install {len(missing_packages)} missing packages...")
    
    try:
        # Try to install from requirements.txt first
        if Path("requirements.txt").exists():
            print("[INFO] Installing from requirements.txt...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("[OK] Successfully installed packages from requirements.txt")
                return True
        
        # Fallback to installing individual packages
        print("[INFO] Installing packages individually...")
        for package in missing_packages:
            print(f"Installing {package}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"[ERROR] Failed to install {package}")
                return False
        
        print("[OK] Successfully installed all missing packages")
        return True
        
    except Exception as e:
        print(f"[ERROR] Installation failed: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("FEDERATED LEARNING FRAMEWORK SETUP")
    print("=" * 60)
    print()
    
    success_count = 0
    total_checks = 0
    
    # 1. Check Python version
    total_checks += 1
    if check_python_version():
        success_count += 1
    
    # 2. Check directory structure
    total_checks += 1
    structure_ok, missing_dirs, missing_files = check_directory_structure()
    if structure_ok:
        success_count += 1
    
    # 3. Check dependencies
    total_checks += 1
    deps_ok, missing_packages = check_dependencies()
    if deps_ok:
        success_count += 1
    elif input("\nAttempt to install missing packages? (y/n): ").lower() == 'y':
        if install_missing_packages(missing_packages):
            print("\n[INFO] Re-checking dependencies after installation...")
            deps_ok, _ = check_dependencies()
            if deps_ok:
                success_count += 1
    
    # 4. Check port availability
    total_checks += 1
    if check_port_availability():
        success_count += 1
    
    # 5. Test run_with_attacks.py (only if structure is OK)
    if structure_ok:
        total_checks += 1
        if test_run_with_attacks():
            success_count += 1
    
    # 6. Create example config
    create_example_config()
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    print(f"Passed: {success_count}/{total_checks} checks")
    
    if success_count == total_checks:
        print("\n[SUCCESS] All checks passed! The framework is ready to use.")
        print_usage_instructions()
        return True
    else:
        print(f"\n[WARNING] {total_checks - success_count} checks failed.")
        
        if missing_dirs or missing_files:
            print("\nMissing files/directories suggest this may not be a complete")
            print("federated learning framework installation.")
        
        if not deps_ok:
            print("\nTo install missing dependencies:")
            print("  pip install -r requirements.txt")
        
        print("\nYou can still try to run experiments, but some features may not work.")
        print_usage_instructions()
        return False

if __name__ == "__main__":
    main()
