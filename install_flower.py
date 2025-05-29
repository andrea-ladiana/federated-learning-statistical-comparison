import subprocess
import sys
import importlib.util
import os

def check_module_installed(module_name):
    """Check if a module is installed"""
    return importlib.util.find_spec(module_name) is not None

def install_flower():
    print("=" * 60)
    print("FLOWER INSTALLATION ASSISTANT")
    print("=" * 60)
    
    # Display Python version and path
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if flower is already installed
    if check_module_installed("flwr"):
        print("\n✅ Flower library is already installed!")
        return True
    
    print("\n❌ Flower library is not installed. Attempting to install...")
    
    # Try to install flower
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "flwr"])
        print("\n✅ Installation successful!")
        
        # Verify installation
        if check_module_installed("flwr"):
            print("✅ Flower library is now properly installed.")
            return True
        else:
            print("❌ Installation verification failed.")
            return False
    except subprocess.CalledProcessError:
        print("\n❌ Installation failed.")
        return False

def create_requirements_file():
    """Create a requirements.txt file with all necessary dependencies"""
    requirements = [
        "flwr>=1.0.0",
        "torch",
        "torchvision",
        "numpy",
        "matplotlib"
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    print("\nCreated requirements.txt file with necessary dependencies.")
    print("You can install all dependencies with: pip install -r requirements.txt")

def troubleshooting_guide():
    print("\n" + "=" * 60)
    print("TROUBLESHOOTING GUIDE")
    print("=" * 60)
    print("If you're still having issues, try these steps:")
    print("1. Make sure you're using Python 3.9 or 3.10 (Flower may have compatibility issues with Python 3.12)")
    print("2. Create a virtual environment:")
    print("   - Windows: python -m venv venv && venv\\Scripts\\activate")
    print("   - Linux/Mac: python -m venv venv && source venv/bin/activate")
    print("3. Install all requirements: pip install -r requirements.txt")
    print("4. If using Anaconda, try: conda install -c conda-forge flwr")
    print("\nAdditional commands that might help:")
    print("- Update pip: python -m pip install --upgrade pip")
    print("- Install with specific version: pip install flwr==1.5.0")
    print("=" * 60)

if __name__ == "__main__":
    success = install_flower()
    create_requirements_file()
    
    if not success:
        troubleshooting_guide()
    else:
        print("\nYou can now run your Federated Learning code!")
