import subprocess
import sys
import os
import platform

def print_section(title):
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

print_section("FLOWER INSTALLATION DIAGNOSTIC")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Python executable: {sys.executable}")

# Check if using Windows Store Python (which might have permission issues)
is_windows_store_python = "WindowsApps" in sys.executable
if is_windows_store_python:
    print_section("WINDOWS STORE PYTHON DETECTED")
    print("You're using Windows Store Python which might have permission restrictions.")
    print("Consider using Python from python.org if installation fails.")
    print("Download from: https://www.python.org/downloads/")

print_section("CHECKING PIP")
try:
    subprocess.run([sys.executable, "-m", "pip", "--version"], check=True)
except Exception as e:
    print(f"Error with pip: {e}")
    print("Please make sure pip is installed and working correctly.")
    sys.exit(1)

print_section("ATTEMPTING TO INSTALL FLOWER")
print("Running: pip install -U flwr")
try:
    # Force installation with --no-cache-dir to avoid using cached packages
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", "flwr"],
        capture_output=True,
        text=True,
        check=False
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print("Installation with pip failed.")
    else:
        print("Installation appears successful!")
except Exception as e:
    print(f"Exception during installation: {e}")

print_section("VERIFYING INSTALLATION")
try:
    # Try to import flwr after installation
    subprocess.run(
        [sys.executable, "-c", "import flwr; print(f'Flower installed successfully. Version: {flwr.__version__}')"],
        check=False
    )
except Exception as e:
    print(f"Exception during verification: {e}")

print_section("ALTERNATIVE INSTALLATION METHODS")
print("If the above installation failed, try one of these alternatives:")
print("\n1. Install using a virtual environment:")
print("   python -m venv flower_env")
print("   flower_env\\Scripts\\activate  # On Windows")
print("   pip install flwr")
print("\n2. Install with specific version:")
print("   pip install flwr==1.4.0")
print("\n3. Install from GitHub:")
print("   pip install git+https://github.com/adap/flower.git")
print("\n4. Check for conflicts with other packages:")
print("   pip list")

print_section("NEXT STEPS")
print("After installation, run the test.py script again to verify.")
print("If problems persist, try using a different Python distribution (not from Windows Store).")
