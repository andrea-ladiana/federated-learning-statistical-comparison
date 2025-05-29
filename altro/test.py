try:
    import flwr as fl  # Libreria Flower
    print(f"Flower successfully imported. Version: {fl.__version__}")
except ImportError as e:
    print("Error importing Flower library (flwr):")
    print(f"Original error: {e}")
    print("\nTry these solutions:")
    print("1. Reinstall flower: pip install -U flwr")
    print("2. Check if you're using the correct Python environment")
    print("3. Try importing with 'import flower as fl' instead")
    try:
        import flower as fl
        print("Successfully imported using 'import flower as fl'")
    except ImportError:
        print("Both 'import flwr' and 'import flower' failed.")
        print("Please make sure Flower is installed correctly: pip install -U flwr")
        import sys
        print(f"Python executable: {sys.executable}")
        print(f"Python path: {sys.path}")
        raise