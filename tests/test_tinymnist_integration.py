#!/usr/bin/env python3
"""
Test di integrazione per verificare che il cambio da MiniResNet20 a TinyMNIST 
sia stato completato correttamente in tutto il sistema di federated learning.
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path for imports
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

def test_model_imports():
    """Test che tutti i modelli siano importabili correttamente."""
    print("🔍 Testing model imports...")
    try:
        from models import TinyMNIST, MiniResNet20
        print("✅ Model imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_server_model():
    """Test che il server usi TinyMNIST."""
    print("🖥️ Testing server model...")
    try:
        from core.server import initial_model
        model_name = initial_model.__class__.__name__
        if model_name == "TinyMNIST":
            print(f"✅ Server uses {model_name}")
            return True
        else:
            print(f"❌ Server uses {model_name} instead of TinyMNIST")
            return False
    except Exception as e:
        print(f"❌ Server test error: {e}")
        return False

def test_client_model():
    """Test che il client usi TinyMNIST."""
    print("👥 Testing client model...")
    try:
        from core.client import LoggingClient
        client = LoggingClient("0", "MNIST")  # Use "0" as client ID instead of "test"
        model_name = client.model.__class__.__name__
        if model_name == "TinyMNIST":
            print(f"✅ Client uses {model_name}")
            return True
        else:
            print(f"❌ Client uses {model_name} instead of TinyMNIST")
            return False
    except Exception as e:
        print(f"❌ Client test error: {e}")
        return False

def test_model_compatibility():
    """Test che TinyMNIST funzioni con MNIST e Fashion-MNIST."""
    print("🔄 Testing model compatibility...")
    try:
        from models import TinyMNIST
        model = TinyMNIST()
        
        # Test MNIST input (28x28 grayscale)
        mnist_input = torch.randn(1, 1, 28, 28)
        output = model(mnist_input)
        
        if output.shape == (1, 10):
            print("✅ TinyMNIST works with MNIST/Fashion-MNIST input")
            return True
        else:
            print(f"❌ Unexpected output shape: {output.shape}")
            return False
    except Exception as e:
        print(f"❌ Compatibility test error: {e}")
        return False

def test_parameter_efficiency():
    """Test l'efficienza parametrica di TinyMNIST vs MiniResNet20."""
    print("📊 Testing parameter efficiency...")
    try:
        from models import TinyMNIST, MiniResNet20
        
        tiny_model = TinyMNIST()
        mini_model = MiniResNet20()
        
        tiny_params = sum(p.numel() for p in tiny_model.parameters())
        mini_params = sum(p.numel() for p in mini_model.parameters())
        
        reduction = ((mini_params - tiny_params) / mini_params * 100)
        
        print(f"   TinyMNIST: {tiny_params:,} parameters")
        print(f"   MiniResNet20: {mini_params:,} parameters")
        print(f"   Reduction: {reduction:.1f}%")
        
        if tiny_params == 702:
            print("✅ TinyMNIST has expected 702 parameters")
            return True
        else:
            print(f"❌ TinyMNIST has {tiny_params} parameters, expected 702")
            return False
    except Exception as e:
        print(f"❌ Efficiency test error: {e}")
        return False

def main():
    """Esegue tutti i test di integrazione."""
    print("🧪 TinyMNIST Integration Test Suite")
    print("=" * 60)
    
    tests = [
        test_model_imports,
        test_server_model,
        test_client_model,
        test_model_compatibility,
        test_parameter_efficiency
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("📋 Test Results Summary")
    print("-" * 30)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ TinyMNIST integration successful")
        print("✅ Ready for federated learning experiments")
    else:
        print(f"\n⚠️ {total-passed} tests failed")
        print("❌ Integration incomplete")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
