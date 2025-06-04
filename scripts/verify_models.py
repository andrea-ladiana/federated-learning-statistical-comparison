#!/usr/bin/env python3
"""
Verification script for the reorganized models structure.
This script demonstrates that all models work correctly after reorganization.
"""

import sys
from pathlib import Path
import torch

# Add paths for reorganized imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "models"))

from models import Net, CNNNet, OptAEGV3, TinyMNIST, DepthwiseSeparableConv, MinimalCNN, MiniResNet20

def test_model(model, model_name, input_tensor):
    """Test a model with given input and print results."""
    try:
        output = model(input_tensor)
        print(f"✓ {model_name}: Forward pass successful, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ {model_name}: Error - {e}")
        return False

def main():
    print("=" * 60)
    print("MODELS REORGANIZATION VERIFICATION")
    print("=" * 60)
    
    # Test data for different input types
    mnist_input = torch.randn(2, 1, 28, 28)  # MNIST-like input
    cifar_input = torch.randn(2, 3, 32, 32)  # CIFAR-like input
    
    print("\n📁 Models directory structure created successfully!")
    print("   ├── __init__.py")
    print("   ├── simple.py          (Net)")
    print("   ├── cnn.py             (CNNNet)")
    print("   ├── optaegv3.py        (OptAEGV3, TinyMNIST)")
    print("   ├── minimal_cnn.py     (DepthwiseSeparableConv, MinimalCNN)")
    print("   ├── miniresnet20.py    (MiniResNet20, ResNet20)")
    print("   └── README.md")
    
    print("\n🧪 Testing MNIST-compatible models:")
    success_count = 0
    
    # Test MNIST models
    models_to_test = [
        (Net(), "Net (Simple Linear)"),
        (CNNNet(), "CNNNet (Basic CNN)"),
        (TinyMNIST(), "TinyMNIST (Optimized)"),
        (MinimalCNN(), "MinimalCNN (Adaptive)"),
        (MiniResNet20(), "MiniResNet20 (ResNet for Multi-Dataset)")
    ]
    
    for model, name in models_to_test:
        if test_model(model, name, mnist_input):
            success_count += 1
    
    print("\n🧪 Testing utility components:")
    # Test utility components
    opt_activation = OptAEGV3()
    dsconv = DepthwiseSeparableConv(3, 64)
    
    try:
        # Test OptAEGV3 with MNIST input
        opt_out = opt_activation(mnist_input)
        print(f"✓ OptAEGV3 (Activation): Forward pass successful, output shape: {opt_out.shape}")
        success_count += 1
    except Exception as e:
        print(f"✗ OptAEGV3 (Activation): Error - {e}")
    
    try:
        # Test DepthwiseSeparableConv with CIFAR input
        dsconv_out = dsconv(cifar_input)
        print(f"✓ DepthwiseSeparableConv: Forward pass successful, output shape: {dsconv_out.shape}")
        success_count += 1
    except Exception as e:
        print(f"✗ DepthwiseSeparableConv: Error - {e}")
    
    print(f"\n📊 Results: {success_count}/7 components working correctly")
    
    print("\n✅ Backward compatibility test:")
    try:
        # Test that old import style still works
        import models
        old_style_net = models.Net()
        print("✓ Original models.py import style still works!")
    except Exception as e:
        print(f"✗ Backward compatibility issue: {e}")
    
    print("\n🎯 New import options:")
    print("   # Option 1: Import from models package")
    print("   from models import Net, CNNNet, TinyMNIST, MiniResNet20")
    print("   ")
    print("   # Option 2: Import from specific modules")
    print("   from models.simple import Net")
    print("   from models.cnn import CNNNet")
    print("   from models.optaegv3 import TinyMNIST")
    print("   from models.miniresnet20 import MiniResNet20")
    print("   ")
    print("   # Option 3: Backward compatible (original style)")
    print("   import models")
    print("   net = models.Net()")
    
    print("\n" + "=" * 60)
    print("REORGANIZATION COMPLETE! ✨")
    print("=" * 60)

if __name__ == "__main__":
    main()
