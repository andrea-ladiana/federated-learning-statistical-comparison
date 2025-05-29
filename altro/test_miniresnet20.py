#!/usr/bin/env python3
"""
Demo script per mostrare le capacità del MiniResNet20 con diversi dataset.
Questo script dimostra l'adattamento automatico degli input del modello.
"""

import torch
import torch.nn.functional as F
from models import MiniResNet20, get_transform_common, get_transform_cifar

def test_different_inputs():
    """Test MiniResNet20 con diversi formati di input."""
    print("🚀 Testing MiniResNet20 with different input formats")
    print("=" * 60)
    
    # Crea il modello
    model = MiniResNet20(num_classes=10)
    model.eval()
    
    # Test 1: Input MNIST-like (1 channel, 28x28)
    print("\n📊 Test 1: MNIST-like input (1, 28, 28)")
    mnist_input = torch.randn(1, 1, 28, 28)
    print(f"   Input shape: {mnist_input.shape}")
    
    with torch.no_grad():
        output = model(mnist_input)
        predictions = F.softmax(output, dim=1)
        print(f"   Output shape: {output.shape}")
        print(f"   Predictions: {predictions.max(1)[1].item()} (confidence: {predictions.max(1)[0].item():.3f})")
    
    # Test 2: Input Fashion-MNIST-like (1 channel, 28x28)
    print("\n👕 Test 2: Fashion-MNIST-like input (1, 28, 28)")
    fmnist_input = torch.randn(1, 1, 28, 28)
    print(f"   Input shape: {fmnist_input.shape}")
    
    with torch.no_grad():
        output = model(fmnist_input)
        predictions = F.softmax(output, dim=1)
        print(f"   Output shape: {output.shape}")
        print(f"   Predictions: {predictions.max(1)[1].item()} (confidence: {predictions.max(1)[0].item():.3f})")
    
    # Test 3: Input CIFAR-10-like (3 channels, 32x32)
    print("\n🖼️ Test 3: CIFAR-10-like input (3, 32, 32)")
    cifar_input = torch.randn(1, 3, 32, 32)
    print(f"   Input shape: {cifar_input.shape}")
    
    with torch.no_grad():
        output = model(cifar_input)
        predictions = F.softmax(output, dim=1)
        print(f"   Output shape: {output.shape}")
        print(f"   Predictions: {predictions.max(1)[1].item()} (confidence: {predictions.max(1)[0].item():.3f})")
    
    # Test 4: Batch processing
    print("\n📦 Test 4: Batch processing with mixed sizes")
    batch_mnist = torch.randn(4, 1, 28, 28)
    batch_cifar = torch.randn(4, 3, 32, 32)
    
    print(f"   MNIST batch shape: {batch_mnist.shape}")
    with torch.no_grad():
        mnist_outputs = model(batch_mnist)
        print(f"   MNIST batch output: {mnist_outputs.shape}")
    
    print(f"   CIFAR batch shape: {batch_cifar.shape}")
    with torch.no_grad():
        cifar_outputs = model(batch_cifar)
        print(f"   CIFAR batch output: {cifar_outputs.shape}")
    
    print("\n✅ All tests completed successfully!")
    print("\n💡 Key features demonstrated:")
    print("   • Automatic conversion: grayscale → RGB")
    print("   • Automatic resizing: 28x28 → 32x32")
    print("   • Consistent output format for all inputs")
    print("   • Efficient processing with depthwise separable convolutions")

def show_model_info():
    """Mostra informazioni sul modello."""
    print("\n🔧 Model Architecture Information")
    print("=" * 60)
    
    model = MiniResNet20()
    
    # Conta i parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📈 Total parameters: {total_params:,}")
    print(f"🏋️ Trainable parameters: {trainable_params:,}")
    print(f"💾 Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print(f"\n🏗️ Architecture:")
    print(f"   • Depthwise Separable Conv: 3→16 channels")
    print(f"   • ResNet Layer 1: 16 channels, 3 blocks")
    print(f"   • ResNet Layer 2: 32 channels, 3 blocks, stride=2")
    print(f"   • ResNet Layer 3: 64 channels, 3 blocks, stride=2")
    print(f"   • Global Average Pooling")
    print(f"   • Linear: 64→10 classes")
    
    print(f"\n🎯 Supported datasets:")
    print(f"   • MNIST (28x28, grayscale)")
    print(f"   • Fashion-MNIST (28x28, grayscale)")
    print(f"   • CIFAR-10 (32x32, RGB)")
    print(f"   • Any similar dataset with automatic adaptation")

if __name__ == "__main__":
    print("🧠 MiniResNet20 - Universal Model Demo")
    print("=" * 60)
    
    test_different_inputs()
    show_model_info()
    
    print(f"\n📚 Usage examples:")
    print(f"   from models import MiniResNet20")
    print(f"   model = MiniResNet20(num_classes=10)")
    print(f"   ")
    print(f"   # Works with any of these:")
    print(f"   mnist_data = torch.randn(batch, 1, 28, 28)")
    print(f"   cifar_data = torch.randn(batch, 3, 32, 32)")
    print(f"   output = model(mnist_data)  # Automatic adaptation!")
