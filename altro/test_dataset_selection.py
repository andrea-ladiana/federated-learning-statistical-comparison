#!/usr/bin/env python3
"""
Test script per verificare che la selezione del dataset funzioni correttamente
con MiniResNet20 per MNIST, Fashion-MNIST e CIFAR-10.
"""

import torch
from models import MiniResNet20, get_transform_common, get_transform_cifar
from torchvision import datasets, transforms

def test_dataset_compatibility():
    """Testa che MiniResNet20 funzioni con tutti i dataset supportati."""
    print("=" * 60)
    print("TEST COMPATIBILITÀ DATASET CON MINIRESNET20")
    print("=" * 60)
    
    model = MiniResNet20()
    print(f"Modello: {model.__class__.__name__}")
    
    # Conteggio parametri
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali: {total_params:,}")
    
    # Test data per ogni dataset
    datasets_to_test = [
        ("MNIST", torch.randn(4, 1, 28, 28)),       # Grayscale 28x28
        ("Fashion-MNIST", torch.randn(4, 1, 28, 28)), # Grayscale 28x28  
        ("CIFAR-10", torch.randn(4, 3, 32, 32))     # RGB 32x32
    ]
    
    print("\n🧪 Testing model compatibility:")
    
    for dataset_name, test_input in datasets_to_test:
        try:
            with torch.no_grad():
                output = model(test_input)
            
            print(f"✓ {dataset_name:12} | Input: {tuple(test_input.shape)} → Output: {tuple(output.shape)}")
            
            # Verifica che l'output abbia la forma corretta (batch_size, 10)
            expected_shape = (test_input.shape[0], 10)
            if output.shape == expected_shape:
                print(f"  → Output shape correct: {output.shape}")
            else:
                print(f"  ✗ Output shape incorrect: {output.shape}, expected: {expected_shape}")
                
        except Exception as e:
            print(f"✗ {dataset_name:12} | Error: {e}")
    
    print("\n🔧 Testing transforms:")
    
    # Test trasformazioni
    transforms_to_test = [
        ("Common (MNIST/F-MNIST)", get_transform_common()),
        ("CIFAR-10", get_transform_cifar())
    ]
    
    for transform_name, transform in transforms_to_test:
        print(f"✓ {transform_name}: {len(transform.transforms)} steps")
        for i, step in enumerate(transform.transforms):
            print(f"  {i+1}. {step}")
    
    print("\n📊 Summary:")
    print("  • MiniResNet20 automatically adapts input dimensions")
    print("  • Converts grayscale (1 channel) → RGB (3 channels)")
    print("  • Resizes 28×28 → 32×32 via bilinear interpolation")
    print("  • Universal architecture for MNIST, Fashion-MNIST, CIFAR-10")
    print("  • Output: 10 classes for all datasets")
    
    print("=" * 60)
    print("TEST COMPLETATO! ✨")
    print("=" * 60)

def test_actual_datasets():
    """Test caricamento dei dataset reali."""
    print("\n🔍 Testing actual dataset loading:")
    
    # Transform per ogni dataset
    transform_common = get_transform_common()
    transform_cifar = get_transform_cifar()
    
    datasets_config = [
        ("MNIST", datasets.MNIST, transform_common),
        ("Fashion-MNIST", datasets.FashionMNIST, transform_common),
        ("CIFAR-10", datasets.CIFAR10, transform_cifar)
    ]
    
    for dataset_name, dataset_class, transform in datasets_config:
        try:
            # Carica solo un piccolo subset per il test
            testset = dataset_class(".", train=False, download=True, transform=transform)
            
            # Prendi un batch di esempio
            from torch.utils.data import DataLoader
            loader = DataLoader(testset, batch_size=2, shuffle=False)
            data, targets = next(iter(loader))
            
            print(f"✓ {dataset_name:12} | Shape: {tuple(data.shape)}, Classes: {len(testset.classes) if hasattr(testset, 'classes') else '10'}")
            
            # Test con il modello
            model = MiniResNet20()
            with torch.no_grad():
                output = model(data)
            print(f"  → Model output: {tuple(output.shape)}")
            
        except Exception as e:
            print(f"✗ {dataset_name:12} | Error: {e}")

if __name__ == "__main__":
    test_dataset_compatibility()
    test_actual_datasets()
