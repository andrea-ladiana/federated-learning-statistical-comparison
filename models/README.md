# Models Package

This directory contains the organized neural network models for the Federated Learning project.

## Structure

- `__init__.py` - Package initialization, imports all models
- `simple.py` - Simple linear model (`Net`)
- `cnn.py` - Basic CNN model (`CNNNet`)
- `optaegv3.py` - Optimized models (`OptAEGV3`, `TinyMNIST`)
- `minimal_cnn.py` - Adaptive CNN models (`DepthwiseSeparableConv`, `MinimalCNN`)

## Models Description

### Net (simple.py)
A simple linear model that flattens input images and applies a single fully connected layer. Suitable for basic classification tasks.

### CNNNet (cnn.py)
A basic convolutional neural network with two convolutional blocks followed by fully connected layers. Good for image classification with moderate complexity.

### OptAEGV3 & TinyMNIST (optaegv3.py)
- **OptAEGV3**: An optimized activation function module derived from transformer concepts
- **TinyMNIST**: An extremely efficient model using OptAEGV3 activation, capable of reaching 98.2% accuracy on MNIST with only 702 parameters

### DepthwiseSeparableConv & MinimalCNN (minimal_cnn.py)
- **DepthwiseSeparableConv**: A depthwise separable convolution block for efficient computation
- **MinimalCNN**: An adaptive CNN that auto-detects input channels and works with MNIST, FashionMNIST, and CIFAR-10

## Usage

You can import models in two ways:

```python
# Direct import from models package
from models import Net, CNNNet, TinyMNIST

# Import from the main models.py file (backward compatibility)
from models import Net, CNNNet, TinyMNIST
```

Both approaches work identically thanks to the re-export in the main `models.py` file.
