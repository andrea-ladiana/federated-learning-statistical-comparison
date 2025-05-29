import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution block:
    - Depthwise conv
    - Pointwise conv
    - BatchNorm + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class MinimalCNN(nn.Module):
    """
    Minimal CNN that auto-detects input channels and adapts for MNIST, FashionMNIST, CIFAR-10
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        
        # Initialize with Identity modules as placeholders for dynamic assignment
        self.in_channels = None
        self.block1 = nn.Identity()
        self.block2 = nn.Identity()
        self.block3 = nn.Identity()
        self.fc1 = nn.Identity()
        self.bn1 = nn.Identity()
        self.fc2 = nn.Identity()
        
    def _initialize(self, in_channels):
        """Initialize layers based on detected input channels"""
        self.in_channels = in_channels

        # Create and register all modules at once
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 32),
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )

        self.fc1 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, self.num_classes)
        
    def forward(self, x):
        # Auto-detect and initialize on first pass
        if self.in_channels is None:
            self._initialize(x.size(1))
            
        # Check if input is large enough for our architecture
        if x.size(2) < 8 or x.size(3) < 8:
            raise ValueError(f"Input image too small: {x.shape}. Minimum size required is 8x8.")

        # Forward pass through convolution blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected with BatchNorm, ReLU, and Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Use log_softmax for better numerical stability
        return F.log_softmax(x, dim=1)
