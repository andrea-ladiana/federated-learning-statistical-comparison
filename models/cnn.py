import torch
import torch.nn as nn


class CNNNet(nn.Module):
    """
    CNN Model - Basic Convolutional Neural Network for MNIST
    """
    def __init__(self):
        super().__init__()
        # Conv-Block 1: 32 filters, 3×3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv-Block 2: 64 filters, 3×3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully-Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Conv-Block 1
        x = self.pool(torch.relu(self.conv1(x)))  # Output: 32×14×14
        
        # Conv-Block 2
        x = self.pool(torch.relu(self.conv2(x)))  # Output: 64×7×7
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)  # Flatten to 3136
        
        # Fully-Connected Layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
