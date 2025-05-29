"""
MiniResNet20 - Un modello ResNet compatto che funziona per MNIST, Fashion-MNIST e CIFAR-10.
Questo modello include adattamento automatico degli input e usa depthwise separable convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class DepthwiseSeparableConv(nn.Module):
    """3×3 depth-wise + 1×1 point-wise conv."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return F.relu_(self.bn(x))


class BasicBlock(nn.Module):
    """Basic building block for ResNet."""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu_(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu_(out)


class MiniResNet20(nn.Module):
    """
    Un'unica rete che gestisce MNIST, Fashion-MNIST e CIFAR-10.
    
    Caratteristiche:
    - Adattamento automatico degli input (da grayscale a RGB, da 28x28 a 32x32)
    - Depthwise separable convolution per efficienza
    - Architettura ResNet con 20 layer
    - Supporto per diversi dataset senza modifiche al codice
    """
    
    def __init__(self, num_classes=10, blocks_per_stage=(3, 3, 3)):
        super().__init__()
        # primo layer depthwise-separable per risparmiare parametri
        self.prep = DepthwiseSeparableConv(3, 16)

        self.in_planes = 16
        self.layer1 = self._make_layer(16, blocks_per_stage[0], stride=1)
        self.layer2 = self._make_layer(32, blocks_per_stage[1], stride=2)
        self.layer3 = self._make_layer(64, blocks_per_stage[2], stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        """Crea un layer con multiple BasicBlock."""
        strides = [stride] + [1]*(num_blocks - 1)
        layers  = []
        for st in strides:
            layers.append(BasicBlock(self.in_planes, planes, st))
            self.in_planes = planes
        return nn.Sequential(*layers)

    @torch.inference_mode(False)
    def forward(self, x):
        """
        Forward pass con adattamento automatico dell'input.
        
        Args:
            x: Input tensor di forma (batch_size, channels, height, width)
               Supporta:
               - MNIST/Fashion-MNIST: (N, 1, 28, 28)
               - CIFAR-10: (N, 3, 32, 32)
        
        Returns:
            Output logits di forma (batch_size, num_classes)
        """
        # --- adattamento automatico forma input ---
        if x.shape[1] == 1:               # da grigio → 3 canali
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] != 32:             # da 28×28 → 32×32
            x = F.interpolate(x, size=32, mode='bilinear', align_corners=False)
        # ------------------------------------------
        
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# Utility functions per le trasformazioni
def get_transform_common():
    """
    Trasformazioni comuni per MNIST e Fashion-MNIST.
    Converte in RGB e ridimensiona a 32x32.
    """
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),   # MNIST & F-MNIST → 3 canali
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])


def get_transform_cifar():
    """
    Trasformazioni specifiche per CIFAR-10.
    Usa normalizzazione ottimizzata per CIFAR-10.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])


# Alias per compatibilità con l'architettura esistente
ResNet20 = MiniResNet20
