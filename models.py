"""
Main models module - imports all model classes from the models package.
This file maintains backward compatibility by re-exporting all models.
"""

# Import all models from the new organized structure
from models import (
    Net,
    CNNNet,
    OptAEGV3,
    TinyMNIST,
    DepthwiseSeparableConv,
    MinimalCNN
)

# Make all models available when importing from models.py
__all__ = [
    'Net',
    'CNNNet',
    'OptAEGV3',
    'TinyMNIST',
    'DepthwiseSeparableConv',
    'MinimalCNN'
]