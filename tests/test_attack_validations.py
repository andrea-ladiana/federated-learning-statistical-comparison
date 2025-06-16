import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attacks.label_flipping import apply_targeted_label_flipping
from attacks.gradient_flipping import (
    apply_gradient_flipping,
    select_clients_for_gradient_flipping,
    MAX_FLIP_INTENSITY,
)


class DummyDataset(TensorDataset):
    def __init__(self):
        data = torch.zeros(10, 1)
        targets = torch.tensor([0, 1] * 5)
        super().__init__(data, targets)
        self.data = data
        self.targets = targets


def test_apply_targeted_label_flipping_invalid_probability():
    ds = DummyDataset()
    indices = list(range(len(ds)))
    with pytest.raises(ValueError):
        apply_targeted_label_flipping(ds, indices, 0, 1, -0.1)
    with pytest.raises(ValueError):
        apply_targeted_label_flipping(ds, indices, 0, 1, 1.1)


def test_gradient_flipping_invalid_fraction():
    with pytest.raises(ValueError):
        select_clients_for_gradient_flipping(5, -0.2)
    with pytest.raises(ValueError):
        select_clients_for_gradient_flipping(5, 1.2)


def test_apply_gradient_flipping_invalid_intensity():
    params = [np.array([1.0, -1.0])]
    with pytest.raises(ValueError):
        apply_gradient_flipping(params, -0.1)
    with pytest.raises(ValueError):
        apply_gradient_flipping(params, MAX_FLIP_INTENSITY + 1)
