import sys
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attacks.data_asymmetry import generate_client_sizes, create_asymmetric_datasets


class DummyDataset(Dataset):
    def __init__(self, n):
        self.data = list(range(n))
        self.targets = [i % 10 for i in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def test_generate_client_sizes_uniform():
    sizes = generate_client_sizes(10, 3, 1.0, 1.0)
    assert sum(sizes) == 10
    assert len(sizes) == 3
    assert max(sizes) - min(sizes) <= 1


def test_generate_client_sizes_zero_handling():
    np.random.seed(0)
    sizes = generate_client_sizes(5, 10, 0.1, 0.1)
    assert sum(sizes) == 5
    assert len(sizes) == 10
    assert all(s in (0, 1) for s in sizes)


def test_create_asymmetric_datasets(tmp_path):
    np.random.seed(1)
    dataset = DummyDataset(20)
    client_sizes = [5, 5, 10]
    dsets, removed = create_asymmetric_datasets(dataset, client_sizes, class_removal_prob=0.5)
    assert len(dsets) == 3
    assert sum(len(d.indices) for d in dsets) <= len(dataset)
    assert set(removed.keys()) <= {0, 1, 2}


def test_generate_client_sizes_invalid_min_factor():
    with pytest.raises(ValueError):
        generate_client_sizes(10, 2, 0, 1)
    with pytest.raises(ValueError):
        generate_client_sizes(10, 2, -0.5, 1)


def test_generate_client_sizes_invalid_max_factor():
    with pytest.raises(ValueError):
        generate_client_sizes(10, 2, 1, 0)
    with pytest.raises(ValueError):
        generate_client_sizes(10, 2, 1, -2)


def test_generate_client_sizes_min_greater_than_max():
    with pytest.raises(ValueError):
        generate_client_sizes(10, 2, 2, 1)
