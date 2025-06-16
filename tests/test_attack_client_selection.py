import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attacks.label_flipping import select_clients_for_label_flipping
from attacks.noise_injection import select_clients_for_noise_injection


def test_label_flipping_selection():
    np.random.seed(0)
    clients = select_clients_for_label_flipping(5, 0.4)
    assert len(clients) == 2
    assert all(0 <= c < 5 for c in clients)


def test_noise_injection_selection():
    np.random.seed(1)
    clients = select_clients_for_noise_injection(8, 0.25)
    assert len(clients) == 2
    assert all(0 <= c < 8 for c in clients)


def test_zero_fraction_returns_empty():
    np.random.seed(0)
    assert select_clients_for_label_flipping(6, 0.0) == []
    assert select_clients_for_noise_injection(6, 0.0) == []
