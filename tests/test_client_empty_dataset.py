import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.client import LoggingClient

class EmptyDataset(Dataset):
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        raise IndexError

def test_fit_and_evaluate_with_empty_dataset(monkeypatch):
    empty_loader = DataLoader(EmptyDataset(), batch_size=32)

    def fake_load_data(*args, **kwargs):
        return empty_loader, empty_loader

    monkeypatch.setattr(
        sys.modules['core.client'], 'load_data', fake_load_data
    )

    client = LoggingClient(cid="0", num_clients=1, dataset_name="MNIST")
    params = client.get_parameters({})

    upd_params, total_fit, metrics_fit = client.fit(params, {})
    assert total_fit == 0
    assert metrics_fit["loss"] == 0.0
    assert metrics_fit["accuracy"] == 0.0

    loss, total_eval, metrics_eval = client.evaluate(params, {})
    assert total_eval == 0
    assert metrics_eval["loss"] == 0.0
    assert metrics_eval["accuracy"] == 0.0
