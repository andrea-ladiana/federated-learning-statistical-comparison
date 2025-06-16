import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.client import LoggingClient
from configuration.attack_config import create_attack_config
import utilities.fl_attacks as fl_attacks


class SmallDataset(Dataset):
    def __len__(self):
        return 4
    def __getitem__(self, idx):
        return torch.zeros(1, 28, 28), 0


def test_noise_clients_constant_within_round(monkeypatch):
    loader = DataLoader(SmallDataset(), batch_size=2)

    monkeypatch.setattr(
        sys.modules['core.client'],
        'load_data',
        lambda *a, **k: (loader, loader)
    )

    cfg = create_attack_config()
    cfg.enable_attacks = True
    cfg.noise_injection['enabled'] = True
    cfg.noise_injection['attack_fraction'] = 1.0

    call_count = 0
    def fake_select(num_clients, frac):
        nonlocal call_count
        call_count += 1
        return [0]

    monkeypatch.setattr(fl_attacks, 'select_clients_for_noise_injection', fake_select)

    applied = 0
    def fake_apply(data, std):
        nonlocal applied
        applied += 1
        return data

    monkeypatch.setattr(fl_attacks, 'apply_noise_injection', fake_apply)

    client = LoggingClient(cid="0", num_clients=1, dataset_name="MNIST", attack_cfg=cfg)
    params = client.get_parameters({})
    client.fit(params, {"round": 1})

    assert call_count == 1
    assert applied == len(loader)
    assert client.noise_clients == [0]
