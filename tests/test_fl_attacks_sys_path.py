import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_fl_attacks_path_not_duplicated():
    import utilities.fl_attacks as fl_attacks
    attack_path = str(ROOT / "attacks")
    # Should be inserted exactly once after initial import
    assert sys.path.count(attack_path) == 1

    importlib.reload(fl_attacks)
    # Path should still appear only once
    assert sys.path.count(attack_path) == 1
