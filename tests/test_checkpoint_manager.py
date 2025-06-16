import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utilities.checkpoint_manager import CheckpointManager


def test_register_and_complete_runs(tmp_path):
    manager = CheckpointManager(tmp_path)
    config = {"strategy": "fedavg", "attack": "none", "dataset": "mnist"}
    manager.register_experiments([config], num_runs=2)
    exp_id = "fedavg_none_mnist"
    assert exp_id in manager.experiment_status
    manager.mark_run_completed(exp_id, 0)
    manager.mark_run_completed(exp_id, 1)
    status = manager.experiment_status[exp_id]
    assert status.is_complete()
    assert status.success is True

    manager2 = CheckpointManager(tmp_path)
    assert manager2.experiment_status[exp_id].completed_runs == {0, 1}


def test_backup_results(tmp_path):
    manager = CheckpointManager(tmp_path)
    df = pd.DataFrame({"a": [1, 2]})
    manager.backup_results(df, "_test")
    backups = list((tmp_path / "results_backup").glob("results_backup_*_test.csv"))
    assert backups
