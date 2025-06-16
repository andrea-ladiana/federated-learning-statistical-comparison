import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_runners.basic_experiment_runner import ExperimentRunner, ExperimentConfig

class SimpleProc:
    def __init__(self):
        self.killed = False
        self.wait_called = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.wait_called = True
        return 0

def test_handle_timeout_kills_process(tmp_path):
    runner = ExperimentRunner(base_dir='.', results_dir=tmp_path)
    proc = SimpleProc()
    runner._handle_timeout(proc, deque(['a', 'b']))
    assert proc.killed
    assert proc.wait_called

def test_parse_and_store_metrics(tmp_path):
    runner = ExperimentRunner(base_dir='.', results_dir=tmp_path)
    cfg = ExperimentConfig(strategy='fedavg', attack='none', dataset='MNIST')
    setattr(cfg, '_actual_strategy', cfg.strategy)
    runner.parse_and_store_metrics('[ROUND 1]', cfg, run_id=0)
    runner.parse_and_store_metrics('[Client 1] fit complete | loss=0.5, accuracy=0.9', cfg, run_id=0)
    assert len(runner.results_df) == 2
    assert set(runner.results_df['metric']) == {'loss', 'accuracy'}
    assert all(runner.results_df['round'] == 1)
