import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_runners.run_with_attacks import terminate_process

class DummyProc:
    def __init__(self):
        self.killed = False
    def terminate(self):
        pass
    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd="dummy", timeout=timeout)
    def kill(self):
        self.killed = True

class DummyProcKillSuccess(DummyProc):
    def wait(self, timeout=None):
        if self.killed:
            return 0
        else:
            raise subprocess.TimeoutExpired(cmd="dummy", timeout=timeout)


def test_terminate_process_kills_after_timeout(monkeypatch):
    proc = DummyProcKillSuccess()
    terminate_process(proc, timeout=0)
    assert proc.killed
