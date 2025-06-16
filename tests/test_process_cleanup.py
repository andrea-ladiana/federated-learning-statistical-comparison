import subprocess
import sys
from pathlib import Path
import types
import psutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_runners.basic_experiment_runner import ExperimentConfig, ExperimentRunner

class DummyProc:
    def __init__(self):
        self.stdout = self
        self.killed = False
        self.wait_after_kill_called = False
        self.initial_wait_called = False
        self.iter = 0
        self.pid = 12345

    # file-like interface for stdout
    def readline(self):
        self.iter += 1
        if self.iter > 2:
            self.stdout = None
            return ""
        return "line\n"

    def poll(self):
        return None

    def wait(self, timeout=None):
        if not self.killed:
            self.initial_wait_called = True
            raise subprocess.TimeoutExpired(cmd="dummy", timeout=timeout)
        else:
            self.wait_after_kill_called = True
            return 0

    def kill(self):
        self.killed = True

    def close(self):
        pass

class DummyRunner(ExperimentRunner):
    def kill_flower_processes(self):
        pass

    def wait_for_port(self, port: int, timeout: int = 60):
        pass

    def parse_and_store_metrics(self, log_line, config, run_id):
        pass

    def build_attack_command(self, config):
        return [sys.executable, "-c", "pass"]

def test_process_terminated_on_timeout(monkeypatch):
    dummy = DummyProc()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: dummy)

    cfg = ExperimentConfig(strategy="fedavg", attack="none", dataset="MNIST")
    runner = DummyRunner(base_dir=".", results_dir="./tmp", process_timeout=1)

    success = runner.run_single_experiment(cfg, run_id=0)
    assert not success
    assert dummy.killed
    assert dummy.wait_after_kill_called


class MockPsProcess:
    def __init__(self, pid):
        self.pid = pid
        self.terminated = False
        self.killed = False

    def cmdline(self):
        return ["python", "server.py"]

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def test_kill_flower_processes_does_not_touch_external_process(monkeypatch):
    runner = ExperimentRunner(base_dir=".", results_dir="./tmp")

    managed = MockPsProcess(111)
    external = MockPsProcess(222)

    process_map = {111: managed, 222: external}

    monkeypatch.setattr(psutil, "Process", lambda pid: process_map[pid])
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: pid in process_map)

    runner._managed_pids.add(111)

    runner.kill_flower_processes()

    assert managed.terminated or managed.killed
    assert not external.terminated
    assert not external.killed
