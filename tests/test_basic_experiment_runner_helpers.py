import subprocess
import sys
from collections import deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_runners.basic_experiment_runner import ExperimentRunner, ExperimentConfig

class DummyStdout:
    def __init__(self, lines):
        self.lines = lines
        self.idx = 0

    def readline(self):
        if self.idx < len(self.lines):
            line = self.lines[self.idx]
            self.idx += 1
            return line
        return ""

class DummyProcessStream:
    def __init__(self, lines):
        self.stdout = DummyStdout(lines)
        self.pid = 1

    def poll(self):
        return 0 if self.stdout.idx >= len(self.stdout.lines) else None

class DummyProcessTimeout:
    def __init__(self):
        self.killed = False
        self.pid = 2

    def wait(self, timeout=None):
        if not self.killed:
            raise subprocess.TimeoutExpired(cmd="dummy", timeout=timeout)
        return 0

    def kill(self):
        self.killed = True


class DummyRunner(ExperimentRunner):
    def parse_and_store_metrics(self, log_line, config, run_id):
        self.parsed.append(log_line)

    def _start_process(self, cmd):
        # Not used in these tests
        pass


def test_stream_process_output_parses_lines():
    runner = DummyRunner(base_dir=".", results_dir="./tmp")
    runner.parsed = []
    proc = DummyProcessStream(["line1\n", "line2\n"])
    cfg = ExperimentConfig(strategy="fedavg", attack="none", dataset="MNIST")
    count, lines = runner._stream_process_output(proc, cfg, run_id=0)
    assert count == 2
    assert list(lines) == ["line1", "line2"]
    assert runner.parsed == ["line1", "line2"]


def test_wait_for_process_handles_timeout():
    runner = ExperimentRunner(base_dir=".", results_dir="./tmp", process_timeout=1)
    proc = DummyProcessTimeout()
    success = runner._wait_for_process(proc, deque(), 0, "exp", 0)
    assert not success
    assert proc.killed

