import sys
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_runners.enhanced_experiment_runner import EnhancedExperimentRunner, EnhancedConfigManager

class DummyPortManager:
    def is_port_free(self, port):
        return True

class DummyProcess:
    def __init__(self, pid=1, name=None, cmdline=None):
        self.info = {'pid': pid, 'name': name, 'cmdline': cmdline or []}
        self.terminated = False
    def terminate(self):
        self.terminated = True
    def is_running(self):
        return False
    def kill(self):
        pass

def test_kill_flower_processes_handles_none_name(monkeypatch):
    cfg_manager = EnhancedConfigManager(Path('configuration/enhanced_config.yaml'))
    runner = EnhancedExperimentRunner(config_manager=cfg_manager, _test_mode=True)
    runner.port_manager = DummyPortManager()

    processes = [
        DummyProcess(pid=1, name=None, cmdline=['python', 'server.py']),
        DummyProcess(pid=2, name='python', cmdline=['run_with_attacks.py'])
    ]

    monkeypatch.setattr(psutil, 'process_iter', lambda attrs: processes)
    runner.kill_flower_processes()
    assert processes[1].terminated
