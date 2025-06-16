import sys
import socket
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_runners.enhanced_experiment_runner import PortManager, ResourceError


def test_acquire_and_release_port():
    pm = PortManager(base_port=15000, num_ports=2)
    p1 = pm.acquire_port()
    p2 = pm.acquire_port()
    assert {p1, p2} == {15000, 15001}
    with pytest.raises(ResourceError):
        pm.acquire_port()
    pm.release_port(p1)
    assert pm.acquire_port() == p1


def test_port_context_manager():
    pm = PortManager(base_port=16000, num_ports=1)
    with pm.get_port() as p:
        assert p == 16000
        assert pm.available_ports == []
    assert pm.available_ports == [16000]


def test_is_port_free():
    pm = PortManager(base_port=17000, num_ports=1)
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    assert pm.is_port_free(port) is False
    s.close()
    assert pm.is_port_free(port) is True
