import sys
import runpy
from pathlib import Path

import flwr as fl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_server_uses_cli_port(monkeypatch):
    captured = {}

    def fake_start_server(server_address, config=None, strategy=None):
        captured["addr"] = server_address

    monkeypatch.setattr(fl.server, "start_server", fake_start_server)
    monkeypatch.setattr("core.server.create_strategy", lambda *a, **k: None)
    testargs = ["server.py", "--rounds", "1", "--port", "9091"]
    monkeypatch.setattr(sys, "argv", testargs)
    runpy.run_module("core.server", run_name="__main__")
    assert captured["addr"] == "0.0.0.0:9091"
