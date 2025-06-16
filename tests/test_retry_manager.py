import sys
import time
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utilities.retry_manager import RetryManager, RetryConfig, FailureType


def test_classify_failure():
    manager = RetryManager()
    assert manager.classify_failure("timeout error") == FailureType.TIMEOUT
    assert manager.classify_failure("permission denied") == FailureType.PERMISSION_ERROR
    assert manager.classify_failure("connection lost") == FailureType.NETWORK_ERROR
    assert manager.classify_failure("memory issue") == FailureType.RESOURCE_ERROR
    assert manager.classify_failure("process failed", return_code=1) == FailureType.PROCESS_ERROR
    assert manager.classify_failure("something unknown") == FailureType.UNKNOWN_ERROR


def test_execute_with_retry_success(monkeypatch):
    calls = {"count": 0}

    def func():
        calls["count"] += 1
        if calls["count"] < 3:
            return False, "timeout"
        return True, "ok"

    cfg = RetryConfig(max_retries=3, base_delay=0, exponential_backoff=False)
    manager = RetryManager(cfg)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    success, out = manager.execute_with_retry("exp", func)
    assert success is True
    assert out == "ok"
    assert calls["count"] == 3
    assert len(manager.failure_history["exp"]) == 2


def test_execute_with_non_retryable(monkeypatch):
    def func():
        return False, "process failed"

    cfg = RetryConfig(max_retries=2, retry_on_types={FailureType.TIMEOUT})
    manager = RetryManager(cfg)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    success, out = manager.execute_with_retry("exp2", func)
    assert success is False
    assert len(manager.failure_history["exp2"]) == 1
