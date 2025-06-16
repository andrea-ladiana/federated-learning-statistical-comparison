import time
from configuration.config_manager import ConfigManager

def test_check_system_resources_fast():
    cm = ConfigManager()
    start = time.perf_counter()
    cm.check_system_resources()
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0

