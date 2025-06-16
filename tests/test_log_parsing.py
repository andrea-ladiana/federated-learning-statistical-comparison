import sys
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_runners.stable_experiment_runner import MetricsCollector


def test_parse_client_log_supports_sign_and_exponent():
    collector = MetricsCollector()
    line = (
        "[Client 1] fit complete | avg_loss=-1.23e-2, accuracy=+9.87e-01 Round 3"
    )
    res = collector.parse_client_log(line, client_id=1, run_id=0)
    assert res is not None
    assert res["client_id"] == 1
    assert res["run"] == 0
    assert res["round"] == 3
    assert res["metric_type"] == "training"
    assert math.isclose(res["loss"], -1.23e-2)
    assert math.isclose(res["accuracy"], 9.87e-01)


def test_parse_server_log_supports_sign_and_exponent():
    collector = MetricsCollector()
    line = "evaluate server Round 2 accuracy -5.5e-01"
    res = collector.parse_server_log(line, run_id=1)
    assert res is not None
    assert res["run"] == 1
    assert res["round"] == 2
    assert res["metric"] == "accuracy"
    assert res["phase"] == "evaluation"
    assert math.isclose(res["value"], -5.5e-01)
