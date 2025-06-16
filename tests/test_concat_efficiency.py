import pandas as pd
import types
from experiment_runners.enhanced_experiment_runner import EnhancedExperimentRunner, EnhancedExperimentConfig

def test_concat_called_once_per_line(monkeypatch):
    runner = EnhancedExperimentRunner(_test_mode=True)
    config = EnhancedExperimentConfig(strategy="fedavg", attack="none", dataset="MNIST")

    call_count = 0
    original_concat = pd.concat

    def counting_concat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_concat(*args, **kwargs)

    monkeypatch.setattr(pd, "concat", counting_concat)

    lines = [
        "[ROUND 1]",
        "[Server] Round 1 aggregate fit -> loss=0.5, accuracy=0.1",
        "[Client 0] fit complete | loss=0.4, accuracy=0.5",
        "[Client 0] evaluate complete | loss=0.6, accuracy=0.4",
    ]

    for line in lines:
        runner.parse_and_store_metrics(line, config, run_id=0)

    assert call_count == 3
