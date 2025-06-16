import pandas as pd
from experiment_runners.enhanced_experiment_runner import EnhancedExperimentRunner, EnhancedExperimentConfig


def test_parse_negative_and_scientific_metrics():
    runner = EnhancedExperimentRunner(_test_mode=True)
    config = EnhancedExperimentConfig(strategy="fedavg", attack="none", dataset="MNIST")

    runner.parse_and_store_metrics("[Server] Round 1 aggregate fit -> loss=-0.42, accuracy=0.95", config, 0)
    runner.parse_and_store_metrics("[Client 2] fit complete | loss=1.2e-3, accuracy=8e-2", config, 0)

    loss_values = runner.results_df[runner.results_df["metric"] == "fit_loss"]["value"].tolist()

    assert any(abs(v + 0.42) < 1e-6 for v in loss_values)
    assert any(abs(v - 1.2e-3) < 1e-6 for v in loss_values)
