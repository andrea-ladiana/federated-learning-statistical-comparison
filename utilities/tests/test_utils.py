import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

import numpy as np

import utilities.utils as utils


def test_weighted_average_empty_metrics():
    assert utils.weighted_average([]) == {}


def test_weighted_average_zero_examples():
    metrics = [(0, {"accuracy": 0.5}), (0, {"accuracy": 1.0})]
    assert utils.weighted_average(metrics) == {}


def test_weighted_average_normal_case():
    metrics = [(2, {"acc": 0.5}), (8, {"acc": 0.75})]
    result = utils.weighted_average(metrics)
    expected = {"acc": (2 * 0.5 + 8 * 0.75) / 10}
    assert result == expected


def test_aggregate_ndarrays_weighted_normalization():
    weights = [[np.array([1.0])], [np.array([3.0])]]
    factors = [1.0, 3.0]
    result = utils.aggregate_ndarrays_weighted(weights, factors)
    expected = [np.array([1.0]) * 0.25 + np.array([3.0]) * 0.75]
    assert len(result) == 1
    assert np.allclose(result[0], expected[0])


def test_aggregate_ndarrays_weighted_negative_factor():
    weights = [[np.array([1.0])]]
    factors = [-1.0]
    result = utils.aggregate_ndarrays_weighted(weights, factors)
    assert result == []
