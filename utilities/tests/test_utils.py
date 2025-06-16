import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

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


def test_weighted_average_with_none_and_negative_values():
    metrics = [
        (-1, {"acc": 0.8}),
        (3, {"acc": 0.6}),
        (2, None),
        (5, {"acc": 0.9}),
    ]
    result = utils.weighted_average(metrics)
    expected = {"acc": (3 * 0.6 + 5 * 0.9) / (3 + 5)}
    assert result == expected


def test_weighted_average_all_invalid():
    metrics = [
        (-2, {"acc": 0.5}),
        (0, None),
    ]
    assert utils.weighted_average(metrics) == {}
