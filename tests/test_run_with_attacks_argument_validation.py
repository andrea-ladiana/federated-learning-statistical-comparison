import argparse
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_runners.run_with_attacks import configure_attacks


def default_args():
    return argparse.Namespace(
        attack="none",
        noise_std=0.1,
        noise_fraction=0.2,
        missed_prob=0.3,
        failure_prob=0.2,
        asymmetry_min=0.5,
        asymmetry_max=3.0,
        labelflip_fraction=0.2,
        flip_prob=0.8,
        source_class=None,
        target_class=None,
        gradflip_fraction=0.2,
        gradflip_intensity=1.0,
    )


def test_invalid_noise_fraction():
    args = default_args()
    args.attack = "noise"
    args.noise_fraction = 1.5
    with pytest.raises(ValueError):
        configure_attacks(args)


def test_invalid_noise_std():
    args = default_args()
    args.attack = "noise"
    args.noise_std = 0
    with pytest.raises(ValueError):
        configure_attacks(args)


def test_invalid_labelflip_fraction():
    args = default_args()
    args.attack = "labelflip"
    args.labelflip_fraction = -0.1
    with pytest.raises(ValueError):
        configure_attacks(args)


def test_invalid_gradflip_intensity():
    args = default_args()
    args.attack = "gradflip"
    args.gradflip_intensity = -1
    with pytest.raises(ValueError):
        configure_attacks(args)
