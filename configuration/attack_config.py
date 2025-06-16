"""Configuration for federated learning attacks.

This module defines a small data structure holding default attack
parameters.  A factory function returns a new instance each time so
that modifications do not affect other modules.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Set


@dataclass
class AttackConfig:
    """Container for attack settings."""

    enable_attacks: bool = True
    noise_injection: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "noise_std": 0.1,
            "attack_fraction": 0.2,
        }
    )
    missed_class: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "class_removal_prob": 0.3,
        }
    )
    client_failure: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "failure_prob": 0.1,
            "debug_mode": True,
        }
    )
    data_asymmetry: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "min_factor": 0.5,
            "max_factor": 3.0,
            "class_removal_prob": 0.0,
        }
    )
    label_flipping: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "attack_fraction": 0.2,
            "flip_probability": 0.8,
            "fixed_source": None,
            "fixed_target": None,
            "change_each_round": True,
        }
    )
    gradient_flipping: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "attack_fraction": 0.2,
            "flip_intensity": 1.0,
        }
    )
    attack_state: Dict[str, Any] = field(
        default_factory=lambda: {
            "client_failure_history": {},
            "source_target_history": {},
            "current_round": 0,
            "broken_clients_in_current_round": set(),
            "gradient_flipping_history": {},
        }
    )


def create_attack_config() -> AttackConfig:
    """Return a fresh :class:`AttackConfig` instance."""
    return AttackConfig()
