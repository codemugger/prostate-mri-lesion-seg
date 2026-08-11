"""Compatibility import for code that referenced the former engine module.

The pre-manifest implementation was intentionally retired because it could
turn missing lesion labels into negatives and did not satisfy deployment
contracts. All entry points now route to the guarded engine.
"""

from .engine import (  # noqa: F401
    TrainArtifacts,
    build_model,
    train_from_config,
    train_lesion_from_config,
    train_organ_from_config,
)

__all__ = [
    "TrainArtifacts",
    "build_model",
    "train_from_config",
    "train_lesion_from_config",
    "train_organ_from_config",
]
