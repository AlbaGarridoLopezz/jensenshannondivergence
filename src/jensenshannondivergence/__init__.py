from __future__ import annotations

from .estimators import (
    JSDEstimator,
    JSDEstimatorResult,
    estimate_jensen_shannon,
    supported_discriminators,
)

__all__ = [
    "JSDEstimator",
    "JSDEstimatorResult",
    "estimate_jensen_shannon",
    "supported_discriminators",
]

# Semantic package version used by users and CI
__version__ = "0.1.0"
