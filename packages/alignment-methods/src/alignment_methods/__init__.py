"""Alignment methods package for ML playground."""

from .ncc import (
    ncc,
    ncc_batched,
    ncc_efficient,
    ncc_with_size_penalty,
    ncc_with_size_preference,
)
from .phase import phase_correlation, phase_correlation_peak
from .two_stage import two_stage_correlation

__all__ = [
    "ncc",
    "ncc_efficient",
    "ncc_batched",
    "ncc_with_size_preference",
    "ncc_with_size_penalty",
    "phase_correlation",
    "phase_correlation_peak",
    "two_stage_correlation",
    "hello",
]


def hello() -> str:
    return "Hello from alignment-methods!"
