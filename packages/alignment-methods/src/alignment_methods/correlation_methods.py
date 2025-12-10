"""Backwards-compatibility wrapper for legacy imports.

All implementations have been split into dedicated modules:

- :mod:`alignment_methods.ncc` for NCC variants and scoring helpers
- :mod:`alignment_methods.phase` for phase correlation utilities
- :mod:`alignment_methods.two_stage` for the two-stage hybrid approach
"""

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
]
