"""
WiRainbow: Single-Antenna Direction-Aware Wi-Fi Sensing via Dispersion Effect

This module implements the WiRainbow system as described in the paper:
https://arxiv.org/abs/2511.20671

The system enables directional awareness for Wi-Fi sensing using only a single
frequency-scanning antenna (FSA) by leveraging the dispersion effect.
"""

from .antenna import FrequencyScanningAntenna
from .csi_processor import CSIProcessor
from .direction_estimator import DirectionEstimator
from .wirainbow import WiRainbow

__all__ = [
    "FrequencyScanningAntenna",
    "CSIProcessor",
    "DirectionEstimator",
    "WiRainbow",
]
