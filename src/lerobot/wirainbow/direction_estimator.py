"""
Direction estimation module for WiRainbow.

Implements the Sensing-Signal-to-Noise Ratio (SSNR) based direction estimation
algorithm using multi-interval variance analysis.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .antenna import FrequencyScanningAntenna
from .csi_processor import CSIProcessor


@dataclass
class DirectionEstimate:
    """
    Container for direction estimation results.

    Attributes:
        angle: Estimated target direction in degrees
        frequency: Subcarrier frequency corresponding to target direction
        subcarrier_index: Index of best subcarrier
        ssnr: SSNR value for the selected subcarrier
        ssnr_all: SSNR values for all subcarriers
        confidence: Confidence score (0-1) based on SSNR contrast
    """
    angle: float
    frequency: float
    subcarrier_index: int
    ssnr: float
    ssnr_all: np.ndarray
    confidence: float


class DirectionEstimator:
    """
    Estimates target direction using SSNR-based analysis.

    The algorithm:
    1. Computes TD-CSI for multiple time intervals
    2. Calculates phase differences between consecutive samples
    3. Computes variance of phase differences for each subcarrier
    4. Averages variances across intervals to obtain SSNR
    5. Identifies subcarrier with highest SSNR
    6. Maps frequency to angle using antenna calibration
    """

    def __init__(
        self,
        antenna: FrequencyScanningAntenna,
        csi_processor: CSIProcessor,
        min_interval: float = 0.005,  # 5 ms
        max_interval: float = 0.1,    # 100 ms
        num_intervals: int = 20,
        reference_antenna: int = 0
    ):
        """
        Initialize direction estimator.

        Args:
            antenna: Frequency-scanning antenna model
            csi_processor: CSI processor instance
            min_interval: Minimum time interval for TD-CSI
            max_interval: Maximum time interval for TD-CSI
            num_intervals: Number of different intervals to use
            reference_antenna: Index of antenna to use for estimation
        """
        self.antenna = antenna
        self.csi_processor = csi_processor
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.num_intervals = num_intervals
        self.reference_antenna = reference_antenna

    def compute_ssnr(
        self,
        csi: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Compute Sensing-Signal-to-Noise Ratio (SSNR) for all subcarriers.

        The SSNR is computed as the inverse of phase variance, which indicates
        the strength of motion-induced signal variations.

        Args:
            csi: Complex CSI array, shape (num_antennas, num_subcarriers, num_packets)
            timestamps: Packet timestamps in seconds

        Returns:
            SSNR values for each subcarrier, shape (num_subcarriers,)
        """
        num_antennas, num_subcarriers, num_packets = csi.shape

        # Use specified antenna
        csi_single = csi[self.reference_antenna]  # (num_subcarriers, num_packets)

        # Compute multi-interval TD-CSI
        td_csi_results = self.csi_processor.compute_multi_interval_td_csi(
            csi_single[np.newaxis, :, :],  # Add antenna dimension back
            timestamps,
            self.min_interval,
            self.max_interval,
            self.num_intervals
        )

        if len(td_csi_results) == 0:
            # No valid TD-CSI, return zeros
            return np.zeros(num_subcarriers)

        # Compute variance for each subcarrier across all intervals
        variance_sums = np.zeros(num_subcarriers)
        valid_intervals = 0

        for td_csi, valid_ts, interval in td_csi_results:
            if td_csi.shape[-1] < 2:
                continue  # Need at least 2 samples

            # Extract single antenna
            td_csi_single = td_csi[0]  # (num_subcarriers, num_samples)

            # Compute phase differences
            phase_diff = self.csi_processor.compute_phase_difference(
                td_csi_single, axis=-1
            )

            # Compute variance of phase differences for each subcarrier
            variance = np.var(phase_diff, axis=-1)

            # Accumulate
            variance_sums += variance
            valid_intervals += 1

        if valid_intervals == 0:
            return np.zeros(num_subcarriers)

        # Average variance across intervals
        avg_variance = variance_sums / valid_intervals

        # SSNR is inverse of variance (higher variance = stronger signal)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ssnr = 1.0 / (avg_variance + epsilon)

        # Normalize SSNR to [0, 1] range
        ssnr_normalized = (ssnr - np.min(ssnr)) / (np.max(ssnr) - np.min(ssnr) + epsilon)

        return ssnr_normalized

    def estimate_direction(
        self,
        csi: np.ndarray,
        timestamps: np.ndarray,
        frequencies: Optional[np.ndarray] = None
    ) -> DirectionEstimate:
        """
        Estimate target direction from CSI measurements.

        Args:
            csi: Complex CSI array, shape (num_antennas, num_subcarriers, num_packets)
            timestamps: Packet timestamps in seconds
            frequencies: Subcarrier frequencies in Hz (uses processor default if None)

        Returns:
            DirectionEstimate object containing estimated angle and metadata
        """
        num_subcarriers = csi.shape[1]

        if frequencies is None:
            frequencies = self.csi_processor.subcarrier_frequencies

        # Sanitize CSI
        csi = self.csi_processor.sanitize_csi(csi)

        # Compute SSNR for all subcarriers
        ssnr_all = self.compute_ssnr(csi, timestamps)

        # Find subcarrier with maximum SSNR
        best_subcarrier_idx = np.argmax(ssnr_all)
        best_ssnr = ssnr_all[best_subcarrier_idx]
        best_frequency = frequencies[best_subcarrier_idx]

        # Map frequency to angle using antenna model
        estimated_angle = self.antenna.get_angle_from_frequency(best_frequency)

        # Compute confidence based on SSNR contrast
        # Higher contrast = more confident
        ssnr_mean = np.mean(ssnr_all)
        ssnr_std = np.std(ssnr_all)
        if ssnr_std > 0:
            confidence = min(1.0, (best_ssnr - ssnr_mean) / (3 * ssnr_std))
        else:
            confidence = 0.0

        confidence = max(0.0, confidence)  # Ensure non-negative

        return DirectionEstimate(
            angle=estimated_angle,
            frequency=best_frequency,
            subcarrier_index=best_subcarrier_idx,
            ssnr=best_ssnr,
            ssnr_all=ssnr_all,
            confidence=confidence
        )

    def estimate_multiple_targets(
        self,
        csi: np.ndarray,
        timestamps: np.ndarray,
        num_targets: int,
        frequencies: Optional[np.ndarray] = None,
        min_separation: float = 10.0  # degrees
    ) -> List[DirectionEstimate]:
        """
        Estimate directions of multiple targets.

        Uses peak detection on SSNR to identify multiple targets.

        Args:
            csi: Complex CSI array
            timestamps: Packet timestamps
            num_targets: Number of targets to detect
            frequencies: Subcarrier frequencies
            min_separation: Minimum angular separation between targets (degrees)

        Returns:
            List of DirectionEstimate objects, sorted by SSNR (highest first)
        """
        num_subcarriers = csi.shape[1]

        if frequencies is None:
            frequencies = self.csi_processor.subcarrier_frequencies

        # Sanitize CSI
        csi = self.csi_processor.sanitize_csi(csi)

        # Compute SSNR
        ssnr_all = self.compute_ssnr(csi, timestamps)

        # Find peaks in SSNR
        from scipy.signal import find_peaks

        # Find peaks with minimum height threshold
        threshold = np.mean(ssnr_all) + 0.5 * np.std(ssnr_all)
        peaks, properties = find_peaks(ssnr_all, height=threshold)

        if len(peaks) == 0:
            # Fallback to single target
            return [self.estimate_direction(csi, timestamps, frequencies)]

        # Sort peaks by SSNR value
        peak_ssnrs = ssnr_all[peaks]
        sorted_indices = np.argsort(peak_ssnrs)[::-1]  # Descending order
        sorted_peaks = peaks[sorted_indices]

        # Select top targets with minimum separation
        selected_targets = []
        selected_angles = []

        for peak_idx in sorted_peaks:
            if len(selected_targets) >= num_targets:
                break

            freq = frequencies[peak_idx]
            angle = self.antenna.get_angle_from_frequency(freq)

            # Check separation from already selected targets
            too_close = False
            for prev_angle in selected_angles:
                if abs(angle - prev_angle) < min_separation:
                    too_close = True
                    break

            if not too_close:
                ssnr_mean = np.mean(ssnr_all)
                ssnr_std = np.std(ssnr_all)
                if ssnr_std > 0:
                    confidence = min(1.0, (ssnr_all[peak_idx] - ssnr_mean) / (3 * ssnr_std))
                else:
                    confidence = 0.0
                confidence = max(0.0, confidence)

                estimate = DirectionEstimate(
                    angle=angle,
                    frequency=freq,
                    subcarrier_index=peak_idx,
                    ssnr=ssnr_all[peak_idx],
                    ssnr_all=ssnr_all,
                    confidence=confidence
                )
                selected_targets.append(estimate)
                selected_angles.append(angle)

        # If fewer targets found than requested, fill with best estimate
        if len(selected_targets) == 0:
            selected_targets.append(self.estimate_direction(csi, timestamps, frequencies))

        return selected_targets

    def track_direction(
        self,
        csi: np.ndarray,
        timestamps: np.ndarray,
        previous_estimate: Optional[DirectionEstimate] = None,
        search_range: float = 30.0  # degrees
    ) -> DirectionEstimate:
        """
        Track target direction with temporal smoothing.

        Args:
            csi: Complex CSI array
            timestamps: Packet timestamps
            previous_estimate: Previous direction estimate for tracking
            search_range: Search range around previous estimate (degrees)

        Returns:
            Updated DirectionEstimate
        """
        current_estimate = self.estimate_direction(csi, timestamps)

        if previous_estimate is None:
            return current_estimate

        # Check if current estimate is within search range
        angle_diff = abs(current_estimate.angle - previous_estimate.angle)

        if angle_diff > search_range:
            # Large jump detected, may be unreliable
            # Use weighted average based on confidence
            alpha = current_estimate.confidence
            smoothed_angle = alpha * current_estimate.angle + (1 - alpha) * previous_estimate.angle
            current_estimate.angle = smoothed_angle

        return current_estimate
