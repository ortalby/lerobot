"""
Channel State Information (CSI) processor for WiRainbow.

Implements CSI extraction, preprocessing, and time-domain difference calculations
for motion-based Wi-Fi sensing.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class CSIData:
    """
    Container for Channel State Information data.

    Attributes:
        csi: Complex CSI values, shape (num_antennas, num_subcarriers, num_packets)
        frequencies: Subcarrier frequencies in Hz, shape (num_subcarriers,)
        timestamps: Packet timestamps in seconds, shape (num_packets,)
        subcarrier_indices: Indices of subcarriers
    """
    csi: np.ndarray
    frequencies: np.ndarray
    timestamps: np.ndarray
    subcarrier_indices: np.ndarray


class CSIProcessor:
    """
    Processes Channel State Information for Wi-Fi sensing.

    Implements:
    - Phase offset cancellation using CSI ratio method
    - Static component removal via Time-Domain CSI (TD-CSI)
    - CSI amplitude and phase extraction
    """

    def __init__(
        self,
        sampling_rate: float = 200.0,  # Hz
        center_frequency: float = 5.57e9,  # Hz (5.57 GHz)
        bandwidth: float = 80e6,  # Hz (80 MHz for 802.11ac)
        num_subcarriers: int = 234,  # For 802.11ac 80 MHz
    ):
        """
        Initialize CSI processor.

        Args:
            sampling_rate: Packet sampling rate in Hz
            center_frequency: Channel center frequency in Hz
            bandwidth: Channel bandwidth in Hz
            num_subcarriers: Number of OFDM subcarriers
        """
        self.sampling_rate = sampling_rate
        self.center_frequency = center_frequency
        self.bandwidth = bandwidth
        self.num_subcarriers = num_subcarriers

        # Compute subcarrier frequencies
        self.subcarrier_spacing = bandwidth / num_subcarriers
        self.subcarrier_frequencies = self._compute_subcarrier_frequencies()

    def _compute_subcarrier_frequencies(self) -> np.ndarray:
        """
        Compute frequencies of all subcarriers.

        Returns:
            Array of subcarrier frequencies in Hz
        """
        # Subcarriers are centered around the center frequency
        subcarrier_indices = np.arange(self.num_subcarriers) - self.num_subcarriers // 2
        frequencies = self.center_frequency + subcarrier_indices * self.subcarrier_spacing
        return frequencies

    def cancel_phase_offset(
        self,
        csi: np.ndarray,
        reference_antenna: int = 0
    ) -> np.ndarray:
        """
        Cancel hardware-induced phase offsets using CSI ratio method.

        Divides CSI from one antenna by another to remove synchronization-related
        phase offsets between transceivers.

        Args:
            csi: Complex CSI array, shape (num_antennas, num_subcarriers, num_packets)
            reference_antenna: Index of reference antenna

        Returns:
            Phase-offset-corrected CSI, shape (num_antennas, num_subcarriers, num_packets)
        """
        num_antennas = csi.shape[0]
        corrected_csi = np.zeros_like(csi, dtype=complex)

        # Reference antenna stays unchanged
        corrected_csi[reference_antenna] = csi[reference_antenna]

        # Divide other antennas by reference
        for i in range(num_antennas):
            if i != reference_antenna:
                # Avoid division by zero
                mask = np.abs(csi[reference_antenna]) > 1e-10
                corrected_csi[i] = np.where(
                    mask,
                    csi[i] / csi[reference_antenna],
                    csi[i]
                )

        return corrected_csi

    def compute_td_csi(
        self,
        csi: np.ndarray,
        time_delta: float,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Time-Domain Difference CSI (TD-CSI) to remove static components.

        Based on Equation 6 from the paper:
        ΔH(f,t) = H(f,t+Δt) - H(f,t)

        Args:
            csi: Complex CSI array, shape (num_antennas, num_subcarriers, num_packets)
            time_delta: Time interval in seconds
            timestamps: Packet timestamps in seconds

        Returns:
            Tuple of (td_csi, valid_timestamps)
            - td_csi: Time-domain difference CSI
            - valid_timestamps: Timestamps of valid TD-CSI samples
        """
        num_antennas, num_subcarriers, num_packets = csi.shape

        # Find packet pairs separated by approximately time_delta
        td_csi_list = []
        valid_timestamps_list = []

        for i in range(num_packets):
            # Find next packet approximately time_delta seconds later
            time_diff = timestamps - timestamps[i]
            candidates = np.where(
                (time_diff >= time_delta * 0.9) &
                (time_diff <= time_delta * 1.1)
            )[0]

            if len(candidates) > 0:
                j = candidates[0]
                # Compute time-domain difference
                delta_h = csi[:, :, j] - csi[:, :, i]
                td_csi_list.append(delta_h)
                valid_timestamps_list.append(timestamps[i])

        if len(td_csi_list) == 0:
            # No valid pairs found
            return np.array([]), np.array([])

        # Stack results
        td_csi = np.stack(td_csi_list, axis=-1)  # (num_antennas, num_subcarriers, num_samples)
        valid_timestamps = np.array(valid_timestamps_list)

        return td_csi, valid_timestamps

    def compute_multi_interval_td_csi(
        self,
        csi: np.ndarray,
        timestamps: np.ndarray,
        min_interval: float = 0.005,  # 5 ms
        max_interval: float = 0.1,    # 100 ms
        num_intervals: int = 20
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Compute TD-CSI for multiple time intervals.

        This multi-interval approach adapts to varying motion speeds and
        mitigates noise domination at slow speeds.

        Args:
            csi: Complex CSI array, shape (num_antennas, num_subcarriers, num_packets)
            timestamps: Packet timestamps in seconds
            min_interval: Minimum time interval in seconds
            max_interval: Maximum time interval in seconds
            num_intervals: Number of different intervals to use

        Returns:
            List of tuples (td_csi, valid_timestamps, interval) for each interval
        """
        intervals = np.linspace(min_interval, max_interval, num_intervals)
        results = []

        for interval in intervals:
            td_csi, valid_ts = self.compute_td_csi(csi, interval, timestamps)
            if len(valid_ts) > 0:
                results.append((td_csi, valid_ts, interval))

        return results

    def extract_amplitude(self, csi: np.ndarray) -> np.ndarray:
        """
        Extract amplitude from complex CSI.

        Args:
            csi: Complex CSI array

        Returns:
            Amplitude array with same shape as input
        """
        return np.abs(csi)

    def extract_phase(self, csi: np.ndarray) -> np.ndarray:
        """
        Extract phase from complex CSI.

        Args:
            csi: Complex CSI array

        Returns:
            Phase array (in radians) with same shape as input
        """
        return np.angle(csi)

    def compute_phase_difference(
        self,
        td_csi: np.ndarray,
        axis: int = -1
    ) -> np.ndarray:
        """
        Compute phase differences between consecutive TD-CSI samples.

        Args:
            td_csi: Time-domain difference CSI
            axis: Axis along which to compute differences

        Returns:
            Phase difference array
        """
        # Extract phase
        phase = self.extract_phase(td_csi)

        # Compute differences along specified axis
        phase_diff = np.diff(phase, axis=axis)

        # Unwrap phase to handle 2π discontinuities
        phase_diff = np.angle(np.exp(1j * phase_diff))

        return phase_diff

    def apply_bandpass_filter(
        self,
        signal: np.ndarray,
        low_freq: float,
        high_freq: float,
        axis: int = -1
    ) -> np.ndarray:
        """
        Apply bandpass filter to signal (e.g., for respiration monitoring).

        Args:
            signal: Input signal
            low_freq: Lower cutoff frequency in Hz
            high_freq: Upper cutoff frequency in Hz
            axis: Time axis

        Returns:
            Filtered signal
        """
        from scipy import signal as scipy_signal

        # Design Butterworth bandpass filter
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        b, a = scipy_signal.butter(4, [low, high], btype='band')

        # Apply filter
        filtered = scipy_signal.filtfilt(b, a, signal, axis=axis)

        return filtered

    def sanitize_csi(self, csi: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
        """
        Remove invalid CSI values (zeros, NaNs, Infs).

        Args:
            csi: Complex CSI array
            threshold: Minimum valid amplitude

        Returns:
            Sanitized CSI array
        """
        # Replace NaN and Inf with zeros
        csi = np.where(np.isfinite(csi), csi, 0.0 + 0.0j)

        # Mask very small values
        amplitude = np.abs(csi)
        csi = np.where(amplitude > threshold, csi, 0.0 + 0.0j)

        return csi
