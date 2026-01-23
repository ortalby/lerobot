"""
Main WiRainbow system implementation.

Integrates all components for single-antenna direction-aware Wi-Fi sensing.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import json

from .antenna import FrequencyScanningAntenna
from .csi_processor import CSIProcessor, CSIData
from .direction_estimator import DirectionEstimator, DirectionEstimate


@dataclass
class WiRainbowConfig:
    """
    Configuration for WiRainbow system.

    Attributes:
        # Antenna parameters
        num_resonators: Number of resonator elements
        element_spacing: Distance between resonators (meters)
        quality_factor: Q factor of resonators
        resonant_freq: Central resonant frequency (Hz)

        # Signal processing parameters
        sampling_rate: Packet sampling rate (Hz)
        center_frequency: Channel center frequency (Hz)
        bandwidth: Channel bandwidth (Hz)
        num_subcarriers: Number of OFDM subcarriers

        # Direction estimation parameters
        min_interval: Minimum time interval for TD-CSI (seconds)
        max_interval: Maximum time interval for TD-CSI (seconds)
        num_intervals: Number of different intervals
        reference_antenna: Index of antenna for direction estimation

        # Calibration
        calibration_angles: Angles for antenna calibration (degrees)
        calibration_frequencies: Frequencies for calibration (Hz)
    """
    # Antenna parameters
    num_resonators: int = 12
    element_spacing: float = 0.0145
    quality_factor: float = 200.0
    resonant_freq: float = 5.57e9

    # Signal processing parameters
    sampling_rate: float = 200.0
    center_frequency: float = 5.57e9
    bandwidth: float = 80e6
    num_subcarriers: int = 234

    # Direction estimation parameters
    min_interval: float = 0.005
    max_interval: float = 0.1
    num_intervals: int = 20
    reference_antenna: int = 0

    # Calibration
    calibration_angles: List[float] = field(
        default_factory=lambda: list(np.linspace(-60, 60, 121))
    )
    calibration_frequencies: List[float] = field(
        default_factory=lambda: list(np.linspace(5.37e9, 5.73e9, 360))
    )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'num_resonators': self.num_resonators,
            'element_spacing': self.element_spacing,
            'quality_factor': self.quality_factor,
            'resonant_freq': self.resonant_freq,
            'sampling_rate': self.sampling_rate,
            'center_frequency': self.center_frequency,
            'bandwidth': self.bandwidth,
            'num_subcarriers': self.num_subcarriers,
            'min_interval': self.min_interval,
            'max_interval': self.max_interval,
            'num_intervals': self.num_intervals,
            'reference_antenna': self.reference_antenna,
            'calibration_angles': self.calibration_angles,
            'calibration_frequencies': self.calibration_frequencies,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'WiRainbowConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'WiRainbowConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class WiRainbow:
    """
    Main WiRainbow system for single-antenna direction-aware Wi-Fi sensing.

    This class integrates:
    - Frequency-scanning antenna model
    - CSI processing and preprocessing
    - Direction estimation using SSNR

    Example:
        >>> config = WiRainbowConfig()
        >>> wirainbow = WiRainbow(config)
        >>> wirainbow.calibrate()
        >>>
        >>> # Process CSI data
        >>> csi = np.random.randn(2, 234, 1000) + 1j * np.random.randn(2, 234, 1000)
        >>> timestamps = np.linspace(0, 5, 1000)
        >>> estimate = wirainbow.estimate_direction(csi, timestamps)
        >>> print(f"Target at {estimate.angle:.1f} degrees")
    """

    def __init__(self, config: Optional[WiRainbowConfig] = None):
        """
        Initialize WiRainbow system.

        Args:
            config: System configuration (uses default if None)
        """
        self.config = config if config is not None else WiRainbowConfig()

        # Initialize components
        self.antenna = FrequencyScanningAntenna(
            num_resonators=self.config.num_resonators,
            element_spacing=self.config.element_spacing,
            quality_factor=self.config.quality_factor,
            resonant_freq=self.config.resonant_freq,
        )

        self.csi_processor = CSIProcessor(
            sampling_rate=self.config.sampling_rate,
            center_frequency=self.config.center_frequency,
            bandwidth=self.config.bandwidth,
            num_subcarriers=self.config.num_subcarriers,
        )

        self.direction_estimator = DirectionEstimator(
            antenna=self.antenna,
            csi_processor=self.csi_processor,
            min_interval=self.config.min_interval,
            max_interval=self.config.max_interval,
            num_intervals=self.config.num_intervals,
            reference_antenna=self.config.reference_antenna,
        )

        self._is_calibrated = False

    def calibrate(self):
        """
        Calibrate antenna frequency-to-direction mapping.

        This would typically be done in an anechoic chamber in practice.
        Here we use the theoretical antenna model.
        """
        calibration_map = self.antenna.calibrate_frequency_direction_mapping(
            frequencies=np.array(self.config.calibration_frequencies),
            angles=np.array(self.config.calibration_angles)
        )
        self._is_calibrated = True
        return calibration_map

    def preprocess_csi(
        self,
        csi: np.ndarray,
        cancel_phase_offset: bool = True
    ) -> np.ndarray:
        """
        Preprocess CSI data.

        Args:
            csi: Raw CSI array, shape (num_antennas, num_subcarriers, num_packets)
            cancel_phase_offset: Whether to apply phase offset cancellation

        Returns:
            Preprocessed CSI array
        """
        # Sanitize CSI
        csi = self.csi_processor.sanitize_csi(csi)

        # Cancel phase offsets if requested
        if cancel_phase_offset and csi.shape[0] > 1:
            csi = self.csi_processor.cancel_phase_offset(
                csi, reference_antenna=self.config.reference_antenna
            )

        return csi

    def estimate_direction(
        self,
        csi: np.ndarray,
        timestamps: np.ndarray,
        preprocess: bool = True
    ) -> DirectionEstimate:
        """
        Estimate target direction from CSI measurements.

        Args:
            csi: CSI array, shape (num_antennas, num_subcarriers, num_packets)
            timestamps: Packet timestamps in seconds
            preprocess: Whether to preprocess CSI

        Returns:
            DirectionEstimate object
        """
        if not self._is_calibrated:
            self.calibrate()

        if preprocess:
            csi = self.preprocess_csi(csi)

        estimate = self.direction_estimator.estimate_direction(
            csi, timestamps, self.csi_processor.subcarrier_frequencies
        )

        return estimate

    def estimate_multiple_targets(
        self,
        csi: np.ndarray,
        timestamps: np.ndarray,
        num_targets: int = 2,
        preprocess: bool = True
    ) -> List[DirectionEstimate]:
        """
        Estimate directions of multiple targets.

        Args:
            csi: CSI array
            timestamps: Packet timestamps
            num_targets: Number of targets to detect
            preprocess: Whether to preprocess CSI

        Returns:
            List of DirectionEstimate objects
        """
        if not self._is_calibrated:
            self.calibrate()

        if preprocess:
            csi = self.preprocess_csi(csi)

        estimates = self.direction_estimator.estimate_multiple_targets(
            csi, timestamps, num_targets,
            self.csi_processor.subcarrier_frequencies
        )

        return estimates

    def monitor_respiration(
        self,
        csi: np.ndarray,
        timestamps: np.ndarray,
        target_direction: Optional[float] = None,
        respiration_band: Tuple[float, float] = (0.2, 0.5)  # 12-30 bpm
    ) -> Dict:
        """
        Monitor respiration rate in a specific direction.

        Args:
            csi: CSI array
            timestamps: Packet timestamps
            target_direction: Target direction in degrees (auto-detect if None)
            respiration_band: Frequency band for respiration (Hz)

        Returns:
            Dictionary with respiration rate and signal
        """
        if not self._is_calibrated:
            self.calibrate()

        # Estimate target direction if not provided
        if target_direction is None:
            estimate = self.estimate_direction(csi, timestamps)
            target_direction = estimate.angle
            target_freq = estimate.frequency
            subcarrier_idx = estimate.subcarrier_index
        else:
            # Find frequency corresponding to target direction
            # (Inverse mapping from angle to frequency)
            calibration_map = self.antenna._freq_to_angle_map
            if calibration_map is not None:
                # Find frequency closest to target angle
                best_freq = None
                min_diff = float('inf')
                for freq, data in calibration_map.items():
                    diff = abs(data['angle'] - target_direction)
                    if diff < min_diff:
                        min_diff = diff
                        best_freq = freq
                target_freq = best_freq
                # Find subcarrier index
                subcarrier_idx = np.argmin(
                    np.abs(self.csi_processor.subcarrier_frequencies - target_freq)
                )
            else:
                # Use center frequency
                subcarrier_idx = self.config.num_subcarriers // 2
                target_freq = self.csi_processor.subcarrier_frequencies[subcarrier_idx]

        # Extract CSI for target subcarrier
        csi_target = csi[self.config.reference_antenna, subcarrier_idx, :]

        # Extract amplitude (respiratory signal is in amplitude)
        amplitude = np.abs(csi_target)

        # Apply bandpass filter for respiration
        filtered_signal = self.csi_processor.apply_bandpass_filter(
            amplitude,
            low_freq=respiration_band[0],
            high_freq=respiration_band[1]
        )

        # Estimate respiration rate using FFT
        fft_freq = np.fft.rfftfreq(len(filtered_signal), 1.0 / self.config.sampling_rate)
        fft_mag = np.abs(np.fft.rfft(filtered_signal))

        # Find peak in respiration band
        band_mask = (fft_freq >= respiration_band[0]) & (fft_freq <= respiration_band[1])
        peak_idx = np.argmax(fft_mag[band_mask])
        respiration_freq = fft_freq[band_mask][peak_idx]
        respiration_rate_bpm = respiration_freq * 60  # Convert to breaths per minute

        return {
            'respiration_rate_bpm': respiration_rate_bpm,
            'respiration_freq_hz': respiration_freq,
            'filtered_signal': filtered_signal,
            'target_direction': target_direction,
            'target_frequency': target_freq,
            'fft_frequencies': fft_freq,
            'fft_magnitude': fft_mag,
        }

    def get_field_of_view(self) -> Tuple[float, float]:
        """
        Get field of view of the antenna system.

        Returns:
            Tuple of (min_angle, max_angle) in degrees
        """
        freq_range = (
            self.config.center_frequency - self.config.bandwidth / 2,
            self.config.center_frequency + self.config.bandwidth / 2
        )
        return self.antenna.get_field_of_view(freq_range)

    def visualize_ssnr(
        self,
        csi: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict:
        """
        Compute SSNR and frequency-angle mapping for visualization.

        Args:
            csi: CSI array
            timestamps: Packet timestamps

        Returns:
            Dictionary with SSNR values, frequencies, and angles
        """
        if not self._is_calibrated:
            self.calibrate()

        # Compute SSNR
        ssnr = self.direction_estimator.compute_ssnr(csi, timestamps)

        # Get angles for each frequency
        frequencies = self.csi_processor.subcarrier_frequencies
        angles = [self.antenna.get_angle_from_frequency(f) for f in frequencies]

        return {
            'ssnr': ssnr,
            'frequencies': frequencies,
            'angles': angles,
            'subcarrier_indices': np.arange(len(ssnr))
        }
