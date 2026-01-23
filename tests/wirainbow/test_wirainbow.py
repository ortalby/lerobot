"""Tests for main WiRainbow system."""

import pytest
import numpy as np
from lerobot.wirainbow import WiRainbow, WiRainbowConfig
from lerobot.wirainbow.direction_estimator import DirectionEstimate


class TestWiRainbowConfig:
    """Test suite for WiRainbowConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = WiRainbowConfig()

        assert config.num_resonators == 12
        assert config.quality_factor == 200.0
        assert config.sampling_rate == 200.0
        assert config.num_subcarriers == 234

    def test_custom_config(self):
        """Test custom configuration."""
        config = WiRainbowConfig(
            num_resonators=16,
            sampling_rate=100.0,
            num_intervals=10
        )

        assert config.num_resonators == 16
        assert config.sampling_rate == 100.0
        assert config.num_intervals == 10

    def test_config_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = WiRainbowConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert 'num_resonators' in config_dict
        assert 'sampling_rate' in config_dict
        assert config_dict['num_resonators'] == 12

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            'num_resonators': 16,
            'sampling_rate': 100.0,
            'quality_factor': 150.0,
            'resonant_freq': 5.5e9,
            'center_frequency': 5.5e9,
            'bandwidth': 80e6,
            'num_subcarriers': 234,
            'min_interval': 0.005,
            'max_interval': 0.1,
            'num_intervals': 10,
            'reference_antenna': 0,
            'element_spacing': 0.0145,
            'calibration_angles': list(np.linspace(-60, 60, 121)),
            'calibration_frequencies': list(np.linspace(5.37e9, 5.73e9, 360))
        }

        config = WiRainbowConfig.from_dict(config_dict)

        assert config.num_resonators == 16
        assert config.sampling_rate == 100.0
        assert config.quality_factor == 150.0


class TestWiRainbow:
    """Test suite for WiRainbow class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = WiRainbowConfig()
        self.wirainbow = WiRainbow(self.config)

    def test_initialization(self):
        """Test WiRainbow initialization."""
        assert self.wirainbow.antenna is not None
        assert self.wirainbow.csi_processor is not None
        assert self.wirainbow.direction_estimator is not None
        assert self.wirainbow._is_calibrated is False

    def test_calibration(self):
        """Test antenna calibration."""
        calibration_map = self.wirainbow.calibrate()

        assert isinstance(calibration_map, dict)
        assert len(calibration_map) > 0
        assert self.wirainbow._is_calibrated is True

    def test_preprocess_csi(self):
        """Test CSI preprocessing."""
        num_antennas = 2
        num_subcarriers = 234
        num_packets = 100

        csi = np.random.randn(num_antennas, num_subcarriers, num_packets) + \
              1j * np.random.randn(num_antennas, num_subcarriers, num_packets)

        preprocessed = self.wirainbow.preprocess_csi(csi)

        assert preprocessed.shape == csi.shape
        assert preprocessed.dtype == complex

    def test_estimate_direction(self):
        """Test direction estimation."""
        # Create synthetic CSI with simulated target
        num_antennas = 2
        num_subcarriers = 234
        num_packets = 500

        csi, timestamps = self._create_synthetic_csi(
            num_antennas, num_subcarriers, num_packets,
            target_angle=20.0
        )

        estimate = self.wirainbow.estimate_direction(csi, timestamps)

        assert isinstance(estimate, DirectionEstimate)
        assert -90 <= estimate.angle <= 90
        assert 0 <= estimate.ssnr <= 1.1
        assert 0 <= estimate.confidence <= 1.0
        assert estimate.frequency > 0

    def test_estimate_multiple_targets(self):
        """Test multiple target estimation."""
        num_antennas = 2
        num_subcarriers = 234
        num_packets = 500

        csi, timestamps = self._create_synthetic_csi(
            num_antennas, num_subcarriers, num_packets,
            target_angle=0.0
        )

        estimates = self.wirainbow.estimate_multiple_targets(
            csi, timestamps, num_targets=2
        )

        assert isinstance(estimates, list)
        assert len(estimates) > 0
        assert all(isinstance(e, DirectionEstimate) for e in estimates)

    def test_get_field_of_view(self):
        """Test field of view retrieval."""
        fov_min, fov_max = self.wirainbow.get_field_of_view()

        assert -90 <= fov_min <= 90
        assert -90 <= fov_max <= 90
        assert fov_min < fov_max

    def test_visualize_ssnr(self):
        """Test SSNR visualization data."""
        num_antennas = 2
        num_subcarriers = 234
        num_packets = 500

        csi, timestamps = self._create_synthetic_csi(
            num_antennas, num_subcarriers, num_packets,
            target_angle=0.0
        )

        viz_data = self.wirainbow.visualize_ssnr(csi, timestamps)

        assert 'ssnr' in viz_data
        assert 'frequencies' in viz_data
        assert 'angles' in viz_data
        assert 'subcarrier_indices' in viz_data

        assert len(viz_data['ssnr']) == num_subcarriers
        assert len(viz_data['frequencies']) == num_subcarriers
        assert len(viz_data['angles']) == num_subcarriers

    def test_monitor_respiration(self):
        """Test respiration monitoring."""
        num_antennas = 2
        num_subcarriers = 234
        num_packets = 2000  # Longer duration for respiration

        csi, timestamps = self._create_synthetic_csi_with_respiration(
            num_antennas, num_subcarriers, num_packets,
            respiration_rate_bpm=18.0
        )

        respiration_data = self.wirainbow.monitor_respiration(
            csi, timestamps,
            target_direction=None  # Auto-detect
        )

        assert 'respiration_rate_bpm' in respiration_data
        assert 'respiration_freq_hz' in respiration_data
        assert 'filtered_signal' in respiration_data
        assert 'target_direction' in respiration_data

        # Respiration rate should be positive and reasonable
        assert 0 < respiration_data['respiration_rate_bpm'] < 60

    def test_auto_calibration_on_first_use(self):
        """Test that calibration happens automatically."""
        wirainbow = WiRainbow()
        assert wirainbow._is_calibrated is False

        # Create dummy CSI
        csi = np.random.randn(2, 234, 100) + 1j * np.random.randn(2, 234, 100)
        timestamps = np.arange(100) / 200.0

        # First call should trigger calibration
        estimate = wirainbow.estimate_direction(csi, timestamps)

        assert wirainbow._is_calibrated is True

    @staticmethod
    def _create_synthetic_csi(
        num_antennas, num_subcarriers, num_packets, target_angle=0.0
    ):
        """Helper method to create synthetic CSI data."""
        timestamps = np.arange(num_packets) / 200.0

        # Create CSI with motion signature
        csi = np.zeros((num_antennas, num_subcarriers, num_packets), dtype=complex)

        for i in range(num_subcarriers):
            # Frequency-dependent beam angle
            beam_angle = -60 + (i / num_subcarriers) * 120

            # Angular sensitivity
            sensitivity = np.exp(-((beam_angle - target_angle) ** 2) / (2 * 20**2))

            # Add motion (1 Hz)
            motion_phase = 2 * np.pi * 1.0 * timestamps
            motion_signal = sensitivity * 0.3 * np.sin(motion_phase)

            for ant in range(num_antennas):
                # Static + dynamic
                static = 1.0 * np.exp(1j * np.random.rand())
                dynamic = motion_signal * np.exp(1j * (motion_phase + ant * np.pi / 4))

                csi[ant, i, :] = static + dynamic

                # Add noise
                noise = 0.1 * (np.random.randn(num_packets) + 1j * np.random.randn(num_packets))
                csi[ant, i, :] += noise

        return csi, timestamps

    @staticmethod
    def _create_synthetic_csi_with_respiration(
        num_antennas, num_subcarriers, num_packets, respiration_rate_bpm=18.0
    ):
        """Helper method to create synthetic CSI with respiration signal."""
        timestamps = np.arange(num_packets) / 200.0
        respiration_freq = respiration_rate_bpm / 60.0

        csi = np.zeros((num_antennas, num_subcarriers, num_packets), dtype=complex)

        # Add respiration signal to middle subcarriers
        respiration_phase = 2 * np.pi * respiration_freq * timestamps
        respiration_signal = 0.2 * np.sin(respiration_phase)

        for i in range(num_subcarriers):
            for ant in range(num_antennas):
                static = 1.0 * np.exp(1j * np.random.rand())
                dynamic = respiration_signal * np.exp(1j * respiration_phase)

                csi[ant, i, :] = static + dynamic

                # Add noise
                noise = 0.15 * (np.random.randn(num_packets) + 1j * np.random.randn(num_packets))
                csi[ant, i, :] += noise

        return csi, timestamps
