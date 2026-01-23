"""Tests for FrequencyScanningAntenna module."""

import pytest
import numpy as np
from lerobot.wirainbow.antenna import FrequencyScanningAntenna


class TestFrequencyScanningAntenna:
    """Test suite for FrequencyScanningAntenna class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.antenna = FrequencyScanningAntenna()

    def test_initialization(self):
        """Test antenna initialization with default parameters."""
        assert self.antenna.num_resonators == 12
        assert self.antenna.quality_factor == 200.0
        assert self.antenna.resonant_freq == 5.57e9
        assert self.antenna.speed_of_light == 3e8

    def test_custom_initialization(self):
        """Test antenna initialization with custom parameters."""
        antenna = FrequencyScanningAntenna(
            num_resonators=16,
            quality_factor=150.0,
            resonant_freq=5.5e9
        )
        assert antenna.num_resonators == 16
        assert antenna.quality_factor == 150.0
        assert antenna.resonant_freq == 5.5e9

    def test_compute_resonator_phase_delay(self):
        """Test resonator phase delay computation."""
        # At resonant frequency, phase delay should be 0
        phase = self.antenna.compute_resonator_phase_delay(self.antenna.resonant_freq)
        assert abs(phase) < 1e-6

        # Below resonant frequency, phase should be positive
        phase_low = self.antenna.compute_resonator_phase_delay(5.4e9)
        assert phase_low > 0

        # Above resonant frequency, phase should be negative
        phase_high = self.antenna.compute_resonator_phase_delay(5.7e9)
        assert phase_high < 0

    def test_compute_beam_direction(self):
        """Test beam direction computation."""
        # Compute beam direction for a range of frequencies
        freq_low = 5.49e9
        freq_high = 5.65e9

        angle_low = self.antenna.compute_beam_direction(freq_low)
        angle_high = self.antenna.compute_beam_direction(freq_high)

        # Angles should be different for different frequencies
        assert abs(angle_low - angle_high) > 1.0  # At least 1 degree difference

        # Angles should be within reasonable range
        assert -90 <= angle_low <= 90
        assert -90 <= angle_high <= 90

    def test_compute_beam_pattern(self):
        """Test beam pattern computation."""
        frequency = 5.57e9
        angle = 0.0

        beam_pattern = self.antenna.compute_beam_pattern(frequency, angle)

        # Beam pattern should be complex
        assert isinstance(beam_pattern, complex)

        # Magnitude should be reasonable (normalized by num_resonators)
        magnitude = abs(beam_pattern)
        assert 0 <= magnitude <= 1.5

    def test_beam_pattern_directivity(self):
        """Test that beam pattern has directional properties."""
        frequency = 5.57e9

        # Get beam pattern at different angles
        angles = np.linspace(-60, 60, 121)
        magnitudes = []

        for angle in angles:
            beam_pattern = self.antenna.compute_beam_pattern(frequency, angle)
            magnitudes.append(abs(beam_pattern) ** 2)

        magnitudes = np.array(magnitudes)

        # There should be variation in beam pattern (directivity)
        assert np.std(magnitudes) > 0.01

    def test_calibrate_frequency_direction_mapping(self):
        """Test frequency-direction calibration."""
        frequencies = np.linspace(5.49e9, 5.65e9, 50)
        angles = np.linspace(-60, 60, 121)

        calibration_map = self.antenna.calibrate_frequency_direction_mapping(
            frequencies, angles
        )

        # Should return a dictionary
        assert isinstance(calibration_map, dict)

        # Should have entries for all frequencies
        assert len(calibration_map) == len(frequencies)

        # Each entry should have angle and gain
        for freq, data in calibration_map.items():
            assert 'angle' in data
            assert 'gain' in data
            assert -90 <= data['angle'] <= 90
            assert data['gain'] >= 0

    def test_get_angle_from_frequency(self):
        """Test getting angle from frequency."""
        # Without calibration, should use theoretical model
        frequency = 5.57e9
        angle = self.antenna.get_angle_from_frequency(frequency)
        assert -90 <= angle <= 90

        # With calibration
        frequencies = np.linspace(5.49e9, 5.65e9, 50)
        angles = np.linspace(-60, 60, 121)
        self.antenna.calibrate_frequency_direction_mapping(frequencies, angles)

        angle_calibrated = self.antenna.get_angle_from_frequency(frequency)
        assert -90 <= angle_calibrated <= 90

    def test_get_field_of_view(self):
        """Test field of view computation."""
        freq_range = (5.49e9, 5.65e9)
        fov_min, fov_max = self.antenna.get_field_of_view(freq_range)

        # FoV should be symmetric or asymmetric based on design
        assert -90 <= fov_min <= 90
        assert -90 <= fov_max <= 90
        assert fov_min < fov_max

        # FoV should be non-trivial (at least a few degrees)
        fov_span = fov_max - fov_min
        assert fov_span > 5.0

    def test_frequency_scanning_property(self):
        """Test that different frequencies map to different angles."""
        frequencies = np.linspace(5.49e9, 5.65e9, 10)
        angles = [self.antenna.compute_beam_direction(f) for f in frequencies]

        # Angles should vary with frequency
        angle_range = max(angles) - min(angles)
        assert angle_range > 10.0  # At least 10 degrees range
