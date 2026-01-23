"""Tests for CSI processor module."""

import pytest
import numpy as np
from lerobot.wirainbow.csi_processor import CSIProcessor, CSIData


class TestCSIProcessor:
    """Test suite for CSIProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = CSIProcessor(
            sampling_rate=200.0,
            center_frequency=5.57e9,
            bandwidth=80e6,
            num_subcarriers=234
        )

    def test_initialization(self):
        """Test CSI processor initialization."""
        assert self.processor.sampling_rate == 200.0
        assert self.processor.center_frequency == 5.57e9
        assert self.processor.bandwidth == 80e6
        assert self.processor.num_subcarriers == 234

    def test_compute_subcarrier_frequencies(self):
        """Test subcarrier frequency computation."""
        frequencies = self.processor.subcarrier_frequencies

        assert len(frequencies) == self.processor.num_subcarriers

        # Should be centered around center frequency
        mean_freq = np.mean(frequencies)
        assert abs(mean_freq - self.processor.center_frequency) < 1e6  # Within 1 MHz

        # Should span the bandwidth
        freq_span = np.max(frequencies) - np.min(frequencies)
        assert abs(freq_span - self.processor.bandwidth) < 1e6  # Within 1 MHz

    def test_cancel_phase_offset(self):
        """Test phase offset cancellation."""
        num_antennas = 2
        num_subcarriers = 234
        num_packets = 100

        # Create synthetic CSI with phase offset
        csi = np.random.randn(num_antennas, num_subcarriers, num_packets) + \
              1j * np.random.randn(num_antennas, num_subcarriers, num_packets)

        # Apply phase offset to second antenna
        phase_offset = np.pi / 4
        csi[1, :, :] *= np.exp(1j * phase_offset)

        # Cancel phase offset
        corrected_csi = self.processor.cancel_phase_offset(csi, reference_antenna=0)

        # Reference antenna should remain unchanged
        np.testing.assert_array_almost_equal(corrected_csi[0], csi[0])

        # Second antenna should have offset removed (ratio should have consistent phase)
        ratio = corrected_csi[1] / corrected_csi[0]
        ratio_phase = np.angle(ratio)

        # Phase should be more consistent after correction
        assert np.std(ratio_phase) < np.std(np.angle(csi[1] / csi[0]))

    def test_compute_td_csi(self):
        """Test Time-Domain CSI computation."""
        num_antennas = 2
        num_subcarriers = 234
        num_packets = 200

        # Create synthetic CSI
        csi = np.random.randn(num_antennas, num_subcarriers, num_packets) + \
              1j * np.random.randn(num_antennas, num_subcarriers, num_packets)

        timestamps = np.arange(num_packets) / 200.0  # 200 Hz sampling

        # Compute TD-CSI with 50ms interval
        td_csi, valid_ts = self.processor.compute_td_csi(csi, 0.05, timestamps)

        # Should have valid output
        assert len(td_csi.shape) == 3
        assert td_csi.shape[0] == num_antennas
        assert td_csi.shape[1] == num_subcarriers
        assert len(valid_ts) > 0

        # TD-CSI should be complex
        assert td_csi.dtype == complex

    def test_compute_multi_interval_td_csi(self):
        """Test multi-interval TD-CSI computation."""
        num_antennas = 2
        num_subcarriers = 234
        num_packets = 500

        csi = np.random.randn(num_antennas, num_subcarriers, num_packets) + \
              1j * np.random.randn(num_antennas, num_subcarriers, num_packets)

        timestamps = np.arange(num_packets) / 200.0

        # Compute multi-interval TD-CSI
        results = self.processor.compute_multi_interval_td_csi(
            csi, timestamps,
            min_interval=0.01,
            max_interval=0.1,
            num_intervals=10
        )

        # Should return list of results
        assert isinstance(results, list)
        assert len(results) > 0

        # Each result should be a tuple
        for td_csi, valid_ts, interval in results:
            assert td_csi.dtype == complex
            assert len(valid_ts) > 0
            assert 0.01 <= interval <= 0.1

    def test_extract_amplitude(self):
        """Test amplitude extraction."""
        csi = 3.0 + 4.0j  # Magnitude should be 5.0

        amplitude = self.processor.extract_amplitude(csi)

        assert abs(amplitude - 5.0) < 1e-6

    def test_extract_phase(self):
        """Test phase extraction."""
        csi = 1.0 + 1.0j  # Phase should be π/4

        phase = self.processor.extract_phase(csi)

        assert abs(phase - np.pi/4) < 1e-6

    def test_compute_phase_difference(self):
        """Test phase difference computation."""
        # Create TD-CSI with known phase progression
        num_samples = 10
        phases = np.linspace(0, np.pi, num_samples)
        td_csi = np.exp(1j * phases)

        phase_diff = self.processor.compute_phase_difference(td_csi, axis=0)

        # Should have one less sample than input
        assert len(phase_diff) == num_samples - 1

        # Phase differences should be approximately constant
        assert np.std(phase_diff) < 0.1

    def test_sanitize_csi(self):
        """Test CSI sanitization."""
        csi = np.array([1.0+1.0j, np.nan, np.inf, 1e-15+1e-15j, 2.0+2.0j])

        sanitized = self.processor.sanitize_csi(csi)

        # NaN and Inf should be replaced with zero
        assert sanitized[1] == 0.0
        assert sanitized[2] == 0.0

        # Very small values should be replaced with zero
        assert sanitized[3] == 0.0

        # Valid values should remain
        assert sanitized[0] == 1.0+1.0j
        assert sanitized[4] == 2.0+2.0j

    def test_csi_data_container(self):
        """Test CSIData container."""
        csi = np.random.randn(2, 234, 100) + 1j * np.random.randn(2, 234, 100)
        frequencies = np.linspace(5.49e9, 5.65e9, 234)
        timestamps = np.arange(100) / 200.0
        subcarrier_indices = np.arange(234)

        csi_data = CSIData(
            csi=csi,
            frequencies=frequencies,
            timestamps=timestamps,
            subcarrier_indices=subcarrier_indices
        )

        assert csi_data.csi.shape == (2, 234, 100)
        assert len(csi_data.frequencies) == 234
        assert len(csi_data.timestamps) == 100
        assert len(csi_data.subcarrier_indices) == 234
