"""
Frequency-Scanning Antenna (FSA) model for WiRainbow.

Implements the coupled-resonator antenna design that creates frequency-dependent
radiation patterns for directional Wi-Fi sensing.
"""

import numpy as np
from typing import Optional, Tuple


class FrequencyScanningAntenna:
    """
    Frequency-Scanning Antenna with coupled resonators.

    This class models the antenna design from the WiRainbow paper, which uses
    alternating electric and magnetic coupling between resonators to create
    frequency-dependent beam patterns.

    Attributes:
        num_resonators: Number of resonator elements (default: 12)
        element_spacing: Distance between resonator elements in meters
        quality_factor: Q factor of resonators (default: 200)
        resonant_freq: Central resonant frequency in Hz (default: 5.57 GHz)
        substrate_dielectric: Dielectric constant of substrate (default: 2.2)
    """

    def __init__(
        self,
        num_resonators: int = 12,
        element_spacing: float = 0.0145,  # ~17.4cm / 12 elements
        quality_factor: float = 200.0,
        resonant_freq: float = 5.57e9,  # 5.57 GHz (center of 5.49-5.65 GHz)
        substrate_dielectric: float = 2.2,
    ):
        """
        Initialize the Frequency-Scanning Antenna model.

        Args:
            num_resonators: Number of resonator elements
            element_spacing: Distance between adjacent resonators (meters)
            quality_factor: Quality factor Q of resonators
            resonant_freq: Central resonant frequency (Hz)
            substrate_dielectric: Dielectric constant of substrate
        """
        self.num_resonators = num_resonators
        self.element_spacing = element_spacing
        self.quality_factor = quality_factor
        self.resonant_freq = resonant_freq
        self.substrate_dielectric = substrate_dielectric

        # Physical constants
        self.speed_of_light = 3e8  # m/s

        # Frequency-to-direction mapping (calibrated)
        # This would typically be measured in an anechoic chamber
        self._freq_to_angle_map: Optional[dict] = None

    def compute_resonator_phase_delay(self, frequency: float) -> float:
        """
        Compute phase delay for a single resonator at given frequency.

        Based on Equation 3 from the paper:
        Δφ_rf = arctan{Q(f₀/f - f/f₀)}

        Args:
            frequency: Operating frequency in Hz

        Returns:
            Phase delay in radians
        """
        f = frequency
        f0 = self.resonant_freq
        Q = self.quality_factor

        # Equation 3: Δφ_rf = arctan{Q(f₀/f - f/f₀)}
        delta_phi = np.arctan(Q * (f0 / f - f / f0))

        return delta_phi

    def compute_beam_direction(self, frequency: float) -> float:
        """
        Compute beam direction angle for given frequency.

        Based on Equation 2 from the paper:
        θ_f = arcsin(-λ_f·Δφ_f / 2πl)

        Args:
            frequency: Operating frequency in Hz

        Returns:
            Beam direction angle in degrees
        """
        # Compute wavelength
        wavelength = self.speed_of_light / frequency

        # Compute phase delay
        delta_phi = self.compute_resonator_phase_delay(frequency)

        # Equation 2: θ_f = arcsin(-λ_f·Δφ_f / 2πl)
        l = self.element_spacing
        sin_theta = -(wavelength * delta_phi) / (2 * np.pi * l)

        # Clip to valid range for arcsin
        sin_theta = np.clip(sin_theta, -1.0, 1.0)

        # Compute angle in radians then convert to degrees
        theta_rad = np.arcsin(sin_theta)
        theta_deg = np.degrees(theta_rad)

        return theta_deg

    def compute_beam_pattern(
        self,
        frequency: float,
        angle: float
    ) -> complex:
        """
        Compute frequency-dependent beam pattern D(f, θ).

        This represents the antenna gain in a specific direction at a given frequency.
        The model uses a simplified array factor approach.

        Args:
            frequency: Operating frequency in Hz
            angle: Direction angle in degrees

        Returns:
            Complex beam pattern value D(f, θ)
        """
        # Convert angle to radians
        theta_rad = np.radians(angle)

        # Compute wavelength
        wavelength = self.speed_of_light / frequency

        # Compute wave number
        k = 2 * np.pi / wavelength

        # Compute progressive phase shift for array factor
        # This includes both spatial and frequency-dependent phase
        beta = k * self.element_spacing * np.sin(theta_rad)

        # Add resonator-induced phase delay
        delta_phi = self.compute_resonator_phase_delay(frequency)

        # Array factor for linear array with progressive phase
        array_factor = 0.0 + 0.0j
        for n in range(self.num_resonators):
            # Alternating coupling creates symmetric beam distribution
            coupling_factor = 1.0 if n % 2 == 0 else -1.0
            phase = n * (beta + delta_phi) * coupling_factor
            array_factor += np.exp(1j * phase)

        # Normalize by number of elements
        beam_pattern = array_factor / self.num_resonators

        return beam_pattern

    def calibrate_frequency_direction_mapping(
        self,
        frequencies: np.ndarray,
        angles: np.ndarray
    ) -> dict:
        """
        Create frequency-to-direction calibration mapping.

        In practice, this would be measured in an anechoic chamber. Here we
        compute it based on the theoretical model.

        Args:
            frequencies: Array of frequencies to calibrate (Hz)
            angles: Array of angles to measure (degrees)

        Returns:
            Dictionary mapping frequencies to optimal beam angles
        """
        calibration_map = {}

        for freq in frequencies:
            # Find angle with maximum gain
            max_gain = -np.inf
            best_angle = 0.0

            for angle in angles:
                beam_pattern = self.compute_beam_pattern(freq, angle)
                gain = np.abs(beam_pattern) ** 2

                if gain > max_gain:
                    max_gain = gain
                    best_angle = angle

            calibration_map[freq] = {
                'angle': best_angle,
                'gain': max_gain
            }

        self._freq_to_angle_map = calibration_map
        return calibration_map

    def get_angle_from_frequency(self, frequency: float) -> float:
        """
        Get beam angle for given frequency using calibration map.

        Args:
            frequency: Operating frequency in Hz

        Returns:
            Beam direction angle in degrees
        """
        if self._freq_to_angle_map is None:
            # Fall back to theoretical model
            return self.compute_beam_direction(frequency)

        # Find closest calibrated frequency
        calibrated_freqs = np.array(list(self._freq_to_angle_map.keys()))
        closest_idx = np.argmin(np.abs(calibrated_freqs - frequency))
        closest_freq = calibrated_freqs[closest_idx]

        return self._freq_to_angle_map[closest_freq]['angle']

    def get_field_of_view(
        self,
        frequency_range: Tuple[float, float],
        num_points: int = 100
    ) -> Tuple[float, float]:
        """
        Compute field of view (FoV) for given frequency range.

        Args:
            frequency_range: Tuple of (min_freq, max_freq) in Hz
            num_points: Number of frequency points to evaluate

        Returns:
            Tuple of (min_angle, max_angle) in degrees
        """
        frequencies = np.linspace(frequency_range[0], frequency_range[1], num_points)
        angles = [self.compute_beam_direction(f) for f in frequencies]

        return (min(angles), max(angles))
