"""
Respiration monitoring example for WiRainbow.

Demonstrates how to use WiRainbow for contactless respiration rate monitoring.
"""

import numpy as np
import matplotlib.pyplot as plt
from lerobot.wirainbow import WiRainbow, WiRainbowConfig


def simulate_respiration_csi(
    num_antennas: int = 2,
    num_subcarriers: int = 234,
    duration: float = 30.0,  # seconds
    sampling_rate: float = 200.0,  # Hz
    target_angle: float = 0.0,  # degrees
    respiration_rate_bpm: float = 18.0,  # breaths per minute
    respiration_amplitude: float = 0.005,  # meters (5mm chest movement)
    noise_level: float = 0.15
) -> tuple:
    """
    Simulate CSI data with respiration signal.

    Args:
        num_antennas: Number of receiving antennas
        num_subcarriers: Number of OFDM subcarriers
        duration: Duration of recording in seconds
        sampling_rate: Packet sampling rate in Hz
        target_angle: Target direction in degrees
        respiration_rate_bpm: Respiration rate in breaths per minute
        respiration_amplitude: Chest wall displacement amplitude
        noise_level: Noise standard deviation

    Returns:
        Tuple of (csi, timestamps)
    """
    num_packets = int(duration * sampling_rate)
    timestamps = np.arange(num_packets) / sampling_rate

    # Initialize CSI
    csi = np.zeros((num_antennas, num_subcarriers, num_packets), dtype=complex)

    # Subcarrier frequencies
    center_freq = 5.57e9
    bandwidth = 80e6
    subcarrier_spacing = bandwidth / num_subcarriers
    subcarrier_freqs = center_freq + (np.arange(num_subcarriers) - num_subcarriers // 2) * subcarrier_spacing

    # Speed of light
    c = 3e8

    # Convert respiration rate to frequency
    respiration_freq = respiration_rate_bpm / 60.0  # Hz

    # Simulate respiration (sinusoidal chest movement)
    respiration_motion = respiration_amplitude * np.sin(2 * np.pi * respiration_freq * timestamps)

    # Target distance
    target_distance = 1.5  # meters

    for i, freq in enumerate(subcarrier_freqs):
        wavelength = c / freq

        # Static multipath components
        static_phase = 2 * np.pi * np.random.rand()
        static_amplitude = 1.0 + 0.3 * np.random.randn()

        # Angle-dependent sensitivity
        beam_angle = -60 + (i / num_subcarriers) * 120
        angle_sensitivity = np.exp(-((beam_angle - target_angle) ** 2) / (2 * 15**2))

        for j, t in enumerate(timestamps):
            # Respiration-induced distance change
            distance_change = respiration_motion[j]
            total_distance = target_distance + distance_change

            # Phase from two-way path
            phase = 2 * np.pi * 2 * total_distance / wavelength

            # Dynamic amplitude
            dynamic_amplitude = 0.4 * angle_sensitivity

            for ant in range(num_antennas):
                ant_phase = ant * np.pi / 3

                h = (static_amplitude * np.exp(1j * static_phase) +
                     dynamic_amplitude * np.exp(1j * (phase + ant_phase)))

                # Add noise and random body movements
                noise = noise_level * (np.random.randn() + 1j * np.random.randn())
                random_motion = 0.05 * np.sin(2 * np.pi * 0.1 * t) * np.exp(1j * np.random.rand())

                csi[ant, i, j] = h + noise + random_motion

    return csi, timestamps


def main():
    """Run respiration monitoring example."""
    print("=" * 70)
    print("WiRainbow Respiration Monitoring Example")
    print("=" * 70)

    # Create configuration
    config = WiRainbowConfig(sampling_rate=200.0)

    # Initialize WiRainbow
    print("\n1. Initializing WiRainbow system...")
    wirainbow = WiRainbow(config)
    wirainbow.calibrate()

    # Simulate respiration data
    print("\n2. Simulating respiration signal...")
    true_respiration_rate = 18.0  # bpm (normal adult respiration rate)
    target_angle = 0.0  # degrees (directly in front)

    print(f"   True respiration rate: {true_respiration_rate:.1f} bpm")
    print(f"   Target angle: {target_angle:.1f}°")

    csi, timestamps = simulate_respiration_csi(
        duration=30.0,
        sampling_rate=200.0,
        target_angle=target_angle,
        respiration_rate_bpm=true_respiration_rate,
        respiration_amplitude=0.005,
        noise_level=0.15
    )

    # Monitor respiration
    print("\n3. Monitoring respiration...")
    respiration_data = wirainbow.monitor_respiration(
        csi, timestamps,
        target_direction=None,  # Auto-detect
        respiration_band=(0.2, 0.5)  # 12-30 bpm
    )

    estimated_rate = respiration_data['respiration_rate_bpm']
    detected_angle = respiration_data['target_direction']

    print(f"   Detected direction: {detected_angle:.1f}°")
    print(f"   Estimated respiration rate: {estimated_rate:.1f} bpm")
    print(f"   Estimation error: {abs(estimated_rate - true_respiration_rate):.2f} bpm")

    # Visualize results
    print("\n4. Generating visualization...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Filtered respiration signal
    time_plot = timestamps[:len(respiration_data['filtered_signal'])]
    axes[0].plot(time_plot, respiration_data['filtered_signal'], linewidth=1)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Filtered Respiration Signal')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Frequency spectrum
    fft_freqs = respiration_data['fft_frequencies']
    fft_mag = respiration_data['fft_magnitude']
    resp_freq_hz = respiration_data['respiration_freq_hz']

    axes[1].plot(fft_freqs * 60, fft_mag, linewidth=1.5)  # Convert to bpm
    axes[1].axvline(estimated_rate, color='red', linestyle='--',
                    label=f'Estimated: {estimated_rate:.1f} bpm')
    axes[1].axvline(true_respiration_rate, color='green', linestyle='--',
                    label=f'True: {true_respiration_rate:.1f} bpm')
    axes[1].set_xlabel('Respiration Rate (bpm)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Respiration Frequency Spectrum')
    axes[1].set_xlim([0, 40])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: SSNR for direction detection
    viz_data = wirainbow.visualize_ssnr(csi, timestamps)
    axes[2].plot(viz_data['angles'], viz_data['ssnr'], linewidth=2)
    axes[2].axvline(detected_angle, color='red', linestyle='--',
                    label=f'Detected: {detected_angle:.1f}°')
    axes[2].axvline(target_angle, color='green', linestyle='--',
                    label=f'True: {target_angle:.1f}°')
    axes[2].set_xlabel('Angle (degrees)')
    axes[2].set_ylabel('SSNR')
    axes[2].set_title('Direction Detection via SSNR')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('/tmp/wirainbow_respiration_example.png', dpi=150)
    print(f"   Saved visualization to /tmp/wirainbow_respiration_example.png")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
