"""
Basic usage example for WiRainbow system.

This example demonstrates:
1. Creating a WiRainbow instance
2. Simulating CSI data
3. Estimating target direction
4. Visualizing results
"""

import numpy as np
import matplotlib.pyplot as plt
from lerobot.wirainbow import WiRainbow, WiRainbowConfig


def simulate_csi_with_target(
    num_antennas: int = 2,
    num_subcarriers: int = 234,
    num_packets: int = 1000,
    target_angle: float = 30.0,  # degrees
    target_distance: float = 2.0,  # meters
    motion_amplitude: float = 0.01,  # meters
    motion_frequency: float = 1.0,  # Hz
    sampling_rate: float = 200.0,  # Hz
    noise_level: float = 0.1
) -> tuple:
    """
    Simulate CSI data with a moving target.

    Args:
        num_antennas: Number of receiving antennas
        num_subcarriers: Number of OFDM subcarriers
        num_packets: Number of packets to simulate
        target_angle: Target direction in degrees
        target_distance: Target distance in meters
        motion_amplitude: Amplitude of target motion
        motion_frequency: Frequency of target motion
        sampling_rate: Packet sampling rate in Hz
        noise_level: Noise standard deviation

    Returns:
        Tuple of (csi, timestamps)
    """
    # Create timestamps
    timestamps = np.arange(num_packets) / sampling_rate

    # Initialize CSI array
    csi = np.zeros((num_antennas, num_subcarriers, num_packets), dtype=complex)

    # Simulate subcarrier frequencies (5.49-5.65 GHz for 80 MHz bandwidth)
    center_freq = 5.57e9  # Hz
    bandwidth = 80e6  # Hz
    subcarrier_spacing = bandwidth / num_subcarriers
    subcarrier_freqs = center_freq + (np.arange(num_subcarriers) - num_subcarriers // 2) * subcarrier_spacing

    # Speed of light
    c = 3e8  # m/s

    # Simulate target motion (simple sinusoidal motion)
    target_motion = motion_amplitude * np.sin(2 * np.pi * motion_frequency * timestamps)

    # Convert angle to radians
    theta_rad = np.radians(target_angle)

    for i, freq in enumerate(subcarrier_freqs):
        # Wavelength
        wavelength = c / freq

        # Static component (path from transmitter to receiver)
        static_phase = 2 * np.pi * np.random.rand()
        static_amplitude = 1.0 + 0.2 * np.random.randn()

        # Dynamic component (reflection from moving target)
        # Phase depends on target distance and motion
        for j, t in enumerate(timestamps):
            distance_variation = target_motion[j]
            dynamic_distance = target_distance + distance_variation

            # Two-way path (transmitter -> target -> receiver)
            phase = 2 * np.pi * 2 * dynamic_distance / wavelength

            # Angle-dependent amplitude (simulates beam pattern)
            # Subcarriers at different frequencies have different beam angles
            # This creates frequency-dependent directional sensitivity
            beam_angle = -60 + (i / num_subcarriers) * 120  # Map subcarrier to angle
            angle_sensitivity = np.exp(-((beam_angle - target_angle) ** 2) / (2 * 20**2))

            dynamic_amplitude = 0.3 * angle_sensitivity

            # Combine static and dynamic components
            for ant in range(num_antennas):
                # Add antenna-specific phase offset
                ant_phase = ant * np.pi / 4

                h = (static_amplitude * np.exp(1j * static_phase) +
                     dynamic_amplitude * np.exp(1j * (phase + ant_phase)))

                # Add noise
                noise = noise_level * (np.random.randn() + 1j * np.random.randn())
                csi[ant, i, j] = h + noise

    return csi, timestamps


def main():
    """Run basic WiRainbow example."""
    print("=" * 70)
    print("WiRainbow Basic Usage Example")
    print("=" * 70)

    # Create configuration
    config = WiRainbowConfig(
        sampling_rate=200.0,
        num_intervals=20,
    )

    # Initialize WiRainbow system
    print("\n1. Initializing WiRainbow system...")
    wirainbow = WiRainbow(config)

    # Calibrate antenna
    print("2. Calibrating frequency-to-direction mapping...")
    wirainbow.calibrate()

    # Get field of view
    fov_min, fov_max = wirainbow.get_field_of_view()
    print(f"   Field of View: {fov_min:.1f}° to {fov_max:.1f}°")

    # Simulate CSI data with target at 30 degrees
    print("\n3. Simulating CSI data...")
    target_angle = 30.0
    print(f"   True target angle: {target_angle:.1f}°")

    csi, timestamps = simulate_csi_with_target(
        num_antennas=2,
        num_subcarriers=234,
        num_packets=1000,
        target_angle=target_angle,
        motion_frequency=1.0,  # 1 Hz motion
        sampling_rate=200.0,
        noise_level=0.1
    )

    # Estimate target direction
    print("\n4. Estimating target direction...")
    estimate = wirainbow.estimate_direction(csi, timestamps)

    print(f"   Estimated angle: {estimate.angle:.1f}°")
    print(f"   Estimation error: {abs(estimate.angle - target_angle):.1f}°")
    print(f"   SSNR value: {estimate.ssnr:.3f}")
    print(f"   Confidence: {estimate.confidence:.3f}")
    print(f"   Best frequency: {estimate.frequency / 1e9:.3f} GHz")

    # Visualize SSNR
    print("\n5. Generating visualization...")
    viz_data = wirainbow.visualize_ssnr(csi, timestamps)

    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: SSNR vs Subcarrier Index
    axes[0].plot(viz_data['subcarrier_indices'], viz_data['ssnr'], linewidth=2)
    axes[0].axvline(estimate.subcarrier_index, color='red', linestyle='--',
                    label=f'Detected peak (subcarrier {estimate.subcarrier_index})')
    axes[0].set_xlabel('Subcarrier Index')
    axes[0].set_ylabel('SSNR (normalized)')
    axes[0].set_title('Sensing Signal-to-Noise Ratio vs Subcarrier')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: SSNR vs Angle
    axes[1].plot(viz_data['angles'], viz_data['ssnr'], linewidth=2)
    axes[1].axvline(estimate.angle, color='red', linestyle='--',
                    label=f'Estimated angle: {estimate.angle:.1f}°')
    axes[1].axvline(target_angle, color='green', linestyle='--',
                    label=f'True angle: {target_angle:.1f}°')
    axes[1].set_xlabel('Angle (degrees)')
    axes[1].set_ylabel('SSNR (normalized)')
    axes[1].set_title('Sensing Signal-to-Noise Ratio vs Direction')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('/tmp/wirainbow_basic_example.png', dpi=150)
    print(f"   Saved visualization to /tmp/wirainbow_basic_example.png")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
