"""
Multi-target detection example for WiRainbow.

Demonstrates simultaneous direction estimation for multiple targets.
"""

import numpy as np
import matplotlib.pyplot as plt
from lerobot.wirainbow import WiRainbow, WiRainbowConfig


def simulate_multi_target_csi(
    num_antennas: int = 2,
    num_subcarriers: int = 234,
    num_packets: int = 1000,
    target_angles: list = None,
    target_distances: list = None,
    motion_frequencies: list = None,
    sampling_rate: float = 200.0,
    noise_level: float = 0.1
) -> tuple:
    """
    Simulate CSI data with multiple moving targets.

    Args:
        num_antennas: Number of receiving antennas
        num_subcarriers: Number of OFDM subcarriers
        num_packets: Number of packets to simulate
        target_angles: List of target angles in degrees
        target_distances: List of target distances in meters
        motion_frequencies: List of motion frequencies in Hz
        sampling_rate: Packet sampling rate in Hz
        noise_level: Noise standard deviation

    Returns:
        Tuple of (csi, timestamps)
    """
    if target_angles is None:
        target_angles = [-30.0, 20.0]
    if target_distances is None:
        target_distances = [2.0, 2.5]
    if motion_frequencies is None:
        motion_frequencies = [1.0, 1.5]

    num_targets = len(target_angles)
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

    # Simulate motion for each target
    target_motions = []
    for freq in motion_frequencies:
        motion = 0.01 * np.sin(2 * np.pi * freq * timestamps)
        target_motions.append(motion)

    for i, freq in enumerate(subcarrier_freqs):
        wavelength = c / freq

        # Static component
        static_phase = 2 * np.pi * np.random.rand()
        static_amplitude = 1.0 + 0.2 * np.random.randn()

        for j, t in enumerate(timestamps):
            # Contributions from all targets
            dynamic_signal = 0.0 + 0.0j

            for target_idx in range(num_targets):
                angle = target_angles[target_idx]
                distance = target_distances[target_idx]
                motion = target_motions[target_idx][j]

                # Total distance with motion
                total_distance = distance + motion

                # Phase from two-way path
                phase = 2 * np.pi * 2 * total_distance / wavelength

                # Angle-dependent sensitivity
                beam_angle = -60 + (i / num_subcarriers) * 120
                angle_sensitivity = np.exp(-((beam_angle - angle) ** 2) / (2 * 20**2))

                dynamic_amplitude = 0.3 * angle_sensitivity

                # Add contribution from this target
                dynamic_signal += dynamic_amplitude * np.exp(1j * phase)

            # Combine static and dynamic for all antennas
            for ant in range(num_antennas):
                ant_phase = ant * np.pi / 4

                h = (static_amplitude * np.exp(1j * static_phase) +
                     dynamic_signal * np.exp(1j * ant_phase))

                # Add noise
                noise = noise_level * (np.random.randn() + 1j * np.random.randn())
                csi[ant, i, j] = h + noise

    return csi, timestamps


def main():
    """Run multi-target detection example."""
    print("=" * 70)
    print("WiRainbow Multi-Target Detection Example")
    print("=" * 70)

    # Create configuration
    config = WiRainbowConfig(sampling_rate=200.0)

    # Initialize WiRainbow
    print("\n1. Initializing WiRainbow system...")
    wirainbow = WiRainbow(config)
    wirainbow.calibrate()

    # Simulate multi-target scenario
    print("\n2. Simulating multi-target CSI data...")
    target_angles = [-30.0, 20.0]
    target_distances = [2.0, 2.5]
    motion_frequencies = [1.0, 1.5]
    num_targets = len(target_angles)

    print(f"   Number of targets: {num_targets}")
    for i, angle in enumerate(target_angles):
        print(f"   Target {i+1}: {angle:.1f}° at {target_distances[i]:.1f}m")

    csi, timestamps = simulate_multi_target_csi(
        num_packets=1000,
        target_angles=target_angles,
        target_distances=target_distances,
        motion_frequencies=motion_frequencies,
        sampling_rate=200.0,
        noise_level=0.12
    )

    # Detect multiple targets
    print("\n3. Detecting multiple targets...")
    estimates = wirainbow.estimate_multiple_targets(
        csi, timestamps,
        num_targets=num_targets
    )

    print(f"   Detected {len(estimates)} targets:")
    for i, est in enumerate(estimates):
        print(f"   Target {i+1}:")
        print(f"     Estimated angle: {est.angle:.1f}°")
        print(f"     SSNR: {est.ssnr:.3f}")
        print(f"     Confidence: {est.confidence:.3f}")

    # Match detected targets to ground truth
    print("\n4. Matching detected targets to ground truth...")
    detected_angles = [est.angle for est in estimates]
    errors = []

    for true_angle in target_angles:
        # Find closest detected angle
        diffs = [abs(detected - true_angle) for detected in detected_angles]
        min_error = min(diffs)
        errors.append(min_error)
        best_match_idx = diffs.index(min_error)
        print(f"   True: {true_angle:.1f}° -> Detected: {detected_angles[best_match_idx]:.1f}° (error: {min_error:.1f}°)")

    avg_error = np.mean(errors)
    print(f"   Average error: {avg_error:.1f}°")

    # Visualize results
    print("\n5. Generating visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: SSNR spectrum
    viz_data = wirainbow.visualize_ssnr(csi, timestamps)

    axes[0].plot(viz_data['angles'], viz_data['ssnr'], linewidth=2, label='SSNR')

    # Mark true target positions
    for i, angle in enumerate(target_angles):
        axes[0].axvline(angle, color='green', linestyle='--', alpha=0.7,
                        label=f'True target {i+1}' if i < 2 else '')

    # Mark detected target positions
    for i, est in enumerate(estimates):
        axes[0].axvline(est.angle, color='red', linestyle=':', alpha=0.7,
                        label=f'Detected target {i+1}' if i < 2 else '')

    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('SSNR (normalized)')
    axes[0].set_title('Multi-Target Detection via SSNR')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Detection comparison
    x_pos = np.arange(num_targets)
    width = 0.35

    axes[1].bar(x_pos - width/2, target_angles, width, label='True angles', color='green', alpha=0.7)
    axes[1].bar(x_pos + width/2, detected_angles[:num_targets], width, label='Detected angles', color='red', alpha=0.7)

    axes[1].set_xlabel('Target Index')
    axes[1].set_ylabel('Angle (degrees)')
    axes[1].set_title('True vs Detected Target Angles')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'Target {i+1}' for i in range(num_targets)])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/tmp/wirainbow_multitarget_example.png', dpi=150)
    print(f"   Saved visualization to /tmp/wirainbow_multitarget_example.png")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
