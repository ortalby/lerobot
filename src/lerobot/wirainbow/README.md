# WiRainbow: Single-Antenna Direction-Aware Wi-Fi Sensing

Implementation of the WiRainbow system as described in the paper:

**"WiRainbow: Single-Antenna Direction-Aware Wi-Fi Sensing via Dispersion Effect"**
Zhaoxin Chang, Shuguang Xiao, Fusang Zhang, Xujun Ma, Badii Jouaber, Qingfeng Zhang, and Daqing Zhang
arXiv:2511.20671 (2025)
[Paper Link](https://arxiv.org/abs/2511.20671)

## Overview

WiRainbow enables directional awareness for Wi-Fi sensing using only a **single frequency-scanning antenna (FSA)**, eliminating the need for costly multi-antenna arrays. The system leverages the dispersion effect of FSAs, which naturally steer Wi-Fi subcarriers toward distinct angles during signal transmission.

### Key Features

- **Single-antenna design**: No need for expensive antenna arrays
- **Direction estimation**: Accurate target localization using SSNR-based algorithm
- **Multi-target detection**: Simultaneous tracking of multiple targets
- **Respiration monitoring**: Contactless vital sign monitoring
- **Wide field of view**: 120° coverage across 360 MHz bandwidth (5.37-5.73 GHz)

## System Architecture

The WiRainbow system consists of three main components:

1. **Frequency-Scanning Antenna (FSA)**
   - Coupled-resonator design with 12 elements
   - Frequency-dependent beam patterns
   - Quality factor Q = 200

2. **CSI Processor**
   - Channel State Information (CSI) extraction
   - Phase offset cancellation using CSI ratio method
   - Time-Domain CSI (TD-CSI) for static component removal
   - Multi-interval variance analysis

3. **Direction Estimator**
   - Sensing-Signal-to-Noise Ratio (SSNR) computation
   - Multi-interval variance-based direction estimation
   - Frequency-to-angle mapping using antenna calibration

## Installation

The WiRainbow module is included in the LeRobot package. Install the required dependencies:

```bash
pip install numpy scipy matplotlib
```

## Quick Start

### Basic Direction Estimation

```python
from lerobot.wirainbow import WiRainbow
import numpy as np

# Initialize WiRainbow system
wirainbow = WiRainbow()

# Calibrate antenna (done automatically on first use)
wirainbow.calibrate()

# Prepare your CSI data
# Shape: (num_antennas, num_subcarriers, num_packets)
csi = np.load('your_csi_data.npy')
timestamps = np.load('your_timestamps.npy')

# Estimate target direction
estimate = wirainbow.estimate_direction(csi, timestamps)

print(f"Target detected at {estimate.angle:.1f}° with confidence {estimate.confidence:.2f}")
```

### Multi-Target Detection

```python
# Detect multiple targets simultaneously
estimates = wirainbow.estimate_multiple_targets(
    csi, timestamps,
    num_targets=2
)

for i, est in enumerate(estimates):
    print(f"Target {i+1}: {est.angle:.1f}° (SSNR: {est.ssnr:.3f})")
```

### Respiration Monitoring

```python
# Monitor respiration rate in a specific direction
respiration_data = wirainbow.monitor_respiration(
    csi, timestamps,
    target_direction=0.0,  # or None for auto-detection
    respiration_band=(0.2, 0.5)  # 12-30 bpm
)

print(f"Respiration rate: {respiration_data['respiration_rate_bpm']:.1f} bpm")
```

## Configuration

Customize the WiRainbow system using `WiRainbowConfig`:

```python
from lerobot.wirainbow import WiRainbowConfig, WiRainbow

config = WiRainbowConfig(
    # Antenna parameters
    num_resonators=12,
    element_spacing=0.0145,  # meters
    quality_factor=200.0,
    resonant_freq=5.57e9,  # Hz

    # Signal processing
    sampling_rate=200.0,  # Hz
    center_frequency=5.57e9,  # Hz
    bandwidth=80e6,  # Hz (80 MHz)
    num_subcarriers=234,  # 802.11ac

    # Direction estimation
    min_interval=0.005,  # 5 ms
    max_interval=0.1,    # 100 ms
    num_intervals=20,
)

wirainbow = WiRainbow(config)
```

Save and load configurations:

```python
# Save configuration
config.save('wirainbow_config.json')

# Load configuration
config = WiRainbowConfig.load('wirainbow_config.json')
```

## Examples

See the `examples/wirainbow/` directory for complete examples:

- `basic_usage.py`: Basic direction estimation with visualization
- `respiration_monitoring.py`: Contactless respiration rate monitoring
- `multi_target_detection.py`: Simultaneous multi-target tracking

Run an example:

```bash
python examples/wirainbow/basic_usage.py
```

## API Reference

### WiRainbow

Main class for Wi-Fi sensing operations.

**Methods:**

- `calibrate()`: Calibrate frequency-to-direction mapping
- `estimate_direction(csi, timestamps)`: Estimate single target direction
- `estimate_multiple_targets(csi, timestamps, num_targets)`: Detect multiple targets
- `monitor_respiration(csi, timestamps, target_direction)`: Monitor respiration rate
- `get_field_of_view()`: Get antenna field of view
- `visualize_ssnr(csi, timestamps)`: Get SSNR visualization data

### FrequencyScanningAntenna

Models the frequency-scanning antenna with coupled resonators.

**Methods:**

- `compute_beam_direction(frequency)`: Compute beam angle for frequency
- `compute_beam_pattern(frequency, angle)`: Compute beam pattern D(f,θ)
- `calibrate_frequency_direction_mapping(frequencies, angles)`: Create calibration map
- `get_field_of_view(frequency_range)`: Compute field of view

### CSIProcessor

Processes Channel State Information.

**Methods:**

- `cancel_phase_offset(csi)`: Remove hardware phase offsets
- `compute_td_csi(csi, time_delta, timestamps)`: Compute Time-Domain CSI
- `compute_multi_interval_td_csi(csi, timestamps)`: Multi-interval TD-CSI
- `sanitize_csi(csi)`: Remove invalid CSI values

### DirectionEstimator

Estimates target direction using SSNR.

**Methods:**

- `compute_ssnr(csi, timestamps)`: Compute SSNR for all subcarriers
- `estimate_direction(csi, timestamps)`: Estimate target direction
- `estimate_multiple_targets(csi, timestamps, num_targets)`: Detect multiple targets
- `track_direction(csi, timestamps, previous_estimate)`: Track with temporal smoothing

## Technical Details

### Mathematical Foundations

**Beam Direction (Equation 2):**
```
θ_f = arcsin(-λ_f·Δφ_f / 2πl)
```

**Resonator Phase Delay (Equation 3):**
```
Δφ_rf = arctan{Q(f₀/f - f/f₀)}
```

**CSI Model with FSA (Equation 4):**
```
H(f,t) = H_s(f) + D(f,θ_d)·A(f,t)·e^(-j2π·d(t)/λ_f) + n(f,t)
```

**Sensing-SNR (Equation 5):**
```
SSNR = |D(f,θ_d)·A(f,t)|² / |n(f,t)|²
```

### Direction Estimation Algorithm

1. Construct multiple TD-CSI signals using time intervals (5-100 ms)
2. Calculate phase differences between consecutive TD-CSI samples
3. Compute variance of phase differences for each subcarrier
4. Average variances across intervals to obtain SSNR
5. Identify subcarrier with highest SSNR
6. Map frequency to angle using calibration

## Performance

Based on the original paper:

- **Direction estimation error**: < 4.4° for large-scale motion, < 4.6° for small-scale motion
- **Field of view**: 120° across 360 MHz bandwidth
- **Respiration monitoring error**: < 0.62 bpm
- **Multi-target capability**: Simultaneous tracking of multiple targets

## Limitations

- Requires CSI data from Wi-Fi transceivers (e.g., Intel 5300 NIC, Qualcomm Atheros AR9380)
- Performance depends on multipath environment
- Current implementation uses theoretical antenna model (anechoic chamber calibration recommended for production)

## CSI Data Format

WiRainbow expects CSI data in the following format:

```python
csi: numpy.ndarray
    Complex CSI values
    Shape: (num_antennas, num_subcarriers, num_packets)
    dtype: complex128 or complex64

timestamps: numpy.ndarray
    Packet timestamps in seconds
    Shape: (num_packets,)
    dtype: float64 or float32
```

### Obtaining CSI Data

CSI can be extracted from Wi-Fi hardware using tools such as:

- **Linux 802.11n CSI Tool** (Intel 5300): https://dhalperi.github.io/linux-80211n-csitool/
- **Atheros CSI Tool** (Qualcomm Atheros AR9380): https://wands.sg/research/wifi/AtherosCSI/
- **PicoScenes** (Multiple platforms): https://ps.zpj.io/

## Testing

Run the test suite:

```bash
pytest tests/wirainbow/
```

Run specific test modules:

```bash
pytest tests/wirainbow/test_antenna.py
pytest tests/wirainbow/test_csi_processor.py
pytest tests/wirainbow/test_wirainbow.py
```

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{chang2025wirainbow,
  title={WiRainbow: Single-Antenna Direction-Aware Wi-Fi Sensing via Dispersion Effect},
  author={Chang, Zhaoxin and Xiao, Shuguang and Zhang, Fusang and Ma, Xujun and Jouaber, Badii and Zhang, Qingfeng and Zhang, Daqing},
  journal={arXiv preprint arXiv:2511.20671},
  year={2025}
}
```

## License

This implementation is released under the same license as the LeRobot project (Apache 2.0).

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Performance improvements
- Additional features
- Documentation improvements
- Example applications

## Acknowledgments

- Original WiRainbow paper authors for the innovative approach
- LeRobot team for the excellent robotics framework
- Contributors to CSI extraction tools (Linux 802.11n CSI Tool, PicoScenes, etc.)

## Contact

For questions or issues specific to this implementation, please open an issue on the LeRobot GitHub repository.

For questions about the original WiRainbow research, please contact the paper authors.
