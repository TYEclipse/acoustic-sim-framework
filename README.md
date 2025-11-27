# Acoustic Simulation Framework

A highly configurable, Python-based acoustic simulation framework for generating synthetic training data for **vehicle noise source localization and separation** deep learning models.

## 🎯 Overview

This framework addresses the critical challenge of data scarcity in training AI models for in-vehicle noise analysis. By programmatically generating large-scale, high-fidelity acoustic datasets with pixel-perfect ground truth labels, it enables:

- **Automated generation** of thousands of training samples
- **Precise control** over acoustic environments, source positions, and signal characteristics
- **High-quality 8-channel microphone array** simulations at 48kHz/24-bit
- **Comprehensive ground truth** including source positions, orientations, and clean signals
- **Realistic acoustic propagation** using Image Source Method for room impulse responses

## ✨ Key Features

### 🔊 Sound Source Generation
- **8+ noise source types**: Engine, road, wind, HVAC, motor whine, BSR, speech, alert tones
- **Concurrent multi-source simulation**: Support for 3+ simultaneous sources
- **Spectral and temporal control**: Configurable frequency ranges and time-domain characteristics
- **Synthetic and hybrid approaches**: Procedural generation + recorded sample integration

### 🏠 Acoustic Environment Modeling
- **Vehicle cabin simulation**: Configurable dimensions and material properties
- **Reverberation modeling**: Image Source Method with adjustable reflection orders
- **Frequency-dependent absorption**: Realistic material acoustic properties
- **RT60 control**: Adjustable reverberation time (0.1-0.2s typical for vehicles)

### 🎤 Microphone Array Configuration
- **8-channel array**: Default configuration optimized for vehicle cabins
- **3D spatial positioning**: Precise coordinate definition in vehicle frame
- **Flexible geometry**: Easy reconfiguration for different array layouts
- **Coordinate transformations**: Cartesian ↔ Spherical conversions

### 📊 Data Output
- **Multi-channel audio**: 8-channel WAV files (48kHz, 24-bit)
- **Clean source signals**: Individual source recordings for separation training
- **Comprehensive labels**: JSON/YAML metadata with:
  - 3D source positions (x, y, z)
  - Spherical coordinates (distance, azimuth, elevation)
  - Source type classifications
  - Room acoustic properties
  - Microphone array configuration

## 📁 Repository Structure

```
acoustic_sim_framework/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── config/
│   └── default.yaml            # Default configuration template
├── src/
│   ├── __init__.py
│   ├── signal_generation/
│   │   ├── __init__.py
│   │   └── noise_sources.py    # Sound source generators
│   ├── propagation_model/
│   │   ├── __init__.py
│   │   └── room_impulse_response.py  # Acoustic propagation simulator
│   ├── array_geometry/
│   │   ├── __init__.py
│   │   └── microphone_array.py  # Microphone array configuration
│   ├── data_synthesis/
│   │   ├── __init__.py
│   │   └── mixer.py            # Audio mixing and post-processing
│   └── utils/
│       ├── __init__.py
│       ├── audio_io.py         # Audio file I/O utilities
│       └── labels.py           # Label generation and management
├── examples/
│   └── run_simulation_example.py  # Single simulation demo
├── tests/
│   ├── __init__.py
│   └── test_basic_functionality.py  # Unit tests
└── scripts/
    └── generate_dataset.py     # Batch dataset generation tool
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/acoustic-sim-framework.git
cd acoustic-sim-framework
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

### Requirements

- Python 3.8+
- NumPy, SciPy, Librosa
- pyroomacoustics (for acoustic simulation)
- soundfile (for audio I/O)
- PyYAML (for configuration)
- tqdm (for progress bars)

### Run a Single Simulation

Generate one example simulation to verify installation:

```bash
python examples/run_simulation_example.py
```

This will create:
- `output/example/example_001_mixed.wav` - 8-channel mixed audio
- `output/example/example_001_source_*.wav` - Individual clean source signals
- `output/example/example_001_label.json` - Ground truth labels

### Generate a Dataset

Generate a batch of training samples:

```bash
python scripts/generate_dataset.py \
    --config config/default.yaml \
    --output output/dataset \
    --num-samples 100 \
    --num-workers 4 \
    --seed 42
```

**Arguments**:
- `--config`: Path to configuration file
- `--output`: Output directory for dataset
- `--num-samples`: Number of samples to generate
- `--num-workers`: Number of parallel workers (for faster generation)
- `--seed`: Random seed for reproducibility

## ⚙️ Configuration

The framework is highly configurable via YAML files. See `config/default.yaml` for all options.

### Key Configuration Sections

#### Audio Settings
```yaml
audio:
  sampling_rate: 48000  # Hz
  bit_depth: 24         # bits
  duration: 5.0         # seconds per clip
```

#### Microphone Array
```yaml
microphone_array:
  name: "default_8mic_array"
  positions:  # [x, y, z] in meters
    - [0.10, 0.05, 0.0]   # Driver headrest right
    - [0.10, -0.05, 0.0]  # Driver headrest left
    # ... 6 more microphones
```

#### Room Acoustics
```yaml
room:
  dimensions: [4.5, 1.8, 1.5]  # [length, width, height] in meters
  absorption:
    default: [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]  # Frequency-dependent
  max_order: 15  # Reflection order for ISM
  rt60_range: [0.10, 0.20]  # Reverberation time range
```

#### Sound Sources
```yaml
sources:
  num_sources_range: [1, 5]  # Min and max concurrent sources
  types:
    engine_noise:
      enabled: true
      frequency_range: [100, 4000]
      position_range:
        x: [1.2, 2.0]
        y: [-0.3, 0.3]
        z: [-0.8, -0.4]
    # ... more source types
```

## 🧪 Testing

Run unit tests to verify functionality:

```bash
python -m pytest tests/
```

Or run tests directly:
```bash
python tests/test_basic_functionality.py
```

## 📖 Usage Examples

### Example 1: Custom Microphone Array

```python
from src.array_geometry.microphone_array import MicrophoneArray

# Define custom array positions
positions = [
    [0.0, 0.1, 0.0],
    [0.0, -0.1, 0.0],
    [0.5, 0.5, 0.2],
    [0.5, -0.5, 0.2]
]

mic_array = MicrophoneArray(positions, name="custom_4mic")
print(mic_array.visualize_array())
```

### Example 2: Generate Specific Noise Type

```python
from src.signal_generation.noise_sources import NoiseSourceGenerator

generator = NoiseSourceGenerator(sampling_rate=48000)

# Generate 5 seconds of engine noise
engine_signal = generator.generate_source(
    source_type='engine_noise',
    duration=5.0,
    frequency_range=(100, 3000),
    volume=0.8,
    rpm=2000  # Optional: specify RPM
)
```

### Example 3: Simulate Custom Acoustic Scene

```python
from src.propagation_model.room_impulse_response import RoomAcousticSimulator
import numpy as np

# Create room
room_sim = RoomAcousticSimulator(
    room_dimensions=[4.0, 1.8, 1.4],
    mic_array=mic_array,
    sampling_rate=48000,
    max_order=12
)

# Add sources
room_sim.add_source(
    position=np.array([1.5, 0.0, -0.5]),
    signal=engine_signal
)

# Simulate
mixed_audio = room_sim.simulate()
```

## 🎓 Technical Background

This framework implements the **Image Source Method (ISM)** for acoustic propagation, which:

1. **Models direct path**: Sound traveling directly from source to microphone
2. **Computes early reflections**: Sound bouncing off walls, ceiling, floor
3. **Simulates late reverberation**: Diffuse sound field from multiple reflections

The simulation pipeline:

```
Source Generation → RIR Computation → Convolution → Mixing → Post-processing
     (Dry)              (ISM)         (Wet)      (8-ch)    (Normalize)
```

## 📊 Output Data Format

### Audio Files
- **Format**: WAV (PCM)
- **Channels**: 8
- **Sample Rate**: 48000 Hz
- **Bit Depth**: 24-bit
- **Naming**: `{clip_id}_mixed.wav`, `{clip_id}_source_{id}_{type}.wav`

### Label Files (JSON)
```json
{
  "clip_id": "sim_000001",
  "audio_filepath": "/path/to/sim_000001_mixed.wav",
  "sampling_rate": 48000,
  "bit_depth": 24,
  "mic_array_setup": {
    "name": "default_8mic_array",
    "positions": [[0.1, 0.05, 0.0], ...]
  },
  "room_properties": {
    "dimensions": [4.5, 1.8, 1.5],
    "rt60_s": 0.15
  },
  "sources": [
    {
      "source_id": 0,
      "label": "engine_noise",
      "position_xyz": [1.5, 0.2, -0.5],
      "orientation_az_el": [15.0, -10.0],
      "distance_m": 1.8,
      "clean_signal_path": "/path/to/source_0_engine_noise.wav"
    }
  ]
}
```

## 🔬 Validation & Quality Assurance

The framework includes validation strategies:

1. **Physical Validation**: Compare spectral and spatial features with real recordings
2. **Model Validation**: Train baseline models on synthetic data, test on real data (Sim-to-Real)
3. **Cross-Validation**: Mix synthetic and real data for optimal performance

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

1. Allen, J. B., & Berkley, D. A. (1979). Image method for efficiently simulating small-room acoustics. *JASA*.
2. Grumiaux, P. A., et al. (2022). A survey of sound source localization with deep learning methods. *JASA*.
3. Luo, Y., & Mesgarani, N. (2019). Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation.

## 🙋 Support

For questions, issues, or feature requests, please:
- Open an issue on GitHub
- Contact: dev@manus.ai

## 🎉 Acknowledgments

Developed by **Manus AI** as part of the vehicle NVH deep learning initiative.

---

**Happy Simulating! 🎵🚗**
