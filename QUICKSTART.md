# Quick Start Guide

## Installation

### 1. System Requirements

- Ubuntu 20.04+ or similar Linux distribution
- Python 3.8+
- Build tools (gcc, g++, python3-dev)

### 2. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y build-essential g++ python3.11-dev
```

### 3. Install Python Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/acoustic-sim-framework.git
cd acoustic-sim-framework

# Install dependencies
sudo pip3 install -r requirements.txt

# Or install as a package
pip install -e .
```

## Verify Installation

```bash
python3 -c "
from src.signal_generation.noise_sources import NoiseSourceGenerator
from src.array_geometry.microphone_array import MicrophoneArray
from src.propagation_model.room_impulse_response import RoomAcousticSimulator
from src.data_synthesis.mixer import AudioMixer
from src.utils.labels import LabelGenerator
print('✓ All modules imported successfully!')
"
```

## Run Your First Simulation

### Option 1: Single Example

```bash
python3 examples/run_simulation_example.py
```

This will generate:
- `output/example/example_001_mixed.wav` - 8-channel mixed audio
- `output/example/example_001_source_*.wav` - Clean source signals
- `output/example/example_001_label.json` - Ground truth labels

### Option 2: Batch Dataset Generation

Generate 10 samples for testing:

```bash
python3 scripts/generate_dataset.py \
    --config config/default.yaml \
    --output output/test_dataset \
    --num-samples 10 \
    --num-workers 1 \
    --seed 42
```

Generate 1000 samples for training:

```bash
python3 scripts/generate_dataset.py \
    --config config/default.yaml \
    --output output/train_dataset \
    --num-samples 1000 \
    --num-workers 4 \
    --seed 42
```

## Configuration

Edit `config/default.yaml` to customize:

- **Audio settings**: Sample rate, bit depth, duration
- **Microphone array**: Number and positions of microphones
- **Room acoustics**: Dimensions, absorption, RT60
- **Sound sources**: Types, frequency ranges, position ranges
- **Generation**: Number of samples, output format

## Next Steps

1. **Customize Configuration**: Modify `config/default.yaml` for your specific use case
2. **Generate Training Data**: Use `scripts/generate_dataset.py` to create large datasets
3. **Train Your Model**: Use the generated data to train your deep learning model
4. **Validate**: Compare model performance on synthetic vs. real data

## Troubleshooting

### ImportError: No module named 'pyroomacoustics'

```bash
sudo pip3 install pyroomacoustics
```

### Permission denied when installing

Use `sudo` for system-wide installation:

```bash
sudo pip3 install -r requirements.txt
```

### C++ compiler errors

Install build tools:

```bash
sudo apt-get install build-essential g++ python3-dev
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/acoustic-sim-framework/issues
- Email: dev@manus.ai
