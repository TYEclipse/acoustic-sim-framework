"""
Example Script: Single Simulation Run

This script demonstrates how to use the acoustic simulation framework
to generate a single multi-channel audio sample with ground truth labels.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import yaml

from src.array_geometry.microphone_array import MicrophoneArray
from src.data_synthesis.mixer import AudioMixer
from src.propagation_model.room_impulse_response import RoomAcousticSimulator
from src.signal_generation.noise_sources import NoiseSourceGenerator
from src.utils.audio_io import write_audio
from src.utils.labels import LabelGenerator


def ensure_length(sig, target_samples):
    """
    Ensure 1D signal has exactly target_samples length (pad or trim).
    """
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    if len(sig) > target_samples:
        return sig[:target_samples]
    if len(sig) < target_samples:
        return np.pad(sig, (0, target_samples - len(sig)), mode="constant")
    return sig


def main():
    """
    Run a single simulation example.
    """
    parser = argparse.ArgumentParser(
        description="Run single acoustic simulation example")
    parser.add_argument("--config", "-c", type=str,
                        default=None, help="Path to config yaml")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    logging.info("Acoustic Simulation Framework - Single Simulation Example")

    # set seed if provided for reproducibility
    if args.seed is not None:
        np.random.seed(int(args.seed))
        logging.info(f"Random seed set to {args.seed}")

    # ========================================================================
    # 1. Load Configuration
    # ========================================================================
    logging.info("[1/7] Loading configuration...")
    # resolve config path: CLI > project config/default.yaml
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).resolve(
        ).parent.parent / "config" / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # basic config validation with helpful errors
    for key in ("audio", "microphone_array", "room", "background"):
        if key not in config:
            raise KeyError(
                f"Missing required config section: '{key}' in {config_path}")

    sampling_rate = config['audio']['sampling_rate']
    bit_depth = config['audio']['bit_depth']
    duration = config['audio']['duration']
    target_samples = int(duration * sampling_rate)

    logging.info(f"  - Sampling rate: {sampling_rate} Hz")
    logging.info(f"  - Bit depth: {bit_depth} bits")
    logging.info(f"  - Duration: {duration} seconds")

    # ========================================================================
    # 2. Initialize Microphone Array
    # ========================================================================
    logging.info("[2/7] Initializing microphone array...")
    mic_array = MicrophoneArray.from_config(config['microphone_array'])
    logging.info(f"  - Array: {mic_array.name}")
    logging.info(f"  - Number of microphones: {mic_array.num_mics}")
    logging.info(f"  - Array aperture: {mic_array.aperture:.3f} m")

    # ========================================================================
    # 3. Create Room Acoustic Simulator
    # ========================================================================
    logging.info("[3/7] Creating room acoustic simulator...")
    room_sim = RoomAcousticSimulator.from_config(
        config['room'],
        mic_array,
        sampling_rate
    )
    room_info = room_sim.get_room_info()
    logging.info(f"  - Room dimensions: {room_info['dimensions']} m")
    logging.info(f"  - RT60: {room_info['rt60_s']:.3f} seconds")
    logging.info(f"  - Max reflection order: {room_info['max_order']}")

    # ========================================================================
    # 4. Generate Sound Sources
    # ========================================================================
    logging.info("[4/7] Generating sound sources...")
    source_generator = NoiseSourceGenerator(sampling_rate)

    # Define sources for this example
    source_configs = [
        {
            'type': 'engine_noise',
            'position': [1.5, 0.0, -0.5],
            'frequency_range': [100, 3000],
            'volume': 0.8
        },
        {
            'type': 'wind_noise',
            'position': [0.8, -0.9, 0.3],
            'frequency_range': [500, 8000],
            'volume': 0.6
        },
        {
            'type': 'speech',
            'position': [-0.3, 0.5, 0.0],
            'frequency_range': [200, 8000],
            'volume': 0.7
        }
    ]

    sources = []
    source_labels = []

    for i, src_config in enumerate(source_configs):
        logging.info(
            f"  - Source {i}: {src_config['type']} at {src_config['position']}")

        # Generate source signal
        signal = source_generator.generate_source(
            source_type=src_config['type'],
            duration=duration,
            frequency_range=src_config['frequency_range'],
            volume=src_config['volume']
        )
        # ensure source has exact target length
        signal = ensure_length(signal, target_samples)

        # Add source to room
        source_idx = room_sim.add_source(
            position=np.array(src_config['position']),
            signal=signal,
            source_id=i
        )

        sources.append({
            'signal': signal,
            'config': src_config,
            'index': source_idx
        })

        # Calculate spherical coordinates for label
        distance, azimuth, elevation = mic_array.cartesian_to_spherical(
            np.array(src_config['position'])
        )

        # Create label entry
        source_labels.append({
            'source_id': i,
            'label': src_config['type'],
            'position_xyz': src_config['position'],
            'orientation_az_el': [float(azimuth), float(elevation)],
            'distance_m': float(distance),
            'clean_signal_path': None
        })

    # ========================================================================
    # 5. Simulate Acoustic Propagation
    # ========================================================================
    logging.info("[5/7] Simulating acoustic propagation...")
    logging.info("  - Computing room impulse responses...")
    logging.info("  - Convolving sources with RIRs...")
    logging.info("  - Mixing signals at microphone positions...")

    # Run simulation
    mixed_signal = room_sim.simulate()

    # Ensure shape and orientation: prefer (channels, samples)
    mixed_signal = np.asarray(mixed_signal)
    if mixed_signal.ndim == 1:
        mixed_signal = np.expand_dims(mixed_signal, 0)
    else:
        mixed_signal = np.atleast_2d(mixed_signal)
        # 如果通道在 axis=1，则转置；否则若两边都不匹配则发出警告
        if mixed_signal.shape[0] != mic_array.num_mics:
            if mixed_signal.shape[1] == mic_array.num_mics:
                mixed_signal = mixed_signal.T
            else:
                logging.warning(
                    f"Mixed signal channels ({mixed_signal.shape[0]}) != mic_array.num_mics ({mic_array.num_mics})"
                )
    logging.info(f"  - Output shape: {mixed_signal.shape}")
    logging.info(
        f"  - Output duration: {mixed_signal.shape[1] / sampling_rate:.2f} seconds")

    # ========================================================================
    # 6. Add Background Noise and Post-Processing
    # ========================================================================
    logging.info("[6/7] Post-processing...")
    mixer = AudioMixer(num_channels=mic_array.num_mics,
                       sampling_rate=sampling_rate)

    # Add background noise
    if config['background']['enabled']:
        logging.info(
            f"  - Adding {config['background']['type']} background noise...")
        mixed_signal = mixer.add_background_noise(
            mixed_signal,
            noise_level_db=config['background']['level'],
            noise_type=config['background']['type']
        )

    # Normalize
    logging.info("  - Normalizing signal...")
    mixed_signal = mixer.normalize_signal(
        mixed_signal, target_level_db=-3.0, mode='peak')

    # Trim to exact duration（使用提前计算的 target_samples）
    mixed_signal = mixer.trim_or_pad(mixed_signal, target_samples)
    # ensure dtype for downstream (write_audio may expect floats in [-1,1])
    mixed_signal = mixed_signal.astype(np.float32)

    # ========================================================================
    # 7. Save Outputs
    # ========================================================================
    logging.info("[7/7] Saving outputs...")

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).resolve(
        ).parent.parent / "output" / "example"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_id = "example_001"

    # 变更：统一使用 Path 生成文件路径
    mixed_audio_path = output_dir / f"{clip_id}_mixed.wav"
    write_audio(str(mixed_audio_path), mixed_signal, sampling_rate, bit_depth)
    logging.info(f"  - Mixed audio: {mixed_audio_path}")

    # Save individual source signals
    for i, src in enumerate(sources):
        source_path = output_dir / \
            f"{clip_id}_source_{i}_{src['config']['type']}.wav"
        write_audio(str(source_path), ensure_length(
            src['signal'], target_samples), sampling_rate, bit_depth)
        # 写入后再更新标签里对应的路径
        source_labels[i]['clean_signal_path'] = str(source_path)
    logging.info(f"  - Saved {len(sources)} clean source signals")

    # Generate and save label
    label_gen = LabelGenerator(format='json')
    label = label_gen.create_label(
        clip_id=clip_id,
        audio_filepath=str(mixed_audio_path),  # 传入字符串，避免 Path 被误用
        sources=source_labels,
        mic_array_config=mic_array.get_array_info(),
        room_properties=room_sim.get_room_info(),
        sampling_rate=sampling_rate,
        bit_depth=bit_depth
    )

    label_path = output_dir / f"{clip_id}_label.json"
    label_gen.save_label(label, str(label_path))
    logging.info(f"  - Label file: {label_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    logging.info("Simulation completed successfully!")
    logging.info(f"Generated files in: {output_dir}")
    logging.info(
        f"  - 1 mixed multi-channel audio file ({bit_depth}-bit, {sampling_rate} Hz)")
    logging.info(f"  - {len(sources)} clean source audio files")
    logging.info(f"  - 1 JSON label file with ground truth")
    logging.info(
        "You can now: 1) Listen to the mixed audio 2) Inspect the label file 3) Use data for training")


if __name__ == "__main__":
    main()
