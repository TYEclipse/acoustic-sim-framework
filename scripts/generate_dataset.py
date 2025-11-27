#!/usr/bin/env python3
"""
Dataset Generation Script

This script automates the generation of large-scale acoustic datasets
for training deep learning models in vehicle noise source localization
and separation tasks.
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys

import numpy as np
import yaml
from tqdm import tqdm

from src.array_geometry.microphone_array import MicrophoneArray
from src.data_synthesis.mixer import AudioMixer
from src.propagation_model.room_impulse_response import RoomAcousticSimulator
from src.signal_generation.noise_sources import NoiseSourceGenerator
from src.utils.audio_io import write_audio
from src.utils.labels import LabelGenerator, create_dataset_manifest

# Ensure project root is on sys.path before importing local 'src' package
script_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if script_root not in sys.path:
    sys.path.insert(0, script_root)

# Import local modules (after sys.path adjustment)

# Module-level worker generator used by child processes (avoid pickling self)
_WORKER_GENERATOR = None


def _worker_init(config_path, output_dir, random_seed):
    """
    Initializer for worker processes: create a DatasetGenerator instance
    local to each worker to avoid pickling the parent self.
    """
    global _WORKER_GENERATOR
    # Adjust seed with pid to avoid identical sequences across workers
    seed = None
    if random_seed is not None:
        try:
            seed = int(random_seed) + os.getpid()
        except Exception:
            seed = random_seed
    _WORKER_GENERATOR = DatasetGenerator(
        config_path=config_path,
        output_dir=output_dir,
        random_seed=seed
    )


def _worker_task(sample_idx):
    """
    Worker task that delegates to the per-process DatasetGenerator.
    """
    global _WORKER_GENERATOR
    if _WORKER_GENERATOR is None:
        raise RuntimeError(
            "Worker not initialized. Did you set initializer for the Pool?")
    return _WORKER_GENERATOR.generate_single_sample(sample_idx)


class DatasetGenerator:
    """
    Automated dataset generator for acoustic simulations.
    """

    def __init__(self, config_path: str, output_dir: str, random_seed: int = None):
        """
        Initialize dataset generator.

        Args:
            config_path: Path to configuration YAML file
            output_dir: Output directory for generated dataset
            random_seed: Random seed for reproducibility
        """
        # Save config path for potential worker re-creation
        self.config_path = config_path

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = output_dir
        self.random_seed = random_seed

        # Per-instance RNG to avoid global state collisions and make unit-testable
        if random_seed is None:
            # fallback to unpredictable seed
            self.rng = np.random.RandomState()
        else:
            self.rng = np.random.RandomState(int(random_seed))

        # Extract key parameters
        self.sampling_rate = self.config['audio']['sampling_rate']
        self.bit_depth = self.config['audio']['bit_depth']
        self.duration = self.config['audio']['duration']

        # Create output directories
        self.audio_dir = os.path.join(output_dir, 'audio')
        self.label_dir = os.path.join(output_dir, 'labels')
        self.source_dir = os.path.join(output_dir, 'clean_sources')

        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        if self.config['generation']['save_clean_sources']:
            os.makedirs(self.source_dir, exist_ok=True)

    def _randomize_room_properties(self) -> dict:
        """
        Randomize room properties within configured ranges.

        Returns:
            Dictionary of room properties
        """
        # Use default dimensions or randomize if range provided
        dimensions = self.config['room']['dimensions']

        # Randomize RT60 within range
        rt60_range = self.config['room'].get('rt60_range', [0.10, 0.20])
        target_rt60 = float(self.rng.uniform(rt60_range[0], rt60_range[1]))

        # Adjust absorption to achieve target RT60 (simplified)
        # Higher absorption = lower RT60
        base_absorption = self.config['room']['absorption']['default']
        scale_factor = 0.15 / target_rt60  # Assuming base RT60 ~ 0.15s
        absorption = [min(0.95, a * scale_factor) for a in base_absorption]

        return {
            'dimensions': dimensions,
            'absorption': {'default': absorption},
            'max_order': self.config['room']['max_order']
        }

    def _select_random_sources(self) -> list:
        """
        Randomly select and configure sound sources.

        Returns:
            List of source configuration dictionaries
        """
        # Determine number of sources
        num_range = self.config['sources']['num_sources_range']
        # RandomState.randint upper bound is exclusive; original code used randint(low, high+1)
        num_sources = int(self.rng.randint(num_range[0], num_range[1] + 1))

        # Room dimensions used to validate positions (expect list-like [x, y, z])
        room_dims = self.config['room'].get('dimensions', [None, None, None])

        # Get enabled source types
        enabled_types = [
            src_type for src_type, src_config in self.config['sources']['types'].items()
            if src_config.get('enabled', True)
        ]

        if len(enabled_types) == 0:
            raise ValueError("No source types are enabled in configuration")

        # Randomly select source types (with replacement if needed)
        selected_types = self.rng.choice(
            enabled_types, size=num_sources, replace=True)

        # Configure each source
        sources = []
        for i, src_type in enumerate(selected_types):
            src_config = self.config['sources']['types'][src_type]

            # Randomize position within range
            pos_range = src_config['position_range']
            # Try sampling a valid position up to N attempts; otherwise clamp to room bounds
            position = None
            max_attempts = 20
            for attempt in range(max_attempts):
                cand = [
                    float(self.rng.uniform(
                        pos_range['x'][0], pos_range['x'][1])),
                    float(self.rng.uniform(
                        pos_range['y'][0], pos_range['y'][1])),
                    float(self.rng.uniform(
                        pos_range['z'][0], pos_range['z'][1]))
                ]
                # If room dimensions are available, check containment
                if all(d is not None for d in room_dims):
                    if 0.0 <= cand[0] <= room_dims[0] and 0.0 <= cand[1] <= room_dims[1] and 0.0 <= cand[2] <= room_dims[2]:
                        position = cand
                        break
                else:
                    # No room dims -> accept candidate
                    position = cand
                    break

            if position is None:
                # Clamp to room bounds as fallback and warn
                position = [
                    float(min(max(self.rng.uniform(pos_range['x'][0], pos_range['x'][1]), 0.0), room_dims[0]
                          if room_dims[0] is not None else self.rng.uniform(pos_range['x'][0], pos_range['x'][1]))),
                    float(min(max(self.rng.uniform(pos_range['y'][0], pos_range['y'][1]), 0.0), room_dims[1]
                          if room_dims[1] is not None else self.rng.uniform(pos_range['y'][0], pos_range['y'][1]))),
                    float(min(max(self.rng.uniform(pos_range['z'][0], pos_range['z'][1]), 0.0), room_dims[2]
                          if room_dims[2] is not None else self.rng.uniform(pos_range['z'][0], pos_range['z'][1])))
                ]
                logging.warning(
                    "Position sampling failed to produce an in-room point after %d attempts; clamped to %s (room_dims=%s)",
                    max_attempts, position, room_dims
                )

            # Randomize volume (SNR)
            snr_range = self.config['sources']['snr_range']
            snr_db = float(self.rng.uniform(snr_range[0], snr_range[1]))
            volume = 10 ** (snr_db / 20) * float(self.rng.uniform(0.5, 1.0))

            sources.append({
                'id': i,
                'type': src_type,
                'position': position,
                'frequency_range': src_config['frequency_range'],
                'volume': volume
            })

        return sources

    def generate_single_sample(self, sample_idx: int) -> dict:
        """
        Generate a single simulation sample.

        Args:
            sample_idx: Index of the sample

        Returns:
            Dictionary with generation results
        """
        try:
            clip_id = f"sim_{sample_idx:06d}"

            # Initialize components
            mic_array = MicrophoneArray.from_config(
                self.config['microphone_array'])
            room_props = self._randomize_room_properties()
            room_sim = RoomAcousticSimulator(
                room_dimensions=room_props['dimensions'],
                mic_array=mic_array,
                sampling_rate=self.sampling_rate,
                max_order=room_props['max_order'],
                absorption=room_props['absorption']
            )

            source_generator = NoiseSourceGenerator(self.sampling_rate)
            mixer = AudioMixer(num_channels=mic_array.num_mics,
                               sampling_rate=self.sampling_rate)

            # Generate sources
            source_configs = self._select_random_sources()
            sources = []
            source_labels = []

            for src_config in source_configs:
                # Generate signal
                signal = source_generator.generate_source(
                    source_type=src_config['type'],
                    duration=self.duration,
                    frequency_range=src_config['frequency_range'],
                    volume=src_config['volume']
                )

                # Add to room
                room_sim.add_source(
                    position=np.array(src_config['position']),
                    signal=signal
                )

                sources.append({
                    'signal': signal,
                    'config': src_config
                })

                # Calculate orientation
                distance, azimuth, elevation = mic_array.cartesian_to_spherical(
                    np.array(src_config['position'])
                )

                # Prepare label entry
                source_label = {
                    'source_id': src_config['id'],
                    'label': src_config['type'],
                    'position_xyz': src_config['position'],
                    'orientation_az_el': [float(azimuth), float(elevation)],
                    'distance_m': float(distance)
                }

                # Save clean source if configured
                if self.config['generation']['save_clean_sources']:
                    source_filename = f"{clip_id}_source_{src_config['id']}_{src_config['type']}.wav"
                    source_path = os.path.join(
                        self.source_dir, source_filename)
                    write_audio(source_path, signal,
                                self.sampling_rate, self.bit_depth)
                    source_label['clean_signal_path'] = source_path

                source_labels.append(source_label)

            # Simulate
            mixed_signal = room_sim.simulate()

            # Add background noise
            if self.config['background']['enabled']:
                mixed_signal = mixer.add_background_noise(
                    mixed_signal,
                    noise_level_db=self.config['background']['level'],
                    noise_type=self.config['background']['type']
                )

            # Normalize and trim
            mixed_signal = mixer.normalize_signal(
                mixed_signal, target_level_db=-3.0, mode='peak')
            target_samples = int(self.duration * self.sampling_rate)
            mixed_signal = mixer.trim_or_pad(mixed_signal, target_samples)

            # Save mixed audio
            audio_filename = f"{clip_id}_mixed.wav"
            audio_path = os.path.join(self.audio_dir, audio_filename)
            write_audio(audio_path, mixed_signal,
                        self.sampling_rate, self.bit_depth)

            # Generate and save label
            label_gen = LabelGenerator(
                format=self.config['generation']['metadata_format'])
            label = label_gen.create_label(
                clip_id=clip_id,
                audio_filepath=audio_path,
                sources=source_labels,
                mic_array_config=mic_array.get_array_info(),
                room_properties=room_sim.get_room_info(),
                sampling_rate=self.sampling_rate,
                bit_depth=self.bit_depth
            )

            label_filename = f"{clip_id}_label.{self.config['generation']['metadata_format']}"
            label_path = os.path.join(self.label_dir, label_filename)
            label_gen.save_label(label, label_path)

            return {
                'success': True,
                'clip_id': clip_id,
                'audio_path': audio_path,
                'label_path': label_path,
                'num_sources': len(sources)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'sample_idx': sample_idx
            }

    def generate_dataset(self, num_samples: int = None, num_workers: int = 1):
        """
        Generate complete dataset.

        Args:
            num_samples: Number of samples to generate (if None, use config)
            num_workers: Number of parallel workers
        """
        if num_samples is None:
            num_samples = self.config['generation']['num_samples']

        logging.info("=" * 70)
        logging.info("Acoustic Dataset Generation")
        logging.info("=" * 70)
        logging.info("Output directory: %s", self.output_dir)
        logging.info("Number of samples: %d", num_samples)
        logging.info("Sampling rate: %d Hz", self.sampling_rate)
        logging.info("Bit depth: %d bits", self.bit_depth)
        logging.info("Duration per sample: %s seconds", self.duration)
        logging.info("Parallel workers: %d", num_workers)
        logging.info("=" * 70)

        # Generate samples
        logging.info("\nGenerating samples...")
        results = []

        if num_workers > 1:
            # Parallel processing using initializer to avoid pickling the whole object
            with mp.Pool(processes=num_workers, initializer=_worker_init,
                         initargs=(self.config_path, self.output_dir, self.random_seed)) as pool:
                results = list(tqdm(
                    pool.imap(_worker_task, range(num_samples)),
                    total=num_samples,
                    desc="Progress"
                ))
        else:
            # Sequential processing
            for i in tqdm(range(num_samples), desc="Progress"):
                results.append(self.generate_single_sample(i))

        # Analyze results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        logging.info("\n" + "=" * 70)
        logging.info("Generation Summary")
        logging.info("=" * 70)
        logging.info("Total samples: %d", num_samples)
        logging.info("Successful: %d", len(successful))
        logging.info("Failed: %d", len(failed))

        if failed:
            logging.info("\nFailed samples:")
            for f in failed[:10]:  # Show first 10 failures
                logging.info("  - Sample %s: %s", f.get('sample_idx',
                             'N/A'), f.get('error', 'Unknown error'))

        # Create dataset manifest
        if successful:
            logging.info("\nCreating dataset manifest...")
            label_files = [r['label_path'] for r in successful]
            manifest_path = os.path.join(self.output_dir, 'manifest.json')
            create_dataset_manifest(
                self.output_dir, label_files, manifest_path)
            logging.info("Manifest saved to: %s", manifest_path)

        logging.info("\n" + "=" * 70)
        logging.info("Dataset generation completed!")
        logging.info("=" * 70)
        logging.info("\nDataset location: %s", self.output_dir)
        logging.info("  - Audio files: %s", self.audio_dir)
        logging.info("  - Label files: %s", self.label_dir)
        if self.config['generation']['save_clean_sources']:
            logging.info("  - Clean sources: %s", self.source_dir)
        logging.info("")


def main():
    """
    Main entry point for dataset generation script.
    """
    parser = argparse.ArgumentParser(
        description="Generate acoustic simulation dataset for training deep learning models"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/dataset',
        help='Output directory for generated dataset'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to generate (overrides config)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Resolve config/output paths robustly (support absolute paths)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(args.config):
        config_path = os.path.abspath(args.config)
    else:
        config_path = os.path.abspath(
            os.path.join(script_dir, '..', args.config))

    if os.path.isabs(args.output):
        output_dir = os.path.abspath(args.output)
    else:
        output_dir = os.path.abspath(
            os.path.join(script_dir, '..', args.output))

    # Setup basic logging
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(message)s')

    # Create generator
    generator = DatasetGenerator(
        config_path=config_path,
        output_dir=output_dir,
        random_seed=args.seed
    )

    # Generate dataset
    generator.generate_dataset(
        num_samples=args.num_samples,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
