"""
Example Script: Single Simulation Run

This script demonstrates how to use the acoustic simulation framework
to generate a single multi-channel audio sample with ground truth labels.
"""

import sys
import os
import numpy as np
import yaml

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.signal_generation.noise_sources import NoiseSourceGenerator
from src.array_geometry.microphone_array import MicrophoneArray
from src.propagation_model.room_impulse_response import RoomAcousticSimulator
from src.data_synthesis.mixer import AudioMixer
from src.utils.audio_io import write_audio
from src.utils.labels import LabelGenerator


def main():
    """
    Run a single simulation example.
    """
    print("=" * 70)
    print("Acoustic Simulation Framework - Single Simulation Example")
    print("=" * 70)
    
    # ========================================================================
    # 1. Load Configuration
    # ========================================================================
    print("\n[1/7] Loading configuration...")
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sampling_rate = config['audio']['sampling_rate']
    bit_depth = config['audio']['bit_depth']
    duration = config['audio']['duration']
    
    print(f"  - Sampling rate: {sampling_rate} Hz")
    print(f"  - Bit depth: {bit_depth} bits")
    print(f"  - Duration: {duration} seconds")
    
    # ========================================================================
    # 2. Initialize Microphone Array
    # ========================================================================
    print("\n[2/7] Initializing microphone array...")
    mic_array = MicrophoneArray.from_config(config['microphone_array'])
    print(f"  - Array: {mic_array.name}")
    print(f"  - Number of microphones: {mic_array.num_mics}")
    print(f"  - Array aperture: {mic_array.aperture:.3f} m")
    
    # ========================================================================
    # 3. Create Room Acoustic Simulator
    # ========================================================================
    print("\n[3/7] Creating room acoustic simulator...")
    room_sim = RoomAcousticSimulator.from_config(
        config['room'],
        mic_array,
        sampling_rate
    )
    room_info = room_sim.get_room_info()
    print(f"  - Room dimensions: {room_info['dimensions']} m")
    print(f"  - RT60: {room_info['rt60_s']:.3f} seconds")
    print(f"  - Max reflection order: {room_info['max_order']}")
    
    # ========================================================================
    # 4. Generate Sound Sources
    # ========================================================================
    print("\n[4/7] Generating sound sources...")
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
        print(f"  - Source {i}: {src_config['type']} at {src_config['position']}")
        
        # Generate source signal
        signal = source_generator.generate_source(
            source_type=src_config['type'],
            duration=duration,
            frequency_range=src_config['frequency_range'],
            volume=src_config['volume']
        )
        
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
            'clean_signal_path': f"source_{i}_{src_config['type']}.wav"
        })
    
    # ========================================================================
    # 5. Simulate Acoustic Propagation
    # ========================================================================
    print("\n[5/7] Simulating acoustic propagation...")
    print("  - Computing room impulse responses...")
    print("  - Convolving sources with RIRs...")
    print("  - Mixing signals at microphone positions...")
    
    # Run simulation
    mixed_signal = room_sim.simulate()
    
    print(f"  - Output shape: {mixed_signal.shape}")
    print(f"  - Output duration: {mixed_signal.shape[1] / sampling_rate:.2f} seconds")
    
    # ========================================================================
    # 6. Add Background Noise and Post-Processing
    # ========================================================================
    print("\n[6/7] Post-processing...")
    mixer = AudioMixer(num_channels=mic_array.num_mics, sampling_rate=sampling_rate)
    
    # Add background noise
    if config['background']['enabled']:
        print(f"  - Adding {config['background']['type']} background noise...")
        mixed_signal = mixer.add_background_noise(
            mixed_signal,
            noise_level_db=config['background']['level'],
            noise_type=config['background']['type']
        )
    
    # Normalize
    print("  - Normalizing signal...")
    mixed_signal = mixer.normalize_signal(mixed_signal, target_level_db=-3.0, mode='peak')
    
    # Trim to exact duration
    target_samples = int(duration * sampling_rate)
    mixed_signal = mixer.trim_or_pad(mixed_signal, target_samples)
    
    # ========================================================================
    # 7. Save Outputs
    # ========================================================================
    print("\n[7/7] Saving outputs...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'example')
    os.makedirs(output_dir, exist_ok=True)
    
    clip_id = "example_001"
    
    # Save mixed audio
    mixed_audio_path = os.path.join(output_dir, f"{clip_id}_mixed.wav")
    write_audio(mixed_audio_path, mixed_signal, sampling_rate, bit_depth)
    print(f"  - Mixed audio: {mixed_audio_path}")
    
    # Save individual source signals
    for i, src in enumerate(sources):
        source_path = os.path.join(output_dir, f"{clip_id}_source_{i}_{src['config']['type']}.wav")
        write_audio(source_path, src['signal'], sampling_rate, bit_depth)
        source_labels[i]['clean_signal_path'] = source_path
    print(f"  - Saved {len(sources)} clean source signals")
    
    # Generate and save label
    label_gen = LabelGenerator(format='json')
    label = label_gen.create_label(
        clip_id=clip_id,
        audio_filepath=mixed_audio_path,
        sources=source_labels,
        mic_array_config=mic_array.get_array_info(),
        room_properties=room_sim.get_room_info(),
        sampling_rate=sampling_rate,
        bit_depth=bit_depth
    )
    
    label_path = os.path.join(output_dir, f"{clip_id}_label.json")
    label_gen.save_label(label, label_path)
    print(f"  - Label file: {label_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Simulation completed successfully!")
    print("=" * 70)
    print(f"\nGenerated files in: {output_dir}")
    print(f"  - 1 mixed 8-channel audio file ({bit_depth}-bit, {sampling_rate} Hz)")
    print(f"  - {len(sources)} clean source audio files")
    print(f"  - 1 JSON label file with ground truth")
    print("\nYou can now:")
    print("  1. Listen to the mixed audio to verify realism")
    print("  2. Inspect the label file to see ground truth annotations")
    print("  3. Use this data to train your deep learning model")
    print()


if __name__ == "__main__":
    main()
