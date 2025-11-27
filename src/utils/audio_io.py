"""
Audio I/O Utilities

This module provides utilities for reading and writing audio files
in various formats with support for multi-channel and high bit-depth audio.
"""

import numpy as np
import soundfile as sf
from typing import Tuple, Optional
import os


def read_audio(file_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Read audio file and optionally resample.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sampling rate (if None, use original)
        
    Returns:
        Tuple of (audio_data, sampling_rate)
        audio_data shape: (num_samples,) for mono, (num_samples, num_channels) for multi-channel
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Read audio file
    audio_data, sr = sf.read(file_path, dtype='float64')
    
    # Resample if needed
    if target_sr is not None and sr != target_sr:
        import librosa
        audio_data = librosa.resample(audio_data.T, orig_sr=sr, target_sr=target_sr).T
        sr = target_sr
    
    return audio_data, sr


def write_audio(
    file_path: str,
    audio_data: np.ndarray,
    sampling_rate: int,
    bit_depth: int = 24,
    normalize: bool = False
) -> None:
    """
    Write audio data to file.
    
    Args:
        file_path: Output file path
        audio_data: Audio data to write
                   Shape: (num_samples,) for mono
                          (num_samples, num_channels) for multi-channel
                          (num_channels, num_samples) also accepted (will be transposed)
        sampling_rate: Sampling rate in Hz
        bit_depth: Bit depth (16, 24, or 32)
        normalize: Whether to normalize before writing
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    # Handle different input shapes
    if audio_data.ndim == 2:
        # If channels are first dimension (num_channels, num_samples), transpose
        if audio_data.shape[0] < audio_data.shape[1]:
            audio_data = audio_data.T
    
    # Normalize if requested
    if normalize:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95  # Leave some headroom
    
    # Clip to valid range
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # Determine subtype based on bit depth
    if bit_depth == 16:
        subtype = 'PCM_16'
    elif bit_depth == 24:
        subtype = 'PCM_24'
    elif bit_depth == 32:
        subtype = 'PCM_32'
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    
    # Write audio file
    sf.write(file_path, audio_data, sampling_rate, subtype=subtype)


def get_audio_info(file_path: str) -> dict:
    """
    Get information about an audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary containing audio file metadata
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    info = sf.info(file_path)
    
    return {
        'file_path': file_path,
        'sampling_rate': info.samplerate,
        'channels': info.channels,
        'duration': info.duration,
        'frames': info.frames,
        'format': info.format,
        'subtype': info.subtype
    }


def create_multichannel_audio(
    channels: list,
    sampling_rate: int
) -> np.ndarray:
    """
    Create multi-channel audio from list of mono channels.
    
    Args:
        channels: List of mono audio arrays
        sampling_rate: Sampling rate (must be same for all channels)
        
    Returns:
        Multi-channel audio array (num_samples, num_channels)
    """
    if len(channels) == 0:
        raise ValueError("At least one channel is required")
    
    # Ensure all channels have the same length
    max_length = max(len(ch) for ch in channels)
    
    # Pad shorter channels with zeros
    padded_channels = []
    for ch in channels:
        if len(ch) < max_length:
            padding = np.zeros(max_length - len(ch))
            ch = np.concatenate([ch, padding])
        padded_channels.append(ch)
    
    # Stack channels
    multi_channel = np.column_stack(padded_channels)
    
    return multi_channel


def split_multichannel_audio(audio_data: np.ndarray) -> list:
    """
    Split multi-channel audio into list of mono channels.
    
    Args:
        audio_data: Multi-channel audio (num_samples, num_channels)
        
    Returns:
        List of mono audio arrays
    """
    if audio_data.ndim == 1:
        return [audio_data]
    
    channels = []
    for i in range(audio_data.shape[1]):
        channels.append(audio_data[:, i])
    
    return channels


def calculate_rms(audio_data: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) level of audio signal.
    
    Args:
        audio_data: Audio signal
        
    Returns:
        RMS level
    """
    return np.sqrt(np.mean(audio_data ** 2))


def calculate_peak(audio_data: np.ndarray) -> float:
    """
    Calculate peak level of audio signal.
    
    Args:
        audio_data: Audio signal
        
    Returns:
        Peak level (absolute maximum)
    """
    return np.max(np.abs(audio_data))


def db_to_linear(db: float) -> float:
    """
    Convert decibels to linear scale.
    
    Args:
        db: Value in decibels
        
    Returns:
        Linear scale value
    """
    return 10 ** (db / 20)


def linear_to_db(linear: float) -> float:
    """
    Convert linear scale to decibels.
    
    Args:
        linear: Linear scale value
        
    Returns:
        Value in decibels
    """
    if linear <= 0:
        return -np.inf
    return 20 * np.log10(linear)
