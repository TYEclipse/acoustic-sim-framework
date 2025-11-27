"""
Audio Mixer Module

This module handles the mixing of multiple source signals through
acoustic propagation to create final multi-channel microphone signals.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.signal import fftconvolve


class AudioMixer:
    """
    Mixes multiple audio sources with room impulse responses to create
    realistic multi-channel microphone array recordings.
    """
    
    def __init__(self, num_channels: int = 8, sampling_rate: int = 48000):
        """
        Initialize the audio mixer.
        
        Args:
            num_channels: Number of microphone channels
            sampling_rate: Audio sampling rate in Hz
        """
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        
    def mix_sources_with_rir(
        self,
        sources: List[np.ndarray],
        rirs: List[np.ndarray],
        background_noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Mix multiple source signals using their respective RIRs.
        
        Args:
            sources: List of source signals (each is 1D array)
            rirs: List of RIRs for each source (each is 2D array: num_mics x rir_length)
            background_noise: Optional background noise to add (num_mics x length)
            
        Returns:
            Mixed multi-channel signal (num_mics x num_samples)
        """
        if len(sources) != len(rirs):
            raise ValueError(f"Number of sources ({len(sources)}) must match number of RIRs ({len(rirs)})")
        
        if len(sources) == 0:
            raise ValueError("At least one source is required")
        
        # Determine output length (max of all convolved signals)
        max_length = 0
        for source, rir in zip(sources, rirs):
            convolved_length = len(source) + rir.shape[1] - 1
            max_length = max(max_length, convolved_length)
        
        # Initialize output array
        mixed_signal = np.zeros((self.num_channels, max_length))
        
        # Mix each source
        for source, rir in zip(sources, rirs):
            # Convolve source with RIR for each microphone channel
            for ch in range(self.num_channels):
                # Use FFT-based convolution for efficiency
                convolved = fftconvolve(source, rir[ch], mode='full')
                
                # Add to mixed signal
                mixed_signal[ch, :len(convolved)] += convolved
        
        # Add background noise if provided
        if background_noise is not None:
            noise_length = min(background_noise.shape[1], max_length)
            mixed_signal[:, :noise_length] += background_noise[:, :noise_length]
        
        return mixed_signal
    
    def add_background_noise(
        self,
        signal: np.ndarray,
        noise_level_db: float = -40,
        noise_type: str = "pink"
    ) -> np.ndarray:
        """
        Add background noise to a multi-channel signal.
        
        Args:
            signal: Input multi-channel signal (num_channels x num_samples)
            noise_level_db: Noise level in dB relative to signal RMS
            noise_type: Type of noise ('white', 'pink', 'brown')
            
        Returns:
            Signal with added background noise
        """
        num_channels, num_samples = signal.shape
        
        # Generate noise
        if noise_type == "white":
            noise = np.random.randn(num_channels, num_samples)
        elif noise_type == "pink":
            noise = self._generate_pink_noise(num_channels, num_samples)
        elif noise_type == "brown":
            noise = self._generate_brown_noise(num_channels, num_samples)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Calculate signal RMS
        signal_rms = np.sqrt(np.mean(signal ** 2))
        
        # Calculate noise scaling factor
        noise_linear = 10 ** (noise_level_db / 20)
        noise_rms = signal_rms * noise_linear
        
        # Scale noise
        current_noise_rms = np.sqrt(np.mean(noise ** 2))
        if current_noise_rms > 0:
            noise = noise * (noise_rms / current_noise_rms)
        
        return signal + noise
    
    def _generate_pink_noise(self, num_channels: int, num_samples: int) -> np.ndarray:
        """
        Generate pink noise (1/f spectrum) for multiple channels.
        
        Args:
            num_channels: Number of channels
            num_samples: Number of samples per channel
            
        Returns:
            Pink noise array (num_channels x num_samples)
        """
        pink_noise = np.zeros((num_channels, num_samples))
        
        for ch in range(num_channels):
            # Generate pink noise using Voss-McCartney algorithm
            num_octaves = 16
            pink = np.zeros(num_samples)
            
            for i in range(num_octaves):
                step = 2 ** i
                noise = np.random.randn(num_samples // step + 1)
                noise_upsampled = np.repeat(noise, step)[:num_samples]
                pink += noise_upsampled
            
            pink_noise[ch] = pink / num_octaves
        
        return pink_noise
    
    def _generate_brown_noise(self, num_channels: int, num_samples: int) -> np.ndarray:
        """
        Generate brown noise (1/f^2 spectrum) for multiple channels.
        
        Args:
            num_channels: Number of channels
            num_samples: Number of samples per channel
            
        Returns:
            Brown noise array (num_channels x num_samples)
        """
        brown_noise = np.zeros((num_channels, num_samples))
        
        for ch in range(num_channels):
            # Generate brown noise by integrating white noise
            white = np.random.randn(num_samples)
            brown = np.cumsum(white)
            
            # Normalize
            brown = brown / np.std(brown)
            brown_noise[ch] = brown
        
        return brown_noise
    
    def normalize_signal(
        self,
        signal: np.ndarray,
        target_level_db: float = -3.0,
        mode: str = "peak"
    ) -> np.ndarray:
        """
        Normalize multi-channel signal to target level.
        
        Args:
            signal: Input multi-channel signal
            target_level_db: Target level in dB (0 dB = full scale)
            mode: Normalization mode ('peak' or 'rms')
            
        Returns:
            Normalized signal
        """
        if mode == "peak":
            # Peak normalization
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                target_linear = 10 ** (target_level_db / 20)
                signal = signal * (target_linear / max_val)
        elif mode == "rms":
            # RMS normalization
            rms = np.sqrt(np.mean(signal ** 2))
            if rms > 0:
                target_linear = 10 ** (target_level_db / 20)
                signal = signal * (target_linear / rms)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        
        return signal
    
    def apply_snr(
        self,
        signal: np.ndarray,
        noise: np.ndarray,
        target_snr_db: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust signal and noise levels to achieve target SNR.
        
        Args:
            signal: Clean signal
            noise: Noise signal
            target_snr_db: Target signal-to-noise ratio in dB
            
        Returns:
            Tuple of (adjusted_signal, adjusted_noise)
        """
        # Calculate current RMS levels
        signal_rms = np.sqrt(np.mean(signal ** 2))
        noise_rms = np.sqrt(np.mean(noise ** 2))
        
        if signal_rms == 0 or noise_rms == 0:
            return signal, noise
        
        # Calculate required noise scaling
        target_ratio = 10 ** (target_snr_db / 20)
        noise_scale = signal_rms / (noise_rms * target_ratio)
        
        return signal, noise * noise_scale
    
    def trim_or_pad(
        self,
        signal: np.ndarray,
        target_length: int,
        pad_mode: str = "zeros"
    ) -> np.ndarray:
        """
        Trim or pad signal to target length.
        
        Args:
            signal: Input signal (num_channels x num_samples)
            target_length: Target number of samples
            pad_mode: Padding mode ('zeros', 'repeat', 'reflect')
            
        Returns:
            Signal with target length
        """
        current_length = signal.shape[1]
        
        if current_length == target_length:
            return signal
        elif current_length > target_length:
            # Trim
            return signal[:, :target_length]
        else:
            # Pad
            num_channels = signal.shape[0]
            pad_length = target_length - current_length
            
            if pad_mode == "zeros":
                padding = np.zeros((num_channels, pad_length))
            elif pad_mode == "repeat":
                num_repeats = pad_length // current_length + 1
                repeated = np.tile(signal, num_repeats)
                padding = repeated[:, :pad_length]
            elif pad_mode == "reflect":
                padding = np.flip(signal, axis=1)[:, :pad_length]
            else:
                raise ValueError(f"Unknown pad mode: {pad_mode}")
            
            return np.concatenate([signal, padding], axis=1)
    
    def convert_to_target_format(
        self,
        signal: np.ndarray,
        bit_depth: int = 24
    ) -> np.ndarray:
        """
        Convert signal to target bit depth format.
        
        Args:
            signal: Input signal (float, range approximately -1 to 1)
            bit_depth: Target bit depth (16, 24, or 32)
            
        Returns:
            Signal scaled for target bit depth (still as float)
        """
        # Clip to [-1, 1] range
        signal = np.clip(signal, -1.0, 1.0)
        
        # Scale according to bit depth
        if bit_depth == 16:
            max_val = 2 ** 15 - 1
        elif bit_depth == 24:
            max_val = 2 ** 23 - 1
        elif bit_depth == 32:
            # For 32-bit float, keep as is
            return signal
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        # Scale to integer range and back to float (simulates quantization)
        signal_int = np.round(signal * max_val)
        signal_float = signal_int / max_val
        
        return signal_float
