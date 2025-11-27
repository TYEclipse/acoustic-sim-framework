"""
Noise Source Generation Module

This module provides functionality for generating various types of vehicle noise sources,
including both synthetic generation and loading from pre-recorded samples.
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional
import warnings


class NoiseSourceGenerator:
    """
    Generator for various types of vehicle noise sources.
    
    Supports both synthetic generation (for tones, motor whines, alerts) and
    spectral shaping of noise (for engine, road, wind, HVAC, BSR).
    """
    
    def __init__(self, sampling_rate: int = 48000):
        """
        Initialize the noise source generator.
        
        Args:
            sampling_rate: Audio sampling rate in Hz (default: 48000)
        """
        self.sampling_rate = sampling_rate
        
    def generate_source(
        self,
        source_type: str,
        duration: float,
        frequency_range: Tuple[float, float],
        volume: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """
        Generate a noise source signal based on type and parameters.
        
        Args:
            source_type: Type of noise source (e.g., 'engine_noise', 'motor_whine')
            duration: Duration of the signal in seconds
            frequency_range: Tuple of (low_freq, high_freq) in Hz
            volume: Amplitude scaling factor (0.0 to 1.0)
            **kwargs: Additional parameters specific to each source type
            
        Returns:
            Generated audio signal as numpy array
        """
        num_samples = int(duration * self.sampling_rate)
        
        # Route to appropriate generation method based on source type
        if source_type == "engine_noise":
            signal_data = self._generate_engine_noise(num_samples, frequency_range, **kwargs)
        elif source_type == "road_noise":
            signal_data = self._generate_road_noise(num_samples, frequency_range, **kwargs)
        elif source_type == "wind_noise":
            signal_data = self._generate_wind_noise(num_samples, frequency_range, **kwargs)
        elif source_type == "hvac_noise":
            signal_data = self._generate_hvac_noise(num_samples, frequency_range, **kwargs)
        elif source_type == "motor_whine":
            signal_data = self._generate_motor_whine(num_samples, frequency_range, **kwargs)
        elif source_type == "bsr_noise":
            signal_data = self._generate_bsr_noise(num_samples, frequency_range, **kwargs)
        elif source_type == "speech":
            signal_data = self._generate_speech_like(num_samples, frequency_range, **kwargs)
        elif source_type == "alert_tone":
            signal_data = self._generate_alert_tone(num_samples, frequency_range, **kwargs)
        else:
            warnings.warn(f"Unknown source type '{source_type}', generating white noise")
            signal_data = self._generate_white_noise(num_samples)
        
        # Apply volume scaling
        signal_data = signal_data * volume
        
        # Apply fade in/out to avoid clicks
        signal_data = self._apply_fade(signal_data)
        
        return signal_data
    
    def _generate_engine_noise(
        self,
        num_samples: int,
        frequency_range: Tuple[float, float],
        rpm: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate engine noise with harmonic structure.
        
        Engine noise is characterized by a fundamental frequency (related to RPM)
        and multiple harmonics, modulated with low-frequency amplitude variations.
        """
        # Base frequency from RPM or random within range
        if rpm is not None:
            fundamental_freq = rpm / 60.0  # Convert RPM to Hz
        else:
            fundamental_freq = np.random.uniform(frequency_range[0] / 4, frequency_range[0])
        
        t = np.arange(num_samples) / self.sampling_rate
        signal_data = np.zeros(num_samples)
        
        # Add harmonics with decreasing amplitude
        num_harmonics = 8
        for h in range(1, num_harmonics + 1):
            harmonic_freq = fundamental_freq * h
            if harmonic_freq > frequency_range[1]:
                break
            amplitude = 1.0 / h  # Amplitude decreases with harmonic order
            signal_data += amplitude * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Add low-frequency amplitude modulation (engine vibration)
        modulation_freq = np.random.uniform(2, 8)  # 2-8 Hz modulation
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * modulation_freq * t)
        signal_data *= modulation
        
        # Add some noise for realism
        noise = np.random.randn(num_samples) * 0.1
        signal_data += noise
        
        # Apply bandpass filter
        signal_data = self._apply_bandpass_filter(signal_data, frequency_range)
        
        return self._normalize(signal_data)
    
    def _generate_road_noise(
        self,
        num_samples: int,
        frequency_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate road noise (tire-road interaction).
        
        Road noise is primarily broadband with emphasis on low frequencies,
        with occasional transient impacts.
        """
        # Start with pink noise (1/f spectrum)
        signal_data = self._generate_pink_noise(num_samples)
        
        # Apply bandpass filter
        signal_data = self._apply_bandpass_filter(signal_data, frequency_range)
        
        # Add random transient impacts (road bumps)
        num_impacts = np.random.randint(5, 15)
        for _ in range(num_impacts):
            impact_position = np.random.randint(0, num_samples - 1000)
            impact_duration = np.random.randint(100, 500)
            impact_signal = np.random.randn(impact_duration) * 2.0
            impact_signal *= np.exp(-np.arange(impact_duration) / 100)  # Exponential decay
            signal_data[impact_position:impact_position + impact_duration] += impact_signal
        
        return self._normalize(signal_data)
    
    def _generate_wind_noise(
        self,
        num_samples: int,
        frequency_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate wind noise (aerodynamic noise).
        
        Wind noise is broadband with more energy at higher frequencies,
        with slow amplitude variations.
        """
        # Start with white noise
        signal_data = self._generate_white_noise(num_samples)
        
        # Apply high-pass characteristic (wind noise has more high-frequency content)
        signal_data = self._apply_bandpass_filter(signal_data, frequency_range)
        
        # Add slow amplitude modulation (wind gusts)
        t = np.arange(num_samples) / self.sampling_rate
        modulation_freq = np.random.uniform(0.5, 2.0)  # Slow modulation
        modulation = 0.6 + 0.4 * np.sin(2 * np.pi * modulation_freq * t)
        signal_data *= modulation
        
        return self._normalize(signal_data)
    
    def _generate_hvac_noise(
        self,
        num_samples: int,
        frequency_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate HVAC system noise (fan noise).
        
        HVAC noise is characterized by broadband noise with tonal components
        from fan blade passing frequency.
        """
        # Broadband component
        signal_data = self._generate_pink_noise(num_samples)
        
        # Add tonal components (fan blade passing frequency)
        t = np.arange(num_samples) / self.sampling_rate
        num_tones = np.random.randint(2, 5)
        for _ in range(num_tones):
            tone_freq = np.random.uniform(frequency_range[0], frequency_range[1])
            tone_amplitude = np.random.uniform(0.3, 0.7)
            signal_data += tone_amplitude * np.sin(2 * np.pi * tone_freq * t)
        
        # Apply bandpass filter
        signal_data = self._apply_bandpass_filter(signal_data, frequency_range)
        
        return self._normalize(signal_data)
    
    def _generate_motor_whine(
        self,
        num_samples: int,
        frequency_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate electric motor whine (high-frequency tonal noise).
        
        Motor whine is characterized by a pure tone that may vary in frequency
        (order tracking with motor speed).
        """
        t = np.arange(num_samples) / self.sampling_rate
        
        # Base frequency (may vary over time)
        base_freq = np.random.uniform(frequency_range[0], frequency_range[1])
        
        # Add frequency modulation to simulate speed variation
        freq_modulation = base_freq + np.random.uniform(-200, 200) * np.sin(2 * np.pi * 0.5 * t)
        
        # Generate tone with frequency modulation
        phase = 2 * np.pi * np.cumsum(freq_modulation) / self.sampling_rate
        signal_data = np.sin(phase)
        
        # Add harmonics
        for h in [2, 3]:
            harmonic_phase = h * phase
            signal_data += 0.3 / h * np.sin(harmonic_phase)
        
        return self._normalize(signal_data)
    
    def _generate_bsr_noise(
        self,
        num_samples: int,
        frequency_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate BSR (Buzz, Squeak, Rattle) noise.
        
        BSR noises are transient, impulsive events with random occurrence.
        """
        signal_data = np.zeros(num_samples)
        
        # Generate random impulsive events
        num_events = np.random.randint(10, 30)
        for _ in range(num_events):
            event_position = np.random.randint(0, num_samples - 2000)
            event_type = np.random.choice(['buzz', 'squeak', 'rattle'])
            
            if event_type == 'buzz':
                # High-frequency oscillation
                event_duration = np.random.randint(500, 1500)
                event_freq = np.random.uniform(1000, 3000)
                t_event = np.arange(event_duration) / self.sampling_rate
                event_signal = np.sin(2 * np.pi * event_freq * t_event)
                event_signal *= np.exp(-t_event * 5)  # Decay
            elif event_type == 'squeak':
                # Frequency sweep
                event_duration = np.random.randint(300, 1000)
                f0 = np.random.uniform(500, 1500)
                f1 = f0 + np.random.uniform(-500, 500)
                t_event = np.arange(event_duration) / self.sampling_rate
                freq_sweep = np.linspace(f0, f1, event_duration)
                phase = 2 * np.pi * np.cumsum(freq_sweep) / self.sampling_rate
                event_signal = np.sin(phase)
                event_signal *= np.exp(-t_event * 8)
            else:  # rattle
                # Short impulsive noise burst
                event_duration = np.random.randint(100, 300)
                event_signal = np.random.randn(event_duration)
                event_signal *= np.exp(-np.arange(event_duration) / 50)
            
            # Add event to signal
            end_position = min(event_position + len(event_signal), num_samples)
            signal_data[event_position:end_position] += event_signal[:end_position - event_position]
        
        # Apply bandpass filter
        signal_data = self._apply_bandpass_filter(signal_data, frequency_range)
        
        return self._normalize(signal_data)
    
    def _generate_speech_like(
        self,
        num_samples: int,
        frequency_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate speech-like signal (simplified model).
        
        This is a simplified model using formant synthesis.
        For production, consider using actual speech samples.
        """
        # Generate excitation signal (voiced/unvoiced)
        t = np.arange(num_samples) / self.sampling_rate
        
        # Fundamental frequency (pitch)
        f0 = np.random.uniform(100, 250)  # Typical speech pitch range
        
        # Generate glottal pulse train
        excitation = np.zeros(num_samples)
        pulse_period = int(self.sampling_rate / f0)
        for i in range(0, num_samples, pulse_period):
            if i < num_samples:
                excitation[i] = 1.0
        
        # Add unvoiced component
        unvoiced = np.random.randn(num_samples) * 0.3
        excitation += unvoiced
        
        # Apply formant filters (simplified - 3 formants)
        formant_freqs = [700, 1200, 2500]  # Typical formant frequencies
        formant_bws = [100, 150, 200]  # Bandwidths
        
        signal_data = excitation.copy()
        for fc, bw in zip(formant_freqs, formant_bws):
            if fc < frequency_range[1]:
                signal_data = self._apply_bandpass_filter(signal_data, (fc - bw, fc + bw))
        
        # Add amplitude modulation (syllable structure)
        modulation_freq = np.random.uniform(3, 6)  # Syllable rate
        modulation = 0.3 + 0.7 * (np.sin(2 * np.pi * modulation_freq * t) + 1) / 2
        signal_data *= modulation
        
        return self._normalize(signal_data)
    
    def _generate_alert_tone(
        self,
        num_samples: int,
        frequency_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate alert/warning tone.
        
        Alert tones are typically pure tones or dual tones with on/off pattern.
        """
        t = np.arange(num_samples) / self.sampling_rate
        
        # Choose tone frequency
        tone_freq = np.random.uniform(frequency_range[0], frequency_range[1])
        
        # Generate base tone
        signal_data = np.sin(2 * np.pi * tone_freq * t)
        
        # Add second tone for dual-tone alert
        if np.random.rand() > 0.5:
            tone_freq2 = tone_freq * np.random.choice([1.25, 1.5, 2.0])
            if tone_freq2 < frequency_range[1]:
                signal_data += np.sin(2 * np.pi * tone_freq2 * t)
        
        # Apply on/off pattern
        pattern_freq = np.random.uniform(2, 5)  # Hz
        pattern = (np.sin(2 * np.pi * pattern_freq * t) > 0).astype(float)
        signal_data *= pattern
        
        return self._normalize(signal_data)
    
    def _generate_white_noise(self, num_samples: int) -> np.ndarray:
        """Generate white noise."""
        return np.random.randn(num_samples)
    
    def _generate_pink_noise(self, num_samples: int) -> np.ndarray:
        """
        Generate pink noise (1/f spectrum).
        
        Uses the Voss-McCartney algorithm.
        """
        # Simple approximation using multiple octave bands
        num_octaves = 16
        pink = np.zeros(num_samples)
        
        for i in range(num_octaves):
            step = 2 ** i
            noise = np.random.randn(num_samples // step + 1)
            noise_upsampled = np.repeat(noise, step)[:num_samples]
            pink += noise_upsampled
        
        return pink / num_octaves
    
    def _apply_bandpass_filter(
        self,
        signal_data: np.ndarray,
        frequency_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Apply bandpass filter to signal.
        
        Args:
            signal_data: Input signal
            frequency_range: Tuple of (low_freq, high_freq) in Hz
            
        Returns:
            Filtered signal
        """
        nyquist = self.sampling_rate / 2
        low = max(frequency_range[0] / nyquist, 0.001)  # Avoid zero
        high = min(frequency_range[1] / nyquist, 0.999)  # Avoid Nyquist
        
        if low >= high:
            return signal_data
        
        # Design Butterworth bandpass filter
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        filtered = signal.sosfilt(sos, signal_data)
        
        return filtered
    
    def _apply_fade(
        self,
        signal_data: np.ndarray,
        fade_duration: float = 0.01
    ) -> np.ndarray:
        """
        Apply fade in/out to avoid clicks.
        
        Args:
            signal_data: Input signal
            fade_duration: Duration of fade in seconds
            
        Returns:
            Signal with fade applied
        """
        fade_samples = int(fade_duration * self.sampling_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        signal_data[:fade_samples] *= fade_in
        signal_data[-fade_samples:] *= fade_out
        
        return signal_data
    
    def _normalize(self, signal_data: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """
        Normalize signal to target level.
        
        Args:
            signal_data: Input signal
            target_level: Target peak level (0.0 to 1.0)
            
        Returns:
            Normalized signal
        """
        max_val = np.max(np.abs(signal_data))
        if max_val > 0:
            return signal_data * (target_level / max_val)
        return signal_data
