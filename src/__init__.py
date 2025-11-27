"""
Acoustic Simulation Framework

A highly configurable framework for generating synthetic acoustic data
for training deep learning models in vehicle noise source localization
and separation tasks.
"""

__version__ = "1.0.0"
__author__ = "Manus AI"

from .signal_generation.noise_sources import NoiseSourceGenerator
from .propagation_model.room_impulse_response import RoomAcousticSimulator
from .array_geometry.microphone_array import MicrophoneArray
from .data_synthesis.mixer import AudioMixer

__all__ = [
    "NoiseSourceGenerator",
    "RoomAcousticSimulator",
    "MicrophoneArray",
    "AudioMixer",
]
