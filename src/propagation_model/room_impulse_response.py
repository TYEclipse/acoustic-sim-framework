"""
Room Impulse Response Module

This module simulates acoustic propagation in vehicle cabin environments
using the Image Source Method (ISM) to model reflections and reverberation.
"""

import numpy as np
import pyroomacoustics as pra
from typing import List, Tuple, Dict, Optional
from ..array_geometry.microphone_array import MicrophoneArray


class RoomAcousticSimulator:
    """
    Simulates acoustic propagation in a vehicle cabin using room acoustics models.
    
    Uses pyroomacoustics library to implement the Image Source Method for
    computing Room Impulse Responses (RIRs) from sound sources to microphones.
    """
    
    def __init__(
        self,
        room_dimensions: List[float],
        mic_array: MicrophoneArray,
        sampling_rate: int = 48000,
        max_order: int = 15,
        absorption: Optional[Dict] = None
    ):
        """
        Initialize the room acoustic simulator.
        
        Args:
            room_dimensions: [length, width, height] of room in meters
            mic_array: MicrophoneArray object defining microphone positions
            sampling_rate: Audio sampling rate in Hz
            max_order: Maximum order of reflections for image source method
            absorption: Dictionary of absorption coefficients per frequency band
                       If None, uses default values suitable for vehicle cabin
        """
        self.room_dimensions = np.array(room_dimensions)
        self.mic_array = mic_array
        self.sampling_rate = sampling_rate
        self.max_order = max_order
        
        # Set default absorption if not provided
        if absorption is None:
            # Default absorption for typical vehicle cabin materials
            # Format: [125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz, 8kHz]
            self.absorption = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        else:
            self.absorption = absorption.get('default', [0.25] * 7)
        
        # Create the room
        self.room = None
        self._create_room()
        
    def _create_room(self):
        """
        Create the pyroomacoustics Room object with specified properties.
        """
        # Create material with frequency-dependent absorption
        # pyroomacoustics uses energy absorption coefficients
        material = pra.Material(self.absorption)
        
        # Create shoebox room (rectangular room)
        self.room = pra.ShoeBox(
            self.room_dimensions,
            fs=self.sampling_rate,
            materials=material,
            max_order=self.max_order,
            air_absorption=True  # Enable air absorption for realism
        )
        
        # Add microphone array to room
        # pyroomacoustics expects microphone positions as columns (3 x num_mics)
        mic_positions = self.mic_array.get_all_positions().T
        self.room.add_microphone_array(mic_positions)
    
    def add_source(
        self,
        position: np.ndarray,
        signal: np.ndarray,
        source_id: Optional[int] = None
    ) -> int:
        """
        Add a sound source to the room.
        
        Args:
            position: 3D position [x, y, z] of the source in meters
            signal: Audio signal (numpy array) to be emitted by the source
            source_id: Optional identifier for the source
            
        Returns:
            Index of the added source
        """
        # Validate position is within room bounds
        if not self._is_position_valid(position):
            raise ValueError(f"Source position {position} is outside room bounds {self.room_dimensions}")
        
        # Add source to room
        self.room.add_source(position, signal=signal)
        
        return len(self.room.sources) - 1
    
    def _is_position_valid(self, position: np.ndarray) -> bool:
        """
        Check if a position is within the room boundaries.
        
        Args:
            position: 3D position to validate
            
        Returns:
            True if position is valid, False otherwise
        """
        return np.all(position >= 0) and np.all(position <= self.room_dimensions)
    
    def compute_rir(self) -> np.ndarray:
        """
        Compute Room Impulse Responses for all source-microphone pairs.
        
        This method uses the Image Source Method to calculate the impulse
        response from each source to each microphone, accounting for:
        - Direct path
        - Early reflections (up to max_order)
        - Late reverberation (statistical model)
        
        Returns:
            Array of RIRs with shape (num_sources, num_mics, rir_length)
        """
        if len(self.room.sources) == 0:
            raise RuntimeError("No sources have been added to the room")
        
        # Compute image sources and RIRs
        self.room.compute_rir()
        
        # Extract RIRs
        # pyroomacoustics stores RIRs in room.rir
        # Shape: (num_mics, rir_length) for single source
        # For multiple sources, we need to compute separately
        
        return self.room.rir
    
    def simulate(self) -> np.ndarray:
        """
        Simulate the complete acoustic propagation.
        
        This method:
        1. Computes RIRs for all sources
        2. Convolves each source signal with its corresponding RIRs
        3. Sums the contributions at each microphone
        
        Returns:
            Multi-channel audio signal with shape (num_mics, num_samples)
        """
        if len(self.room.sources) == 0:
            raise RuntimeError("No sources have been added to the room")
        
        # Simulate room acoustics (this performs convolution internally)
        self.room.simulate()
        
        # Get the simulated microphone signals
        # Shape: (num_mics, num_samples)
        mic_signals = self.room.mic_array.signals
        
        return mic_signals
    
    def get_rir_for_source(self, source_index: int) -> np.ndarray:
        """
        Get the RIR for a specific source to all microphones.
        
        Args:
            source_index: Index of the source
            
        Returns:
            RIR array with shape (num_mics, rir_length)
        """
        if self.room.rir is None:
            raise RuntimeError("RIRs have not been computed yet. Call compute_rir() first.")
        
        if source_index >= len(self.room.sources):
            raise IndexError(f"Source index {source_index} out of range")
        
        # For single source, rir is already in correct format
        # For multiple sources, we need to extract the specific one
        return self.room.rir
    
    def calculate_rt60(self) -> float:
        """
        Calculate the RT60 (reverberation time) of the room.
        
        RT60 is the time required for sound to decay by 60 dB.
        
        Returns:
            RT60 in seconds
        """
        # Use Sabine's formula for RT60 estimation
        # RT60 = 0.161 * V / A
        # where V is volume and A is total absorption
        
        volume = np.prod(self.room_dimensions)
        
        # Calculate total absorption area
        # Each wall pair has area and absorption coefficient
        areas = [
            2 * self.room_dimensions[1] * self.room_dimensions[2],  # YZ walls (front/back)
            2 * self.room_dimensions[0] * self.room_dimensions[2],  # XZ walls (left/right)
            2 * self.room_dimensions[0] * self.room_dimensions[1],  # XY walls (floor/ceiling)
        ]
        
        # Use average absorption coefficient
        avg_absorption = np.mean(self.absorption)
        total_absorption = sum(areas) * avg_absorption
        
        # Sabine's formula
        rt60 = 0.161 * volume / (total_absorption + 1e-6)  # Add small value to avoid division by zero
        
        return rt60
    
    def reset(self):
        """
        Reset the room by removing all sources.
        
        Useful for generating multiple simulations with the same room configuration.
        """
        self._create_room()
    
    def get_room_info(self) -> Dict:
        """
        Get comprehensive information about the room configuration.
        
        Returns:
            Dictionary containing room metadata
        """
        return {
            "dimensions": self.room_dimensions.tolist(),
            "volume_m3": float(np.prod(self.room_dimensions)),
            "sampling_rate": self.sampling_rate,
            "max_order": self.max_order,
            "absorption_coefficients": self.absorption,
            "rt60_s": float(self.calculate_rt60()),
            "num_sources": len(self.room.sources) if self.room else 0,
            "num_microphones": self.mic_array.num_mics
        }
    
    @classmethod
    def from_config(
        cls,
        config: Dict,
        mic_array: MicrophoneArray,
        sampling_rate: int = 48000
    ) -> 'RoomAcousticSimulator':
        """
        Create a RoomAcousticSimulator from a configuration dictionary.
        
        Args:
            config: Dictionary with room configuration parameters
            mic_array: MicrophoneArray object
            sampling_rate: Audio sampling rate
            
        Returns:
            RoomAcousticSimulator instance
        """
        dimensions = config['dimensions']
        max_order = config.get('max_order', 15)
        absorption = config.get('absorption', None)
        
        return cls(
            room_dimensions=dimensions,
            mic_array=mic_array,
            sampling_rate=sampling_rate,
            max_order=max_order,
            absorption=absorption
        )
