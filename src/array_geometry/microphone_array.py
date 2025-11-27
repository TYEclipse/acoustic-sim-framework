"""
Microphone Array Geometry Module

This module defines the spatial configuration of the microphone array
and provides utilities for coordinate transformations.
"""

import numpy as np
from typing import List, Tuple, Dict


class MicrophoneArray:
    """
    Represents a microphone array with 3D spatial configuration.
    
    Coordinates are defined in the vehicle coordinate system:
    - X: forward (positive towards front of vehicle)
    - Y: left (positive towards left side)
    - Z: up (positive towards ceiling)
    - Origin: typically at driver's headrest center
    """
    
    def __init__(self, positions: List[List[float]], name: str = "default_array"):
        """
        Initialize microphone array with spatial positions.
        
        Args:
            positions: List of [x, y, z] coordinates for each microphone (in meters)
            name: Descriptive name for the array configuration
        """
        self.name = name
        self.positions = np.array(positions, dtype=np.float64)
        self.num_mics = len(positions)
        
        # Validate array configuration
        if self.num_mics < 2:
            raise ValueError("Array must have at least 2 microphones")
        
        if self.positions.shape[1] != 3:
            raise ValueError("Each microphone position must have 3 coordinates (x, y, z)")
        
        # Calculate array center (geometric center)
        self.center = np.mean(self.positions, axis=0)
        
        # Calculate array aperture (maximum distance between any two mics)
        self.aperture = self._calculate_aperture()
        
    def _calculate_aperture(self) -> float:
        """
        Calculate the maximum distance between any two microphones.
        
        Returns:
            Maximum inter-microphone distance in meters
        """
        max_distance = 0.0
        for i in range(self.num_mics):
            for j in range(i + 1, self.num_mics):
                distance = np.linalg.norm(self.positions[i] - self.positions[j])
                max_distance = max(max_distance, distance)
        return max_distance
    
    def get_mic_position(self, mic_index: int) -> np.ndarray:
        """
        Get the 3D position of a specific microphone.
        
        Args:
            mic_index: Index of the microphone (0-based)
            
        Returns:
            3D position as numpy array [x, y, z]
        """
        if mic_index < 0 or mic_index >= self.num_mics:
            raise IndexError(f"Microphone index {mic_index} out of range [0, {self.num_mics-1}]")
        return self.positions[mic_index]
    
    def get_all_positions(self) -> np.ndarray:
        """
        Get all microphone positions.
        
        Returns:
            Array of shape (num_mics, 3) containing all positions
        """
        return self.positions.copy()
    
    def cartesian_to_spherical(self, point: np.ndarray, reference: np.ndarray = None) -> Tuple[float, float, float]:
        """
        Convert Cartesian coordinates to spherical coordinates.
        
        Args:
            point: 3D point in Cartesian coordinates [x, y, z]
            reference: Reference point (default: array center)
            
        Returns:
            Tuple of (distance, azimuth, elevation) where:
                - distance: radial distance in meters
                - azimuth: angle in degrees (-180 to 180, 0=front, positive=left)
                - elevation: angle in degrees (-90 to 90, 0=horizontal, positive=up)
        """
        if reference is None:
            reference = self.center
        
        # Calculate relative position
        relative = point - reference
        x, y, z = relative
        
        # Calculate spherical coordinates
        distance = np.linalg.norm(relative)
        
        if distance < 1e-6:  # Avoid division by zero
            return 0.0, 0.0, 0.0
        
        # Azimuth: angle in XY plane from X axis (front)
        # atan2(y, x) gives angle from X axis, positive counter-clockwise
        azimuth = np.degrees(np.arctan2(y, x))
        
        # Elevation: angle from XY plane
        elevation = np.degrees(np.arcsin(z / distance))
        
        return distance, azimuth, elevation
    
    def spherical_to_cartesian(
        self,
        distance: float,
        azimuth: float,
        elevation: float,
        reference: np.ndarray = None
    ) -> np.ndarray:
        """
        Convert spherical coordinates to Cartesian coordinates.
        
        Args:
            distance: Radial distance in meters
            azimuth: Azimuth angle in degrees
            elevation: Elevation angle in degrees
            reference: Reference point (default: array center)
            
        Returns:
            3D point in Cartesian coordinates [x, y, z]
        """
        if reference is None:
            reference = self.center
        
        # Convert angles to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Calculate Cartesian coordinates
        x = distance * np.cos(el_rad) * np.cos(az_rad)
        y = distance * np.cos(el_rad) * np.sin(az_rad)
        z = distance * np.sin(el_rad)
        
        return reference + np.array([x, y, z])
    
    def calculate_tdoa(
        self,
        source_position: np.ndarray,
        speed_of_sound: float = 343.0
    ) -> np.ndarray:
        """
        Calculate Time Difference of Arrival (TDOA) for all microphone pairs.
        
        Args:
            source_position: 3D position of sound source
            speed_of_sound: Speed of sound in m/s (default: 343 m/s at 20°C)
            
        Returns:
            Array of TDOAs in seconds, relative to first microphone
        """
        # Calculate distances from source to each microphone
        distances = np.linalg.norm(self.positions - source_position, axis=1)
        
        # Calculate time delays relative to first microphone
        tdoa = (distances - distances[0]) / speed_of_sound
        
        return tdoa
    
    def get_array_info(self) -> Dict:
        """
        Get comprehensive information about the array configuration.
        
        Returns:
            Dictionary containing array metadata
        """
        return {
            "name": self.name,
            "num_microphones": self.num_mics,
            "positions": self.positions.tolist(),
            "center": self.center.tolist(),
            "aperture_m": float(self.aperture),
            "coordinate_system": "vehicle (X:forward, Y:left, Z:up)"
        }
    
    def visualize_array(self) -> str:
        """
        Generate a text-based visualization of the array layout.
        
        Returns:
            String representation of the array
        """
        lines = [f"\nMicrophone Array: {self.name}"]
        lines.append(f"Number of microphones: {self.num_mics}")
        lines.append(f"Array aperture: {self.aperture:.3f} m")
        lines.append(f"Array center: [{self.center[0]:.3f}, {self.center[1]:.3f}, {self.center[2]:.3f}]")
        lines.append("\nMicrophone positions:")
        lines.append("ID |    X (m)  |    Y (m)  |    Z (m)  ")
        lines.append("---|-----------|-----------|----------")
        
        for i, pos in enumerate(self.positions):
            lines.append(f"{i:2d} | {pos[0]:9.3f} | {pos[1]:9.3f} | {pos[2]:9.3f}")
        
        return "\n".join(lines)
    
    @classmethod
    def from_config(cls, config: Dict) -> 'MicrophoneArray':
        """
        Create a MicrophoneArray from a configuration dictionary.
        
        Args:
            config: Dictionary with 'positions' and optional 'name' keys
            
        Returns:
            MicrophoneArray instance
        """
        positions = config['positions']
        name = config.get('name', 'default_array')
        return cls(positions, name)
