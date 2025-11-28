"""
Microphone Array Geometry Module

This module defines the spatial configuration of the microphone array
and provides utilities for coordinate transformations.
"""

from typing import Dict, List, Tuple

import numpy as np


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

        # Validate and normalize positions early with helpful errors
        arr = np.asarray(positions, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("positions must be an iterable of shape (N, 3)")
        self.positions = arr.copy()
        self.num_mics = int(self.positions.shape[0])

        # Validate array configuration
        if self.num_mics < 2:
            raise ValueError("Array must have at least 2 microphones")

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
        # Use a vectorized, numerically stable squared-distance computation:
        # ||p_i - p_j||^2 = ||p_i||^2 + ||p_j||^2 - 2 p_i·p_j
        P = self.positions
        # For small n, vectorized method is fine; for very large n, warn or fallback if needed.
        norms = np.sum(P * P, axis=1)  # shape (n,)
        # Compute matrix of squared distances
        sq_dists = norms[:, None] + norms[None, :] - 2.0 * (P @ P.T)
        # Numerical errors can produce tiny negative values; clamp to zero
        sq_dists = np.maximum(sq_dists, 0.0)
        # The diagonal is zero; max of matrix is aperture^2
        aperture_sq = np.max(sq_dists)
        return float(np.sqrt(aperture_sq))

    def get_mic_position(self, mic_index: int) -> np.ndarray:
        """
        Get the 3D position of a specific microphone.

        Args:
            mic_index: Index of the microphone (0-based)

        Returns:
            3D position as numpy array [x, y, z]
        """
        if mic_index < 0 or mic_index >= self.num_mics:
            raise IndexError(
                f"Microphone index {mic_index} out of range [0, {self.num_mics-1}]")
        # return a copy to avoid external mutation of internal state
        return self.positions[mic_index].copy()

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
            reference: Reference point (default: world origin [0,0,0])

        Returns:
            Tuple of (distance, azimuth, elevation) where:
                - distance: radial distance in meters
                - azimuth: angle in degrees (-180 to 180, 0=front, positive=left)
                - elevation: angle in degrees (-90 to 90, 0=horizontal, positive=up)
        """
        pt = np.asarray(point, dtype=float)
        if pt.shape != (3,):
            raise ValueError("point must be a length-3 iterable (x,y,z)")

        if reference is None:
            reference = np.zeros(3, dtype=float)
        ref = np.asarray(reference, dtype=float)
        if ref.shape != (3,):
            raise ValueError("reference must be a length-3 iterable (x,y,z)")

        # Calculate relative position
        relative = pt - ref
        x, y, z = relative

        # Calculate spherical coordinates
        distance = float(np.linalg.norm(relative))

        if distance < 1e-12:  # Avoid division by zero and treat as origin
            return 0.0, 0.0, 0.0

        # Azimuth: angle in XY plane from X axis (front)
        azimuth = float(np.degrees(np.arctan2(y, x)))
        # normalize azimuth to [-180, 180]
        azimuth = ((azimuth + 180.0) % 360.0) - 180.0

        # Elevation: use atan2 for better numerical stability:
        # elevation = atan2(z, sqrt(x^2 + y^2))
        horiz_dist = np.hypot(x, y)
        elevation = float(np.degrees(np.arctan2(z, horiz_dist)))

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
            distance: Radial distance in meters (must be >= 0)
            azimuth: Azimuth angle in degrees
            elevation: Elevation angle in degrees
            reference: Reference point (default: world origin [0,0,0])

        Returns:
            3D point in Cartesian coordinates [x, y, z]
        """
        if distance < 0:
            raise ValueError("distance must be a non-negative scalar")

        if reference is None:
            reference = np.zeros(3, dtype=float)
        ref = np.asarray(reference, dtype=float)
        if ref.shape != (3,):
            raise ValueError("reference must be a length-3 iterable (x,y,z)")

        # Convert angles to radians
        az_rad = np.radians(float(azimuth))
        el_rad = np.radians(float(elevation))

        # Calculate Cartesian coordinates
        x = distance * np.cos(el_rad) * np.cos(az_rad)
        y = distance * np.cos(el_rad) * np.sin(az_rad)
        z = distance * np.sin(el_rad)

        return ref + np.array([x, y, z], dtype=float)

    def calculate_tdoa(
        self,
        source_position: np.ndarray,
        speed_of_sound: float = 343.0,
        reference_index: int = 0
    ) -> np.ndarray:
        """
        Calculate Time Difference of Arrival (TDOA) for all microphones relative to a reference mic.

        Args:
            source_position: 3D position of sound source
            speed_of_sound: Speed of sound in m/s (must be > 0)
            reference_index: index of microphone used as time reference (default 0)

        Returns:
            Array of TDOAs in seconds of shape (num_mics,), delays = (distances - distances[reference_index]) / speed_of_sound
        """
        if speed_of_sound <= 0:
            raise ValueError("speed_of_sound must be positive")

        if reference_index < 0 or reference_index >= self.num_mics:
            raise IndexError("reference_index out of range")

        src = np.asarray(source_position, dtype=float)
        if src.shape != (3,):
            raise ValueError(
                "source_position must be a length-3 iterable (x,y,z)")

        # Calculate distances from source to each microphone
        distances = np.linalg.norm(self.positions - src, axis=1)

        # Calculate time delays relative to reference microphone
        tdoa = (distances - distances[reference_index]) / float(speed_of_sound)

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
        lines.append(
            f"Array center: [{self.center[0]:.3f}, {self.center[1]:.3f}, {self.center[2]:.3f}]")
        lines.append("\nMicrophone positions:")
        lines.append("ID |    X (m)  |    Y (m)  |    Z (m)  ")
        lines.append("---|-----------|-----------|----------")

        for i, pos in enumerate(self.positions):
            lines.append(
                f"{i:2d} | {pos[0]:9.3f} | {pos[1]:9.3f} | {pos[2]:9.3f}")

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
        if 'positions' not in config:
            raise KeyError(
                "config must contain a 'positions' key (iterable of shape (N,3))")
        positions = config['positions']
        name = config.get('name', 'default_array')
        return cls(positions, name)

    def __repr__(self) -> str:
        return f"MicrophoneArray(name={self.name!r}, num_mics={self.num_mics}, aperture_m={self.aperture:.4f})"
