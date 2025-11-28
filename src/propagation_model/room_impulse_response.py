"""
Room Impulse Response Module

This module simulates acoustic propagation in vehicle cabin environments
using the Image Source Method (ISM) to model reflections and reverberation.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyroomacoustics as pra

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
            absorption: Dictionary/list/scalar of absorption coefficients
        """
        # Basic validation
        self.room_dimensions = np.array(room_dimensions, dtype=float)
        if self.room_dimensions.shape != (3,):
            raise ValueError(
                "room_dimensions must be an iterable of three positive numbers [length, width, height].")
        if np.any(self.room_dimensions <= 0):
            raise ValueError("All room dimension values must be positive.")

        if not hasattr(mic_array, "get_all_positions") or not hasattr(mic_array, "num_mics"):
            raise TypeError(
                "mic_array must implement get_all_positions() and have num_mics attribute.")
        self.mic_array = mic_array

        self.sampling_rate = int(sampling_rate)
        self.max_order = int(max_order)

        # Normalize absorption input: store either np.ndarray of band values or a float scalar.
        default_abs = np.array(
            [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], dtype=float)
        if absorption is None:
            self.absorption: Any = default_abs
        elif isinstance(absorption, dict):
            for k in ("default", "coeffs", "values", "value"):
                if k in absorption:
                    raw = absorption[k]
                    break
            else:
                raw = default_abs
            if isinstance(raw, (list, tuple, np.ndarray)):
                self.absorption = np.asarray(raw, dtype=float)
            else:
                self.absorption = float(raw)
        elif isinstance(absorption, (list, tuple, np.ndarray)):
            self.absorption = np.asarray(absorption, dtype=float)
        else:
            self.absorption = float(absorption)

        # pyroomacoustics Room object
        self.room: Optional[pra.Room] = None
        # Map from internal source index -> user provided source_id (if any)
        self._source_id_map: Dict[int, Any] = {}
        self._create_room()

    def _create_room(self):
        """
        Create the pyroomacoustics Room object with specified properties.
        """
        # Prepare absorption values (either ndarray of floats or scalar)
        if isinstance(self.absorption, np.ndarray):
            # 将非有限值替换为默认值并裁剪到合法区间
            arr = np.nan_to_num(self.absorption, nan=0.25,
                                posinf=0.25, neginf=0.25)
            arr = np.clip(arr, 0.0, 0.99)
            abs_for_material = arr.tolist() if arr.size > 0 else [0.25]
        else:
            abs_for_material = float(self.absorption)

        # 尝试使用给定参数创建 Material，失败时退回为标量平均值
        try:
            material = pra.Material(abs_for_material)
        except Exception:
            if isinstance(abs_for_material, (list, tuple, np.ndarray)):
                avg = float(np.mean(abs_for_material))
            else:
                avg = float(abs_for_material)
            avg = max(0.0, min(0.99, avg))
            material = pra.Material(avg)

        # Create shoebox room
        self.room = pra.ShoeBox(
            self.room_dimensions.tolist(),
            fs=self.sampling_rate,
            materials=material,
            max_order=self.max_order,
            air_absorption=True
        )

        # Add microphone array robustly
        raw_positions = self.mic_array.get_all_positions()
        mp_arr = np.asarray(raw_positions, dtype=float)
        # Accept either (N,3) or (3,N)
        if mp_arr.ndim != 2 or (3 not in mp_arr.shape):
            raise ValueError(
                "get_all_positions() must return a 2D array-like with shape (N,3) or (3,N).")

        # normalize to shape (3, M)
        if mp_arr.shape[0] == 3:
            mp = mp_arr.copy()
        else:
            mp = mp_arr.T.copy()

        # sanitize and convert: replace non-finite with room center, convert to corner coords, clip
        half = self.room_dimensions / 2.0
        finite_mask = np.isfinite(mp)
        mp = np.where(finite_mask, mp, half[:, None])

        # convert and clip per-column (保留逐列处理以保持对 _to_room_coords 的兼容)
        for i in range(mp.shape[1]):
            mp[:, i] = np.clip(self._to_room_coords(
                mp[:, i]), 0.0, self.room_dimensions)

        if mp.shape[1] == 0:
            raise RuntimeError(
                "No microphone positions available from mic_array.")

        # If mic_array advertises num_mics, ensure consistency with provided positions.
        if hasattr(self.mic_array, "num_mics"):
            try:
                reported = int(self.mic_array.num_mics)
            except Exception:
                reported = None
            if reported is not None and reported != mp.shape[1]:
                raise ValueError(
                    f"mic_array.num_mics ({reported}) does not match number of positions provided ({mp.shape[1]}).")

        pra_mic_array = pra.MicrophoneArray(mp, fs=self.sampling_rate)
        self.room.add_microphone_array(pra_mic_array)

    def add_source(
        self,
        position: np.ndarray,
        signal: np.ndarray,
        source_id: Optional[int] = None
    ) -> int:
        """
        Add a sound source to the room.
        """
        pos = np.asarray(position, dtype=float)
        if pos.shape != (3,):
            raise ValueError("position must be a 3-element iterable [x,y,z].")

        if not self._is_position_valid(pos):
            raise ValueError(
                f"Source position {pos} is outside room bounds {self.room_dimensions}")

        pos_room = self._to_room_coords(pos)
        # pyroomacoustics expects 3-element sequence
        self.room.add_source(
            pos_room.tolist(), signal=np.asarray(signal, dtype=float))
        idx = len(self.room.sources) - 1
        # 记录用户指定的 source_id（如果有）
        if source_id is not None:
            self._source_id_map[idx] = source_id
        return idx

    def _is_position_valid(self, position: np.ndarray) -> bool:
        """
        Check if a position is within the room boundaries.
        Accepts corner-based [0, dims] or center-based [-dims/2, dims/2].
        """
        pos = np.asarray(position, dtype=float)
        if np.all(pos >= 0.0) and np.all(pos <= self.room_dimensions):
            return True
        half = self.room_dimensions / 2.0
        if np.all(pos >= -half) and np.all(pos <= half):
            return True
        return False

    def compute_rir(self) -> List[List[np.ndarray]]:
        """
        Compute Room Impulse Responses for all source-microphone pairs and
        return list-per-source of list-per-microphone RIR arrays.
        """
        if self.room is None or len(self.room.sources) == 0:
            raise RuntimeError("No sources have been added to the room")

        self.room.compute_rir()

        # room.rir is room.rir[mic_idx][src_idx]
        num_mics = len(self.room.rir)
        num_sources = len(self.room.sources)
        rirs_by_source: List[List[np.ndarray]] = []
        for s in range(num_sources):
            per_mic = []
            for m in range(num_mics):
                try:
                    per_mic.append(self.room.rir[m][s])
                except Exception:
                    raise RuntimeError(
                        "Unexpected room.rir structure while assembling RIRs.")
            rirs_by_source.append(per_mic)
        return rirs_by_source

    def simulate(self) -> np.ndarray:
        """
        Simulate the complete acoustic propagation and return mic signals
        as ndarray with shape (num_mics, num_samples) when available.
        """
        if self.room is None or len(self.room.sources) == 0:
            raise RuntimeError("No sources have been added to the room")

        self.room.simulate()
        mic_signals = self.room.mic_array.signals
        return np.asarray(mic_signals)

    def get_rir_for_source(self, source_index: int) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Return per-microphone RIRs for a given source index.
        """
        if self.room is None or not hasattr(self.room, "rir") or not self.room.rir:
            raise RuntimeError(
                "RIRs have not been computed yet. Call compute_rir() first.")

        if source_index < 0 or source_index >= len(self.room.sources):
            raise IndexError(f"Source index {source_index} out of range")

        rirs_for_source = [self.room.rir[m][source_index]
                           for m in range(len(self.room.rir))]
        # 如果所有 RIR 都是 ndarray 并且长度一致，则返回堆叠后的 ndarray（便于后续处理）
        if all(isinstance(r, np.ndarray) for r in rirs_for_source):
            lengths = {r.shape[0] for r in rirs_for_source}
            if len(lengths) == 1:
                return np.vstack(rirs_for_source)
        return rirs_for_source

    def calculate_rt60(self) -> float:
        """
        Estimate RT60 using Sabine's formula: RT60 = 0.161 * V / A
        """
        volume = float(np.prod(self.room_dimensions))
        total_area = 2.0 * (self.room_dimensions[0] * self.room_dimensions[1] +
                            self.room_dimensions[0] * self.room_dimensions[2] +
                            self.room_dimensions[1] * self.room_dimensions[2])
        # 根据 absorption 的类型计算平均吸收系数
        if isinstance(self.absorption, np.ndarray):
            avg_absorption = float(np.mean(self.absorption))
        else:
            avg_absorption = float(self.absorption)
        effective_abs = max(1e-6, avg_absorption)
        rt60 = 0.161 * volume / (total_area * effective_abs)
        return float(rt60)

    def reset(self):
        """
        Reset room (recreate pyroomacoustics Room and re-add mic array).
        """
        self._create_room()

    def get_room_info(self) -> Dict[str, Any]:
        """
        Return room metadata dict.
        """
        info = {
            "dimensions": self.room_dimensions.tolist(),
            "volume_m3": float(np.prod(self.room_dimensions)),
            "sampling_rate": self.sampling_rate,
            "max_order": self.max_order,
            "absorption_coefficients": self.absorption,
            "rt60_s": float(self.calculate_rt60()),
            "num_sources": len(self.room.sources) if self.room else 0,
            "num_microphones": int(self.mic_array.num_mics)
        }
        # 添加可选的 source id 映射，便于追踪用户外部 id
        if self._source_id_map:
            info["source_id_map"] = dict(self._source_id_map)
        return info

    @classmethod
    def from_config(
        cls,
        config: Dict,
        mic_array: MicrophoneArray,
        sampling_rate: int = 48000
    ) -> 'RoomAcousticSimulator':
        """
        Create a RoomAcousticSimulator from a configuration dictionary.
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

    def _to_room_coords(self, position: np.ndarray) -> np.ndarray:
        """
        Convert a 3D position to corner-based coordinates expected by ShoeBox.

        Behavior:
        - If position already in [0, dims], return as-is.
        - Else if position in [-half, half], treat as center-based and shift by +half.
        - Otherwise clip to [0, dims].
        """
        pos = np.asarray(position, dtype=float)
        half = self.room_dimensions / 2.0

        # Prefer corner-based interpretation first
        if np.all(pos >= -1e-9) and np.all(pos <= self.room_dimensions + 1e-9):
            return pos.astype(float)

        # Then center-based
        if np.all(pos >= -half - 1e-9) and np.all(pos <= half + 1e-9):
            return (pos + half).astype(float)

        # Fallback: clip
        return np.clip(pos, 0.0, self.room_dimensions).astype(float)
