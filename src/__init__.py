"""
Acoustic Simulation Framework — utilities for generating synthetic acoustic
data for vehicle noise localization and separation tasks.
"""
__version__ = "1.0.0"
__author__ = "Manus AI"

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 类型检查时可见的符号（运行时不导入）
    from .array_geometry.microphone_array import \
        MicrophoneArray  # type: ignore
    from .data_synthesis.mixer import AudioMixer  # type: ignore
    from .propagation_model.room_impulse_response import \
        RoomAcousticSimulator  # type: ignore
    from .signal_generation.noise_sources import \
        NoiseSourceGenerator  # type: ignore
else:
    _lazy_map = {
        "MicrophoneArray": "array_geometry.microphone_array",
        "AudioMixer": "data_synthesis.mixer",
        "RoomAcousticSimulator": "propagation_model.room_impulse_response",
        "NoiseSourceGenerator": "signal_generation.noise_sources",
    }

    def __getattr__(name: str):
        if name in _lazy_map:
            mod_path = f"{__name__}.{_lazy_map[name]}"
            try:
                module = importlib.import_module(mod_path)
                obj = getattr(module, name)
                globals()[name] = obj  # 缓存以避免重复导入
                return obj
            except Exception as exc:
                raise ImportError(
                    f"Failed to import '{name}' from '{mod_path}' while importing package '{__name__}': {exc}"
                ) from exc
        raise AttributeError(f"module {__name__} has no attribute {name}")

    def __dir__():
        # 将延迟导出的符号包含在 dir() 输出中
        return sorted(set(globals()) | set(_lazy_map.keys()))

__all__ = [
    "NoiseSourceGenerator",
    "RoomAcousticSimulator",
    "MicrophoneArray",
    "AudioMixer",
]
