"""Utility modules for audio I/O and label generation."""

from importlib import import_module
from types import ModuleType
from typing import Any, Dict, List

# 显式列出对外暴露的 API（保持原有导出列表以兼容现有代码）
__all__ = [
    "read_audio",
    "write_audio",
    "get_audio_info",
    "create_multichannel_audio",
    "split_multichannel_audio",
    "calculate_rms",
    "calculate_peak",
    "db_to_linear",
    "linear_to_db",
    "LabelGenerator",
    "create_dataset_manifest",
    "extract_source_statistics"
]

# 映射符号名到子模块名（相对模块名）
_lazy_mapping = {
    # 来自 audio_io
    "read_audio": "audio_io",
    "write_audio": "audio_io",
    "get_audio_info": "audio_io",
    "create_multichannel_audio": "audio_io",
    "split_multichannel_audio": "audio_io",
    "calculate_rms": "audio_io",
    "calculate_peak": "audio_io",
    "db_to_linear": "audio_io",
    "linear_to_db": "audio_io",
    # 来自 labels
    "LabelGenerator": "labels",
    "create_dataset_manifest": "labels",
    "extract_source_statistics": "labels",
}

# 缓存已导入的子模块，避免对同一子模块重复 import_module
_module_cache: Dict[str, ModuleType] = {}


def _import_attr(name: str) -> Any:
    """按需导入并返回属性；导入后缓存到 globals() 以避免重复解析属性。"""
    if name not in _lazy_mapping:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submodule = _lazy_mapping[name]
    # 使用模块缓存避免对同一子模块重复 import_module
    mod = _module_cache.get(submodule)
    if mod is None:
        try:
            mod = import_module(f".{submodule}", package=__package__)
        except Exception as exc:
            raise ImportError(
                f"Failed to import submodule '{submodule}' for attribute '{name}': {exc}"
            ) from exc
        _module_cache[submodule] = mod
    try:
        attr = getattr(mod, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module '{mod.__name__}' has no attribute '{name}'") from exc
    # 缓存到当前模块命名空间，下一次直接使用
    globals()[name] = attr
    return attr


def __getattr__(name: str) -> Any:
    """模块级延迟导入入口（PEP 562）。"""
    return _import_attr(name)


def __dir__() -> List[str]:
    """确保 dir(utils) 包含延迟导入的公开符号，改善交互体验和自动补全。"""
    names = set(globals().keys())
    names.update(__all__)
    return sorted(names)
