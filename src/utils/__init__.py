"""Utility modules for audio I/O and label generation."""

from .audio_io import (
    read_audio,
    write_audio,
    get_audio_info,
    create_multichannel_audio,
    split_multichannel_audio,
    calculate_rms,
    calculate_peak,
    db_to_linear,
    linear_to_db
)
from .labels import (
    LabelGenerator,
    create_dataset_manifest,
    extract_source_statistics
)

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
