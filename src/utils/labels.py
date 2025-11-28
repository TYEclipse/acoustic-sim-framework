"""
Label Generation Utilities

This module provides utilities for generating and managing ground truth labels
for simulated acoustic data.
"""

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    Generates ground truth labels for simulated acoustic data.

    Labels include:
    - Source positions and orientations
    - Source types/categories
    - Clean source signal paths
    - Room acoustic properties
    - Microphone array configuration
    """

    def __init__(self, format: str = "json"):
        """
        Initialize label generator.

        Args:
            format: Output format ('json' or 'yaml')
        """
        if format not in ['json', 'yaml']:
            raise ValueError(
                f"Unsupported format: {format}. Use 'json' or 'yaml'")
        self.format = format

    def create_label(
        self,
        clip_id: str,
        audio_filepath: str,
        sources: List[Dict[str, Any]],
        mic_array_config: Dict[str, Any],
        room_properties: Dict[str, Any],
        sampling_rate: int,
        bit_depth: int,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a complete label dictionary for a simulation.

        Args:
            clip_id: Unique identifier for this clip
            audio_filepath: Path to the mixed audio file
            sources: List of source dictionaries, each containing:
                - source_id: int
                - label: str (source type)
                - position_xyz: [x, y, z]
                - orientation_az_el: [azimuth, elevation] (optional)
                - clean_signal_path: str
            mic_array_config: Microphone array configuration
            room_properties: Room acoustic properties
            sampling_rate: Audio sampling rate
            bit_depth: Audio bit depth
            additional_metadata: Optional additional metadata

        Returns:
            Complete label dictionary
        """
        label = {
            "clip_id": clip_id,
            "audio_filepath": audio_filepath,
            "sampling_rate": sampling_rate,
            "bit_depth": bit_depth,
            "timestamp": datetime.now().isoformat(),
            "mic_array_setup": mic_array_config,
            "room_properties": room_properties,
            "sources": sources
        }

        # Add additional metadata if provided
        if additional_metadata:
            label.update(additional_metadata)

        return label

    # serialize helper: convert Paths, numpy types, bytes, etc. into JSON/YAML-friendly types
    def _serialize_for_output(self, obj: Any) -> Any:
        # Path -> str
        if isinstance(obj, Path):
            return str(obj)

        # numpy ndarray -> recurse via list
        if isinstance(obj, np.ndarray):
            return self._serialize_for_output(obj.tolist())

        # numpy scalar -> native python using item()
        if isinstance(obj, np.generic):
            val = obj.item()
            # For floats, guard NaN/inf
            if isinstance(val, float) and not math.isfinite(val):
                return None
            return val

        # bytes -> utf-8 string fallback to str()
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode("utf-8")
            except Exception:
                return str(obj)

        # dict / list / tuple -> recurse
        if isinstance(obj, dict):
            return {k: self._serialize_for_output(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serialize_for_output(v) for v in obj]

        # basic python types (guard floats)
        if isinstance(obj, float):
            if not math.isfinite(obj):
                return None
            return obj
        if isinstance(obj, (str, int, bool, type(None))):
            return obj

        # fallback: string representation
        return str(obj)

    def save_label(
        self,
        label: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """
        Save label to file.

        Args:
            label: Label dictionary
            output_path: Output file path (extension will be added if not present)
        """
        out_p = Path(output_path)
        suffix = out_p.suffix.lower()
        if self.format == "json":
            if suffix != ".json":
                out_p = out_p.with_suffix(".json")
        else:
            if suffix not in (".yaml", ".yml"):
                out_p = out_p.with_suffix(".yaml")

        parent = out_p.parent
        parent.mkdir(parents=True, exist_ok=True)

        label_to_write = self._serialize_for_output(label)

        # Write label file atomically: create temp file next to target then replace
        # use name-based tmp to avoid odd behaviors with empty suffix
        temp_out = out_p.with_name(out_p.name + ".tmp")
        if self.format == "json":
            with temp_out.open("w", encoding="utf-8") as f:
                json.dump(label_to_write, f, indent=2, ensure_ascii=False)
        else:
            with temp_out.open("w", encoding="utf-8") as f:
                yaml.safe_dump(label_to_write, f, default_flow_style=False,
                               allow_unicode=True, sort_keys=False)
        try:
            os.replace(str(temp_out), str(out_p))
        except Exception:
            if temp_out.exists():
                temp_out.rename(out_p)

    def load_label(self, label_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load label from file.

        Args:
            label_path: Path to label file

        Returns:
            Label dictionary
        """
        p = Path(label_path)
        if not p.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # Determine format from extension
        suffix = p.suffix.lower()
        if suffix == '.json':
            with p.open('r', encoding='utf-8') as f:
                label = json.load(f)
        elif suffix in ('.yaml', '.yml'):
            with p.open('r', encoding='utf-8') as f:
                label = yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown label file format: {label_path}")

        # Guard against empty YAML/JSON files which may return None
        if label is None:
            label = {}
        elif not isinstance(label, dict):
            # Ensure we return a dict for downstream consumers; if not, log and wrap
            logger.warning(
                "Label file %s did not contain a mapping at top-level; wrapping into dict", label_path)
            label = {"__value__": label}

        return label

    def create_source_entry(
        self,
        source_id: int,
        label: str,
        position_xyz: List[float],
        clean_signal_path: str,
        orientation_az_el: List[float] = None,
        additional_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a source entry for the label.

        Args:
            source_id: Unique identifier for this source
            label: Source type/category
            position_xyz: 3D position [x, y, z] in meters
            clean_signal_path: Path to clean source signal file
            orientation_az_el: Optional [azimuth, elevation] in degrees
            additional_info: Optional additional information

        Returns:
            Source entry dictionary
        """
        source_entry = {
            "source_id": source_id,
            "label": label,
            "position_xyz": position_xyz,
            "clean_signal_path": clean_signal_path
        }

        if orientation_az_el is not None:
            source_entry["orientation_az_el"] = orientation_az_el

        if additional_info:
            source_entry.update(additional_info)

        return source_entry

    def validate_label(self, label: Dict[str, Any]) -> bool:
        """
        Validate that a label contains all required fields.

        Args:
            label: Label dictionary to validate

        Returns:
            True if valid, raises ValueError if invalid
        """
        required_fields = [
            "clip_id",
            "audio_filepath",
            "sampling_rate",
            "mic_array_setup",
            "room_properties",
            "sources"
        ]

        for field in required_fields:
            if field not in label:
                raise ValueError(f"Missing required field: {field}")

        # 基础类型检查（增强）
        if not isinstance(label["sampling_rate"], int) or label["sampling_rate"] <= 0:
            raise ValueError("'sampling_rate' must be a positive integer")
        if "bit_depth" in label and not isinstance(label["bit_depth"], int):
            raise ValueError("'bit_depth' must be integer if provided")

        # Validate sources
        if not isinstance(label["sources"], list):
            raise ValueError("'sources' must be a list")

        for i, source in enumerate(label["sources"]):
            required_source_fields = ["source_id", "label", "position_xyz"]
            for field in required_source_fields:
                if field not in source:
                    raise ValueError(
                        f"Source {i} missing required field: {field}")

            # Validate position format
            if not isinstance(source["position_xyz"], (list, tuple)) or len(source["position_xyz"]) != 3:
                raise ValueError(
                    f"Source {i} position_xyz must be a list/tuple of 3 numbers")
            # Validate numeric entries
            for coord in source["position_xyz"]:
                if not isinstance(coord, (int, float)):
                    raise ValueError(
                        f"Source {i} position_xyz must contain numeric values")

        return True


def create_dataset_manifest(
    dataset_dir: str,
    label_files: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Create a manifest file for the entire dataset.

    Args:
        dataset_dir: Root directory of the dataset
        label_files: List of label file paths
        output_path: Optional path to save manifest (if None, returns dict only)

    Returns:
        Manifest dictionary
    """
    manifest = {
        "dataset_dir": dataset_dir,
        "num_samples": len(label_files),
        "created": datetime.now().isoformat(),
        "samples": []
    }

    # Collect information from each label file
    label_gen = LabelGenerator()
    for label_file in label_files:
        try:
            label = label_gen.load_label(label_file)
            if not isinstance(label, dict):
                logger.warning(
                    "Skipping non-mapping label from %s", label_file)
                continue
            clip_id = label.get("clip_id")
            if clip_id is None:
                logger.warning("Missing clip_id in %s", label_file)
            manifest["samples"].append({
                "clip_id": clip_id,
                "label_path": str(Path(label_file)),
                "audio_path": label.get("audio_filepath"),
                "num_sources": len(label.get("sources", [])),
                "source_types": [s.get("label") for s in label.get("sources", [])]
            })
        except Exception as e:
            logger.warning("Failed to process %s: %s", label_file, e)

    # Save manifest if output path provided
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open('w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def extract_source_statistics(label_files: List[Union[str, Path]]) -> Dict[str, Any]:
    """
    Extract statistics about sources from a list of label files.

    Args:
        label_files: List of label file paths

    Returns:
        Dictionary containing statistics
    """
    label_gen = LabelGenerator()

    source_type_counts = {}
    total_sources = 0
    position_ranges = {
        'x': {'min': float('inf'), 'max': float('-inf')},
        'y': {'min': float('inf'), 'max': float('-inf')},
        'z': {'min': float('inf'), 'max': float('-inf')}
    }

    for label_file in label_files:
        try:
            label = label_gen.load_label(label_file)
            if not isinstance(label, dict):
                logger.warning(
                    "Skipping non-mapping label from %s", label_file)
                continue

            for source in label.get("sources", []):
                # Validate position_xyz is usable
                pos = source.get("position_xyz")
                if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                    logger.warning(
                        "Skipping source with invalid position_xyz in %s", label_file)
                    continue
                if not all(isinstance(v, (int, float)) for v in pos):
                    logger.warning(
                        "Skipping source with non-numeric position in %s", label_file)
                    continue

                # Count source types
                source_type = source.get("label", "unknown")
                source_type_counts[source_type] = source_type_counts.get(
                    source_type, 0) + 1
                total_sources += 1

                # Track position ranges
                position_ranges['x']['min'] = min(
                    position_ranges['x']['min'], pos[0])
                position_ranges['x']['max'] = max(
                    position_ranges['x']['max'], pos[0])
                position_ranges['y']['min'] = min(
                    position_ranges['y']['min'], pos[1])
                position_ranges['y']['max'] = max(
                    position_ranges['y']['max'], pos[1])
                position_ranges['z']['min'] = min(
                    position_ranges['z']['min'], pos[2])
                position_ranges['z']['max'] = max(
                    position_ranges['z']['max'], pos[2])

        except Exception as e:
            logger.warning("Failed to process %s: %s", label_file, e)

    # 如果没有有效 source，则返回更合适的空值
    if total_sources == 0:
        position_ranges = None

    return {
        "total_sources": total_sources,
        "source_type_distribution": source_type_counts,
        "position_ranges": position_ranges
    }
