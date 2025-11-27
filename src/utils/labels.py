"""
Label Generation Utilities

This module provides utilities for generating and managing ground truth labels
for simulated acoustic data.
"""

import json
import yaml
import os
from typing import Dict, List, Any
from datetime import datetime


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
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'")
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
    
    def save_label(
        self,
        label: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Save label to file.
        
        Args:
            label: Label dictionary
            output_path: Output file path (extension will be added if not present)
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Add extension if not present
        if self.format == "json" and not output_path.endswith('.json'):
            output_path += '.json'
        elif self.format == "yaml" and not output_path.endswith('.yaml'):
            output_path += '.yaml'
        
        # Write label file
        if self.format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(label, f, indent=2, ensure_ascii=False)
        else:  # yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(label, f, default_flow_style=False, allow_unicode=True)
    
    def load_label(self, label_path: str) -> Dict[str, Any]:
        """
        Load label from file.
        
        Args:
            label_path: Path to label file
            
        Returns:
            Label dictionary
        """
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        # Determine format from extension
        if label_path.endswith('.json'):
            with open(label_path, 'r', encoding='utf-8') as f:
                label = json.load(f)
        elif label_path.endswith('.yaml') or label_path.endswith('.yml'):
            with open(label_path, 'r', encoding='utf-8') as f:
                label = yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown label file format: {label_path}")
        
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
        
        # Validate sources
        if not isinstance(label["sources"], list):
            raise ValueError("'sources' must be a list")
        
        for i, source in enumerate(label["sources"]):
            required_source_fields = ["source_id", "label", "position_xyz"]
            for field in required_source_fields:
                if field not in source:
                    raise ValueError(f"Source {i} missing required field: {field}")
            
            # Validate position format
            if not isinstance(source["position_xyz"], list) or len(source["position_xyz"]) != 3:
                raise ValueError(f"Source {i} position_xyz must be a list of 3 numbers")
        
        return True


def create_dataset_manifest(
    dataset_dir: str,
    label_files: List[str],
    output_path: str = None
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
            manifest["samples"].append({
                "clip_id": label["clip_id"],
                "label_path": label_file,
                "audio_path": label["audio_filepath"],
                "num_sources": len(label["sources"]),
                "source_types": [s["label"] for s in label["sources"]]
            })
        except Exception as e:
            print(f"Warning: Failed to process {label_file}: {e}")
    
    # Save manifest if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return manifest


def extract_source_statistics(label_files: List[str]) -> Dict[str, Any]:
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
            
            for source in label["sources"]:
                # Count source types
                source_type = source["label"]
                source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
                total_sources += 1
                
                # Track position ranges
                pos = source["position_xyz"]
                position_ranges['x']['min'] = min(position_ranges['x']['min'], pos[0])
                position_ranges['x']['max'] = max(position_ranges['x']['max'], pos[0])
                position_ranges['y']['min'] = min(position_ranges['y']['min'], pos[1])
                position_ranges['y']['max'] = max(position_ranges['y']['max'], pos[1])
                position_ranges['z']['min'] = min(position_ranges['z']['min'], pos[2])
                position_ranges['z']['max'] = max(position_ranges['z']['max'], pos[2])
        
        except Exception as e:
            print(f"Warning: Failed to process {label_file}: {e}")
    
    return {
        "total_sources": total_sources,
        "source_type_distribution": source_type_counts,
        "position_ranges": position_ranges
    }
