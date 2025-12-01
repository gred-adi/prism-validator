"""
qa_yaml_utils.py

Utility functions for YAML handling, and type conversion.
"""

import yaml
from pathlib import Path
from typing import Union, Dict, Any


class NumpyYAMLSafeLoader(yaml.SafeLoader):
    """Custom YAML Loader that can handle NumPy data types."""
    pass


def safe_load_numpy_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely load YAML files containing NumPy data types.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the YAML file.
        
    Returns
    -------
    dict
        Loaded YAML data with NumPy values converted to Python native types.
    
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If an error occurs while loading the YAML file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r') as file:
        try:
            return yaml.load(file, Loader=NumpyYAMLSafeLoader)
        except Exception as e:
            raise ValueError(f"Error loading YAML file: {str(e)}")


def convert_numpy_for_yaml(data_dict: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Convert NumPy types to native Python types for YAML compatibility.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing NumPy values.
        
    Returns
    -------
    dict
        Dictionary with YAML-safe Python native types.
    """
    converted_dict: Dict[Any, Any] = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            converted_dict[key] = convert_numpy_for_yaml(value)
        else:
            if hasattr(value, 'item'):
                converted_dict[key] = value.item()
            else:
                converted_dict[key] = value
    return converted_dict


def convert_timestamps_for_yaml(data_stats: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Convert timestamp data to YAML-safe string format.
    
    Parameters
    ----------
    data_stats : dict
        Dictionary containing timestamp data.
        
    Returns
    -------
    dict
        Dictionary with timestamp data converted to string format.
    """
    def convert_value(value: Any) -> str:
        if hasattr(value, 'isoformat'):
            return value.isoformat()
        elif hasattr(value, 'total_seconds'):
            days = value.days
            hours = value.seconds // 3600
            minutes = (value.seconds % 3600) // 60
            seconds = value.seconds % 60
            return f"{days}d {hours}h {minutes}m {seconds}s"
        return str(value)
    
    converted_stats: Dict[Any, Any] = {}
    for key, subsection in data_stats.items():
        converted_stats[key] = {
            subkey: convert_value(val)
            for subkey, val in subsection.items()
        }
    
    return converted_stats
