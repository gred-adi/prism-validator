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
    """Safely loads a YAML file that may contain NumPy data types.

    Args:
        file_path (Union[str, Path]): The path to the YAML file.

    Returns:
        Dict[str, Any]: The loaded YAML data as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If there is an error loading the YAML file.
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
    """Converts NumPy types in a dictionary to native Python types.

    This function recursively traverses a dictionary and converts any NumPy
    numeric types to their native Python equivalents to ensure YAML
    compatibility.

    Args:
        data_dict (Dict[Any, Any]): The dictionary to convert.

    Returns:
        Dict[Any, Any]: The converted dictionary.
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
    """Converts timestamp and timedelta objects in a dictionary to strings.

    This function is used to prepare a dictionary for YAML serialization by
    converting datetime and timedelta objects to a human-readable string
    format.

    Args:
        data_stats (Dict[Any, Any]): The dictionary to convert.

    Returns:
        Dict[Any, Any]: The converted dictionary.
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
