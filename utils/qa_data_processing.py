"""
qa_data_processing.py

Functions for loading and processing data.
"""

import os
from typing import List
import pandas as pd
import numpy as np

from utils.data_preparation import (
    extract_training_timestamps,
    create_in_window_val_set,
)
def generate_raw_validation_set(
    file_path: str,
    train_data_fpath: str,
    raw_data_fpath: str,
) -> pd.DataFrame:
    """
    Generate a raw validation set by extracting timestamps from the training 
    dataset and filtering the raw data to include only the relevant timestamps.

    Parameters:
    ----------
    file_path : str
        The directory path where the data files are located.
    train_data_fpath : str
        The filename of the training dataset to extract timestamps from.
    raw_data_fpath : str
        The filename of the raw dataset to filter.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the filtered raw validation set.
    """
    train_timestamps = extract_training_timestamps(os.path.join(file_path, train_data_fpath))
    in_window_df = create_in_window_val_set(os.path.join(file_path, raw_data_fpath), train_timestamps)
    
    return in_window_df


def load_and_process_data(
    file_fpath: str,
    constraint_cols: List[str],
    timestamp_col: str = "Point Name",
) -> pd.DataFrame:
    """
    Load and process a CSV file containing gross load data.

    Parameters
    ----------
    file_fpath : str
        The directory path for the input data file.
    constraint_cols : list of str
        The names of the constraint columns.
    timestamp_col : str, optional
        The name of the timestamp column. Defaults to "Point Name".

    Returns
    -------
    pd.DataFrame
        A processed DataFrame with timestamp and gross load.
    
    Raises
    ------
    ValueError
        If the specified columns are not found in the expected row.
    """
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_fpath, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # If no encoding worked
        raise UnicodeDecodeError(f"Could not read file {file_fpath} with any of the following encodings: {encodings}")
    
    constraint_cols = [col.upper() for col in constraint_cols]
    timestamp_col = timestamp_col.upper()
    df.columns = df.columns.str.upper()

    missing_cols = [col for col in constraint_cols + [timestamp_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {file_fpath}: {missing_cols}")

    df = df.loc[4:, [timestamp_col] + constraint_cols]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.columns = ["timestamp"] + constraint_cols
    df[constraint_cols] = df[constraint_cols].astype(float)

    return df


def apply_conditions(
        df: pd.DataFrame, 
        constraint_cols: List[str], 
        condition_limits: List[float], 
        operators: List[str],
    ) -> pd.Series:
    """
    Apply multiple conditions to determine when the model is off based on specified constraints.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing constraint values.
    constraint_cols : list of str
        The names of the constraint columns in the DataFrame.
    condition_limits : list of float
        The corresponding limit values for each constraint.
    operators : list of str
        The comparison operators (e.g., "<", ">", "=") to apply for each constraint limit.

    Returns
    -------
    pd.Series
        A boolean series indicating whether each row meets any "model off" condition.

    """
    
    conditions = []
    for col, limit, op in zip(constraint_cols, condition_limits, operators):
        if op == "<":
            conditions.append(df[col] < limit)
        elif op == ">":
            conditions.append(df[col] > limit)
        elif op == "=":
            conditions.append(df[col] == limit)
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return pd.Series(np.logical_or.reduce(conditions), index=df.index)