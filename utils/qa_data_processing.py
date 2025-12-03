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
    """Generates a raw validation set from training and raw data files.

    This function extracts timestamps from the training dataset and uses them
    to filter the raw data, creating a validation set that is aligned with
    the training data.

    Args:
        file_path (str): The directory path where the data files are located.
        train_data_fpath (str): The filename of the training dataset.
        raw_data_fpath (str): The filename of the raw dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered raw validation set.
    """
    train_timestamps = extract_training_timestamps(os.path.join(file_path, train_data_fpath))
    in_window_df = create_in_window_val_set(os.path.join(file_path, raw_data_fpath), train_timestamps)
    
    return in_window_df


def load_and_process_data(
    file_fpath: str,
    constraint_cols: List[str],
    timestamp_col: str = "Point Name",
) -> pd.DataFrame:
    """Loads and processes a CSV file for QA analysis.

    This function reads a CSV file, handling various encodings, and processes
    it by converting the timestamp column to datetime objects and the
    constraint columns to numeric types.

    Args:
        file_fpath (str): The path to the input data file.
        constraint_cols (List[str]): A list of constraint column names.
        timestamp_col (str, optional): The name of the timestamp column.

    Returns:
        pd.DataFrame: A processed DataFrame.

    Raises:
        ValueError: If the specified columns are not found in the file.
        UnicodeDecodeError: If the file cannot be read with any of the
            supported encodings.
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
    """Applies multiple conditions to a DataFrame to identify "model off" states.

    This function takes a DataFrame and a set of constraints, and returns a
    boolean Series indicating which rows meet any of the "model off" conditions.

    Args:
        df (pd.DataFrame): The DataFrame to apply conditions to.
        constraint_cols (List[str]): A list of constraint column names.
        condition_limits (List[float]): A list of limit values for each constraint.
        operators (List[str]): A list of comparison operators for each constraint.

    Returns:
        pd.Series: A boolean Series indicating "model off" conditions.

    Raises:
        ValueError: If an unsupported operator is provided.
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