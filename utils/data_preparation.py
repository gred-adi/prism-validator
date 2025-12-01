import pandas as pd
from typing import List

def extract_training_timestamps(
    file_path: str,
    timestamp_col: str = "Point Name",
) -> List:
    """
    Extracts and returns a sorted list of training timestamps from a CSV file.

    Parameters:
    file_path (str): The path to the input CSV file.
    timestamp_col (str, optional): The name of the timestamp column. 
        Defaults to "Point Name".

    Returns:
    List: A sorted list of training timestamps.
    """
    df = pd.read_csv(file_path)[4:]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.sort_values(timestamp_col, inplace=True)

    train_timestamp_list = list(pd.to_datetime(df[timestamp_col]))

    return train_timestamp_list


def create_in_window_val_set(
    file_path: str,
    train_timestamps: List,
    timestamp_col: str = "Point Name",
) -> pd.DataFrame:
    """
    Creates a validation set by filtering the raw data to include only 
    timestamps that are greater than or equal to the minimum training timestamp 
    and excluding the training timestamps.

    Parameters:
    file_path (str): The path to the input CSV file containing raw data.
    train_timestamps (List): A list of training timestamps to exclude from the validation set.
    timestamp_col (str, optional): The name of the timestamp column. 
        Defaults to "Point Name".

    Returns:
    pd.DataFrame: A DataFrame containing the filtered validation set with headers.
    """
    header = pd.read_csv(file_path)[:4]

    df = pd.read_csv(file_path)[4:]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.sort_values(timestamp_col, inplace=True)
    df = df[df[timestamp_col] >= min(train_timestamps)]

    filtered_df = df[~df[timestamp_col].isin(train_timestamps)]

    export_filtered_df = pd.concat([header, filtered_df])

    return export_filtered_df


def prepare_omr_data(
    file_path: str,
) -> pd.DataFrame:
    """
    Prepares OMR data from a tab-delimited CSV file 
    by cleaning and transforming the data.

    Parameters:
    file_path (str): The path to the input OMR data file.

    Returns:
    pd.DataFrame: A cleaned DataFrame containing timestamps and OMR values.
    """
    
    df = pd.read_csv(
        file_path, 
        delimiter="\t", 
        encoding="utf-16", 
        names=["index1", "index2", "timestamp", "omr"],
    )
    
    df = df[df.omr != " "]
    df = df[df.timestamp != " "]

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    
    df.omr = df.omr.astype(float)

    df.drop(["index1", "index2"], axis=1, inplace=True)
    
    return df


