"""
qa_ks_comparison.py

Functions for comparing data distributions using the KS test.
"""

import os
import math
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
from scipy.stats import ks_2samp
from datetime import datetime
from typing import Any
from utils.qa_yaml_utils import convert_timestamps_for_yaml


def is_timestamp(timestamp: str) -> bool:
    """
    Check if a given string is a valid timestamp in one of the specified formats.

    Parameters
    ----------
    timestamp : str
        The string to check for timestamp validity.

    Returns
    -------
    bool
        True if the string is a valid timestamp in one of the formats, False otherwise.
    """
    formats = [
        "%Y-%m-%d %H:%M:%S",   # 24-hour format
        "%m/%d/%Y %H:%M:%S",   # 24-hour format
        "%m/%d/%Y %I:%M:%S %p" # 12-hour format with AM/PM
    ]

    for fmt in formats:
        try:
            _ = datetime.strptime(timestamp, fmt)
            return True
        except ValueError:
            continue
    return False

def get_timestamps(
    df: pd.DataFrame,
    timestamp_col: str,
) -> pd.Series:
    """
    Extract and convert valid timestamps from a DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the timestamp data.
    timestamp_col : str
        The name of the column containing timestamps.

    Returns
    -------
    pd.Series
        A Pandas Series containing valid and converted timestamps.
    """
    cond = ~df[timestamp_col].isna()
    timestamps = df.loc[cond, timestamp_col]
    cond = timestamps.map(is_timestamp)
    timestamps = pd.to_datetime(timestamps[cond])

    return timestamps


def timedelta_to_str(td):
    days = td.days
    hours, rem = divmod(td.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if days: parts.append(f"{days} days")
    if hours: parts.append(f"{hours} hours")
    if minutes: parts.append(f"{minutes} minutes")
    if seconds: parts.append(f"{seconds} seconds")
    return ', '.join(parts) if parts else "0 seconds"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize the filename by replacing invalid characters (like "/") with underscores.

    Parameters
    ----------
    filename : str
        The original filename.

    Returns
    -------
    str
        The sanitized filename.
    """
    return filename.replace("/", "_").replace("\\", "_")


def compare_data_distributions(
    validation_fname: str,
    holdout_fpath: str,
    ks_file_path: str,
    timestamp_col: str = "Point Name",
    description_row: int = 1,
    percentile_lower: float = 0.005,
    percentile_upper: float = 0.995,
    alpha: float = 0.5,
    n_bins: int = 40,
    alpha_grid: float = 0.2,
) -> None:
    """
    Compare data distributions between validation and holdout datasets using the KS test
    and save the results as a CSV file and plots of normalized distributions.

    Parameters
    ----------
    validation_fname : str
        Filename of the cleaned raw validation dataset.
    holdout_fpath : str
        Filename of the cleaned holdout dataset.
    ks_file_path : str
        Path to save the KS distribution comparison plots.
    timestamp_col : str, optional
        Column name containing timestamps. Defaults to "Point Name".
    description_row : int
        Row number in the holdout file that contains the descriptions. Defaults to 1 (second row).
    percentile_lower : float, optional
        Lower quantile for filtering data. Defaults to 0.005.
    percentile_upper : float, optional
        Upper quantile for filtering data. Defaults to 0.995.
    alpha : float, optional
        Alpha transparency for the histogram plots. Defaults to 0.5.
    n_bins : int, optional
        Number of bins for histogram plots. Defaults to 40.
    alpha_grid : float, optional
        Alpha transparency for the grid in plots. Defaults to 0.2.

    Returns
    -------
    None
        The function saves the KS test results and plots to the specified file paths.
    """
    validation_df = pd.read_csv(validation_fname)
    validation_timestamp = get_timestamps(df=validation_df, timestamp_col=timestamp_col)
    validation_df.columns = validation_df.columns.str.upper()
    description_mapping = validation_df.iloc[description_row].to_dict()
    validation_df = validation_df.iloc[4:]
    
    holdout_df = pd.read_csv(holdout_fpath)
    holdout_timestamp = get_timestamps(df=holdout_df, timestamp_col=timestamp_col)
    holdout_df.columns = holdout_df.columns.str.upper()
    # description_mapping = holdout_df.iloc[description_row].to_dict()
    holdout_df = holdout_df.iloc[4:]
    
    variables = holdout_df.columns.tolist()

    ks_results = pd.DataFrame(columns=[
        "Variable",
        "0.5% Quantile (Validation)",
        "0.5% Quantile (Holdout)",
        "99.9% Quantile (Validation)",
        "99.9% Quantile (Holdout)",
        "KS Statistic",
        "P-value"
    ])
    
    data_stats = {
        "validation": {
            "n_records": validation_timestamp.shape[0],
            "start_time": validation_timestamp.min(),
            "end_time": validation_timestamp.max(),
            "window_size": timedelta_to_str(validation_timestamp.max() - validation_timestamp.min()),
        },
        "holdout": {
            "n_records": holdout_timestamp.shape[0],
            "start_time": holdout_timestamp.min(),
            "end_time": holdout_timestamp.max(),
            "window_size": timedelta_to_str(holdout_timestamp.max() - holdout_timestamp.min()),
        },
    }
    yaml_safe_stats = convert_timestamps_for_yaml(data_stats)
    with open(os.path.join(ks_file_path, "data_stats.yaml"), "w") as file:
        yaml.safe_dump(yaml_safe_stats, file, default_flow_style=False)

    for var in variables:
        if var not in validation_df.columns:
            print(f"Variable '{var}' not found in validation dataset.")
            continue

        validation_df[var] = pd.to_numeric(validation_df[var], errors='coerce')
        holdout_df[var] = pd.to_numeric(holdout_df[var], errors='coerce')

        validation_filtered = validation_df[var].dropna()
        holdout_filtered = holdout_df[var].dropna()

        v_lower = validation_filtered.quantile(percentile_lower)
        v_upper = validation_filtered.quantile(percentile_upper)

        holdout_filtered = holdout_filtered[
            (holdout_filtered >= v_lower) & (holdout_filtered <= v_upper)
        ]
        validation_filtered = validation_filtered[
            (validation_filtered >= v_lower) & (validation_filtered <= v_upper)
        ]

        if validation_filtered.empty or holdout_filtered.empty:
            print(f"Skipping KS test for {var} due to empty dataset.")
            continue

        ks_statistic, p_value = ks_2samp(validation_filtered, holdout_filtered)

        quantile_05_val = validation_filtered.quantile(percentile_lower)
        quantile_05_holdout = holdout_filtered.quantile(percentile_lower)
        quantile_99_val = validation_filtered.quantile(percentile_upper)
        quantile_99_holdout = holdout_filtered.quantile(percentile_upper)

        x_min = min(
            validation_filtered.quantile(percentile_lower),
            holdout_filtered.quantile(percentile_lower),
        )
        x_max = max(
            validation_filtered.quantile(percentile_upper),
            holdout_filtered.quantile(percentile_upper),
        )

        percent = 0.05
        delta = x_max - x_min
        x_min -= delta * percent
        x_max += delta * percent
        
        bin_size = (x_max - x_min) / n_bins
        if bin_size <= 0:
            continue

        bins = np.arange(x_min, x_max + bin_size, bin_size)

        plt.figure(figsize=(11, 5.5))
        plt.hist(validation_filtered, bins=bins, alpha=alpha, label='Validation', density=True)
        plt.hist(holdout_filtered, bins=bins, alpha=alpha, label='Holdout', density=True)
        # plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(alpha=alpha_grid)

        # if Extended Name is NaN, it will use the tag name
        description = description_mapping.get(var, var)
        if description is None or (isinstance(description, float) and math.isnan(description)):
            description = var
        sanitized_description = sanitize_filename(str(description))

        plot_fname = os.path.join(ks_file_path, f"distribution_comparison_{sanitized_description}.jpg")
        plt.savefig(plot_fname, dpi=300, bbox_inches="tight")
        plt.close()

        new_row = pd.DataFrame({
            "Variable": [sanitized_description],
            "0.5% Quantile (Validation)": [quantile_05_val],
            "0.5% Quantile (Holdout)": [quantile_05_holdout],
            "99.9% Quantile (Validation)": [quantile_99_val],
            "99.9% Quantile (Holdout)": [quantile_99_holdout],
            "KS Statistic": [ks_statistic],
            "P-value": [p_value]
        })

        ks_results = pd.concat([ks_results, new_row], ignore_index=True)
        
    summary_row = {
        "Variable": "Summary",
        "0.5% Quantile (Validation)": ks_results["0.5% Quantile (Validation)"].mean(),
        "0.5% Quantile (Holdout)": ks_results["0.5% Quantile (Holdout)"].mean(),
        "99.9% Quantile (Validation)": ks_results["99.9% Quantile (Validation)"].mean(),
        "99.9% Quantile (Holdout)": ks_results["99.9% Quantile (Holdout)"].mean(),
        "KS Statistic": ks_results["KS Statistic"].mean(),
        "P-value": ks_results["P-value"].mean()
    }

    ks_results = pd.concat([ks_results, pd.DataFrame([summary_row])], ignore_index=True)
    ks_results.to_csv(os.path.join(ks_file_path, "ks_results.csv"), index=False)
    dfi.export(ks_results, os.path.join(ks_file_path, "ks_results.jpg"), table_conversion='matplotlib')

