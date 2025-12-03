import pandas as pd
import numpy as np
import yaml
from datetime import timedelta, datetime
from typing import List, Tuple


def calculate_fpr(
    omr_values: pd.Series,
    fpr_fpath: str,
    omr_limits: Tuple[float, float] = (0.0, 20.0),
    n_steps: int = 20
) -> Tuple[np.ndarray, List[float]]:
    """Calculates the false positive rate (FPR) for a series of OMR values.

    This function calculates the percentage of OMR values that exceed a range
    of thresholds.

    Args:
        omr_values (pd.Series): A Pandas Series of OMR values.
        fpr_fpath (str): The file path to save the FPR statistics YAML file.
        omr_limits (Tuple[float, float], optional): The range of OMR thresholds.
        n_steps (int, optional): The number of steps to divide the threshold range.

    Returns:
        Tuple[np.ndarray, List[float]]: A tuple containing the thresholds and
        the percentage of OMR values above each threshold.
    """
    n_omr_values = len(omr_values)
    delta = omr_limits[1] - omr_limits[0]
    step_size = delta / n_steps
    thresholds = np.arange(omr_limits[0], omr_limits[1] + step_size, step_size)

    fractions_above_threshold = []
    for threshold in thresholds:
        count_above_threshold = np.sum(omr_values > threshold)
        fraction_above_threshold = (count_above_threshold / n_omr_values) * 100
        fractions_above_threshold.append(fraction_above_threshold)

    warning_idx = np.where(np.isclose(thresholds, 5.0))[0][0]
    alert_idx = np.where(np.isclose(thresholds, 10.0))[0][0]

    fpr_stats = {
        "warning": float(fractions_above_threshold[warning_idx]),
        "alert": float(fractions_above_threshold[alert_idx]),
    }

    with open(fpr_fpath, 'w') as file:
        yaml.dump(fpr_stats, file)

    return thresholds, fractions_above_threshold


def calculate_fpr_with_persistence(
    df: pd.DataFrame,
    sub_ts_length_in_minutes: float,
    n_ts_above_threshold: int,
    time_interval: int,
    fpr_fpath: str,
    omr_limits: Tuple[float, float] = (0.0, 20.0),
    n_steps: int = 20,
) -> Tuple[np.ndarray, List[float]]:
    """Calculates the false positive rate with persistence (FPRP).

    This function calculates the FPRP for OMR values over specified sub-time
    series lengths and thresholds.

    Args:
        df (pd.DataFrame): A DataFrame containing OMR values.
        sub_ts_length_in_minutes (float): The length of the sub-time series.
        n_ts_above_threshold (int): The number of timestamps above the
            threshold required for a sub-time series to be counted.
        time_interval (int): The time interval of the data.
        fpr_fpath (str): The file path to save the FPRP statistics YAML file.
        omr_limits (Tuple[float, float], optional): The range of OMR thresholds.
        n_steps (int, optional): The number of steps to divide the threshold range.

    Returns:
        Tuple[np.ndarray, List[float]]: A tuple containing the thresholds and
        the FPRP values for each threshold.
    """
    df_temp = df.copy()
    df_temp.sort_values("timestamp", ascending=True, inplace=True)
    df_temp.reset_index(drop=True, inplace=True)
    ts_min = df_temp.timestamp.min()
    ts_max = df_temp.timestamp.max()
    sub_ts_length = timedelta(minutes=sub_ts_length_in_minutes)
    start_ts = ts_min + sub_ts_length
    ts_delta_values = df_temp.timestamp.diff().value_counts()
    ts_step = ts_delta_values.index.min()

    delta = omr_limits[1] - omr_limits[0]
    step_size = delta / n_steps
    thresholds = np.arange(omr_limits[0], omr_limits[1] + step_size, step_size)

    fprp_values = []
    sub_series_indexes = []

    ts = start_ts
    while ts <= ts_max:
        start_time = ts - sub_ts_length
        end_time = ts
        cond = (df_temp["timestamp"] >= start_time) & (df_temp["timestamp"] <= end_time)
        sub_series_idxs = df_temp[cond].index
        sub_series_indexes.append(sub_series_idxs)

        ts = ts + ts_step

    total_sub_series = len(sub_series_indexes)

    count_above_threshold_list = np.zeros(len(thresholds))
    for sub_series_idxs in sub_series_indexes:
        sub_series_omr = df_temp.loc[sub_series_idxs, "omr"]

        for i, threshold in enumerate(thresholds):
            if (sub_series_omr > threshold).sum() > n_ts_above_threshold/time_interval:
                count_above_threshold_list[i] += 1
    
    n_timestamp = df_temp.shape[0]
    fprp_values = (count_above_threshold_list / n_timestamp) * 100 if total_sub_series > 0 else count_above_threshold_list
    fprp_values = fprp_values.tolist()

    warning_idx = np.where(np.isclose(thresholds, 5.0))[0][0]
    alert_idx = np.where(np.isclose(thresholds, 10.0))[0][0]

    fprp_stats = {
        "warning": float(fprp_values[warning_idx]),
        "alert": float(fprp_values[alert_idx]),
    }

    with open(fpr_fpath, 'w') as file:
        yaml.dump(fprp_stats, file)

    return thresholds, fprp_values


def compute_alert_persistence(
    df: pd.DataFrame,
    warning_threshold: float = 5.0,
    alert_threshold: float = 10.0,
    sub_ts_length_in_minutes: int = 60,
    n_ts_above_threshold: int = 50,
    time_interval: int = 1,
) -> Tuple[List[datetime], np.ndarray, np.ndarray]:
    """Computes alert persistence for OMR values in a time series dataset.

    This function calculates whether warning and alert conditions are met
    persistently over a sliding window.

    Args:
        df (pd.DataFrame): DataFrame with OMR values and timestamps.
        warning_threshold (float, optional): The warning threshold.
        alert_threshold (float, optional): The alert threshold.
        sub_ts_length_in_minutes (int, optional): The length of the sliding window.
        n_ts_above_threshold (int, optional): The number of timestamps above
            the threshold to trigger an alert.
        time_interval (int, optional): The time interval of the data.

    Returns:
        Tuple[List[datetime], np.ndarray, np.ndarray]: A tuple containing the
        timestamps, a boolean array for warning alerts, and a boolean array for
        alerts.
    """
    sub_ts_delta = timedelta(minutes=sub_ts_length_in_minutes)

    df_temp = df.copy()
    df_temp.sort_values("timestamp", ascending=True, inplace=True)
    df_temp.reset_index(drop=True, inplace=True)
    ts_min = df_temp.timestamp.min()
    ts_max = df_temp.timestamp.max()

    start_ts = ts_min + sub_ts_delta
    ts_step = df_temp.timestamp.diff().min()

    ts_values = []
    warning_alerts = []
    alerts = []
    n_ts_in_warning_threshold_list = []
    n_ts_above_alert_threshold_list = []

    ts = start_ts
    while ts <= ts_max:
        start_time = ts - sub_ts_delta
        end_time = ts

        sub_series_omr = df_temp.loc[
            (df_temp["timestamp"] >= start_time) & 
            (df_temp["timestamp"] <= end_time), "omr"
        ]

        n_in_warning_threshold = (
            (sub_series_omr > warning_threshold) & 
            (sub_series_omr < alert_threshold)
        ).sum()
        n_above_alert_threshold = (sub_series_omr > alert_threshold).sum()

        ts_values.append(ts)
        n_ts_in_warning_threshold_list.append(n_in_warning_threshold)
        n_ts_above_alert_threshold_list.append(n_above_alert_threshold)

        warning_alerts.append(int(n_in_warning_threshold > n_ts_above_threshold/time_interval))
        alerts.append(int(n_above_alert_threshold > n_ts_above_threshold/time_interval))

        ts += ts_step

    return ts_values, np.array(warning_alerts), np.array(alerts)
