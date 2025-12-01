import os
from datetime import timedelta, datetime
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import math

from utils.data_preparation import prepare_omr_data
from utils.model_validation import (
    compute_alert_persistence,
    calculate_fpr,
    calculate_fpr_with_persistence,
)
from utils.qa_yaml_utils import convert_numpy_for_yaml
from utils.qa_data_processing import (
    apply_conditions,
    load_and_process_data,
)


def apply_conditions(
        df: pd.DataFrame, 
        constraint_cols: List[str], 
        condition_limits: List[float], 
        operators: List[str],
    ) -> pd.Series:
    """
    Apply multiple conditions to determine when the model is off based on specified constraints.

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


def plot_omr_and_constraint(
    df: pd.DataFrame,
    omr_df: pd.DataFrame,
    plot_fname: str,
    constraint_cols: List[str],
    condition_limits: List[float],
    operators: List[str],
    timestamp_col: str = "timestamp",
    omr_col: str = "omr",
    warning_thresh: float = 5.0,
    alert_thresh: float = 10.0,
    font_size: int = 12,
) -> None:
    """
    Plot OMR values with constraint condition overlayed, including warning and alert thresholds.
    """

    if df.empty or omr_df.empty:
        plt.figure()
        plt.text(0.5, 0.5, 'No data available', fontsize=20, ha='center', va='center')
        plt.title('OMR Plot - No Data Available')
        plt.savefig(plot_fname, dpi=300, bbox_inches="tight")
        plt.close()
        return
    
    model_off_cond = apply_conditions(df, constraint_cols, condition_limits, operators)
    warning_cond = (omr_df[omr_col] >= warning_thresh) & (omr_df[omr_col] < alert_thresh)
    alert_cond = (omr_df[omr_col] >= alert_thresh)

    plt.figure(figsize=(11, 5.5))
    plt.plot(omr_df[timestamp_col], omr_df[omr_col], color="#1f77b4")

    ymin_omr, ymax_omr = 0, omr_df[omr_col].max()
    percent = 0.25
    y_delta = ymax_omr - ymin_omr
    ymax_omr += percent * y_delta

    plt.fill_between(
        df[timestamp_col], ymin_omr, ymax_omr, 
        where=model_off_cond, color="gray", alpha=0.5, 
        linewidth=0, label="Model is off"
    )
    plt.fill_between(
        omr_df[timestamp_col], ymin_omr, ymax_omr, 
        where=warning_cond, color="orange", 
        alpha=0.7, label=f"Above Warning Threshold ({warning_thresh})"
    )
    plt.fill_between(
        omr_df[timestamp_col], ymin_omr, ymax_omr, 
        where=alert_cond, color="red", 
        alpha=0.6, label=f"Above Alert Threshold ({alert_thresh})"
    )

    # plt.xlabel("Time", fontsize=font_size)
    plt.ylabel("OMR (%)", fontsize=font_size, color="#1f77b4")
    plt.ylim([ymin_omr, ymax_omr])
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tick_params(axis="y", labelcolor="#1f77b4")
    plt.legend(fontsize=font_size, loc="upper right")
    plt.tight_layout()
    plt.savefig(plot_fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_alert_persistence(
    df: pd.DataFrame,
    omr_df: pd.DataFrame,
    ts_values: List[datetime],
    warning_alerts: np.ndarray,
    alerts: np.ndarray,
    plot_fname: str,
    constraint_cols: List[str],
    condition_limits: List[float],
    operators: List[str],
    timestamp_col: str,
    font_size: int = 10,
) -> None:
    """
    Plot OMR values and alert persistence areas for warnings and alerts, with 
    additional model-off condition overlays based on multiple constraints.
    """

    plt.figure(figsize=(11, 5.5))
    plt.plot(omr_df[timestamp_col], omr_df["omr"])

    model_off_cond = apply_conditions(df, constraint_cols, condition_limits, operators)

    omr_max = omr_df.omr.max()

    if omr_df.omr.notnull().any(): 
        if np.any(warning_alerts) or np.any(alerts):
            ymin_omr, ymax_omr = 0, max(np.max([max(warning_alerts), max(alerts)]), omr_max)
        else:
            ymin_omr, ymax_omr = 0, omr_max if not np.isnan(omr_max) else 1
    else:
        ymin_omr, ymax_omr = 0, 1 

    if ymax_omr - ymin_omr < 0.01: 
        ymax_omr = ymin_omr + 0.01 

    percent = 0.25
    y_delta = ymax_omr - ymin_omr
    ymax_omr += percent * y_delta

    plt.fill_between(
        df[timestamp_col], ymin_omr, ymax_omr, 
        where=model_off_cond, color="gray", alpha=0.5, 
        linewidth=0, label="Model is off"
    )
    plt.fill_between(
        ts_values, ymin_omr, ymax_omr, 
        where=warning_alerts == 1, color='orange', 
        alpha=0.5, label='Warning Alert Area'
    )
    plt.fill_between(
        ts_values, ymin_omr, ymax_omr, 
        where=alerts == 1, color='red', 
        alpha=0.5, label='Alert Area'
    )

    plt.ylim([ymin_omr, ymax_omr])
    plt.xlim(df[timestamp_col].min(), df[timestamp_col].max())
    plt.axhline(y=5, c='orange', ls='dashed', lw=2, label="Warning Threshold")
    plt.axhline(y=10, c='red', ls='dashed', lw=2, label="Alert Threshold")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # plt.xlabel("Timestamp", fontsize=font_size)
    plt.ylabel("OMR (%)", fontsize=font_size)
    plt.title("OMR Threshold Exceedance with Warning and Alert Areas")
    plt.legend(fontsize=font_size, loc="upper right")
    plt.savefig(plot_fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_omr_distribution_comparison(
    clean_val_omr_df: pd.DataFrame,
    raw_val_omr_df: pd.DataFrame,
    holdout_omr_df: pd.DataFrame,
    plot_fname: str,
    percentile_lower: float = 0.005,
    percentile_upper: float = 0.995,
    n_bins: int = 100,
    alpha: float = 0.5,
    alpha_grid: float = 0.5,
) -> None:
    """
    Plot the OMR distribution comparison for clean validation, raw validation, and holdout datasets.
    """
    quantile_05_clean_val = clean_val_omr_df.omr.quantile(percentile_lower)
    quantile_05_raw_val = raw_val_omr_df.omr.quantile(percentile_lower)
    quantile_05_holdout = holdout_omr_df.omr.quantile(percentile_lower)
    quantile_99_clean_val = clean_val_omr_df.omr.quantile(percentile_upper)
    quantile_99_raw_val = raw_val_omr_df.omr.quantile(percentile_upper)
    quantile_99_holdout = holdout_omr_df.omr.quantile(percentile_upper)

    lower_quantiles = [quantile_05_clean_val, quantile_05_raw_val, quantile_05_holdout]
    upper_quantiles = [quantile_99_clean_val, quantile_99_raw_val, quantile_99_holdout]

    lower_quantiles = [q for q in lower_quantiles if not np.isnan(q)]
    upper_quantiles = [q for q in upper_quantiles if not np.isnan(q)]

    if lower_quantiles and upper_quantiles:
        x_min = np.min(lower_quantiles)
        x_max = np.max(upper_quantiles)
    else:
        x_min, x_max = 0, 1
        print("All datasets are empty, plotting an empty distribution.")

    percent = 0.05
    if x_min == x_max:
        x_min -= 0.1
        x_max += 0.1

    delta = x_max - x_min
    x_min -= delta * percent
    x_max += delta * percent

    if np.isfinite(x_min) and np.isfinite(x_max) and (x_max > x_min):
        bin_size = (x_max - x_min) / n_bins

        if bin_size > 1e-6:
            bins = np.arange(x_min, x_max + bin_size, bin_size)
        else:
            print(f"bin_size too small: {bin_size}. Skipping plot.")
            bins = None
    else:
        print(f"Invalid x_min ({x_min}) or x_max ({x_max}), using default bin range.")
        bins = np.linspace(0, 1, n_bins)

    plt.figure(figsize=(11, 5.5))
    plt.yscale(value='log')

    if bins is not None:
        if not clean_val_omr_df.empty:
            plt.hist(clean_val_omr_df.omr, bins=bins, alpha=alpha, label='Clean Validation', density=True, lw=3, color="red")
        if not raw_val_omr_df.empty:
            plt.hist(raw_val_omr_df.omr, bins=bins, alpha=alpha, label='Raw Validation', density=True, lw=3, color="blue")
        if not holdout_omr_df.empty:
            plt.hist(holdout_omr_df.omr, bins=bins, alpha=alpha, label='Holdout', density=True, lw=3, color="green")

        if clean_val_omr_df.empty and raw_val_omr_df.empty and holdout_omr_df.empty:
            plt.text(0.5, 0.5, 'No data available', fontsize=12, ha='center', va='center')
            plt.axis('off')

        plt.legend(loc='best')
        plt.grid(alpha=alpha_grid)
    else:
        plt.text(0.5, 0.5, 'Invalid bin range', fontsize=12, ha='center', va='center')
        plt.axis('off')

    plt.savefig(plot_fname, dpi=300, bbox_inches="tight")
    plt.close()
    plt.close()


def generate_report_plots(
    data_fpath: str,
    fpr_fpath: str,
    model_name: str,
    constraint_cols: List[str],
    condition_limits: List[float],
    operators: List[str],
    raw_data_fpath: str,
    holdout_fpath: str,
    holdout_omr_fname: str,
    val_without_outlier_omr_fname: str,
    val_with_outlier_omr_fname: str,
    timestamp_col: str = "timestamp",
    warning_threshold: float = 5,
    alert_threshold: float = 10,
    sub_ts_length_in_minutes: float = 60,
    n_ts_above_threshold: float = 50,
    time_interval: int = 1,
    nsteps: int = 20,
    font_size: int = 12,
) -> None:
    """
    Generate and save plots for OMR and gross load comparisons, along with false positive rates (FPR) for cleaned, 
    raw, and holdout datasets.
    """

    raw_df = load_and_process_data(raw_data_fpath, constraint_cols)
    holdout_df = load_and_process_data(holdout_fpath, constraint_cols)

    cleaned_val_omr_df = prepare_omr_data(os.path.join(data_fpath, val_without_outlier_omr_fname))
    raw_val_omr_df = prepare_omr_data(os.path.join(data_fpath, val_with_outlier_omr_fname))
    holdout_omr_df = prepare_omr_data(os.path.join(data_fpath, holdout_omr_fname))

    datasets_range = {
        "cleaned_omr_range": {
            "min": cleaned_val_omr_df.omr.min(),
            "max": cleaned_val_omr_df.omr.max(),
        },
        "raw_omr_range": {
            "min": raw_val_omr_df.omr.min(),
            "max": raw_val_omr_df.omr.max(),
        },
        "holdout_omr_range": {
            "min": holdout_omr_df.omr.min(),
            "max": holdout_omr_df.omr.max(),
        },
    }

    yaml_safe_ranges = convert_numpy_for_yaml(datasets_range)

    with open(os.path.join(fpr_fpath, "datasets_range.yaml"), "w") as file:
        yaml.safe_dump(yaml_safe_ranges, file, default_flow_style=False)
        
    thresholds, cleaned_fpr = calculate_fpr(cleaned_val_omr_df.omr, os.path.join(fpr_fpath, "fpr_stats_cleaned_val_omr_df.yaml"), n_steps=nsteps)
    thresholds, raw_fpr = calculate_fpr(raw_val_omr_df.omr, os.path.join(fpr_fpath, "fpr_stats_raw_val_omr_df.yaml"), n_steps=nsteps)
    thresholds, holdout_fpr = calculate_fpr(holdout_omr_df.omr, os.path.join(fpr_fpath, "fpr_stats_holdout_omr_df.yaml"), n_steps=nsteps)
    thresholds, holdout_fprp = calculate_fpr_with_persistence(
        df=holdout_omr_df, 
        sub_ts_length_in_minutes=sub_ts_length_in_minutes, 
        n_ts_above_threshold=n_ts_above_threshold, 
        time_interval=time_interval,
        fpr_fpath=os.path.join(fpr_fpath, "fprp_stats_holdout_omr_df.yaml"),
        n_steps=nsteps,
    )

    constraint_cols = [col.upper() for col in constraint_cols]
    plot_omr_and_constraint(
        df=raw_df,
        omr_df=cleaned_val_omr_df,
        plot_fname=os.path.join(fpr_fpath, f"{model_name} - Cleaned Validation OMR.jpg"),
        timestamp_col=timestamp_col,
        constraint_cols=constraint_cols,
        condition_limits=condition_limits,
        operators=operators,
    )
    
    plot_omr_and_constraint(
        df=raw_df,
        omr_df=raw_val_omr_df,
        plot_fname=os.path.join(fpr_fpath, f"{model_name} - Raw Validation OMR.jpg"),
        timestamp_col=timestamp_col,
        constraint_cols=constraint_cols,
        condition_limits=condition_limits,
        operators=operators,
    )
    
    ts_values, warning_alerts, alerts = compute_alert_persistence(df=holdout_omr_df, time_interval=time_interval)
    plot_alert_persistence(
        df=holdout_df,
        omr_df=holdout_omr_df,
        ts_values=ts_values, 
        warning_alerts=warning_alerts, 
        alerts=alerts,  
        constraint_cols=constraint_cols,
        condition_limits=condition_limits,
        operators=operators,
        timestamp_col=timestamp_col,
        plot_fname= os.path.join(fpr_fpath, f"{model_name} - Holdout OMR.jpg"),
        font_size=font_size,
    )

    # FPR with and without Persistence
    ymin_omr = 0
    fpr_values = [cleaned_fpr, raw_fpr, holdout_fpr, holdout_fprp]
    finite_fpr_values = [np.max(np.array(fpr)) for fpr in fpr_values if np.isfinite(np.max(np.array(fpr)))]

    if finite_fpr_values:
        ymax_omr = np.max(finite_fpr_values)
    else:
        ymax_omr = 1  

    percent = 0.30
    y_delta = ymax_omr - ymin_omr
    ymax_omr += percent * y_delta

    if np.isfinite(ymax_omr):
        plt.ylim([ymin_omr, ymax_omr])  
    else:
        print("Warning: ymax_omr is NaN or Inf. Skipping plt.ylim()")

    plt.figure(figsize=(11, 5.5))
    plt.plot(thresholds, cleaned_fpr, lw=5, color="#1f77b4", linestyle="dotted", label="Inwindow Validation Data (without Outlier)")
    plt.plot(thresholds, raw_fpr, lw=5, color="#1f77b4", linestyle="solid", label="Inwindow Validation Data (with Outlier)")
    plt.plot(thresholds, holdout_fpr, lw=5, color="green", linestyle="solid", label="Holdout Data")
    plt.plot(thresholds, holdout_fprp, lw=5, color="green", linestyle="--", label="Holdout Data (persistence)")
    plt.axvline(warning_threshold, color="orange", linestyle="--", label="Warning")
    plt.axvline(alert_threshold, color="red", linestyle="--", label="Alert")

    plt.xlabel("OMR Threshold (%)", fontsize=font_size)
    plt.ylabel("False Positive Rate (%)", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tick_params(axis="y")
    plt.legend(fontsize=font_size, loc="upper right")

    plt.grid(alpha=0.2)
    plot_fname = os.path.join(
        fpr_fpath, f"{model_name} - False Positive Rates (with and without persistence).jpg"
    )
    plt.savefig(plot_fname, dpi=300, bbox_inches="tight")
    plt.close()

    plot_omr_distribution_comparison(
        clean_val_omr_df=cleaned_val_omr_df,
        raw_val_omr_df=raw_val_omr_df,
        holdout_omr_df=holdout_omr_df,
        plot_fname= os.path.join(fpr_fpath, f"{model_name} - OMR Distributions.jpg")
    )


def generate_summary_fprp_plots(
    df: pd.DataFrame,
    plot_fname: str,
    fpr_limit: float = 10,
    font_size: int = 16,
) -> None:
    """
    Generates and saves a horizontal bar plot visualizing False Positive Rate with Persistence (FPRP) 
    statistics for each model.

    This function creates a bar plot with two bars per model, representing FPRP at 5.0% OMR (Warning) 
    and 10.0% OMR (Alert). Models are displayed on the y-axis in ascending order of their FPRP at 10.0% OMR (Alert). 
    The plot is saved as an image file.
    """

    df_sorted = df.sort_values(
        by=['FPR with persistence at 10.0% OMR (Alert)', 'FPR with persistence at 5.0% OMR (Warning)'],
        ascending=[True, True])

    plt.figure(figsize=(20, 10))
    bar_width = 0.4
    index = np.arange(len(df_sorted['Model Name']))
    plt.barh(index + bar_width, df_sorted['FPR with persistence at 10.0% OMR (Alert)'], bar_width, 
             label='FPR with persistence at 10.0% OMR (Alert)', color='red', alpha=0.5)
    plt.barh(index, df_sorted['FPR with persistence at 5.0% OMR (Warning)'], bar_width, 
             label='FPR with persistence at 5.0% OMR (Warning)', color='orange', alpha=0.5)
    
    plt.xlabel('FPR (%)', fontsize=font_size)
    plt.xlim(0, fpr_limit)
    plt.yticks(index + bar_width / 2, df_sorted['Model Name'], fontsize=font_size)
    plt.legend(fontsize=font_size, loc='lower right')
    plt.tight_layout()
    plt.savefig(plot_fname, dpi=300, bbox_inches="tight")
    plt.close()