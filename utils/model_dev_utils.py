import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st
import numpy as np
import base64
from datetime import datetime

from typing import Tuple, Union, Dict, Any
from pathlib import Path
from scipy.stats import pearsonr
# Removed FPDF import
from io import BytesIO
from playwright.sync_api import sync_playwright
from jinja2 import Environment
import sys
import asyncio

def cleaned_dataset_name_split(
        filename:str
        ) -> Tuple [str, str, str]:
    """
    Given a filename, returns the site_name, model_name,
    and inclusive dates.

    filename format is expected to be:
        CLEANED-[model_name]-[inclusive_dates]-RAW.csv
        Example: CLEANED-AP-TVI-U1-BFP_A_MOTOR-20240601-20250801-RAW.csv

    Args:
        filename (str): The name of the cleaned dataset file.

    Returns:
        Tuple[str, str, str]: A tuple containing:
        site_name, model_name, inclusive_dates
    """
    # Remove the 'CLEANED-' prefix and suffix
    base_name = filename.removeprefix("CLEANED-")
    if base_name.endswith("-WITH-OUTLIER.csv"):
        base_name = base_name.removesuffix("-WITH-OUTLIER.csv")
    elif base_name.endswith("-WITHOUT-OUTLIER.csv"):
        base_name = base_name.removesuffix("-WITHOUT-OUTLIER.csv")
    elif base_name.endswith("-RAW.csv"):
        base_name = base_name.removesuffix("-RAW.csv")
    # Split the remaining string by the second to the last "-"
    parts = base_name.rsplit("-", 2)
    model_name = parts[0]
    inclusive_dates = parts[1] + "-" + parts[2]
    # Get the value between the first and second "-" in model_name (TVI in the example)
    site_name = model_name.split("-")[1]
    return site_name, model_name, inclusive_dates

def data_cleaning_read_prism_csv(df: pd.DataFrame, project_points: pd.DataFrame):
    """
    Takes a PRISM DataFrame and returns a DataFrame with mapped metric names and the original PRISM header.
    Optimized for performance.
    """
    # Extract header rows (metadata)
    df_header = df.iloc[:4, :].copy()
    
    # Extract data rows (skip the first 4 rows of metadata)
    df_data = df.iloc[4:, :].copy()
    
    # Rename the first column (Time)
    # Assuming 'Point Name' is the first column in the raw export
    first_col_name = df_data.columns[0]
    df_data.rename(columns={first_col_name: 'DATETIME'}, inplace=True)
    
    # Optimized Datetime Conversion
    # Using errors='coerce' is safer and usually faster than letting pandas infer mixed formats
    df_data['DATETIME'] = pd.to_datetime(df_data['DATETIME'], errors='coerce')
    
    # Reset index after slicing
    df_data.reset_index(drop=True, inplace=True)

    # Optimized Numeric Conversion
    # Instead of apply(pd.to_numeric) on the whole DF (which is very slow), iterate columns
    cols_to_convert = [c for c in df_data.columns if c != 'DATETIME']
    for col in cols_to_convert:
        df_data[col] = pd.to_numeric(df_data[col], errors='coerce')

    # Optimized Column Mapping
    # Create a dictionary for O(1) lookup instead of O(N) filtering inside the loop
    # Handle NaN values in mapping by converting to string 'nan' to match existing logic
    name_to_metric = pd.Series(
        project_points['Metric'].values, 
        index=project_points['Name']
    ).to_dict()

    new_columns = []
    for column in df_data.columns:
        if column == 'DATETIME':
            new_columns.append(column)
        else:
            # Fast lookup
            mapping_val = name_to_metric.get(column)
            # Check if mapping exists and is not nan (pandas might map to nan/float)
            if mapping_val is not None and str(mapping_val) != 'nan':
                new_columns.append(str(mapping_val))
            else:
                new_columns.append(column)

    df_data.columns = new_columns
    
    return df_data, df_header

def _generate_html_report(stats_payload, numeric_filters, datetime_filters, plot_images):
    """Generates HTML content for the report using Jinja2."""
    
    env = Environment()
    template_str = """
    <html>
    <head>
        <style>
            body { font-family: "Helvetica", "Arial", sans-serif; color: #333; margin: 40px; }
            h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
            h2 { color: #2980b9; margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            .stats-box { background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
            .stats-row { display: flex; justify-content: space-between; margin-bottom: 10px; }
            .stat-item { text-align: center; flex: 1; }
            .stat-label { font-size: 0.9em; color: #777; }
            .stat-value { font-size: 1.2em; font-weight: bold; color: #333; }
            .filter-list { background-color: #fff; border: 1px solid #eee; padding: 10px; margin-top: 10px; }
            .img-container { text-align: center; margin-top: 20px; page-break-inside: avoid; }
            img { max-width: 100%; height: auto; border: 1px solid #ccc; }
            .footer { font-size: 0.8em; color: #999; text-align: center; margin-top: 50px; }
        </style>
    </head>
    <body>
        <h1>Data Cleaning Report</h1>
        <p>Generated on: {{ generation_date }}</p>

        <div class="stats-box">
            <h2>Impact Statistics</h2>
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-label">Total Rows</div>
                    <div class="stat-value">{{ total_rows }}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Remaining Rows</div>
                    <div class="stat-value">{{ remaining_rows }}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Overall Retention</div>
                    <div class="stat-value">{{ retention_pct }}%</div>
                </div>
            </div>
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-label">Numeric Removed</div>
                    <div class="stat-value">{{ numeric_removed }} ({{ numeric_pct }}%)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Date Removed</div>
                    <div class="stat-value">{{ date_removed }} ({{ date_pct }}%)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Total Removed</div>
                    <div class="stat-value">{{ total_removed }} ({{ removed_pct }}%)</div>
                </div>
            </div>
        </div>

        <h2>Active Filters</h2>
        <div class="filter-list">
            <strong>Numeric Filters:</strong>
            <ul>
            {% for f in numeric_filters %}
                <li>{{ f }}</li>
            {% else %}
                <li>None</li>
            {% endfor %}
            </ul>
            
            <strong>Date Filters:</strong>
            <ul>
            {% for f in datetime_filters %}
                <li>{{ f }}</li>
            {% else %}
                <li>None</li>
            {% endfor %}
            </ul>
        </div>

        <h2>Visualizations</h2>
        
        {% for title, img_data in plot_images %}
        <div class="img-container">
            <h3>{{ title }}</h3>
            <img src="data:image/png;base64,{{ img_data }}" />
        </div>
        {% endfor %}

        <div class="footer">
            PRISM Validator Tool
        </div>
    </body>
    </html>
    """
    
    template = env.from_string(template_str)
    
    # Format filters for display
    fmt_numeric = [f"{col} {op} {val}" for col, op, val in numeric_filters]
    fmt_date = []
    for op, val in datetime_filters:
        if op == "between (includes edge values)" or op == "between":
            val_str = f"{val[0]} to {val[1]}" if isinstance(val, tuple) or isinstance(val, list) else str(val)
            fmt_date.append(f"Remove between {val_str}")
        else:
            fmt_date.append(f"{op} {val}")

    html = template.render(
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_rows=f"{stats_payload['total_rows']:,}",
        remaining_rows=f"{stats_payload['remaining_rows']:,}",
        retention_pct=f"{stats_payload['retention_pct']:.2f}",
        numeric_removed=f"{stats_payload['numeric_removed']:,}",
        numeric_pct=f"{stats_payload['numeric_pct']:.2f}",
        date_removed=f"{stats_payload['date_removed']:,}",
        date_pct=f"{stats_payload['date_pct']:.2f}",
        total_removed=f"{stats_payload['total_removed']:,}",
        removed_pct=f"{stats_payload['removed_pct']:.2f}",
        numeric_filters=fmt_numeric,
        datetime_filters=fmt_date,
        plot_images=plot_images
    )
    return html

def generate_simple_report(raw_df: pd.DataFrame, numeric_filters: list, datetime_filters: list, pdf_file_path: Path):
    """Generates a simple PDF report using Playwright (consistent with other reports)."""
    # Reuse the full generation logic but with empty metrics list to skip charts
    # But we need to calculate stats first.
    
    # --- Calculate Stats (Replicated Logic for Consistency) ---
    total_rows = len(raw_df)
    
    # 1. Numeric Mask
    numeric_mask = pd.Series(True, index=raw_df.index)
    for col, op, val in numeric_filters:
        if col in raw_df.columns:
            if op == "<": numeric_mask &= ~(raw_df[col] < val)
            elif op == "<=": numeric_mask &= ~(raw_df[col] <= val)
            elif op == "==": numeric_mask &= ~(raw_df[col] == val)
            elif op == ">=": numeric_mask &= ~(raw_df[col] >= val)
            elif op == ">": numeric_mask &= ~(raw_df[col] > val)
    numeric_removed_count = (~numeric_mask).sum()

    # 2. Date Mask
    date_mask = pd.Series(True, index=raw_df.index)
    if 'DATETIME' in raw_df.columns:
        for op, val in datetime_filters:
            if op == "< (remove before)":
                date_mask &= ~(raw_df['DATETIME'] < pd.to_datetime(val))
            elif op == "> (remove after)":
                date_mask &= ~(raw_df['DATETIME'] > pd.to_datetime(val))
            elif op in ["between", "between (includes edge values)"]:
                # Handle list or tuple
                start = pd.to_datetime(val[0])
                end = pd.to_datetime(val[1])
                date_mask &= ~((raw_df['DATETIME'] >= start) & (raw_df['DATETIME'] <= end))
    date_removed_count = (~date_mask).sum()

    final_mask = numeric_mask & date_mask
    remaining_count = final_mask.sum()
    total_removed = total_rows - remaining_count

    stats_payload = {
        "total_rows": total_rows,
        "remaining_rows": remaining_count,
        "numeric_removed": numeric_removed_count,
        "date_removed": date_removed_count,
        "total_removed": total_removed,
        "retention_pct": (remaining_count / total_rows * 100) if total_rows > 0 else 0,
        "numeric_pct": (numeric_removed_count / total_rows * 100) if total_rows > 0 else 0,
        "date_pct": (date_removed_count / total_rows * 100) if total_rows > 0 else 0,
        "removed_pct": (total_removed / total_rows * 100) if total_rows > 0 else 0
    }

    html_content = _generate_html_report(stats_payload, numeric_filters, datetime_filters, [])
    
    # --- FIX: Enforce ProactorEventLoop for Windows (Critical for Playwright) ---
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html_content)
            page.pdf(path=str(pdf_file_path), format="A4", margin={'top': '1cm', 'bottom': '1cm', 'left': '1cm', 'right': '1cm'})
            browser.close()
    except Exception as e:
        st.error(f"Failed to generate PDF: {e}")


def generate_data_cleaning_visualizations(raw_df: pd.DataFrame,
                                          cleaned_df: pd.DataFrame,
                                          numeric_filters: list,
                                          datetime_filters: list,
                                          selected_metrics: list,
                                          generate_report: bool,
                                          pdf_file_path: Path = None):
    """
    Generates and displays all data cleaning visualizations.
    Uses Playwright for PDF generation.
    """
    plot_images = [] # List to store (title, base64_string) for PDF

    # FIX: Define total_rows explicitly from len(raw_df)
    total_rows = len(raw_df)
    if total_rows == 0:
        st.info("No data to visualize.")
        return

    # --- OPTIMIZATION: Downsampling ---
    MAX_PLOT_POINTS = 15000
    use_downsampling = total_rows > MAX_PLOT_POINTS
    
    if use_downsampling:
        step = total_rows // MAX_PLOT_POINTS
        plot_raw_df = raw_df.iloc[::step].copy()
        plot_cleaned_df = cleaned_df.iloc[::step].copy()
        st.toast(f"⚡ Data downsampled for visualization (displaying ~{len(plot_raw_df)} points)", icon="ℹ️")
    else:
        plot_raw_df = raw_df
        plot_cleaned_df = cleaned_df

    # --- STATISTICS (Use FULL Data to match Preview Impact) ---
    # 1. Numeric Mask
    numeric_mask = pd.Series(True, index=raw_df.index)
    for col, op, val in numeric_filters:
        if col in raw_df.columns:
            if op == "<": numeric_mask &= ~(raw_df[col] < val)
            elif op == "<=": numeric_mask &= ~(raw_df[col] <= val)
            elif op == "==": numeric_mask &= ~(raw_df[col] == val)
            elif op == ">=": numeric_mask &= ~(raw_df[col] >= val)
            elif op == ">": numeric_mask &= ~(raw_df[col] > val)
    numeric_removed_count = (~numeric_mask).sum()

    # 2. Date Mask
    date_mask = pd.Series(True, index=raw_df.index)
    if 'DATETIME' in raw_df.columns:
        for op, val in datetime_filters:
            if op == "< (remove before)":
                date_mask &= ~(raw_df['DATETIME'] < pd.to_datetime(val))
            elif op == "> (remove after)":
                date_mask &= ~(raw_df['DATETIME'] > pd.to_datetime(val))
            elif op in ["between", "between (includes edge values)"]:
                start = pd.to_datetime(val[0])
                end = pd.to_datetime(val[1])
                date_mask &= ~((raw_df['DATETIME'] >= start) & (raw_df['DATETIME'] <= end))
    date_removed_count = (~date_mask).sum()

    final_mask = numeric_mask & date_mask
    remaining_count = final_mask.sum()
    total_removed = total_rows - remaining_count

    # Calculate percentages (formatted to 2 decimals)
    # FIX: total_rows variable is now properly defined in scope
    retention_pct = (remaining_count / total_rows * 100) if total_rows > 0 else 0
    numeric_pct = (numeric_removed_count / total_rows * 100) if total_rows > 0 else 0
    date_pct = (date_removed_count / total_rows * 100) if total_rows > 0 else 0
    removed_pct = (total_removed / total_rows * 100) if total_rows > 0 else 0

    stats_payload = {
        "total_rows": total_rows,
        "remaining_rows": remaining_count,
        "numeric_removed": numeric_removed_count,
        "date_removed": date_removed_count,
        "total_removed": total_removed,
        "retention_pct": retention_pct,
        "numeric_pct": numeric_pct,
        "date_pct": date_pct,
        "removed_pct": removed_pct
    }

    # --- Display Stats (Aligned with Preview Impact) ---
    st.markdown("### Filter Impact Statistics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{total_rows:,}")
    m2.metric("Remaining", f"{remaining_count:,}", delta=f"-{removed_pct:.2f}% removed", delta_color="inverse")
    m3.metric("Numeric Removed", f"{numeric_removed_count:,}", delta=f"-{numeric_pct:.2f}% removed", delta_color="inverse")
    m4.metric("Date Removed", f"{date_removed_count:,}", delta=f"-{date_pct:.2f}% removed", delta_color="inverse")

    # --- Generate Timeline Graph (using fill_between) ---
    st.markdown("### Filter Timeline")
    
    # We need boolean arrays for fill_between. 
    # If downsampling, re-calculate masks on the small DF for plotting.
    if use_downsampling:
        timeline_df = plot_raw_df
        t_date_mask = pd.Series(False, index=timeline_df.index)
        if 'DATETIME' in timeline_df.columns:
            for op, val in datetime_filters:
                if op == "< (remove before)": t_date_mask |= (timeline_df['DATETIME'] < pd.to_datetime(val))
                elif op == "> (remove after)": t_date_mask |= (timeline_df['DATETIME'] > pd.to_datetime(val))
                elif op in ["between", "between (includes edge values)"]:
                    t_date_mask |= ((timeline_df['DATETIME'] >= pd.to_datetime(val[0])) & (timeline_df['DATETIME'] <= pd.to_datetime(val[1])))
        
        t_num_mask = pd.Series(False, index=timeline_df.index)
        for col, op, val in numeric_filters:
            if col in timeline_df.columns:
                if op == "<": t_num_mask |= (timeline_df[col] < val)
                elif op == "<=": t_num_mask |= (timeline_df[col] <= val)
                elif op == "==": t_num_mask |= (timeline_df[col] == val)
                elif op == ">=": t_num_mask |= (timeline_df[col] >= val)
                elif op == ">": t_num_mask |= (timeline_df[col] > val)
    else:
        timeline_df = raw_df
        # Invert original KEEP masks to get REMOVE masks
        t_date_mask = ~date_mask 
        t_num_mask = ~numeric_mask

    fig_timeline, ax_timeline = plt.subplots(figsize=(12, 3))
    
    # Fill 1: Removed by Numeric (Gray)
    # MODIFIED: Added linewidth=0
    ax_timeline.fill_between(
        timeline_df['DATETIME'], 0, 1,
        where=t_num_mask,
        color='gray', alpha=0.3, label='Removed (Numeric)', step='mid', edgecolor='none', linewidth=0
    )
    
    # Fill 2: Removed by Date (Purple) - Prioritize if overlapping
    # MODIFIED: Added linewidth=0
    ax_timeline.fill_between(
        timeline_df['DATETIME'], 0, 1,
        where=t_date_mask,
        color='purple', alpha=0.3, label='Removed (Datetime)', step='mid', edgecolor='none', linewidth=0
    )

    ax_timeline.get_yaxis().set_visible(False)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_xlabel('DATETIME')
    ax_timeline.legend(loc='upper right')
    #ax_timeline.grid(axis='x', linestyle='--', alpha=0.5)
    ax_timeline.set_title("Data Removal Timeline")
    
    st.pyplot(fig_timeline)

    if generate_report:
        buf = BytesIO()
        fig_timeline.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        buf.seek(0)
        plot_images.append(("Filter Timeline", base64.b64encode(buf.read()).decode('utf-8')))
        buf.close()

    plt.close(fig_timeline)

    # --- Metric Plots ---
    st.markdown("### Filtered Metric Plots")

    for metric in selected_metrics:
        st.subheader(f"Plot for: {metric}")

        # Calculate metric-specific masks for highlighting
        # Using plot_raw_df (downsampled or full)
        m_spec_mask = pd.Series(False, index=plot_raw_df.index)
        m_other_mask = pd.Series(False, index=plot_raw_df.index)
        m_date_mask = pd.Series(False, index=plot_raw_df.index)
        
        metric_filters_list = []

        # 1. Date Mask for Plotting
        if 'DATETIME' in plot_raw_df.columns:
            for op, val in datetime_filters:
                if op == "< (remove before)": m_date_mask |= (plot_raw_df['DATETIME'] < pd.to_datetime(val))
                elif op == "> (remove after)": m_date_mask |= (plot_raw_df['DATETIME'] > pd.to_datetime(val))
                elif op in ["between", "between (includes edge values)"]: m_date_mask |= ((plot_raw_df['DATETIME'] >= pd.to_datetime(val[0])) & (plot_raw_df['DATETIME'] <= pd.to_datetime(val[1])))

        # 2. Numeric Masks for Plotting
        for col, op, val in numeric_filters:
            if col not in plot_raw_df.columns: continue
            
            if op == "<": mask = (plot_raw_df[col] < val)
            elif op == "<=": mask = (plot_raw_df[col] <= val)
            elif op == "==": mask = (plot_raw_df[col] == val)
            elif op == ">=": mask = (plot_raw_df[col] >= val)
            elif op == ">": mask = (plot_raw_df[col] > val)
            else: continue

            if col == metric:
                m_spec_mask |= mask
                metric_filters_list.append((op, val))
            else:
                m_other_mask |= mask

        # Conditions for coloring
        cond_date = m_date_mask
        cond_spec = m_spec_mask & ~m_date_mask
        cond_other = m_other_mask & ~m_date_mask & ~m_spec_mask

        # Graph 2 (Raw with Highlights)
        fig_metric, ax_metric = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=plot_raw_df, x='DATETIME', y=metric, ax=ax_metric, label='Raw Data', color="green", linewidth=0.5)
        
        # Get limits
        if not plot_raw_df.empty:
            ymin, ymax = plot_raw_df[metric].min(), plot_raw_df[metric].max()
        else:
            ymin, ymax = 0, 1

        # MODIFIED: Added linewidth=0 to all fill_between calls
        ax_metric.fill_between(plot_raw_df['DATETIME'], ymin, ymax, where=cond_date, color='purple', alpha=0.3, label='Datetime Filter', linewidth=0)
        ax_metric.fill_between(plot_raw_df['DATETIME'], ymin, ymax, where=cond_other, color='gray', alpha=0.3, label='Other Numeric Filter', linewidth=0)
        ax_metric.fill_between(plot_raw_df['DATETIME'], ymin, ymax, where=cond_spec, color='blue', alpha=0.3, label=f'{metric} Filter', linewidth=0)
        
        ax_metric.set_title(f"{metric} - Raw Data & Filters")
        ax_metric.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        
        # Graph 3 (Cleaned)
        fig_cleaned, ax_cleaned = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=plot_cleaned_df, x='DATETIME', y=metric, ax=ax_cleaned, label='Cleaned Data', color="green", linewidth=0.5)
        
        # Add threshold lines
        x_pos = plot_cleaned_df['DATETIME'].min() if not plot_cleaned_df.empty else None
        if x_pos:
            for op, val in metric_filters_list:
                ax_cleaned.axhline(y=val, color='red', linestyle='--', linewidth=1)
                ax_cleaned.text(x=x_pos, y=val, s=f" {op} {val}", color='red', verticalalignment='bottom')

        ax_cleaned.set_ylim(ymin, ymax)
        ax_cleaned.set_title(f"{metric} - Cleaned Result")

        c1, c2 = st.columns(2)
        with c1: st.pyplot(fig_metric)
        with c2: st.pyplot(fig_cleaned)

        if generate_report:
            # Save Raw Plot
            buf1 = BytesIO()
            fig_metric.savefig(buf1, format="png", bbox_inches='tight', dpi=100)
            buf1.seek(0)
            plot_images.append((f"{metric} (Raw)", base64.b64encode(buf1.read()).decode('utf-8')))
            buf1.close()

            # Save Cleaned Plot
            buf2 = BytesIO()
            fig_cleaned.savefig(buf2, format="png", bbox_inches='tight', dpi=100)
            buf2.seek(0)
            plot_images.append((f"{metric} (Cleaned)", base64.b64encode(buf2.read()).decode('utf-8')))
            buf2.close()

        plt.close(fig_metric)
        plt.close(fig_cleaned)

    if generate_report:
        html_content = _generate_html_report(stats_payload, numeric_filters, datetime_filters, plot_images)
        
        # --- FIX: Enforce ProactorEventLoop for Windows (Critical for Playwright) ---
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_content(html_content)
                page.pdf(path=str(pdf_file_path), format="A4", margin={'top': '1cm', 'bottom': '1cm', 'left': '1cm', 'right': '1cm'})
                browser.close()
        except Exception as e:
            st.error(f"Failed to save PDF: {e}")

def split_holdout(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    split_mark: Union[float, str, pd.Timestamp],
    date_col: str = "Point Name",
    verbose: bool = False, # Set to True to print stats for logging
    remove_header_rows: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, Dict[str, Dict[str, Any]]]:
    """
    Split cleaned_df into train/validation and holdout sets by a date or percentage split.
    Optionally removes header rows and prepends them to split outputs.
    Also returns split statistics for each set.
    """

    df_header = None
    if remove_header_rows > 0:
        df_header = cleaned_df.iloc[:remove_header_rows].reset_index(drop=True)
        raw_df = raw_df.iloc[remove_header_rows:, :].reset_index(drop=True)
        cleaned_df = cleaned_df.iloc[remove_header_rows:, :].reset_index(drop=True)

    raw_df[date_col] = pd.to_datetime(raw_df[date_col])
    cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col])

    raw_df_sorted = raw_df.sort_values(date_col).reset_index(drop=True)
    cleaned_df_sorted = cleaned_df.sort_values(date_col).reset_index(drop=True)

    if not isinstance(split_mark, float):
        if not isinstance(split_mark, pd.Timestamp):
            split_mark = pd.to_datetime(split_mark)
    else:
        n = len(raw_df_sorted)
        h_raw = int(round(n * split_mark))
        split_idx = n - h_raw - 1 if h_raw < n else n - 1
        split_mark = raw_df_sorted.iloc[split_idx][date_col]

    train_val_df = cleaned_df_sorted[cleaned_df_sorted[date_col] <= split_mark].reset_index(drop=True)
    holdout_df = cleaned_df_sorted[cleaned_df_sorted[date_col] > split_mark].reset_index(drop=True)

    def get_stats(df):
        if len(df) > 0:
            start = df[date_col].iloc[0]
            end = df[date_col].iloc[-1]
            num_days = (end - start).days + 1
            return {"start": start, "end": end, "size": len(df), "num_days": num_days}
        else:
            return {"start": None, "end": None, "size": 0, "num_days": 0}

    stats = {
        "raw": get_stats(raw_df_sorted),
        "cleaned": get_stats(cleaned_df_sorted),
        "train_val": get_stats(train_val_df),
        "holdout": get_stats(holdout_df),
    }

    if verbose:
        for key, stat in stats.items():
            print(f"\n{key.title()} set:")
            if stat['size'] > 0:
                print(f"  Start time: {stat['start']}")
                print(f"  End time:   {stat['end']}")
                print(f"  Size:       {stat['size']} rows")
                print(f"  Num days:   {stat['num_days']}")
            else:
                print("  (Empty set)")

    if df_header is not None:
        train_val_df = pd.concat([df_header, train_val_df], ignore_index=True)
        holdout_df = pd.concat([df_header, holdout_df], ignore_index=True)

    return train_val_df, holdout_df, split_mark, stats

def generate_split_holdout_report(stats: Dict[str, Any], split_mark_used: str, pdf_file_path: Path, fig: plt.figure):
    """
    Generates a PDF report with the data split figure and stats tables.
    """
    # NOTE: Keeping FPDF here for now as requested to only change Data Cleansing, 
    # unless you want this one migrated to Playwright as well?
    # For now, focusing on the Data Cleansing module's consistency.
    pdf = FPDF(orientation="landscape")
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Data Split Report", 0, 1, "C")
    pdf.ln(5)

    # Add figure to pdf
    if fig:
        pdf.set_font("Arial", "B", 14)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png", bbox_inches='tight')
            img_buffer.seek(0)
            image_bytes = img_buffer.read()
            pdf.image(image_bytes, x=10, y=30, w=pdf.w - 20, type="PNG")
        plt.close(fig)
        pdf.ln(120)
    else:
        pdf.cell(0, 10, "No time span chart to display.", 0, 1, "L")

    # Add stats table to pdf
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Split Statistics", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Split mark used: {split_mark_used}", 0, 1)
    pdf.ln(5)

    for set_name, stat in stats.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"{set_name.title()}", 0, 1)
        pdf.set_font("Arial", "", 10)

        if stat.get("size", 0) > 0:
            start_str = stat.get("start").strftime("%Y-%m-%d %H:%M:%S") if stat.get("start") else "N/A"
            end_str = stat.get("end").strftime("%Y-%m-%d %H:%M:%S") if stat.get("end") else "N/A"

            # Create a simple table with borders (1)
            pdf.cell(40, 6, "Metric", 1)
            pdf.cell(0, 6, "Value", 1)
            pdf.ln()

            pdf.cell(40, 6, "Start", 1)
            pdf.cell(0, 6, start_str, 1)
            pdf.ln()

            pdf.cell(40, 6, "End", 1)
            pdf.cell(0, 6, end_str, 1)
            pdf.ln()

            pdf.cell(40, 6, "Rows", 1)
            pdf.cell(0, 6, f"{stat.get('size'):,}", 1) # Added comma formatting
            pdf.ln()

            pdf.cell(40, 6, "Number of days", 1)
            pdf.cell(0, 6, str(stat.get("num_days", "N/A")), 1)
            pdf.ln()

            pdf.ln(5) # Space after table
        else:
            pdf.cell(0, 6, "(Empty set)", 0, 1)
            pdf.ln(5)

    try:
        pdf.output(pdf_file_path)
        return True
    except Exception as e:
        return False

def read_prism_csv(df: pd.DataFrame):
    # df = pd.read_csv(path, index_col=False)

    df_header = df.iloc[:4,:]

    df.columns = df.iloc[1].values
    df = df.iloc[4:,:]

    df.rename(columns={'Extended Name':'DATETIME'}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.apply(pd.to_numeric)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    return df, df_header

def corrfunc(x, y, **kwds):
    cmap = kwds['cmap']
    norm = kwds['norm']
    ax = plt.gca()
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
    r, _ = pearsonr(x, y)
    facecolor = cmap(norm(r))
    ax.set_facecolor(facecolor)
    lightness = (max(facecolor[:3]) + min(facecolor[:3]) ) / 2
    ax.annotate(f"r={r:.2f}", xy=(.5, .5), xycoords=ax.transAxes, color='white' if lightness < 0.7 else 'black', size=30, ha='center', va='center')