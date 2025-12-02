import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st
import numpy as np
import base64
from datetime import datetime
import os

from typing import Tuple, Union, Dict, Any
from pathlib import Path
from scipy.stats import pearsonr
from io import BytesIO
from playwright.sync_api import sync_playwright
from jinja2 import Environment
import sys
import asyncio

def cleaned_dataset_name_split(filename:str) -> Tuple [str, str, str]:
    """
    Given a filename, returns the site_name, model_name,
    and inclusive dates.
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
    try:
        site_name = model_name.split("-")[1]
    except IndexError:
        site_name = "Unknown"
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
    first_col_name = df_data.columns[0]
    df_data.rename(columns={first_col_name: 'DATETIME'}, inplace=True)
    
    # Optimized Datetime Conversion
    df_data['DATETIME'] = pd.to_datetime(df_data['DATETIME'], errors='coerce')
    
    # Reset index after slicing
    df_data.reset_index(drop=True, inplace=True)

    # Optimized Numeric Conversion
    cols_to_convert = [c for c in df_data.columns if c != 'DATETIME']
    for col in cols_to_convert:
        df_data[col] = pd.to_numeric(df_data[col], errors='coerce')

    # Optimized Column Mapping
    name_to_metric = pd.Series(
        project_points['Metric'].values, 
        index=project_points['Name']
    ).to_dict()

    new_columns = []
    for column in df_data.columns:
        if column == 'DATETIME':
            new_columns.append(column)
        else:
            mapping_val = name_to_metric.get(column)
            if mapping_val is not None and str(mapping_val) != 'nan':
                new_columns.append(str(mapping_val))
            else:
                new_columns.append(column)

    df_data.columns = new_columns
    
    return df_data, df_header

def scan_folders_structure(root_path: str) -> pd.DataFrame:
    """
    Scans the folder structure: Root -> Site -> System -> Sprint -> Model
    Checks for .dat files existence in relative_deviation folder.
    Shared by Model Accuracy and Model FPR modules.
    """
    found_models = []
    root = Path(root_path)
    
    if not root.exists():
        return pd.DataFrame()

    # We expect strict depth: Site (1) -> System (2) -> Sprint (3) -> Model (4)
    try:
        # Level 1: Site
        for site_dir in [d for d in root.iterdir() if d.is_dir()]:
            # Level 2: System
            for system_dir in [d for d in site_dir.iterdir() if d.is_dir()]:
                # Level 3: Sprint
                for sprint_dir in [d for d in system_dir.iterdir() if d.is_dir()]:
                    # Level 4: Model
                    for model_dir in [d for d in sprint_dir.iterdir() if d.is_dir()]:
                        model_name = model_dir.name
                        
                        # Check 1: .dat file existence (For Accuracy)
                        rel_dev_path = model_dir / "relative_deviation"
                        dat_file_path = rel_dev_path / f"{model_name}.dat"
                        has_dat = dat_file_path.exists() and dat_file_path.is_file()

                        # Check 2: Dataset folder existence (For FPR)
                        dataset_path = model_dir / "dataset"
                        has_dataset = dataset_path.exists() and dataset_path.is_dir()
                        
                        # Find Files
                        raw_file = None
                        holdout_file = None
                        omr_wo = None
                        omr_w = None
                        omr_hold = None
                        
                        if has_dataset:
                            # Standard CSVs
                            raw_candidates = list(dataset_path.glob("*RAW.csv"))
                            if raw_candidates: raw_file = raw_candidates[0].name
                            
                            hold_candidates = list(dataset_path.glob("*HOLDOUT.csv"))
                            if hold_candidates: holdout_file = hold_candidates[0].name
                            
                            # OMR Files
                            omr_files = list(dataset_path.glob("*OMR*.dat"))
                            omr_wo = next((f.name for f in omr_files if "WITHOUT-OUTLIER" in f.name), None)
                            omr_w = next((f.name for f in omr_files if "WITH-OUTLIER" in f.name), None)
                            omr_hold = next((f.name for f in omr_files if "HOLDOUT" in f.name), None)

                        found_models.append({
                            "Select": False,
                            "Site": site_dir.name,
                            "System": system_dir.name,
                            "Sprint": sprint_dir.name,
                            "Model": model_name,
                            "Dat File Found": has_dat,
                            "Dataset Found": has_dataset,
                            "Raw File": raw_file,
                            "Holdout File": holdout_file,
                            "OMR Cleaned File": omr_wo,
                            "OMR Raw File": omr_w,
                            "OMR Holdout File": omr_hold,
                            "Dat Filename": dat_file_path.name if has_dat else None,
                            "Full Path": str(model_dir),
                            "Dat Path": str(dat_file_path) if has_dat else None,
                            "Dataset Path": str(dataset_path) if has_dataset else None
                        })
    except Exception as e:
        st.error(f"Error scanning directories: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(found_models)
    
    # Sort results
    if not df.empty:
        df = df.sort_values(by=['Site', 'System', 'Sprint', 'Model'])
        
    return df

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
        <h1>Data Cleansing Report</h1>
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
            PRISM Web Toolkit - Data Cleansing Report
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
        color='gray', alpha=0.3, label='Numeric Removed', step='mid', edgecolor='none', linewidth=0
    )

    # Fill 2: Removed by Date (Purple) - Prioritize if overlapping
    # MODIFIED: Added linewidth=0
    ax_timeline.fill_between(
        timeline_df['DATETIME'], 0, 1,
        where=t_date_mask,
        color='purple', alpha=0.3, label='Date Removed', step='mid', edgecolor='none', linewidth=0
    )

    ax_timeline.get_yaxis().set_visible(False)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_xlabel('DATETIME')
    ax_timeline.legend(loc='upper right')
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
        sns.lineplot(data=plot_cleaned_df, x='DATETIME', y=metric, ax=ax_cleaned, color="green", linewidth=0.5, legend=False)

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

def split_holdout(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, split_mark: Union[float, str, pd.Timestamp], date_col: str = "Point Name", verbose: bool = False, remove_header_rows: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, Dict[str, Dict[str, Any]]]:
    """Split cleaned_df into train/validation and holdout sets."""
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

    if df_header is not None:
        train_val_df = pd.concat([df_header, train_val_df], ignore_index=True)
        holdout_df = pd.concat([df_header, holdout_df], ignore_index=True)

    return train_val_df, holdout_df, split_mark, stats

def split_holdout(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, split_mark: Union[float, str, pd.Timestamp], date_col: str = "Point Name", verbose: bool = False, remove_header_rows: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, Dict[str, Dict[str, Any]]]:
    """Split cleaned_df into train/validation and holdout sets."""
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

    if df_header is not None:
        train_val_df = pd.concat([df_header, train_val_df], ignore_index=True)
        holdout_df = pd.concat([df_header, holdout_df], ignore_index=True)

    return train_val_df, holdout_df, split_mark, stats

def generate_split_holdout_report(stats: Dict[str, Any], split_mark_used: str, pdf_file_path: Path, fig: plt.figure):
    """Generates a PDF report with the data split figure and stats tables using Playwright/HTML."""
    env = Environment()
    template_str = """
    <html>
    <head>
        <style>
            body { font-family: "Helvetica", "Arial", sans-serif; color: #333; margin: 40px; }
            h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
            h2 { color: #2980b9; margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            .info-box { background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
            .footer { font-size: 0.8em; color: #999; text-align: center; margin-top: 50px; }
            .stat-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 12px; }
            .stat-table th, .stat-table td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            .stat-table th { background-color: #eee; font-weight: bold; }
            .img-container { text-align: center; margin-top: 20px; page-break-inside: avoid; }
            img { max-width: 100%; height: auto; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <h1>Holdout Split Report</h1>
        <p>Generated on: {{ generation_date }}</p>
        <div class="info-box"><strong>Split Configuration</strong><br>Split Mark Used: {{ split_mark }}</div>
        <h2>Split Statistics</h2>
        <table class="stat-table">
            <thead><tr><th>Dataset</th><th>Start Time</th><th>End Time</th><th>Rows</th><th>Duration (Days)</th></tr></thead>
            <tbody>
                {% for set_name, stat in stats.items() %}
                <tr>
                    <td><strong>{{ set_name }}</strong></td>
                    <td>{{ stat.start }}</td>
                    <td>{{ stat.end }}</td>
                    <td>{{ stat.formatted_size }}</td>
                    <td>{{ stat.num_days }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <h2>Timeline Visualization</h2>
        {% if img_data %}<div class="img-container"><img src="data:image/png;base64,{{ img_data }}" /></div>{% endif %}

        <div class="footer">
            PRISM Web Toolkit - Holdout Split Report
        </div>
    </body>
    </html>
    """
    img_b64 = None
    if fig:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

    formatted_stats = {}
    for key, val in stats.items():
        formatted_stats[key.title()] = {
            "size": val.get("size", 0),
            "formatted_size": f"{val.get('size', 0):,}",
            "start": val.get("start").strftime("%Y-%m-%d %H:%M:%S") if val.get("start") else "N/A",
            "end": val.get("end").strftime("%Y-%m-%d %H:%M:%S") if val.get("end") else "N/A",
            "num_days": val.get("num_days", 0)
        }

    template = env.from_string(template_str)
    html = template.render(generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), split_mark=split_mark_used, stats=formatted_stats, img_data=img_b64)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html)
            page.pdf(path=str(pdf_file_path), format="A4", margin={'top': '1cm', 'bottom': '1cm', 'left': '1cm', 'right': '1cm'})
            browser.close()
        return True
    except Exception as e:
        print(f"PDF generation error: {e}")
        return False

def read_prism_csv(df: pd.DataFrame):
    df_header = df.iloc[:4,:].copy()
    df_data = df.iloc[4:, :].copy()
    first_col_name = df_data.columns[0]
    df_data.rename(columns={first_col_name: 'DATETIME'}, inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    df_data['DATETIME'] = pd.to_datetime(df_data['DATETIME'])
    df_data = df_data.apply(pd.to_numeric)
    df_data['DATETIME'] = pd.to_datetime(df_data['DATETIME'])
    return df_data, df_header

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

# --- NEW FUNCTIONS FOR TRAIN/VAL SPLITTING ---
def _generate_html_tvs_report(stats_payload, numeric_filters, datetime_filters, plot_images, title="Data Cleaning Report"):
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
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 12px; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            th { background-color: #eee; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>{{ title }}</h1>
        <p>Generated on: {{ generation_date }}</p>

        {% if stats_payload %}
        <div class="stats-box">
            <h2>Statistics</h2>
            {% for row_key, row_val in stats_payload.items() %}
                <div class="stats-row">
                    <div class="stat-item">
                        <div class="stat-label">{{ row_key }}</div>
                        <div class="stat-value">{{ row_val }}</div>
                    </div>
                </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if numeric_filters or datetime_filters %}
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
        {% endif %}

        <h2>Visualizations</h2>
        
        {% for title, img_data in plot_images %}
        <div class="img-container">
            <h3>{{ title }}</h3>
            <img src="data:image/png;base64,{{ img_data }}" />
        </div>
        {% endfor %}

        <div class="footer">
            PRISM Web Toolkit - Training/Validation Split Report
        </div>
    </body>
    </html>
    """
    
    template = env.from_string(template_str)
    
    # Format filters for display
    fmt_numeric = [f"{col} {op} {val}" for col, op, val in numeric_filters] if numeric_filters else []
    fmt_date = []
    if datetime_filters:
        for op, val in datetime_filters:
            if op == "between (includes edge values)" or op == "between":
                val_str = f"{val[0]} to {val[1]}" if isinstance(val, tuple) or isinstance(val, list) else str(val)
                fmt_date.append(f"Remove between {val_str}")
            else:
                fmt_date.append(f"{op} {val}")

    html = template.render(
        title=title,
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stats_payload=stats_payload,
        numeric_filters=fmt_numeric,
        datetime_filters=fmt_date,
        plot_images=plot_images
    )
    return html

def generate_tvs_report(stats: Dict[str, Any], plot_images: list, pdf_file_path: Path):
    """
    Generates a PDF report for Training/Validation splitting results.
    """
    stats_payload = {
        "Original Rows (Cleaned w/o Outlier)": f"{stats.get('original_len', 0):,}",
        "Training Set Rows": f"{stats.get('train_len', 0):,}",
        "Validation Set (w/o Outlier) Rows": f"{stats.get('val_wo_len', 0):,}",
        "Validation Set (w/ Outlier) Rows": f"{stats.get('val_w_len', 0):,}"
    }

    html_content = _generate_html_tvs_report(stats_payload, [], [], plot_images, title="Training-Validation Split Report")
    
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html_content)
            page.pdf(path=str(pdf_file_path), format="A4", margin={'top': '1cm', 'bottom': '1cm', 'left': '1cm', 'right': '1cm'})
            browser.close()
        return True
    except Exception as e:
        print(f"Failed to generate TVS PDF: {e}")
        return False

def generate_tvs_visualizations(df_train: pd.DataFrame, selected_metrics: list, df_val: pd.DataFrame = None):
    """
    Generates side-by-side plots for selected metrics.
    Plot 1: Time Series (Train)
    Plot 2: Distribution (Train vs Validation Overlay if df_val provided, else just Train Hist)
    
    Returns a list of (title, base64_img) tuples for the report.
    """
    plot_images = []
    
    # Downsample for performance if needed (Train)
    MAX_POINTS = 5000
    if len(df_train) > MAX_POINTS:
        step = len(df_train) // MAX_POINTS
        plot_df = df_train.iloc[::step].copy()
    else:
        plot_df = df_train

    # Downsample for performance if needed (Validation)
    plot_val = None
    if df_val is not None:
        if len(df_val) > MAX_POINTS:
            step_val = len(df_val) // MAX_POINTS
            plot_val = df_val.iloc[::step_val].copy()
        else:
            plot_val = df_val

    for metric in selected_metrics:
        if metric not in plot_df.columns:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        # 1. Line Plot (Train Data Pattern)
        sns.lineplot(data=plot_df, x='DATETIME', y=metric, ax=ax1, linewidth=0.5, color='tab:blue')
        ax1.set_title(f"{metric} - Time Series (Train)")
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution Plot
        if plot_val is not None and metric in plot_val.columns:
            # Overlay KDE Plot
            sns.kdeplot(data=plot_df, x=metric, ax=ax2, color='tab:blue', label='Train', fill=True, alpha=0.3)
            sns.kdeplot(data=plot_val, x=metric, ax=ax2, color='tab:orange', label='Validation', fill=True, alpha=0.3)
            ax2.set_title(f"{metric} - Distribution Overlay (Train vs Val)")
            ax2.legend()
        else:
            # Fallback to simple Histogram
            sns.histplot(data=plot_df, x=metric, ax=ax2, kde=True, color="orange")
            ax2.set_title(f"{metric} - Distribution (Train)")
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig) # Render in Streamlit
        
        # Save for Report
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        buf.seek(0)
        plot_images.append((f"{metric} Plots", base64.b64encode(buf.read()).decode('utf-8')))
        buf.close()
        plt.close(fig)
        
    return plot_images