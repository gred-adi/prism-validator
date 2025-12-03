import streamlit as st
import pandas as pd
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime, time as dt_time

from utils.model_dev_utils import (
    data_cleaning_read_prism_csv, 
    cleaned_dataset_name_split, 
    generate_data_cleaning_visualizations, 
    generate_simple_report
)

st.set_page_config(page_title="Data Cleansing", page_icon="1️⃣", layout="wide")

st.title("🪄 Data Cleansing Wizard")
st.markdown("""
This wizard guides you through cleaning raw time-series datasets. 
It automates column mapping using your TDT files and provides an interactive interface for applying numeric and datetime filters.

**How to Use:**
1.  **Ingest Data:** Upload your raw CSV dataset. The tool will automatically map column names using the TDT Survey data loaded in the **Global Settings** sidebar.
2.  **Cleanse:** Use the interactive controls to apply numeric thresholds and exclude specific time ranges (e.g., maintenance periods).
3.  **Review & Export:** Check the impact of your filters and download the cleaned dataset along with a PDF report.
""")

# --- Initialize Session State ---
if 'cleansing_step' not in st.session_state: st.session_state.cleansing_step = 1

# Data Processing States
if 'filters_applied' not in st.session_state: st.session_state.filters_applied = False
if 'generate_report' not in st.session_state: st.session_state.generate_report = False
if 'show_visuals' not in st.session_state: st.session_state.show_visuals = False
if 'filters' not in st.session_state: st.session_state.filters = []
if 'datetime_filters' not in st.session_state: st.session_state.datetime_filters = []

# Cached Data States
if 'model' not in st.session_state: st.session_state.model = None
if 'project_points_df' not in st.session_state: st.session_state.project_points_df = None
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'raw_df_header' not in st.session_state: st.session_state.raw_df_header = None
if 'cleaned_df' not in st.session_state: st.session_state.cleaned_df = None

# TDT Data State (Ensure it exists)
if 'survey_df' not in st.session_state: st.session_state.survey_df = None

# Model Metadata States
if 'site_name' not in st.session_state: st.session_state.site_name = ""
if 'system_name' not in st.session_state: st.session_state.system_name = ""
if 'model_name' not in st.session_state: st.session_state.model_name = ""
if 'sprint_name' not in st.session_state: st.session_state.sprint_name = ""
if 'inclusive_dates' not in st.session_state: st.session_state.inclusive_dates = ""

# --- Helper Functions for State Management ---

def set_step(step):
    st.session_state.cleansing_step = step

def next_step():
    st.session_state.cleansing_step += 1

def prev_step():
    st.session_state.cleansing_step -= 1

def add_filter(column, operator, value):
    st.session_state.filters.append((column, operator, value))

def delete_filter(index):
    if 0 <= index < len(st.session_state.filters):
        st.session_state.filters.pop(index)

def clear_all_numeric_filters():
    st.session_state.filters = []

def add_date_filter(op, val):
    st.session_state.datetime_filters.append((op, val))

def delete_date_filter(index):
    if 0 <= index < len(st.session_state.datetime_filters):
        st.session_state.datetime_filters.pop(index)

def clear_all_date_filters():
    st.session_state.datetime_filters = []

def calculate_impact_breakdown(df, num_filters, dt_filters):
    """
    Calculates how many rows would remain after filtering, and breaks down 
    how many are removed by numeric vs date filters specifically.
    """
    if df is None or df.empty:
        return 0, 0, 0
    
    total_rows = len(df)
    
    # 1. Calculate Numeric Mask (Rows to KEEP based on numeric rules)
    numeric_mask = pd.Series(True, index=df.index)
    for col, op, val in num_filters:
        if col in df.columns:
            if op == "<": numeric_mask &= ~(df[col] < val)
            elif op == "<=": numeric_mask &= ~(df[col] <= val)
            elif op == "==": numeric_mask &= ~(df[col] == val)
            elif op == ">=": numeric_mask &= ~(df[col] >= val)
            elif op == ">": numeric_mask &= ~(df[col] > val)

    numeric_removed_count = (~numeric_mask).sum()

    # 2. Calculate Date Mask (Rows to KEEP based on date rules)
    date_mask = pd.Series(True, index=df.index)
    if 'DATETIME' in df.columns:
        for op, val in dt_filters:
            if op == "< (remove before)":
                date_mask &= ~(df['DATETIME'] < val)
            elif op == "> (remove after)":
                date_mask &= ~(df['DATETIME'] > val)
            elif op == "between (includes edge values)" or op == "between":
                date_mask &= ~((df['DATETIME'] >= val[0]) & (df['DATETIME'] <= val[1]))
    
    date_removed_count = (~date_mask).sum()

    # 3. Combined Result
    final_mask = numeric_mask & date_mask
    remaining_count = final_mask.sum()
    
    return remaining_count, numeric_removed_count, date_removed_count

# --- JSON Serialization Helper ---
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def load_filters_from_json(json_data):
    """Parses JSON data and populates session state filters."""
    try:
        if "numeric" in json_data:
            st.session_state.filters = json_data["numeric"]

        if "datetime" in json_data:
            loaded_dt_filters = []
            for op, val in json_data["datetime"]:
                # Handle both verbose and simple "between" keys
                if op in ["between", "between (includes edge values)"] and isinstance(val, list):
                    start_ts = pd.to_datetime(val[0])
                    end_ts = pd.to_datetime(val[1])
                    loaded_dt_filters.append((op, (start_ts, end_ts)))
                else:
                    ts = pd.to_datetime(val)
                    loaded_dt_filters.append((op, ts))
            
            st.session_state.datetime_filters = loaded_dt_filters
            
        st.success("Filters loaded successfully!")
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to load filters: {e}")

def read_process_cache_files(raw_file, point_list_df):
    """
    Processes the raw file and the fetched point list dataframe.
    """
    # Extract info silently without UI blocking
    auto_site_name, auto_model_name, auto_inclusive_dates = cleaned_dataset_name_split(raw_file.name)
    
    # Populate session state if empty, otherwise keep existing (user might have edited it in previous run)
    if not st.session_state.site_name: st.session_state.site_name = auto_site_name
    if not st.session_state.model_name: st.session_state.model_name = auto_model_name
    if not st.session_state.inclusive_dates: st.session_state.inclusive_dates = auto_inclusive_dates

    progress_text = "Starting file processing..."
    my_bar = st.progress(0, text=progress_text)

    try:
        my_bar.progress(20, text="Reading CSV files...")
        raw_file_df = pd.read_csv(raw_file)
        
        # Point list is already a DF now
        project_points_df = point_list_df

        my_bar.progress(50, text="Formatting and mapping data points...")
        # Using the optimized version from previous step implicitly if utils updated
        raw_df, raw_df_header = data_cleaning_read_prism_csv(raw_file_df, project_points_df)

        my_bar.progress(100, text="Processing complete!")
        time.sleep(0.5)
        my_bar.empty()

        st.session_state.model = raw_file.name
        st.session_state.project_points_df = project_points_df
        st.session_state.raw_df = raw_df
        st.session_state.raw_df_header = raw_df_header

        return project_points_df, raw_df, raw_df_header

    except Exception as e:
        my_bar.empty()
        st.error(f"An error occurred during processing: {e}")
        return None, None, None

# Progress Indicator
steps = ["Data Ingestion", "Interactive Cleansing", "Review & Export"]
current_step = st.session_state.cleansing_step
st.progress(current_step / len(steps), text=f"Step {current_step}: {steps[current_step-1]}")

# ==========================================
# STEP 1: DATA INGESTION
# ==========================================
if current_step == 1:
    st.header("Step 1: Data Ingestion")
    st.markdown("Upload your Raw Dataset. The Point List mapping will be retrieved automatically from the **TDT Survey** (loaded on Home) based on the model name in your file.")

    raw_file = st.file_uploader("Upload RAW dataset (.csv)", type=["csv"], accept_multiple_files=False)
    
    # Placeholder for Point List Status
    point_list_container = st.container()

    # --- Simplified Cache / Process Logic ---
    if raw_file is not None:
        
        # 1. Determine Model Name from File
        try:
            _, auto_model_name, _ = cleaned_dataset_name_split(raw_file.name)
        except Exception:
            auto_model_name = "Unknown"
            
        point_list_df = None
        
        # 2. Fetch Point List from TDT Survey (Replaces DB Logic)
        with point_list_container:
            # Ensure survey_df is initialized before checking it
            if st.session_state.survey_df is None:
                st.error("❌ TDT Data not found. Please go to the **Home** page and load your TDT files first.")
                st.stop()
            
            survey_df = st.session_state.survey_df
            
            if auto_model_name != "Unknown":
                # Filter for the specific model
                model_survey = survey_df[survey_df['Model'] == auto_model_name]
                
                if not model_survey.empty:
                    # Identify the correct column for point name
                    point_col = 'Canary Point Name' if 'Canary Point Name' in model_survey.columns else 'Point Name'
                    
                    if point_col in model_survey.columns:
                        # Create a simple dataframe with 'Metric' and 'Name' cols for the utils function
                        point_list_df = model_survey[['Metric', point_col]].rename(columns={point_col: 'Name'})
                        
                        st.success(f"✅ Point List found in TDT! {len(point_list_df)} metrics loaded for `{auto_model_name}`.")
                        with st.expander("View Mapped Points"):
                            st.dataframe(point_list_df)
                    else:
                        st.error(f"❌ Could not find '{point_col}' column in TDT Survey data.")
                        st.stop()
                else:
                    st.error(f"❌ Model `{auto_model_name}` not found in the loaded TDT Survey data. Please ensure the correct TDT file is loaded on the Home page.")
                    st.stop()
            else:
                st.error("Could not parse Model Name from filename. Please ensure file follows format: `CLEANED-{ModelName}-{Dates}-RAW.csv`")
                st.stop()

        # 3. Process Files (Only if point list was successfully fetched)
        if point_list_df is not None:
            # Check if we need to process (New file OR Force Reload)
            if st.session_state.raw_df is None or st.session_state.model != raw_file.name:
                
                # Clear filters only if it's a genuinely new file load (not just a rerun)
                if st.session_state.raw_df is None:
                    if 'filters' in st.session_state:
                        clear_all_numeric_filters()
                        clear_all_date_filters()
                
                read_process_cache_files(raw_file, point_list_df)
                st.rerun()
                
            else:
                # Data is already loaded and matches current file
                st.success(f"✅ Loaded **{raw_file.name}** from cache.")
                
                # Optional: Force Reload Button
                if st.button("↻ Force Re-process File"):
                    st.session_state.raw_df = None
                    st.session_state.model = None
                    st.rerun()

    # --- Preview & Navigation ---
    if st.session_state.raw_df is not None:
        
        with st.expander("View Dataset Preview & Stats", expanded=True):
            tab1, tab2 = st.tabs(["Preview", "Statistics"])
            with tab1:
                st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)
            with tab2:
                st.dataframe(st.session_state.raw_df.describe(), use_container_width=True)
        
        st.markdown("---")
        col_back, col_next = st.columns([1, 5])
        with col_next:
            st.button("Next: Configure Filters ➡️", on_click=next_step, type="primary")

# ==========================================
# STEP 2: INTERACTIVE CLEANSING
# ==========================================
elif current_step == 2:
    st.header("Step 2: Interactive Cleansing")
    
    if st.session_state.raw_df is None:
        st.warning("No data found. Please go back to Step 1.")
        st.button("Back to Step 1", on_click=prev_step)
        st.stop()
        
    raw_df = st.session_state.raw_df

    # --- Filter Management Sidebar/Expander ---
    with st.expander("📂 **Manage Presets (Import / Export Filters)**"):
        col_export, col_import = st.columns(2)
        with col_export:
            current_config = {
                "numeric": st.session_state.filters,
                "datetime": st.session_state.datetime_filters
            }
            json_str = json.dumps(current_config, default=json_serial, indent=2)
            
            # --- UPDATED FILENAME LOGIC ---
            export_fname = "filters.json"
            if st.session_state.model_name and st.session_state.inclusive_dates:
                export_fname = f"filter-{st.session_state.model_name}-{st.session_state.inclusive_dates}.json"
            
            st.download_button("💾 Download Preset", json_str, export_fname, "application/json", use_container_width=True)
        with col_import:
            uploaded_preset = st.file_uploader("Upload .json preset", type=["json"], label_visibility="collapsed")
            if uploaded_preset and st.button("Load Preset", use_container_width=True):
                data = json.load(uploaded_preset)
                load_filters_from_json(data)

    # --- Filter UI ---
    filter_col1, filter_col2 = st.columns(2, gap="medium")

    # Column 1: Numerical
    with filter_col1:
        with st.container(border=True):
            st.subheader("🔢 Numeric Filters")
            num_col = st.selectbox("Column", raw_df.columns[1:], key="filter_col")
            
            # Smart Context: Stats
            col_stats = raw_df[num_col].describe()
            st.caption(f"Min: {col_stats['min']:.2f} | Max: {col_stats['max']:.2f} | Mean: {col_stats['mean']:.2f}")
            
            # Smart Context: Visualizations (Histogram + Line Plot)
            viz_tab1, viz_tab2 = st.tabs(["Histogram", "Time Series"])
            
            with viz_tab1:
                try:
                    fig_hist, ax_hist = plt.subplots(figsize=(5, 2))
                    # Use dropna to avoid plotting issues
                    data_to_plot = raw_df[num_col].dropna()
                    sns.histplot(data_to_plot, bins=30, kde=False, ax=ax_hist, color="#3366ff", edgecolor=None)
                    ax_hist.set_title(f"Distribution", fontsize=8)
                    ax_hist.tick_params(axis='both', which='major', labelsize=7)
                    ax_hist.set_xlabel("")
                    ax_hist.set_ylabel("")
                    sns.despine(left=True, bottom=True)
                    ax_hist.grid(axis='x', alpha=0.3)
                    st.pyplot(fig_hist, use_container_width=True)
                    plt.close(fig_hist)
                except Exception as e:
                    st.caption("Could not generate histogram.")

            with viz_tab2:
                try:
                    # Downsample for preview speed (max 2000 points)
                    preview_df = raw_df[['DATETIME', num_col]].dropna()
                    if len(preview_df) > 2000:
                        preview_df = preview_df.iloc[::len(preview_df)//2000]
                    
                    fig_line, ax_line = plt.subplots(figsize=(5, 2))
                    ax_line.plot(preview_df['DATETIME'], preview_df[num_col], color="#3366ff", linewidth=0.8)
                    ax_line.set_title(f"Trend (Sampled)", fontsize=8)
                    ax_line.tick_params(axis='both', which='major', labelsize=7)
                    
                    # Format dates nicely
                    ax_line.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax_line.set_xlabel("")
                    
                    sns.despine(left=True, bottom=True)
                    ax_line.grid(alpha=0.3)
                    st.pyplot(fig_line, use_container_width=True)
                    plt.close(fig_line)
                except Exception as e:
                    st.caption(f"Could not generate line plot: {e}")

            num_op = st.selectbox("Operator", ["<", "<=", "==", ">=", ">"], key="filter_op")
            num_val = st.number_input("Value", value=0.0, step=1.0, key="filter_val")
            if st.button("Add Numeric Filter", use_container_width=True):
                add_filter(num_col, num_op, num_val)

    # Column 2: Date (UPDATED UI)
    with filter_col2:
        with st.container(border=True):
            st.subheader("📅 Date Filters")
            st.markdown("Use the slider to select a range, then choose to **Remove** data inside it or **Keep** only that range.")
            
            # Get date limits from data
            min_dt = raw_df['DATETIME'].min().to_pydatetime()
            max_dt = raw_df['DATETIME'].max().to_pydatetime()
            
            if min_dt and max_dt and min_dt < max_dt:
                # Action Selector
                filter_mode = st.radio(
                    "Filter Mode", 
                    ["Remove Range", "Keep Range"], 
                    horizontal=True,
                    help="'Remove Range' deletes data inside the blue slider area. 'Keep Range' deletes everything OUTSIDE."
                )
                
                # Bi-directional Slider
                # User requested slider for date, manual for time/precision
                sel_range = st.slider(
                    "Select Date Range",
                    min_value=min_dt,
                    max_value=max_dt,
                    value=(min_dt, max_dt),
                    format="MM/DD/YY"
                )
                
                # Manual Time Inputs for precision
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    start_time = st.time_input("Start Time", value=sel_range[0].time(), key="man_start_time")
                with t_col2:
                    end_time = st.time_input("End Time", value=sel_range[1].time(), key="man_end_time")
                
                # Combine
                start_val = datetime.combine(sel_range[0].date(), start_time)
                end_val = datetime.combine(sel_range[1].date(), end_time)
                
                if st.button("Apply Date Filter", use_container_width=True):
                    if filter_mode == "Remove Range":
                        # Logic: Remove data between start and end
                        add_date_filter("between", (start_val, end_val))
                        st.toast(f"Added: Remove data between {start_val.strftime('%Y-%m-%d %H:%M')} and {end_val.strftime('%Y-%m-%d %H:%M')}", icon="🗑️")
                        
                    else: # Keep Range
                        # Logic: Remove everything BEFORE start AND everything AFTER end
                        # This effectively "trims" the dataset to the selected window
                        add_date_filter("< (remove before)", start_val)
                        add_date_filter("> (remove after)", end_val)
                        st.toast(f"Added: Keep data only between {start_val.strftime('%Y-%m-%d %H:%M')} and {end_val.strftime('%Y-%m-%d %H:%M')}", icon="✅")
            else:
                st.warning("Insufficient date data to render slider.")

    # --- Impact Preview & Active Rules ---
    st.markdown("---")
    
    # Calculate Impact Breakdown
    total_rows = len(raw_df)
    remaining_rows, numeric_removed, date_removed = calculate_impact_breakdown(raw_df, st.session_state.filters, st.session_state.datetime_filters)
    total_removed_rows = total_rows - remaining_rows
    pct_removed = (total_removed_rows / total_rows) * 100 if total_rows > 0 else 0
    pct_numeric_removed = (remaining_rows / total_rows) * 100 if total_rows > 0 else 0
    pct_date_removed = (date_removed / total_rows) * 100 if total_rows > 0 else 0
    
    # Display Impact Metrics in 4 columns to include breakdown
    st.subheader("Preview Impact")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{total_rows:,}")
    m2.metric("Rows Remaining", f"{remaining_rows:,}", delta=f"-{pct_removed:.2f}% removed", delta_color="inverse")
    m3.metric("Numeric Removed", f"{numeric_removed:,}", delta=f"-{pct_numeric_removed:.2f}% removed", delta_color="inverse", help="Rows removed based on numeric filters")
    m4.metric("Date Removed", f"{date_removed:,}", delta=f"-{pct_date_removed:.2f}% removed", delta_color="inverse", help="Rows removed based on date filters")

    st.caption(f"**Overall Retention:** {100 - pct_removed:.2f}% kept ({pct_removed:.2f}% removed)")

    # Active Rules List
    st.subheader("Active Rules")
    if not st.session_state.filters and not st.session_state.datetime_filters:
        st.info("No filters added yet.")
    else:
        ac1, ac2 = st.columns(2)
        with ac1:
            if st.session_state.filters:
                st.markdown("**Numeric:**")
                for i, (col, op, val) in enumerate(st.session_state.filters):
                    c1, c2 = st.columns([0.8, 0.2])
                    c1.code(f"{col} {op} {val}")
                    c2.button("❌", key=f"d_n_{i}", on_click=delete_filter, args=(i,))
        with ac2:
            if st.session_state.datetime_filters:
                st.markdown("**Date:**")
                for i, (op, val) in enumerate(st.session_state.datetime_filters):
                    c1, c2 = st.columns([0.8, 0.2])
                    if isinstance(val, tuple):
                        val_str = f"{val[0].strftime('%Y-%m-%d %H:%M')} to {val[1].strftime('%Y-%m-%d %H:%M')}"
                    else:
                        val_str = val.strftime('%Y-%m-%d %H:%M')
                    
                    c1.code(f"{op} {val_str}")
                    c2.button("❌", key=f"d_d_{i}", on_click=delete_date_filter, args=(i,))

    st.markdown("---")
    col_back, col_next = st.columns([1, 5])
    with col_back:
        st.button("⬅️ Back", on_click=prev_step)
    with col_next:
        st.button("Next: Review & Process ➡️", on_click=next_step, type="primary")

# ==========================================
# STEP 3: REVIEW & EXPORT
# ==========================================
elif current_step == 3:
    st.header("Step 3: Review & Export")
    
    if st.session_state.raw_df is None:
        st.error("No data loaded.")
        st.stop()

    raw_df = st.session_state.raw_df
    raw_df_header = st.session_state.raw_df_header

    st.markdown(f"**Ready to process:** `{st.session_state.model}`")
    
    # --- Export Metadata Configuration (Moved from Step 1) ---
    st.subheader("Export Configuration (Metadata)")
    st.markdown("Confirm details for the output filename and folder structure.")
    
    with st.container(border=True):
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.session_state.site_name = st.text_input("Site Name", value=st.session_state.site_name, help="e.g., TVI, TSI")
            st.session_state.system_name = st.text_input("System Name", value=st.session_state.system_name, help="e.g., BOP, STG, BOILER_FAN, BOILER")
            st.session_state.model_name = st.text_input("Model Name", value=st.session_state.model_name)
        with meta_col2:
            st.session_state.sprint_name = st.text_input("Sprint Name", value=st.session_state.sprint_name, help="e.g., Sprint_1, Sprint_2")
            st.session_state.inclusive_dates = st.text_input("Inclusive Dates (YYYYMMDD-YYYYMMDD)", value=st.session_state.inclusive_dates, help="e.g., 20240101-20240601")

    st.divider()

    # --- Active Rules Summary (Read-Only) ---
    st.subheader("Active Rules Summary")
    if not st.session_state.filters and not st.session_state.datetime_filters:
        st.info("No filters active.")
    else:
        ac1, ac2 = st.columns(2)
        with ac1:
            if st.session_state.filters:
                st.markdown("**Numeric:**")
                for col, op, val in st.session_state.filters:
                    st.code(f"{col} {op} {val}")
            else:
                st.markdown("**Numeric:** None")
        with ac2:
            if st.session_state.datetime_filters:
                st.markdown("**Date:**")
                for op, val in st.session_state.datetime_filters:
                    if isinstance(val, tuple):
                        val_str = f"{val[0].strftime('%Y-%m-%d %H:%M')} to {val[1].strftime('%Y-%m-%d %H:%M')}"
                    else:
                        val_str = val.strftime('%Y-%m-%d %H:%M')
                    st.code(f"{op} {val_str}")
            else:
                st.markdown("**Date:** None")

    # Check if metadata is filled before enabling the button
    meta_filled = all([st.session_state.site_name, st.session_state.system_name, st.session_state.model_name, st.session_state.sprint_name, st.session_state.inclusive_dates])
    
    if not meta_filled:
        st.warning("Please fill in all Export Configuration fields above to enable processing.")
    
    if st.button("🚀 Apply Filters & Process Data", type="primary", disabled=not meta_filled):
        with st.spinner("Processing..."):
            filtered_df = raw_df.copy()

            # Apply numeric
            for col, op, val in st.session_state.filters:
                if op == "<": filtered_df = filtered_df[~(filtered_df[col] < val)]
                elif op == "<=": filtered_df = filtered_df[~(filtered_df[col] <= val)]
                elif op == "==": filtered_df = filtered_df[~(filtered_df[col] == val)]
                elif op == ">=": filtered_df = filtered_df[~(filtered_df[col] >= val)]
                elif op == ">": filtered_df = filtered_df[~(filtered_df[col] > val)]

            # Apply datetime
            for op, val in st.session_state.datetime_filters:
                if op == "< (remove before)":
                    filtered_df = filtered_df[~(filtered_df['DATETIME'] < val)]
                elif op == "> (remove after)":
                    filtered_df = filtered_df[~(filtered_df['DATETIME'] > val)]
                elif op == "between":
                    filtered_df = filtered_df[~((filtered_df['DATETIME'] >= val[0]) & (filtered_df['DATETIME'] <= val[1]))]
                elif op == "between (includes edge values)":
                    filtered_df = filtered_df[~((filtered_df['DATETIME'] >= val[0]) & (filtered_df['DATETIME'] <= val[1]))]

            # Save Logic
            st.session_state.cleaned_df = filtered_df.copy()
            st.session_state.filters_applied = True
            
            # Export
            filtered_df.columns = raw_df_header.columns
            export_filtered_df = pd.concat([raw_df_header, filtered_df])

            # --- USE GLOBAL BASE PATH ---
            base_path = Path(st.session_state.get('base_path', Path.cwd()))
            dataset_path = base_path / st.session_state.site_name / st.session_state.system_name / st.session_state.sprint_name / st.session_state.model_name / "dataset"
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Construct filename
            filename = f"CLEANED-{st.session_state.model_name}-{st.session_state.inclusive_dates}-RAW.csv"
            dataset_file_path = dataset_path / filename
            export_filtered_df.to_csv(dataset_file_path, index=False)
            
            st.success(f"✅ Data processed! {len(raw_df) - len(filtered_df)} rows removed.")
            st.success(f"Saved to: `{dataset_file_path}`")

    # --- Post-Processing Options ---
    if st.session_state.filters_applied:
        st.divider()
        
        # Settings Container
        with st.container(border=True):
            st.subheader("📊 Reporting & Visualization")
            
            col_actions, col_metrics = st.columns([1, 2])
            
            with col_actions:
                st.markdown("**Settings**")
                st.session_state.generate_report = st.checkbox("Include PDF Report in output", value=st.session_state.generate_report)
                st.session_state.show_visuals = st.checkbox("Show Visuals", value=st.session_state.show_visuals)
                
                enable_gen = st.session_state.generate_report or st.session_state.show_visuals
                # Store button state in a variable to use outside the column layout
                gen_button_clicked = st.button("Generate Visuals & Report", disabled=not enable_gen, type="primary", use_container_width=True)

            with col_metrics:
                if st.session_state.show_visuals or st.session_state.generate_report:
                    numeric_cols = raw_df.select_dtypes(include='number').columns.tolist()
                    st.session_state.selected_metrics_for_report = st.multiselect(
                        "Select metrics to include in report/charts:", 
                        numeric_cols, 
                        default=numeric_cols,
                        key="metric_multiselect"
                    )

        # Execution Logic (Outside the columns to ensure full width)
        if gen_button_clicked:
             # Re-construct path for report using GLOBAL BASE PATH
             base_path = Path(st.session_state.get('base_path', Path.cwd()))
             dataset_path = base_path / st.session_state.site_name / st.session_state.system_name / st.session_state.sprint_name / st.session_state.model_name / "dataset"
             report_file_path = dataset_path / f"{st.session_state.model_name}-{st.session_state.inclusive_dates}-DATA-CLEANING-REPORT.pdf"
             
             if st.session_state.generate_report and not st.session_state.show_visuals:
                 generate_simple_report(raw_df, st.session_state.filters, st.session_state.datetime_filters, report_file_path)
             
             if st.session_state.show_visuals:
                 # This will now render in the main area, full width
                 generate_data_cleaning_visualizations(
                     raw_df, 
                     st.session_state.cleaned_df, 
                     st.session_state.filters, 
                     st.session_state.datetime_filters, 
                     st.session_state.get('selected_metrics_for_report', raw_df.select_dtypes(include='number').columns.tolist()[:3]), 
                     st.session_state.generate_report, 
                     report_file_path
                 )
                 st.success("Visualizations Generated.")

             if st.session_state.generate_report:
                 st.success(f"PDF Report generated successfully: `{report_file_path}`")

    st.markdown("---")
    st.button("⬅️ Back to Filters", on_click=prev_step)