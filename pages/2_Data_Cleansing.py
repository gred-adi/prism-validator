import streamlit as st
import pandas as pd
import json
import time

from utils.app_ui import render_sidebar, get_model_info
from utils.model_dev_utils import data_cleaning_read_prism_csv, cleaned_dataset_name_split, generate_data_cleaning_visualizations, generate_simple_report
from pathlib import Path
from datetime import datetime, time as dt_time

st.set_page_config(page_title="Data Cleansing", page_icon="1️⃣", layout="wide")

# --- Helper Functions for State Management ---

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

# --- JSON Serialization Helper ---
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def load_filters_from_json(json_data):
    """Parses JSON data and populates session state filters."""
    try:
        # 1. Numeric Filters (Simple list of lists/tuples)
        # Structure: [[col, op, val], ...]
        if "numeric" in json_data:
            st.session_state.filters = json_data["numeric"]

        # 2. Datetime Filters (Need to convert ISO strings back to Timestamps)
        # Structure: [[op, val], ...] where val can be string or list of strings
        if "datetime" in json_data:
            loaded_dt_filters = []
            for op, val in json_data["datetime"]:
                if op == "between (includes edge values)" and isinstance(val, list):
                    # It's a range (start, end)
                    start_ts = pd.to_datetime(val[0])
                    end_ts = pd.to_datetime(val[1])
                    loaded_dt_filters.append((op, (start_ts, end_ts)))
                else:
                    # It's a single value
                    ts = pd.to_datetime(val)
                    loaded_dt_filters.append((op, ts))
            
            st.session_state.datetime_filters = loaded_dt_filters
            
        st.success("Filters loaded successfully!")
        time.sleep(1) # Brief pause to show success
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to load filters: {e}")

def read_process_cache_files(raw_file, point_list_dataset_file):
    # Split dataset name to get model info
    auto_site_name, auto_model_name, auto_inclusive_dates = cleaned_dataset_name_split(raw_file.name)
    # Get user inputs for model info
    get_model_info(auto_site_name, auto_model_name, auto_inclusive_dates)

    # Progress Bar Initialization
    progress_text = "Starting file processing..."
    my_bar = st.progress(0, text=progress_text)

    try:
        # Step 1: Read files
        my_bar.progress(20, text="Reading CSV files...")
        raw_file_df = pd.read_csv(raw_file)
        project_points_df = pd.read_csv(point_list_dataset_file)

        # Step 2: Process files
        my_bar.progress(50, text="Formatting and mapping data points (this may take a moment)...")
        raw_df, raw_df_header = data_cleaning_read_prism_csv(raw_file_df, project_points_df)

        # Step 3: Finish
        my_bar.progress(100, text="Processing complete!")
        time.sleep(0.5) # Give user time to see 100%
        my_bar.empty()

        # Store in session state
        st.session_state.model = raw_file.name
        st.session_state.project_points_df = project_points_df
        st.session_state.raw_df = raw_df
        st.session_state.raw_df_header = raw_df_header

        # Return values for the current run
        return project_points_df, raw_df, raw_df_header

    except Exception as e:
        my_bar.empty()
        st.error(f"An error occurred during processing: {e}")
        return None, None, None

# --- Main UI ---

st.header("Data Cleansing")

site_name, system_name, model_name, sprint_name, inclusive_dates = None, None, None, None, None
project_points_df, raw_df, raw_df_header = None, None, None

# Initialize Session State
if 'filters_applied' not in st.session_state: st.session_state.filters_applied = False
if 'generate_report' not in st.session_state: st.session_state.generate_report = False
if 'show_visuals' not in st.session_state: st.session_state.show_visuals = False
if 'filters' not in st.session_state: st.session_state.filters = []
if 'datetime_filters' not in st.session_state: st.session_state.datetime_filters = []

raw_file = st.file_uploader("Upload your RAW dataset file", type=["csv"], accept_multiple_files=False)
point_list_dataset_file = st.file_uploader("Upload your POINT LIST dataset file (project_points.csv)", type=["csv"], accept_multiple_files=False)

if raw_file is not None and point_list_dataset_file is not None:
    st.success(f"Files '{raw_file.name}' & '{point_list_dataset_file.name}' uploaded successfully.")

    if 'model' in st.session_state and st.session_state.model == raw_file.name:

        # Ask the user if he wants to use the cached data
        st.info(f"You have loaded {raw_file.name} before. Use the saved data?")
        if 'use_cache_decision' not in st.session_state:
            st.session_state.use_cache_decision = None

        st.markdown(f"""
        **Saved data:**
        * Site: {st.session_state.get('site_name')}
        * Model: {st.session_state.get('model_name')}
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Yes, use saved data",
                      on_click=lambda: st.session_state.update(use_cache_decision='yes'))
        with col2:
            st.button("No, re-process the file",
                      on_click=lambda: st.session_state.update(use_cache_decision='no'))

        # User wants to use cached data
        if st.session_state.use_cache_decision == 'yes':
            try:
                site_name = st.session_state.site_name
                system_name = st.session_state.system_name
                model_name = st.session_state.model_name
                sprint_name = st.session_state.sprint_name
                inclusive_dates = st.session_state.inclusive_dates
                project_points_df = st.session_state.project_points_df
                raw_df = st.session_state.raw_df
                raw_df_header = st.session_state.raw_df_header
            except KeyError:
                st.error("Cache was incomplete. Re-processing file...")
                st.session_state.use_cache_decision = 'no'
                st.rerun()

        # User does not want to use cached data
        elif st.session_state.use_cache_decision == 'no':
            project_points_df, raw_df, raw_df_header = read_process_cache_files(raw_file, point_list_dataset_file)
            st.session_state.use_cache_decision = None
            st.rerun()
        else:
            st.stop()
    else:
        # New file
        if 'filters' in st.session_state:
            clear_all_numeric_filters()
            clear_all_date_filters()
        project_points_df, raw_df, raw_df_header = read_process_cache_files(raw_file, point_list_dataset_file)
        st.session_state.use_cache_decision = None
        st.rerun()

if raw_df is not None:
    # --- 1. Dataset Preview & Statistics ---
    st.markdown("### Dataset Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)
    
    st.markdown("### Dataset Statistics")
    st.markdown("Basic statistical description of the numerical columns in the uploaded dataset.")
    st.dataframe(raw_df.describe(), use_container_width=True)

    st.markdown("---")

    # --- 2. Filter Configuration Section ---
    st.header("Filter Configuration")

    # --- Import / Export Filters ---
    with st.expander("📂 **Manage Filter Presets (Import / Export)**"):
        col_export, col_import = st.columns(2)
        
        with col_export:
            st.markdown("#### Export Current Filters")
            current_config = {
                "numeric": st.session_state.filters,
                "datetime": st.session_state.datetime_filters
            }
            json_str = json.dumps(current_config, default=json_serial, indent=2)
            st.download_button(
                label="💾 Download Filter Preset (.json)",
                data=json_str,
                file_name="data_cleaning_filters.json",
                mime="application/json",
                use_container_width=True
            )
            
        with col_import:
            st.markdown("#### Import Filter Preset")
            uploaded_preset = st.file_uploader("Upload .json preset", type=["json"], label_visibility="collapsed")
            if uploaded_preset:
                if st.button("Load Preset", type="primary", use_container_width=True):
                    data = json.load(uploaded_preset)
                    load_filters_from_json(data)

    st.write("Configure filters to clean the data. You can add multiple filters for both numerical values and dates.")

    # --- Two-Column Filter Layout ---
    filter_col1, filter_col2 = st.columns(2, gap="medium")

    # Column 1: Numerical Filters
    with filter_col1:
        with st.container(border=True):
            st.subheader("🔢 Numerical Filters")
            
            num_col = st.selectbox("Column", raw_df.columns[1:], key="filter_col")
            num_op = st.selectbox("Operator", ["<", "<=", "==", ">=", ">"], key="filter_op")
            num_val = st.number_input("Value", value=0.0, step=1.0, key="filter_val")
            
            if st.button("Add Numerical Filter", type="secondary", use_container_width=True):
                add_filter(num_col, num_op, num_val)

    # Column 2: Date Filters
    with filter_col2:
        with st.container(border=True):
            st.subheader("📅 Date Filters")
            
            dt_op = st.selectbox(
                "Operator",
                ["< (remove before)", "> (remove after)", "between (includes edge values)"],
                key="dt_op",
            )

            dt_val = None
            if dt_op == "between (includes edge values)":
                val_col1, val_col2 = st.columns(2)
                with val_col1:
                    start_date = st.date_input("Start date", value=pd.Timestamp.now().floor('D'), key="dt_val_start")
                    end_date = st.date_input("End date", value=(pd.Timestamp.now() + pd.Timedelta(days=1)).floor('D'), key="dt_val_end_date")
                with val_col2:
                    start_time = st.time_input("Start time", value=dt_time(0,0), key="dt_time_start")
                    end_time = st.time_input("End time", value=dt_time(23,59), key="dt_time_end")
                
                start_datetime = pd.Timestamp.combine(start_date, start_time)
                end_datetime = pd.Timestamp.combine(end_date, end_time)
                dt_val = (start_datetime, end_datetime)
            else:
                col_d, col_t = st.columns(2)
                with col_d:
                    d_val = st.date_input("Date", value=pd.Timestamp.now().floor('D'), key="dt_val_single")
                with col_t:
                    t_val = st.time_input("Time", value=dt_time(0,0), key="dt_time_single")
                dt_val = pd.Timestamp.combine(d_val, t_val)

            if st.button("Add Date Filter", type="secondary", use_container_width=True):
                add_date_filter(dt_op, dt_val)

    # --- Active Filters Display ---
    st.subheader("Active Filters")
    
    if not st.session_state.filters and not st.session_state.datetime_filters:
        st.info("No filters currently applied.")
    else:
        active_col1, active_col2 = st.columns(2, gap="medium")
        
        # Numeric Filters List
        with active_col1:
            with st.container(border=True):
                st.markdown("**Numeric Filters:**")
                if st.session_state.filters:
                    for i, (col, op, val) in enumerate(st.session_state.filters):
                        c1, c2 = st.columns([0.9, 0.1])
                        with c1:
                            st.code(f"{col} {op} {val}", language="text")
                        with c2:
                            st.button("❌", key=f"del_num_{i}", on_click=delete_filter, args=(i,), help="Remove filter")
                    st.button("Clear Numeric Filters", on_click=clear_all_numeric_filters, key="clear_num", use_container_width=True)
                else:
                    st.caption("No numeric filters applied.")

        # Datetime Filters List
        with active_col2:
            with st.container(border=True):
                st.markdown("**Date Filters:**")
                if st.session_state.datetime_filters:
                    for i, (op, val) in enumerate(st.session_state.datetime_filters):
                        c1, c2 = st.columns([0.9, 0.1])
                        with c1:
                            if op == "between (includes edge values)":
                                st.code(f"DATETIME {op} {val[0]} AND {val[1]}", language="text")
                            else:
                                st.code(f"DATETIME {op} {val}", language="text")
                        with c2:
                            st.button("❌", key=f"del_dt_{i}", on_click=delete_date_filter, args=(i,), help="Remove filter")
                    st.button("Clear Date Filters", on_click=clear_all_date_filters, key="clear_dt", use_container_width=True)
                else:
                    st.caption("No date filters applied.")

    st.markdown("---")

    # --- Apply Filters Button ---
    if st.button("Apply Filters & Process", type="primary", use_container_width=True):

        # 1. Filtering Logic
        filtered_df = raw_df.copy()
        
        if not st.session_state.filters and not st.session_state.datetime_filters:
            st.warning("No filters were set, but proceeding to save standard cleaned file.")

        # Apply numeric filters
        for col, op, val in st.session_state.filters:
            if op == "<": filtered_df = filtered_df[~(filtered_df[col] < val)]
            elif op == "<=": filtered_df = filtered_df[~(filtered_df[col] <= val)]
            elif op == "==": filtered_df = filtered_df[~(filtered_df[col] == val)]
            elif op == ">=": filtered_df = filtered_df[~(filtered_df[col] >= val)]
            elif op == ">": filtered_df = filtered_df[~(filtered_df[col] > val)]

        # Apply datetime filters
        for op, val in st.session_state.datetime_filters:
            if op == "< (remove before)":
                filtered_df = filtered_df[~(filtered_df['DATETIME'] < val)]
            elif op == "> (remove after)":
                filtered_df = filtered_df[~(filtered_df['DATETIME'] > val)]
            elif op == "between (includes edge values)":
                filtered_df = filtered_df[~((filtered_df['DATETIME'] >= val[0]) & (filtered_df['DATETIME'] <= val[1]))]

        # 2. Save to File Logic
        st.session_state.cleaned_df = filtered_df.copy()

        filtered_df.columns = raw_df_header.columns
        export_filtered_df = pd.concat([raw_df_header, filtered_df])

        base_path = Path.cwd()
        # FIXED: Replaced utility_name with system_name
        dataset_path = base_path / st.session_state.site_name / st.session_state.system_name / st.session_state.sprint_name / st.session_state.model_name / "dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        dataset_file_path = dataset_path / f"CLEANED-{model_name}-{inclusive_dates}-RAW.csv"
        export_filtered_df.to_csv(dataset_file_path, index=False)
        
        st.success(f"✅ Cleaned dataset saved to: {dataset_file_path}")
        st.session_state.filters_applied = True

# --- Visualization UI (Appears after applying filters) ---
if st.session_state.filters_applied:
    st.header("Generate Report and Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.toggle("Generate Report", value=st.session_state.generate_report, key="generate_report")
    with col2:
        st.toggle("Generate Visualizations (Extends runtime)", value=st.session_state.show_visuals, key="show_visuals")

    selected_metrics = []
    if st.session_state.show_visuals:
        st.subheader("Visualize Filtered Data")
        numeric_cols = raw_df.select_dtypes(include='number').columns.tolist()
        selected_metrics = st.multiselect(
            "Select metrics to visualize:",
            options=numeric_cols,
            default=numeric_cols
        )

    if st.button("Generate Report / Visuals"):
        base_path = Path.cwd()
        # FIXED: Replaced utility_name with system_name
        dataset_path = base_path / st.session_state.site_name / st.session_state.system_name / st.session_state.sprint_name / st.session_state.model_name / "dataset"
        report_file_path = dataset_path / f"CLEANED-{model_name}-{inclusive_dates}-DATA-CLEANING-REPORT.pdf"

        # Generate simple report
        if st.session_state.generate_report and not st.session_state.show_visuals:
            st.info(f"Generating simple report... {report_file_path}")
            generate_simple_report(
                raw_df,
                st.session_state.filters,
                st.session_state.datetime_filters,
                report_file_path
            )
            st.success(f"Simple report saved to {report_file_path}")

        # Generate visual report
        if st.session_state.show_visuals:
            if not selected_metrics:
                st.warning("Please select metrics to visualize.")
            else:
                generate_data_cleaning_visualizations(
                    raw_df,
                    st.session_state.cleaned_df,
                    st.session_state.filters,
                    st.session_state.datetime_filters,
                    selected_metrics,
                    st.session_state.generate_report,
                    report_file_path
                )
                if st.session_state.generate_report:
                    st.success(f"Visual report saved to {report_file_path}")
        
        st.session_state.filters_applied = False
        st.rerun()