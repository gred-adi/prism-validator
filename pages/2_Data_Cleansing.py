import streamlit as st
import pandas as pd

from utils.app_ui import render_sidebar, get_model_info
from utils.model_dev_utils import data_cleaning_read_prism_csv, cleaned_dataset_name_split, generate_data_cleaning_visualizations, generate_simple_report
from pathlib import Path
from datetime import time

st.set_page_config(page_title="Data Cleaning", page_icon="1️⃣")

render_sidebar()

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

def read_process_cache_files(raw_file, point_list_dataset_file):
    # Split dataset name to get model info
    auto_site_name, auto_model_name, auto_inclusive_dates = cleaned_dataset_name_split(raw_file.name)
    # Get user inputs for model info
    get_model_info(auto_site_name, auto_model_name, auto_inclusive_dates)

    # Read files
    with st.spinner("Reading files..."):
        raw_file_df = pd.read_csv(raw_file)
        project_points_df = pd.read_csv(point_list_dataset_file)

    # Reformat raw df and get header for later use
    with st.spinner("Processing files..."):
        raw_df, raw_df_header = data_cleaning_read_prism_csv(raw_file_df, project_points_df)

    # # Caching moved to get_model_info
    # st.session_state.site_name = site_name
    # st.session_state.utility_name = utility_name
    # st.session_state.model_name = model_name
    # st.session_state.sprint_name = sprint_name
    # st.session_state.inclusive_dates = inclusive_dates

    # Store in session state
    st.session_state.model = raw_file.name
    st.session_state.project_points_df = project_points_df
    st.session_state.raw_df = raw_df
    st.session_state.raw_df_header = raw_df_header

    # Return values for the current run
    return project_points_df, raw_df, raw_df_header

st.header("Data Cleaning")

site_name, utility_name, model_name, sprint_name, inclusive_dates = None, None, None, None, None
project_points_df, raw_df, raw_df_header = None, None, None

if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False

if 'generate_report' not in st.session_state:
    st.session_state.generate_report = False

if 'show_visuals' not in st.session_state:
    st.session_state.show_visuals = False

raw_file = st.file_uploader("Upload your RAW dataset file", type=["csv"], accept_multiple_files=False)
point_list_dataset_file = st.file_uploader("Upload your POINT LIST dataset file (project_points.csv)", type=["csv"], accept_multiple_files=False)

if raw_file is not None and point_list_dataset_file is not None:
    st.success(f"File '{raw_file.name}' uploaded successfully.")
    st.success(f"File '{point_list_dataset_file.name}' uploaded successfully.")

    if 'model' in st.session_state and st.session_state.model == raw_file.name:

        # Ask the user if he wants to use the cached data
        st.info(f"You have loaded {raw_file.name} before. Use the saved data?")
        if 'use_cache_decision' not in st.session_state:
            st.session_state.use_cache_decision = None

        st.markdown(f"""
        Saved data: (Please ensure the data is correct for the file to be saved to the proper path.)
        Site Name: {st.session_state.site_name}
        Utility Name: {st.session_state.utility_name}
        Model Name: {st.session_state.model_name}
        Sprint Name: {st.session_state.sprint_name}
        Inclusive Dates: {st.session_state.inclusive_dates}
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
            # Load all variables from session state
            try:
                site_name = st.session_state.site_name
                utility_name = st.session_state.utility_name
                model_name = st.session_state.model_name
                sprint_name = st.session_state.sprint_name
                inclusive_dates = st.session_state.inclusive_dates
                project_points_df = st.session_state.project_points_df
                raw_df = st.session_state.raw_df
                raw_df_header = st.session_state.raw_df_header
            except KeyError:
                # Lacking data
                st.error("Cache was incomplete. Re-processing file...")
                st.session_state.use_cache_decision = 'no' # Set state for next rerun
                st.rerun()

        # User does not want to use cached data
        elif st.session_state.use_cache_decision == 'no':
            # Run the full processing function
            project_points_df, raw_df, raw_df_header = read_process_cache_files(raw_file, point_list_dataset_file)

            # Reset the decision and rerun
            st.session_state.use_cache_decision = None
            st.rerun()

        else:
            st.stop()
    else:
        # New file
        if 'filters' in st.session_state:
            clear_all_numeric_filters() # Clear filters for new files
            clear_all_date_filters()
        project_points_df, raw_df, raw_df_header = read_process_cache_files(raw_file, point_list_dataset_file)
        st.session_state.use_cache_decision = None
        st.rerun()

if raw_df is not None:
    st.write("Dataset Preview")
    st.dataframe(raw_df.head(10))
    # # Print datatypes for each column
    # dtypes_info = pd.DataFrame(raw_df.dtypes, columns=['Data Type'])
    # st.write("Column Data Types:")
    # st.dataframe(dtypes_info)

    if 'filters' not in st.session_state:
        st.session_state.filters = []
    if 'datetime_filters' not in st.session_state:
        st.session_state.datetime_filters = []

    st.header("Add Filters")
    # Add numerical filters
    st.write("Add a numerical filter")
    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])

    with col1:
        column_to_filter = st.selectbox(
            "Column", raw_df.columns[1:], key="filter_col"
        )

    with col2:
        operator = st.selectbox(
            "Operator",
            ["<", "<=", "==", ">=", ">"],
            key="filter_op",
        )

    with col3:
        # Use 0.0 as default to match float type of data
        value_to_filter = st.number_input(
            "Value", value=0.0, step=1.0, key="filter_val"
        )

    with col4:
        st.write("")
        st.write("")
        # When clicked, call 'add_filter' with the current widget values
        st.button(
            "Add",
            type="primary",
            on_click=add_filter,
            args=(
                st.session_state.filter_col,
                st.session_state.filter_op,
                st.session_state.filter_val,
            ),
        )

    # Add datetime filters
    st.write("Add a DATETIME filter")
    dt_op = st.selectbox(
        "Operator",
        ["< (remove before)", "> (remove after)", "between (includes edge values)"],
        key="dt_op",
    )

    if dt_op == "between (includes edge values)":
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            start_date = st.date_input("Start date", value=pd.Timestamp.now().floor('D'), key="dt_val_start")
            end_date = st.date_input("End date", value=(pd.Timestamp.now() + pd.Timedelta(days=1)).floor('D'), key="dt_val_end_date")
        with val_col2:
            start_time = st.time_input("Start time (Type for more specific times)", value=time(0,0), key="dt_time_start")
            end_time = st.time_input("End time (Type for more specific times)", value=time(23,59), key="dt_time_end")
        start_datetime = pd.Timestamp.combine(start_date, start_time)
        end_datetime = pd.Timestamp.combine(end_date, end_time)
        # Store the value as a tuple for 'between'
        dt_val = (start_datetime, end_datetime)
    else:
        dt_col1, dt_col2 = st.columns(2)
        with dt_col1:
            d_val = st.date_input("Value", value=pd.Timestamp.now().floor('D'), key="dt_val_single")
        with dt_col2:
            t_val = st.time_input("Time (Type for more specific times)", value=time(0,0), key="dt_time_single")
        dt_val = pd.Timestamp.combine(d_val, t_val)

    # "Add Date Filter" button
    st.button(
        "Add Date Filter",
        type="primary",
        on_click=add_date_filter,
        args=(dt_op, dt_val) # Pass the operator and value(s)
    )

    #Display filters
    if not st.session_state.filters:
        st.info("No numerical filters applied.")
    else:
        st.write("Current Numeric Filters:")
        for i, (col, op, val) in enumerate(st.session_state.filters):
            col_a, col_b = st.columns([5, 1])
            col_a.text(f"Filter {i+1}: Remove when `{col}` {op} {val}")
            # remove button for each filter
            col_b.button("Remove", key=f"remove_{i}", on_click=delete_filter, args=(i,))

        # clear all numeric filters at once
        st.button("Clear All Filters", on_click=clear_all_numeric_filters)

    if not st.session_state.datetime_filters:
        st.info("No datetime filters applied.")
    else:
        st.write("Current DATETIME Filters:")
        for i, (op, val) in enumerate(st.session_state.datetime_filters):
            col_a, col_b = st.columns([5, 1])
            if op == "between (includes edge values)":
                col_a.text(f"Filter {i+1}: DATETIME {op} {val[0]} and {val[1]}")
            else:
                col_a.text(f"Filter {i+1}: DATETIME {op} {val}")
            # remove button for each filter
            col_b.button("Remove", key=f"remove_dt_{i}", on_click=delete_date_filter, args=(i,))

        # clear all datetime filters at once
        st.button("Clear All DATETIME Filters", on_click=clear_all_date_filters)

if st.button("Apply Filters", type="primary"):

    # --- 1. Your existing filtering logic ---
    filtered_df = raw_df.copy()
    if not st.session_state.filters and not st.session_state.datetime_filters:
        st.info("No filters to apply.")

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

    # --- 2. Your existing save-to-file logic ---
    # Store the cleaned_df in session_state so we can use it, too
    st.session_state.cleaned_df = filtered_df.copy()

    filtered_df.columns = raw_df_header.columns
    export_filtered_df = pd.concat([raw_df_header, filtered_df])

    base_path = Path.cwd()
    dataset_path = base_path / st.session_state.site_name / st.session_state.utility_name / st.session_state.sprint_name / st.session_state.model_name / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    dataset_file_path = dataset_path / f"CLEANED-{model_name}-{inclusive_dates}-RAW.csv"
    export_filtered_df.to_csv(dataset_file_path, index=False)
    st.success(f"Cleaned dataset saved to {dataset_file_path}")

    # --- 3. Set the flag to show the visualization UI ---
    st.session_state.filters_applied = True



# --- 4. NEW VISUALIZATION UI ---
# This block will only appear *after* the "Apply Filters" button
# has been clicked at least once.
if st.session_state.filters_applied:

    st.markdown("---")
    st.header("Generate Report and Visualizations (optional)")
    st.toggle("Generate Report", value=st.session_state.generate_report, key="generate_report")
    st.toggle("Generate Visualizations (Will extend runtime)", value=st.session_state.show_visuals, key="show_visuals")

    if st.session_state.show_visuals:
        st.header("Visualize Filtered Data")
        # Get all numeric columns from the raw data
        numeric_cols = raw_df.select_dtypes(include='number').columns.tolist()

        selected_metrics = st.multiselect(
            "Select metrics to visualize:",
            options=numeric_cols,
            default=numeric_cols
        )

    if st.button("Generate"):
        base_path = Path.cwd()
        dataset_path = base_path / st.session_state.site_name / st.session_state.utility_name / st.session_state.sprint_name / st.session_state.model_name / "dataset"
        report_file_path = dataset_path / f"CLEANED-{model_name}-{inclusive_dates}-DATA-CLEANING-REPORT.pdf"

        # Generate report without graphs
        if st.session_state.generate_report and not st.session_state.show_visuals:
            st.info(f"Generating simple report... {report_file_path}")
            generate_simple_report(
                raw_df,
                st.session_state.filters,
                st.session_state.datetime_filters,
                report_file_path
            )
            st.success(f"Simple report saved to {report_file_path}")

        # Generate graphs and pass in whether or not to generate report
        if st.session_state.show_visuals:
            if not selected_metrics:
                st.warning("Please select metrics to visualize.")
            else:
                # Call the visualization function
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
                    st.success(f"Report saved to {report_file_path}")
        st.session_state.filters_applied = False
        st.stop()