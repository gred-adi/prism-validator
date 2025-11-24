import pandas as pd
import streamlit as st
import re

@st.cache_data
def parse_excel(uploaded_file):
    """Parses the 'Consolidated Statistics' Excel file for threshold data.

    This function reads the 'Statistics' sheet from the uploaded Excel file,
    extracts the base metric name from the 'Point Name' column, and cleans up
    the data. It then groups the resulting DataFrame by 'Model' and returns a
    dictionary of DataFrames, where each key is a model name.

    Args:
        uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile):
            The uploaded 'Consolidated Statistics' Excel file object.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are model names (in
        uppercase) and values are DataFrames containing the threshold data for
        that model.

    Raises:
        ValueError: If the required 'Statistics' sheet is not found in the file.
    """
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = "Statistics"

    if sheet_name not in xls.sheet_names:
        raise ValueError(f"❌ '{sheet_name}' sheet not found in the statistics Excel file.")

    df = pd.read_excel(xls, sheet_name, header=0)

    # --- Data Cleaning and Feature Engineering ---
    # Extract the base metric name from the 'Point Name' column
    def extract_metric(row):
        # rsplit splits from the right, so with maxsplit=1, it finds the last '('
        parts = row['Point Name'].rsplit('(', 1)
        return parts[0].strip() if parts else row['Point Name'].strip()

    df['METRIC_NAME'] = df.apply(extract_metric, axis=1)

    # Select and deduplicate required columns
    df = df[['Model', 'METRIC_NAME', 'HIGH ALERT', 'HIGH WARNING', 'LOW WARNING', 'LOW ALERT']].drop_duplicates()
    df.rename(columns={
        "Model": 'MODEL'
    }, inplace=True)

    # --- Group into a dictionary of DataFrames ---
    grouped_data = df.groupby('MODEL')
    model_subtables = {}
    for model_name, group in grouped_data:
        subtable = group.drop(columns=['MODEL']).reset_index(drop=True)
        # Standardize column names (already done, but good practice)
        model_subtables[model_name.upper()] = subtable

    return model_subtables