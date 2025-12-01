import pandas as pd
import streamlit as st
import re

@st.cache_data
def parse_excel(uploaded_file):
    """Parses the 'Consolidated Statistics' Excel file for threshold data.

    This function reads the 'Statistics' sheet from the uploaded Excel file.
    Instead of reading raw threshold columns, it now calculates them derived
    from the 'Std. Dev.' column using the following logic:
    
    1. HIGH ALERT   = 3 * Std. Dev.
    2. HIGH WARNING = 2 * Std. Dev.
    3. LOW WARNING  = -2 * Std. Dev.
    4. LOW ALERT    = -3 * Std. Dev.

    If the Std. Dev. is 0, it is replaced with 0.1 to avoid zero thresholds.
    All calculated values are rounded to 2 decimal places.

    Args:
        uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile):
            The uploaded 'Consolidated Statistics' Excel file object.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are model names (in
        uppercase) and values are DataFrames containing the calculated threshold 
        data for that model.

    Raises:
        ValueError: If the required 'Statistics' sheet is not found.
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
        parts = str(row['Point Name']).rsplit('(', 1)
        return parts[0].strip() if parts else str(row['Point Name']).strip()

    # Ensure required identity columns exist
    if 'Point Name' not in df.columns or 'Model' not in df.columns:
        st.error("❌ The file must contain 'Model' and 'Point Name' columns.")
        return {}

    df['METRIC_NAME'] = df.apply(extract_metric, axis=1)

    # --- Identify Std. Dev. Column ---
    # User specified it is usually the 7th column, but we prefer finding it by name.
    # We look for common variations or fall back to the 7th column (index 6) if safe.
    std_col = None
    possible_names = ['Std. Dev.', 'Std. Dev', 'Std Dev', 'Standard Deviation', 'StDev']
    
    for name in possible_names:
        if name in df.columns:
            std_col = name
            break
            
    if std_col is None:
        # Fallback to 7th column (index 6) if available
        if len(df.columns) >= 7:
            std_col = df.columns[6]
            st.warning(f"⚠️ Column 'Std. Dev.' not found by name. Using 7th column '{std_col}' for calculations.")
        else:
            st.error("❌ Could not locate 'Std. Dev.' column and file has fewer than 7 columns.")
            return {}

    # Ensure numeric conversion
    df[std_col] = pd.to_numeric(df[std_col], errors='coerce').fillna(0)

    # --- Handle Zero Std. Dev. ---
    # If Std. Dev is 0, use 0.1 instead to prevent zero thresholds
    df[std_col] = df[std_col].replace(0, 0.1)

    # --- Calculate Thresholds (Rounded to 2 decimal places) ---
    df['HIGH ALERT'] = (df[std_col] * 3).round(2)
    df['HIGH WARNING'] = (df[std_col] * 2).round(2)
    df['LOW WARNING'] = (df[std_col] * -2).round(2)
    df['LOW ALERT'] = (df[std_col] * -3).round(2)

    # Select and deduplicate required columns for the validator
    required_cols = ['Model', 'METRIC_NAME', 'HIGH ALERT', 'HIGH WARNING', 'LOW WARNING', 'LOW ALERT']
    df = df[required_cols].drop_duplicates()
    
    df.rename(columns={
        "Model": 'MODEL'
    }, inplace=True)

    # --- Group into a dictionary of DataFrames ---
    grouped_data = df.groupby('MODEL')
    model_subtables = {}
    for model_name, group in grouped_data:
        subtable = group.drop(columns=['MODEL']).reset_index(drop=True)
        model_subtables[model_name.upper()] = subtable

    return model_subtables