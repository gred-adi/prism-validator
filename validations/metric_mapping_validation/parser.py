import pandas as pd
import streamlit as st

@st.cache_data
def parse_excel(uploaded_file):
    """
    Processes the uploaded Excel file for the 'Metric Mapping' section.
    Parses the 'Consolidated Point Survey' sheet and groups by 'Model'.
    """
    xls = pd.ExcelFile(uploaded_file)

    if "Consolidated Point Survey" not in xls.sheet_names:
        raise ValueError("❌ 'Consolidated Point Survey' sheet not found in Excel file.")

    df = pd.read_excel(xls, 'Consolidated Point Survey', header=0)

    # Clean data as per the original script
    df['Model'] = df['Model'].str.upper()
    df['Metric'] = df['Metric'].str.replace('  ', ' ', regex=False).str.strip()

    # Select and deduplicate required columns
    df = df[['Model', 'Metric', 'Canary Point Name', 'Canary Description', 'Function', 'Point Type']].drop_duplicates()

    # Group into a dictionary of DataFrames, with Model as the key
    grouped_data = df.groupby('Model')
    model_subtables = {}
    for model_name, group in grouped_data:
        subtable = group.drop(columns=['Model']).reset_index(drop=True)
        # Standardize column names for easier comparison later
        subtable.columns = ['METRIC_NAME', 'POINT_NAME', 'POINT_DESCRIPTION', 'FUNCTION', 'POINT_TYPE']
        model_subtables[model_name] = subtable

    return model_subtables
