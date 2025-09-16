import pandas as pd
import streamlit as st

@st.cache_data
def parse_excel(uploaded_file):
    """
    Processes the uploaded Excel file for the 'Failure Diagnostics' section.
    Parses the 'Consolidated Failure Diagnostic' sheet, filters for enabled modes,
    and groups the data by TDT.
    """
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = "Consolidated Failure Diagnostic"

    if sheet_name not in xls.sheet_names:
        raise ValueError(f"❌ '{sheet_name}' sheet not found in the diagnostics Excel file.")

    df = pd.read_excel(xls, sheet_name, header=0)

    # --- Data Cleaning and Transformation ---
    df['TDT'] = df['TDT'].str.upper()
    # Filter to only include enabled failure modes, as per original script
    df = df[df['Failure Mode Enabled'] == 'YES']
    # Replace hyphen with arrow to match SQL output
    df['Direction'] = df['Direction'].str.replace("-", "→", regex=False)

    # Select and deduplicate required columns
    df = df[['TDT', 'Failure Mode', 'Metric', 'Direction', 'Weighting']].drop_duplicates()

    # --- Group into a dictionary of DataFrames ---
    grouped_data = df.groupby('TDT')
    tdt_subtables = {}
    for tdt_name, group in grouped_data:
        subtable = group.drop(columns=['TDT']).reset_index(drop=True)
        # Standardize column names for the validator
        subtable.columns = ['FAILURE_MODE', 'METRIC_NAME', 'DIRECTION', 'WEIGHT']
        tdt_subtables[tdt_name] = subtable

    return tdt_subtables