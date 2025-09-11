import pandas as pd
import streamlit as st

@st.cache_data
def parse_excel(uploaded_file):
    """
    Processes the uploaded Excel file for the 'Metric Validation' section.
    Specifically looks for the 'Consolidated Point Survey' sheet.
    """
    xls = pd.ExcelFile(uploaded_file)

    if "Consolidated Point Survey" not in xls.sheet_names:
        raise ValueError("❌ 'Consolidated Point Survey' sheet not found in Excel file.")

    df = pd.read_excel(xls, "Consolidated Point Survey", header=0)

    df['Metric'] = df['Metric'].str.replace("  ", " ").str.strip()
    df = df[['TDT', 'Metric', 'Function', 'Point Type']].drop_duplicates()

    grouped_data = df.groupby("TDT")
    tdt_subtables = {}
    for tdt, group in grouped_data:
        subtable = group.drop(columns=['TDT']).reset_index(drop=True)
        # Standardize column names for easier comparison later
        subtable.columns = ['METRIC_NAME', 'FUNCTION_TDT', 'POINT_TYPE_TDT']
        tdt_subtables[tdt] = subtable

    return tdt_subtables
