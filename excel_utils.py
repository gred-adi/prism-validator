import pandas as pd
import streamlit as st

# @st.cache_data is perfect for file processing. Streamlit hashes the file
# content, so this function only re-runs if a *new* file is uploaded.
@st.cache_data
def process_excel_reference(uploaded_file):
    """
    Reads the 'Consolidated Point Survey' sheet from the uploaded Excel file,
    cleans it, and groups it into a dictionary of DataFrames by 'TDT'.
    """
    if uploaded_file is None:
        return None

    print("--- Processing Excel File ---") # For debugging
    try:
        xls = pd.ExcelFile(uploaded_file)

        if "Consolidated Point Survey" not in xls.sheet_names:
            # Raising an error that can be caught in the main app
            raise ValueError("❌ 'Consolidated Point Survey' sheet not found in Excel file.")

        df = pd.read_excel(xls, "Consolidated Point Survey", header=0)

        # Clean up column names to avoid issues (e.g., trailing spaces)
        df.columns = df.columns.str.strip()

        # Clean up the 'Metric' column data
        df['Metric'] = df['Metric'].str.replace("  ", " ").str.strip()

        # Select and deduplicate required columns
        df = df[['TDT', 'Metric', 'Function', 'Point Type']].drop_duplicates()

        # Group into a dictionary of DataFrames, with TDT as the key
        grouped_data = df.groupby("TDT")
        tdt_subtables = {}
        for tdt, group in grouped_data:
            subtable = group.drop(columns=['TDT']).reset_index(drop=True)
            # Standardize column names for easier comparison later
            subtable.columns = ['METRIC_NAME', 'FUNCTION', 'POINT_TYPE']
            tdt_subtables[tdt] = subtable

        return tdt_subtables

    except Exception as e:
        # Propagate the exception to be handled by the main app's UI
        raise e

