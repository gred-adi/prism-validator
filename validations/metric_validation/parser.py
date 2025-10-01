import pandas as pd
import streamlit as st

@st.cache_data
def parse_excel(uploaded_file):
    """
    Parses the 'Consolidated Point Survey' sheet for metric validation.

    This function reads the specified sheet from the uploaded Excel file,
    cleans the 'Metric' column by removing extra spaces, and selects the
    essential columns ('TDT', 'Metric', 'Function', 'Point Type') for the
    validation.

    The data is then grouped by 'TDT', and the function returns a dictionary
    of DataFrames. Each DataFrame contains the metric configurations for a
    specific TDT, with columns renamed for standardization. The function is
    cached for performance.

    Args:
        uploaded_file (UploadedFile): The file-like object from Streamlit's
            file uploader, which is the 'Consolidated_Point_Survey.xlsx' file.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where each key is a TDT name and
        the value is a DataFrame with standardized columns ('METRIC_NAME',
        'FUNCTION_TDT', 'POINT_TYPE_TDT').

    Raises:
        ValueError: If the 'Consolidated Point Survey' sheet is not found
            in the uploaded Excel file.
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
