import pandas as pd
import streamlit as st

@st.cache_data
def parse_excel(uploaded_file):
    """Parses the 'Consolidated Point Survey' sheet for metric mapping data.

    This function reads the specified sheet, selects the columns relevant for
    metric mapping validation, and performs necessary data cleaning. It then
    groups the data by 'Model', returning a dictionary of DataFrames.

    Args:
        uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile):
            The uploaded 'Consolidated Point Survey' Excel file object.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are model names and
        values are DataFrames containing the mapping data for that model.

    Raises:
        ValueError: If the required 'Consolidated Point Survey' sheet is not found.
    """
    xls = pd.ExcelFile(uploaded_file)

    if "Consolidated Point Survey" not in xls.sheet_names:
        raise ValueError("❌ 'Consolidated Point Survey' sheet not found in Excel file.")

    df = pd.read_excel(xls, 'Consolidated Point Survey', header=0)

    # Clean data as per the original script
    df['Model'] = df['Model'].str.upper()
    df['Metric'] = df['Metric'].str.replace('  ', ' ', regex=False).str.strip()

    # Select and deduplicate required columns
    df = df[['Model', 'Metric', 'Canary Point Name', 'Canary Description', 'Function', 'Point Type', 'Unit']].drop_duplicates()

    # Group into a dictionary of DataFrames, with Model as the key
    grouped_data = df.groupby('Model')
    model_subtables = {}
    for model_name, group in grouped_data:
        subtable = group.drop(columns=['Model']).reset_index(drop=True)
        # Standardize column names for easier comparison later
        subtable.columns = ['METRIC_NAME', 'POINT_NAME', 'POINT_DESCRIPTION', 'FUNCTION', 'POINT_TYPE', 'POINT_UNIT']
        model_subtables[model_name] = subtable

    return model_subtables
