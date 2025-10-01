import pandas as pd
import streamlit as st

@st.cache_data
def parse_excel(uploaded_file):
    """
    Parses the 'Consolidated Point Survey' sheet for metric mapping validation.

    This function reads the specified sheet from the uploaded Excel file, performs
    data cleaning on 'Model' and 'Metric' columns, and selects the columns
    relevant for metric mapping. It then groups the data by model name.

    The function returns a dictionary of DataFrames, where each DataFrame
    contains the mapping details for a specific model with standardized column
    names for easy comparison in the validation step. The function is cached
    for performance.

    Args:
        uploaded_file (UploadedFile): The file-like object from Streamlit's
            file uploader, which is the 'Consolidated_Point_Survey.xlsx' file.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where each key is an uppercase
        model name and the value is a DataFrame with standardized columns
        ('METRIC_NAME', 'POINT_NAME', 'POINT_DESCRIPTION', etc.) for that model.

    Raises:
        ValueError: If the 'Consolidated Point Survey' sheet is not found
            in the uploaded Excel file.
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
