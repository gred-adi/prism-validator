import pandas as pd
import streamlit as st

@st.cache_data
def parse_excel(uploaded_file):
    """
    Parses the 'Filter Summary' sheet from the consolidated survey Excel file.

    This function reads the specified sheet, constructs a descriptive filter
    string from the 'Filter Condition' and 'Filter Value' columns, and then
    formats the data for validation. It handles numerical formatting to ensure
    clean comparisons.

    The data is grouped by model name, and the function returns a dictionary
    of DataFrames, where each DataFrame contains the filter configurations for
    a specific model. The function is cached to improve performance.

    Args:
        uploaded_file (UploadedFile): The file-like object from Streamlit's
            file uploader, which should be the 'Consolidated_Point_Survey.xlsx'
            file.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where each key is an uppercase
        model name and the value is a DataFrame with standardized columns
        ('METRIC_NAME', 'FILTER') for that model.

    Raises:
        ValueError: If the required 'Filter Summary' sheet is not found in
            the uploaded Excel file.
    """
    xls = pd.ExcelFile(uploaded_file)

    if "Filter Summary" not in xls.sheet_names:
        raise ValueError("❌ 'Filter Summary' sheet not found in Excel file.")

    df = pd.read_excel(xls, 'Filter Summary', header=0)

    # Clean data and create the 'Filter' column
    df['Model'] = df['Model'].str.upper()
    
    def create_filter_string(row):
        condition = row['Filter Condition']
        val = row['Filter Value']
        value = str(int(val)) if float(val).is_integer() else str(val).rstrip('0').rstrip('.')
        if condition == 0:
            return f"Equal To {value}"
        elif condition == "<":
            return f"Less Than {value}"
        elif condition == ">":
            return f"Greater Than {value}"
        else:
            return "Unknown"
            
    df['Filter'] = df.apply(create_filter_string, axis=1)

    # Select and deduplicate required columns
    df = df[['Model', 'Metric', 'Filter']].drop_duplicates()

    # Group into a dictionary of DataFrames, with Model as the key
    grouped_data = df.groupby('Model')
    model_subtables = {}
    for model_name, group in grouped_data:
        subtable = group.drop(columns=['Model']).reset_index(drop=True)
        subtable.columns = ['METRIC_NAME', 'FILTER']
        model_subtables[model_name] = subtable

    return model_subtables
