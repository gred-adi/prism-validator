import pandas as pd
import streamlit as st

@st.cache_data
def parse_excel(uploaded_file):
    """Parses the 'Consolidated Point Survey' sheet for metric validation.

    This function reads the specified sheet, selects the columns relevant for
    both standard metric validation and calculation validation, and returns a
    single cleaned DataFrame.

    Args:
        uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile):
            The uploaded 'Consolidated Point Survey' Excel file object.

    Returns:
        pd.DataFrame: A DataFrame containing the necessary data for validation,
        or an empty DataFrame if an error occurs or the sheet is not found.
    """
    try:
        xls = pd.ExcelFile(uploaded_file)
        if "Consolidated Point Survey" not in xls.sheet_names:
            # Return empty DF if sheet is missing, allowing validator to handle gracefully or UI to warn
            return pd.DataFrame()

        df = pd.read_excel(xls, "Consolidated Point Survey", header=0)

        # 1. Basic Cleaning
        if 'Metric' in df.columns:
            df['Metric'] = df['Metric'].astype(str).str.replace("  ", " ").str.strip()
        
        # 2. Define columns to keep
        # Standard Metric Validation cols
        base_cols = ['TDT', 'Model', 'Metric', 'Function', 'Point Type']
        # Calculation Validation cols
        calc_cols = [
            'Calc Point Type', 'Calculation Description', 
            'Pseudo Code', 'Language', 'Input Point', 'PRiSM Code'
        ]

        # 3. Select existing columns only
        existing_cols = [c for c in base_cols + calc_cols if c in df.columns]
        df = df[existing_cols]

        # 4. Drop duplicates
        # We drop duplicates based on TDT and Metric to ensure uniqueness for the main validation,
        # but we include the calc columns in the drop check to preserve distinct calc logic if it exists.
        df = df.drop_duplicates()

        return df

    except Exception as e:
        st.error(f"Error parsing Excel: {e}")
        return pd.DataFrame()