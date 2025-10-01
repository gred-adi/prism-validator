import pandas as pd
import streamlit as st

@st.cache_data
def parse_excel(uploaded_file):
    """
    Parses the 'Consolidated Failure Diagnostic' Excel file.

    This function reads the specified sheet from the uploaded Excel file,
    filters the data to include only enabled failure modes, and performs
    key data transformations. It standardizes TDT and failure mode names,
    and crucially, converts direction indicators (e.g., '-') into Unicode
    arrows to match the format from the SQL query.

    The processed data is then grouped by TDT and returned as a dictionary
    of DataFrames. The function is cached to prevent reprocessing of the
    same file.

    Args:
        uploaded_file (UploadedFile): The file-like object from Streamlit's
            file uploader, containing the consolidated diagnostics data.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where each key is an uppercase
        TDT name and the value is a DataFrame with standardized columns
        ('FAILURE_MODE', 'METRIC_NAME', 'DIRECTION', 'WEIGHT') for that TDT.

    Raises:
        ValueError: If the 'Consolidated Failure Diagnostic' sheet is not
            found in the uploaded Excel file.
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

    df['Failure Mode'] = (
        df['Failure Mode']
        .str.strip()                               # remove leading/trailing spaces
        .str.replace(r'\s+', ' ', regex=True)      # collapse multiple spaces
    )

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