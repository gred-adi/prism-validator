import pandas as pd
import io
import os
import re
import streamlit as st

# --- Helper functions (from your script, largely unchanged) ---
def _process_survey_sheets(file_path, survey_data_list):
    """
    Processes the 'Point Survey' and related sheets from a single TDT Excel file.

    This function reads the 'Version', 'Point Survey', 'Attribute', and
    'Calculation' sheets. It extracts data for each model defined in the
    'Point Survey' sheet, merges it with attribute and calculation details,
    and appends the resulting DataFrame to a list.

    Args:
        file_path (str): The full path to the TDT Excel file.
        survey_data_list (list): A list to which the processed survey
            DataFrames will be appended. This list is modified in place.
    """
    # (This function's internal logic remains the same as your provided script)
    xls = pd.ExcelFile(file_path)
    if 'Point Survey' not in xls.sheet_names:
        st.warning(f"'Point Survey' sheet not found in {os.path.basename(file_path)}")
        return

    df_version = pd.read_excel(xls, 'Version', header=None, nrows=5)
    tdt_name = str(df_version.iloc[4, 3])
    df_point_survey = pd.read_excel(xls, 'Point Survey', header=None)

    df_attribute = pd.DataFrame()
    if 'Attribute' in xls.sheet_names:
        df_attribute = pd.read_excel(xls, 'Attribute', header=None).iloc[3:, [1, 4, 5, 6, 7]]
        df_attribute.columns = ['Metric', 'Function', 'Constraint', 'Filter Condition', 'Filter Value']

    df_calculation = pd.DataFrame()
    if 'Calculation' in xls.sheet_names:
        df_calculation = pd.read_excel(xls, 'Calculation', header=None).iloc[2:, [1, 2, 3, 5, 6, 7, 8]]
        df_calculation.columns = ['Metric', 'Calc Point Type', 'Calculation Description', 'Pseudo Code', 'Language', 'Input Point', 'PRiSM Code']

    start_col = 3
    while start_col < df_point_survey.shape[1]:
        end_col = start_col + 5
        model_name = str(df_point_survey.iloc[0, start_col])
        if pd.notna(model_name) and model_name.strip() != "":
            headers = list(df_point_survey.iloc[1, 1:3].values) + list(df_point_survey.iloc[1, start_col:end_col].values)
            sub_data = pd.concat([df_point_survey.iloc[2:, 1:3], df_point_survey.iloc[2:, start_col:end_col]], axis=1)
            sub_data = sub_data.dropna(subset=sub_data.columns[-5:], how='all')

            if not sub_data.empty:
                sub_table_df = pd.DataFrame(sub_data.values, columns=headers)
                sub_table_df['TDT'] = tdt_name
                sub_table_df['Model'] = model_name

                if not df_attribute.empty:
                    sub_table_df = sub_table_df.merge(df_attribute, how='inner', on='Metric')
                if not df_calculation.empty:
                    sub_table_df = sub_table_df.merge(df_calculation, how='left', on='Metric')
                
                survey_data_list.append(sub_table_df)
        start_col = end_col

def _process_diagnostic_sheet(file_path, diag_data_list):
    """
    Processes the 'Diagnostic' sheet from a single TDT Excel file.

    This function reads the 'Version' and 'Diagnostic' sheets. It extracts
    data for each failure mode defined in the 'Diagnostic' sheet and appends
    the resulting DataFrame to a list.

    Args:
        file_path (str): The full path to the TDT Excel file.
        diag_data_list (list): A list to which the processed diagnostic
            DataFrames will be appended. This list is modified in place.
    """
    # (This function's internal logic remains the same as your provided script)
    xls = pd.ExcelFile(file_path)
    if 'Diagnostic' not in xls.sheet_names:
        st.warning(f"'Diagnostic' sheet not found in {os.path.basename(file_path)}")
        return

    df_version = pd.read_excel(xls, 'Version', header=None, nrows=5)
    tdt_name = str(df_version.iloc[4, 3])
    df_diagnostics = pd.read_excel(xls, 'Diagnostic', header=None)

    start_col = 7
    while start_col < df_diagnostics.shape[1]:
        end_col = start_col + 3
        failure_enable = str(df_diagnostics.iloc[0, start_col])
        failure_name = str(df_diagnostics.iloc[1, start_col])

        if pd.notna(failure_name) and failure_name.strip() != "":
            headers = list(df_diagnostics.iloc[1, 1:3].values) + ["Direction"] + list(df_diagnostics.iloc[1, start_col + 1:end_col].values)
            sub_data = pd.concat([df_diagnostics.iloc[3:, 1:3], df_diagnostics.iloc[3:, start_col:end_col]], axis=1)
            sub_data.columns = headers
            sub_data = sub_data.dropna(subset=["Direction"])

            if not sub_data.empty:
                sub_table_df = pd.DataFrame(sub_data.values, columns=headers)
                sub_table_df['TDT'] = tdt_name
                sub_table_df['Failure Mode'] = failure_name
                sub_table_df['Failure Mode Enabled'] = failure_enable
                diag_data_list.append(sub_table_df)
        start_col = end_col

# --- NEW: Helper function to create the filter summary ---
def _create_filter_summary(all_survey_data):
    """
    Creates a filter summary DataFrame from the consolidated survey data.

    This function filters the main survey DataFrame to include only rows
    where 'Filter Condition' is not null. It then selects and renames
    columns to create a clean summary table.

    Args:
        all_survey_data (pd.DataFrame): The consolidated DataFrame containing
            all survey data from the TDT files.

    Returns:
        pd.DataFrame: A new DataFrame containing a summary of all filter
            conditions. If the 'Filter Condition' column is not found,
            it returns an empty DataFrame with the expected schema.
    """
    if 'Filter Condition' not in all_survey_data.columns:
        # Return an empty df with expected columns if the necessary column is missing
        return pd.DataFrame(columns=['TDT', 'Model', 'Metric', 'Point Name', 'Constraint', 'Filter Condition', 'Filter Value'])
        
    filtered_data = all_survey_data[all_survey_data['Filter Condition'].notnull()].copy()
    
    # Ensure 'Canary Point Name' exists, if not, use a placeholder.
    if 'Canary Point Name' not in filtered_data.columns:
        filtered_data['Canary Point Name'] = 'N/A'

    filter_df = filtered_data[[
        'TDT', 'Model', 'Metric', 'Canary Point Name', 
        'Constraint', 'Filter Condition', 'Filter Value'
    ]].copy()
    filter_df.rename(columns={'Canary Point Name': 'Point Name'}, inplace=True)
    return filter_df

# --- Main Generator Function (unchanged) ---
@st.cache_data
def generate_files_from_folder(folder_path):
    """
    Scans a folder for TDT Excel files and consolidates them into two DataFrames.

    This function iterates over all `.xlsx` files in the specified folder,
    calling helper functions to process the 'Point Survey' and 'Diagnostic'
    sheets from each. It then concatenates the results into two main
    DataFrames. The function is cached by Streamlit to avoid reprocessing
    the same folder.

    Args:
        folder_path (str): The path to the folder containing the TDT Excel files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas
        DataFrames: the consolidated survey DataFrame and the consolidated
        diagnostic DataFrame.

    Raises:
        FileNotFoundError: If no `.xlsx` files are found in the specified folder.
        ValueError: If no valid survey or diagnostic data can be consolidated
            from the files.
    """
    tdt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx') and not f.startswith('~')]
    if not tdt_files:
        raise FileNotFoundError(f"No Excel files (.xlsx) found in the folder: {folder_path}")

    all_survey_data = []
    all_diag_data = []
    for file_path in tdt_files:
        try:
            _process_survey_sheets(file_path, all_survey_data)
            _process_diagnostic_sheet(file_path, all_diag_data)
        except Exception as e:
            st.error(f"Failed to process file {os.path.basename(file_path)}: {e}")
            continue

    if not all_survey_data:
        raise ValueError("No valid survey data could be consolidated from the provided files.")
    if not all_diag_data:
        raise ValueError("No valid diagnostic data could be consolidated from the provided files.")

    survey_df = pd.concat(all_survey_data, ignore_index=True)
    diag_df = pd.concat(all_diag_data, ignore_index=True)

    # --- NEW: Reorder columns for better presentation ---
    survey_cols = ['TDT', 'Model'] + [col for col in survey_df.columns if col not in ['TDT', 'Model']]
    survey_df = survey_df[survey_cols]

    diag_cols = ['TDT', 'Failure Mode'] + [col for col in diag_df.columns if col not in ['TDT', 'Failure Mode']]
    diag_df = diag_df[diag_cols]


    return survey_df, diag_df

# --- UPDATED: Optional Excel Conversion Function ---
def convert_dfs_to_excel_bytes(survey_df, diag_df):
    """
    Converts survey and diagnostic DataFrames into in-memory Excel files.

    This function takes two DataFrames, writes them to separate in-memory
    IO buffers as Excel files, and returns the byte representations. The survey
    Excel buffer includes a main 'Consolidated Point Survey' sheet and a
    'Filter Summary' sheet.

    Args:
        survey_df (pd.DataFrame): The consolidated survey data.
        diag_df (pd.DataFrame): The consolidated diagnostic data.

    Returns:
        tuple[io.BytesIO, io.BytesIO]: A tuple containing two `BytesIO`
        objects. The first is the in-memory Excel file for survey data,
        and the second is for the diagnostic data.
    """
    output_survey = io.BytesIO()
    with pd.ExcelWriter(output_survey, engine='xlsxwriter') as writer:
        # Write the main consolidated sheet
        survey_df.to_excel(writer, sheet_name='Consolidated Point Survey', index=False)
        
        # --- NEW: Create and write the Filter Summary sheet ---
        filter_summary_df = _create_filter_summary(survey_df)
        filter_summary_df.to_excel(writer, sheet_name='Filter Summary', index=False)
    
    output_diag = io.BytesIO()
    with pd.ExcelWriter(output_diag, engine='xlsxwriter') as writer:
        diag_df.to_excel(writer, sheet_name='Consolidated Failure Diagnostic', index=False)

    output_survey.seek(0)
    output_diag.seek(0)

    return output_survey, output_diag

