import pandas as pd
import streamlit as st

def clean_filter_string(series):
    """
    Standardizes a pandas Series of filter strings for reliable comparison.

    This helper function performs several cleaning steps:
    - Converts all values to strings.
    - Converts text to lowercase.
    - Replaces the '=' character with the phrase 'equal to'.
    - Strips leading/trailing whitespace.
    - Handles None or empty inputs gracefully.

    Args:
        series (pd.Series): A pandas Series containing the filter strings.

    Returns:
        pd.Series: The cleaned and standardized pandas Series.
    """
    if series is None:
        return ""
    return series.astype(str).str.lower().str.replace('=', 'equal to', regex=False).str.strip()

@st.cache_data
def validate_data(model_dfs, prism_df):
    """
    Compares TDT and PRISM data for filter configurations.

    This function performs the core validation by comparing the parsed TDT Excel
    data against PRISM database data for each model. It uses a helper function,
    `clean_filter_string`, to standardize the filter text before comparison.

    The comparison logic identifies:
    - **Matches:** Records where the cleaned filter strings are identical.
    - **Mismatches:** Records found in both sources but with different filter strings.
    - **Missing Records:** Records found in one source but not the other.

    The function is cached with `@st.cache_data` for performance optimization.

    Args:
        model_dfs (dict[str, pd.DataFrame]): A dictionary where keys are model
            names and values are DataFrames of parsed filter data from the TDT.
        prism_df (pd.DataFrame): A DataFrame containing the corresponding
            filter data queried from the PRISM database.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]: A tuple
        containing three elements:
        1. summary_df (pd.DataFrame): A DataFrame summarizing the validation
           results per model with match and mismatch counts.
        2. matches_df (pd.DataFrame): A DataFrame of all records that matched
           perfectly between the TDT and PRISM data.
        3. mismatches_dict (dict[str, pd.DataFrame]): A dictionary where each
           key is a mismatch type ('FILTER', 'Missing_in_PRISM', 'Missing_in_TDT')
           and the value is a DataFrame of the mismatched records.
    """
    prism_df = prism_df.copy()

    all_matches = []
    summary_data = []

    # 1. Rename and clean PRISM columns
    prism_df.rename(columns={
        "FORM NAME": 'MODEL',
        "METRIC NAME": "METRIC_NAME",
        "FILTER": "FILTER"
    }, inplace=True)
    
    prism_df['METRIC_NAME'] = prism_df['METRIC_NAME'].str.replace('AP-TVI-', '', regex=False)

    # Dictionary to hold mismatch results
    mismatches_dict = {
        'FILTER': [],
        'Missing_in_PRISM': [],
        'Missing_in_TDT': []
    }

    for model_name, excel_df in model_dfs.items():
        prism_sub_df = prism_df[prism_df['MODEL'] == model_name].copy()

        excel_df['FILTER_CLEAN'] = clean_filter_string(excel_df['FILTER'])
        prism_sub_df['FILTER_CLEAN'] = clean_filter_string(prism_sub_df['FILTER'])

        merged_df = pd.merge(
            excel_df.drop_duplicates(subset=['METRIC_NAME']),
            prism_sub_df.drop_duplicates(subset=['METRIC_NAME']),
            on='METRIC_NAME',
            how='outer',
            suffixes=('_TDT', '_PRISM'),
            indicator=True
        )

        # --- Identify Matches and Mismatches ---
        match_mask = (merged_df['_merge'] == 'both') & \
                     (merged_df['FILTER_CLEAN_TDT'] == merged_df['FILTER_CLEAN_PRISM'])
        
        mismatch_mask = (merged_df['_merge'] == 'both') & \
                        (merged_df['FILTER_CLEAN_TDT'] != merged_df['FILTER_CLEAN_PRISM'])

        match_rows = merged_df[match_mask].copy()
        mismatch_rows = merged_df[mismatch_mask].copy()
        missing_in_prism_rows = merged_df[merged_df['_merge'] == 'left_only'].copy()
        missing_in_tdt_rows = merged_df[merged_df['_merge'] == 'right_only'].copy()

        # --- Process and store results ---
        if not match_rows.empty:
            match_rows['MODEL'] = model_name
            all_matches.append(match_rows)

        if not mismatch_rows.empty:
            mismatch_subset = mismatch_rows[['METRIC_NAME', 'FILTER_TDT', 'FILTER_PRISM']].copy()
            mismatch_subset.rename(columns={'FILTER_TDT': 'TDT_Value', 'FILTER_PRISM': 'PRISM_Value'}, inplace=True)
            mismatch_subset['MODEL'] = model_name
            # Reorder columns
            mismatch_subset = mismatch_subset[['MODEL', 'METRIC_NAME', 'TDT_Value', 'PRISM_Value']]
            mismatches_dict['FILTER'].append(mismatch_subset)
        
        if not missing_in_prism_rows.empty:
            missing_subset = missing_in_prism_rows[['METRIC_NAME', 'FILTER_TDT']].copy()
            missing_subset.rename(columns={'FILTER_TDT': 'FILTER'}, inplace=True)
            missing_subset['MODEL'] = model_name
            # Reorder columns
            missing_subset = missing_subset[['MODEL', 'METRIC_NAME', 'FILTER']]
            mismatches_dict['Missing_in_PRISM'].append(missing_subset)

        if not missing_in_tdt_rows.empty:
            missing_subset = missing_in_tdt_rows[['METRIC_NAME', 'FILTER_PRISM']].copy()
            missing_subset.rename(columns={'FILTER_PRISM': 'FILTER'}, inplace=True)
            missing_subset['MODEL'] = model_name
            # Reorder columns
            missing_subset = missing_subset[['MODEL', 'METRIC_NAME', 'FILTER']]
            mismatches_dict['Missing_in_TDT'].append(missing_subset)
            
        # --- Append Summary Data ---
        summary_data.append({
            'MODEL': model_name,
            "Match Count": len(match_rows),
            "Mismatch Count": len(mismatch_rows) + len(missing_in_prism_rows) + len(missing_in_tdt_rows),
            "Total Model Records": len(excel_df.drop_duplicates(subset=['METRIC_NAME']))
        })

    # --- Create Final DataFrames and Dictionary ---
    summary_df = pd.DataFrame(summary_data)
    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    if not matches_df.empty:
         matches_df = matches_df[['MODEL', 'METRIC_NAME', 'FILTER_TDT', 'FILTER_PRISM']]

    final_mismatches_dict = {key: pd.concat(val, ignore_index=True) if val else pd.DataFrame() for key, val in mismatches_dict.items()}

    return summary_df, matches_df, final_mismatches_dict