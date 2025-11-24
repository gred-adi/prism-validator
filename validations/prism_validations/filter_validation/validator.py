import pandas as pd
import streamlit as st

def clean_filter_string(series):
    """Standardizes filter strings for robust comparison.

    This helper function converts a series of filter strings to a consistent
    format by making them lowercase, replacing '=' with 'equal to', and
    trimming whitespace.

    Args:
        series (pd.Series): A pandas Series containing filter strings.

    Returns:
        pd.Series: The cleaned and standardized Series.
    """
    if series is None:
        return ""
    return series.astype(str).str.lower().str.replace('=', 'equal to', regex=False).str.strip()

@st.cache_data
def validate_data(model_dfs, prism_df):
    """Validates filter configurations between TDT and PRISM data.

    This function compares the filter strings for each metric on a per-model
    basis. It uses a standardized cleaning function to ensure comparisons are
    not affected by minor formatting differences.

    Args:
        model_dfs (dict[str, pd.DataFrame]): A dictionary of DataFrames parsed
            from the TDT, where keys are model names.
        prism_df (pd.DataFrame): A DataFrame containing the filter data queried
            from the PRISM database.

    Returns:
        dict: A dictionary containing the validation results, with keys for
        "summary", "matches", "mismatches", and "all_entries".
    """
    prism_df = prism_df.copy()
    all_matches = []
    summary_data = []
    all_entries_dfs = []  # To store each model's merged df

    # 1. Rename and clean PRISM columns
    prism_df.rename(columns={
        "FORM NAME": 'MODEL',
        "METRIC NAME": "METRIC_NAME",
        "FILTER": "FILTER"
    }, inplace=True)
    prism_df['METRIC_NAME'] = prism_df['METRIC_NAME'].str.replace('AP-TVI-', '', regex=False)

    # 2. Setup for comparison
    mismatches_dict = {
        'FILTER': [],
        'Missing_in_PRISM': [],
        'Missing_in_TDT': []
    }

    # 3. Iterate through each model and compare
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

        # Store the full merged df for the "All Entries" table
        merged_with_model = merged_df.copy()
        merged_with_model['MODEL'] = model_name
        all_entries_dfs.append(merged_with_model)

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
            mismatch_subset = mismatch_subset[['MODEL', 'METRIC_NAME', 'TDT_Value', 'PRISM_Value']]
            mismatches_dict['FILTER'].append(mismatch_subset)
        
        if not missing_in_prism_rows.empty:
            missing_subset = missing_in_prism_rows[['METRIC_NAME', 'FILTER_TDT']].copy()
            missing_subset.rename(columns={'FILTER_TDT': 'FILTER'}, inplace=True)
            missing_subset['MODEL'] = model_name
            missing_subset = missing_subset[['MODEL', 'METRIC_NAME', 'FILTER']]
            mismatches_dict['Missing_in_PRISM'].append(missing_subset)

        if not missing_in_tdt_rows.empty:
            missing_subset = missing_in_tdt_rows[['METRIC_NAME', 'FILTER_PRISM']].copy()
            missing_subset.rename(columns={'FILTER_PRISM': 'FILTER'}, inplace=True)
            missing_subset['MODEL'] = model_name
            missing_subset = missing_subset[['MODEL', 'METRIC_NAME', 'FILTER']]
            mismatches_dict['Missing_in_TDT'].append(missing_subset)
            
        # --- Append Summary Data ---
        summary_data.append({
            'MODEL': model_name,
            "Match Count": len(match_rows),
            "Mismatch Count": len(mismatch_rows) + len(missing_in_prism_rows) + len(missing_in_tdt_rows),
            "Total Model Records": len(excel_df.drop_duplicates(subset=['METRIC_NAME']))
        })

    # 4. Create Final DataFrames and Dictionary
    summary_df = pd.DataFrame(summary_data)
    all_entries_df = pd.concat(all_entries_dfs, ignore_index=True) if all_entries_dfs else pd.DataFrame()

    # Reorder columns for the 'All Entries' table
    if not all_entries_df.empty:
        # Define the ideal column order
        col_order = ['MODEL', 'METRIC_NAME', 'FILTER_TDT', 'FILTER_PRISM']

        # Get existing columns in the ideal order and then the rest
        existing_cols_in_order = [c for c in col_order if c in all_entries_df.columns]


        # Combine to get the final order
        final_order = existing_cols_in_order
        all_entries_df = all_entries_df[final_order]

    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    if not matches_df.empty:
        # Define the ideal column order
        col_order = ['MODEL', 'METRIC_NAME', 'FILTER_TDT', 'FILTER_PRISM']

        # Get existing columns in the ideal order and then the rest
        existing_cols_in_order = [c for c in col_order if c in matches_df.columns]


        # Combine to get the final order
        final_order = existing_cols_in_order
        matches_df = matches_df[final_order]

    final_mismatches_dict = {key: pd.concat(val, ignore_index=True) if val else pd.DataFrame() for key, val in mismatches_dict.items()}

    return {
        "summary": summary_df,
        "matches": matches_df,
        "mismatches": final_mismatches_dict,
        "all_entries": all_entries_df
    }