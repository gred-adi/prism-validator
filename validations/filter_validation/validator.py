import pandas as pd
import streamlit as st

def clean_filter_string(series):
    """ Standardize filter strings for comparison """
    if series is None:
        return ""
    return series.astype(str).str.lower().str.replace('=', 'equal to', regex=False).str.strip()

@st.cache_data
def validate_data(model_dfs, prism_df):
    """
    Performs the comparison logic for the 'Filter Validation' section.
    """
    prism_df = prism_df.copy()

    all_matches = []
    summary_data = []

    # 1. Rename and clean PRISM columns
    prism_df.rename(columns={
        "FORM NAME": "Model",
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
        prism_sub_df = prism_df[prism_df["Model"] == model_name].copy()

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
            match_rows['Model'] = model_name
            all_matches.append(match_rows)

        if not mismatch_rows.empty:
            mismatch_subset = mismatch_rows[['METRIC_NAME', 'FILTER_TDT', 'FILTER_PRISM']].copy()
            mismatch_subset.rename(columns={'FILTER_TDT': 'TDT_Value', 'FILTER_PRISM': 'PRISM_Value'}, inplace=True)
            mismatch_subset['Model'] = model_name
            # Reorder columns
            mismatch_subset = mismatch_subset[['Model', 'METRIC_NAME', 'TDT_Value', 'PRISM_Value']]
            mismatches_dict['FILTER'].append(mismatch_subset)
        
        if not missing_in_prism_rows.empty:
            missing_subset = missing_in_prism_rows[['METRIC_NAME', 'FILTER_TDT']].copy()
            missing_subset.rename(columns={'FILTER_TDT': 'FILTER'}, inplace=True)
            missing_subset['Model'] = model_name
            # Reorder columns
            missing_subset = missing_subset[['Model', 'METRIC_NAME', 'FILTER']]
            mismatches_dict['Missing_in_PRISM'].append(missing_subset)

        if not missing_in_tdt_rows.empty:
            missing_subset = missing_in_tdt_rows[['METRIC_NAME', 'FILTER_PRISM']].copy()
            missing_subset.rename(columns={'FILTER_PRISM': 'FILTER'}, inplace=True)
            missing_subset['Model'] = model_name
            # Reorder columns
            missing_subset = missing_subset[['Model', 'METRIC_NAME', 'FILTER']]
            mismatches_dict['Missing_in_TDT'].append(missing_subset)
            
        # --- Append Summary Data ---
        summary_data.append({
            "Model": model_name,
            "Match Count": len(match_rows),
            "Mismatch Count": len(mismatch_rows) + len(missing_in_prism_rows) + len(missing_in_tdt_rows),
            "Total Model Records": len(excel_df.drop_duplicates(subset=['METRIC_NAME']))
        })

    # --- Create Final DataFrames and Dictionary ---
    summary_df = pd.DataFrame(summary_data)
    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    if not matches_df.empty:
         matches_df = matches_df[['Model', 'METRIC_NAME', 'FILTER_TDT', 'FILTER_PRISM']]

    final_mismatches_dict = {key: pd.concat(val, ignore_index=True) if val else pd.DataFrame() for key, val in mismatches_dict.items()}

    return summary_df, matches_df, final_mismatches_dict