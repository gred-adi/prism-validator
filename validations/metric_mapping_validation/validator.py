import pandas as pd
import streamlit as st

@st.cache_data
def validate_data(model_dfs, prism_df):
    """
    Performs the comparison logic for the 'Metric Mapping' section and
    returns a dictionary of results.
    """
    prism_df = prism_df.copy()
    all_matches = []
    summary_data = []
    all_entries_dfs = []  # To store each model's merged df

    # 1. Rename and clean PRISM columns
    prism_df.rename(columns={
        "FORM NAME": 'MODEL',
        "METRIC NAME": "METRIC_NAME",
        "POINT NAME": "POINT_NAME",
        "POINT DESCRIPTION": "POINT_DESCRIPTION",
        "FUNCTION": "FUNCTION",
        "POINT TYPE": "POINT_TYPE",
        "POINT UNIT": "POINT_UNIT"
    }, inplace=True)
    if 'METRIC_NAME' in prism_df.columns:
        prism_df['METRIC_NAME'] = prism_df['METRIC_NAME'].str.replace('AP-TVI-', '', regex=False)
    if 'POINT_TYPE' in prism_df.columns:
        prism_df['POINT_TYPE'] = prism_df['POINT_TYPE'].str.title().str.replace('Prism Calc', 'PRiSM Calc', regex=False)
    if 'FUNCTION' in prism_df.columns:
        prism_df['FUNCTION'] = prism_df['FUNCTION'].str.title().str.replace('Non-Modeled', 'Not Modeled', regex=False)

    # 2. Setup for comparison
    columns_to_compare = ['POINT_NAME', 'POINT_DESCRIPTION', 'FUNCTION', 'POINT_TYPE', 'POINT_UNIT']
    mismatches_by_column = {col: [] for col in columns_to_compare}
    mismatches_by_column['Missing_in_PRISM'] = []
    mismatches_by_column['Missing_in_TDT'] = []

    # 3. Iterate through each model and compare
    for model_name, excel_df in model_dfs.items():
        prism_sub_df = prism_df[prism_df['MODEL'] == model_name].copy()

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

        # --- Identify Perfect Matches ---
        match_mask = (merged_df['_merge'] == 'both')
        for col in columns_to_compare:
            values_match = (merged_df[f"{col}_TDT"].astype(str).str.upper() == merged_df[f"{col}_PRISM"].astype(str).str.upper())
            is_prism_calc_override = (merged_df['POINT_TYPE_TDT'] == 'PRiSM Calc')
            condition = values_match if col == 'POINT_UNIT' else (values_match | is_prism_calc_override)
            match_mask &= condition
        
        match_rows = merged_df[match_mask].copy()
        if not match_rows.empty:
            match_rows['MODEL'] = model_name
            all_matches.append(match_rows)

        # --- Identify Mismatches by Specific Column ---
        for col in columns_to_compare:
            values_mismatch = (merged_df[f"{col}_TDT"].astype(str).str.upper() != merged_df[f"{col}_PRISM"].astype(str).str.upper())
            col_mismatch_mask = (merged_df['_merge'] == 'both') & values_mismatch
            if col != 'POINT_UNIT':
                col_mismatch_mask &= (merged_df['POINT_TYPE_TDT'] != 'PRiSM Calc')
            
            if col_mismatch_mask.any():
                mismatch_subset = merged_df.loc[col_mismatch_mask, ['METRIC_NAME', f'{col}_TDT', f'{col}_PRISM']].copy()
                mismatch_subset.rename(columns={f'{col}_TDT': 'TDT_Value', f'{col}_PRISM': 'PRISM_Value'}, inplace=True)
                mismatch_subset['MODEL'] = model_name
                mismatch_subset = mismatch_subset[['MODEL', 'METRIC_NAME', 'TDT_Value', 'PRISM_Value']]
                mismatches_by_column[col].append(mismatch_subset)
        
        # --- Identify and format Records Missing from a Source ---
        missing_in_prism_rows = merged_df[merged_df['_merge'] == 'left_only'].copy()
        if not missing_in_prism_rows.empty:
            missing_in_prism_rows['MODEL'] = model_name
            mismatches_by_column['Missing_in_PRISM'].append(missing_in_prism_rows)

        missing_in_tdt_rows = merged_df[merged_df['_merge'] == 'right_only'].copy()
        if not missing_in_tdt_rows.empty:
            missing_in_tdt_rows['MODEL'] = model_name
            mismatches_by_column['Missing_in_TDT'].append(missing_in_tdt_rows)
            
        # --- Append Summary Data ---
        total_mismatch_count = len(merged_df[~match_mask])
        summary_data.append({
            'MODEL': model_name,
            "Match Count": len(match_rows),
            "Mismatch Count": total_mismatch_count,
            "Total Model Records": len(excel_df.drop_duplicates(subset=['METRIC_NAME']))
        })

    # 4. Create Final DataFrames and Dictionary
    summary_df = pd.DataFrame(summary_data)
    all_entries_df = pd.concat(all_entries_dfs, ignore_index=True) if all_entries_dfs else pd.DataFrame()
    
    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    if not matches_df.empty:
        col_order = ['MODEL', 'METRIC_NAME'] + [f"{c}_TDT" for c in columns_to_compare] + [f"{c}_PRISM" for c in columns_to_compare]
        matches_df = matches_df[[c for c in col_order if c in matches_df.columns]]

    final_mismatches_dict = {}
    for mismatch_type, df_list in mismatches_by_column.items():
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            col_order = ['MODEL', 'METRIC_NAME']    
            if mismatch_type in columns_to_compare:
                col_order.extend(['TDT_Value', 'PRISM_Value'])
            else:
                tdt_cols = sorted([c for c in df.columns if '_TDT' in c])
                prism_cols = sorted([c for c in df.columns if '_PRISM' in c])
                col_order.extend(tdt_cols + prism_cols)
            final_mismatches_dict[mismatch_type] = df[[c for c in col_order if c in df.columns]]
        else:
            final_mismatches_dict[mismatch_type] = pd.DataFrame()

    return {
        "summary": summary_df,
        "matches": matches_df,
        "mismatches": final_mismatches_dict,
        "all_entries": all_entries_df
    }