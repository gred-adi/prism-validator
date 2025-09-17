import pandas as pd
import streamlit as st

@st.cache_data
def validate_data(tdt_dfs, prism_df):
    """
    Performs the comparison logic for the 'Failure Diagnostics' section.
    Compares Direction and Weight, separates mismatches by column, and reorders columns.
    """
    prism_df = prism_df.copy()
    all_matches = []
    summary_data = []

    # 1. Rename and clean PRISM columns
    prism_df.rename(columns={
        "FORM NAME": "TDT",
        "METRIC NAME": "METRIC_NAME",
        "FAILURE MODE": "FAILURE_MODE",
        "DIRECTION": "DIRECTION",
        "WEIGHT": "WEIGHT"
    }, inplace=True)
    
    prism_df['METRIC_NAME'] = prism_df['METRIC_NAME'].str.replace('AP-TVI-', '', regex=False)
    prism_df['DIRECTION'] = prism_df['DIRECTION'].apply(lambda x: x.replace(" ", ""))

    # Dictionary to hold lists of mismatch dataframes
    columns_to_compare = ['DIRECTION', 'WEIGHT']
    mismatches_by_column = {col: [] for col in columns_to_compare}
    mismatches_by_column['Missing_in_PRISM'] = []
    mismatches_by_column['Missing_in_TDT'] = []
    
    # Columns to join on
    join_keys = ['FAILURE_MODE', 'METRIC_NAME']

    for tdt_name, excel_df in tdt_dfs.items():
        prism_sub_df = prism_df[prism_df["TDT"] == tdt_name].copy()

        merged_df = pd.merge(
            excel_df.drop_duplicates(subset=join_keys),
            prism_sub_df.drop_duplicates(subset=join_keys),
            on=join_keys,
            how='outer',
            suffixes=('_TDT', '_PRISM'),
            indicator=True
        )

        merged_df['WEIGHT_TDT'] = pd.to_numeric(merged_df['WEIGHT_TDT'], errors='coerce')
        merged_df['WEIGHT_PRISM'] = pd.to_numeric(merged_df['WEIGHT_PRISM'], errors='coerce')

        # --- Identify Perfect Matches ---
        match_mask = (merged_df['_merge'] == 'both')
        for col in columns_to_compare:
            col_tdt = merged_df[f"{col}_TDT"]
            col_prism = merged_df[f"{col}_PRISM"]
            # FIX: A match occurs if values are equal OR if both values are missing (NaN).
            match_mask &= (col_tdt == col_prism) | (col_tdt.isna() & col_prism.isna())
        
        match_rows = merged_df[match_mask].copy()
        if not match_rows.empty:
            match_rows['TDT'] = tdt_name
            all_matches.append(match_rows)

        # --- Identify Mismatches by Specific Column ---
        for col in columns_to_compare:
            col_tdt = merged_df[f"{col}_TDT"]
            col_prism = merged_df[f"{col}_PRISM"]
            # FIX: A mismatch occurs if values are different, but NOT if both are just missing.
            mismatch_condition = (col_tdt != col_prism) & ~(col_tdt.isna() & col_prism.isna())
            col_mismatch_mask = (merged_df['_merge'] == 'both') & mismatch_condition

            if col_mismatch_mask.any():
                mismatch_subset = merged_df.loc[col_mismatch_mask, ['FAILURE_MODE', 'METRIC_NAME', f'{col}_TDT', f'{col}_PRISM']].copy()
                mismatch_subset['TDT'] = tdt_name
                mismatches_by_column[col].append(mismatch_subset)

        # --- Identify Records Missing from a Source ---
        missing_in_prism_rows = merged_df[merged_df['_merge'] == 'left_only'].copy()
        if not missing_in_prism_rows.empty:
            missing_in_prism_rows['TDT'] = tdt_name
            mismatches_by_column['Missing_in_PRISM'].append(missing_in_prism_rows)

        missing_in_tdt_rows = merged_df[merged_df['_merge'] == 'right_only'].copy()
        if not missing_in_tdt_rows.empty:
            missing_in_tdt_rows['TDT'] = tdt_name
            mismatches_by_column['Missing_in_TDT'].append(missing_in_tdt_rows)
            
        # --- Append Summary Data ---
        total_mismatch_count = len(merged_df[~match_mask])
        summary_data.append({
            "TDT": tdt_name,
            "Match Count": len(match_rows),
            "Mismatch Count": total_mismatch_count,
            "Total TDT Records": len(excel_df.drop_duplicates(subset=join_keys))
        })

    # --- Create Final DataFrames and Dictionary ---
    summary_df = pd.DataFrame(summary_data)
    
    # Reorder columns for matches
    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    if not matches_df.empty:
        match_col_order = ['TDT'] + join_keys
        for col in columns_to_compare:
            match_col_order.extend([f'{col}_TDT', f'{col}_PRISM'])
        # Filter to only existing columns to prevent errors
        matches_df = matches_df[[c for c in match_col_order if c in matches_df.columns]]

    # Reorder columns for each type of mismatch
    final_mismatches_dict = {}
    for mismatch_type, df_list in mismatches_by_column.items():
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            # Define the ideal column order for each mismatch type
            col_order = ['TDT'] + join_keys
            if mismatch_type in columns_to_compare:
                col_order.extend(['TDT_Value', 'PRISM_Value'])
            else: # For missing records, show all available columns
                tdt_cols = sorted([c for c in df.columns if '_TDT' in c])
                prism_cols = sorted([c for c in df.columns if '_PRISM' in c])
                col_order.extend(tdt_cols + prism_cols)

            # Filter to only existing columns before reordering
            final_mismatches_dict[mismatch_type] = df[[c for c in col_order if c in df.columns]]
        else:
            final_mismatches_dict[mismatch_type] = pd.DataFrame()

    return summary_df, matches_df, final_mismatches_dict