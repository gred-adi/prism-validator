import pandas as pd
import streamlit as st

@st.cache_data
def validate_data(tdt_dfs, prism_df):
    """Validates failure diagnostics between TDT and PRISM data.

    This function compares diagnostic `DIRECTION` and `WEIGHT` for each metric
    within a failure mode. It identifies matches, various types of mismatches,
    and missing records. It also extracts the 'Prescriptive' information (failure
    description and next steps) from the PRISM data for display.

    Args:
        tdt_dfs (dict[str, pd.DataFrame]): A dictionary of DataFrames parsed
            from the TDT, where keys are TDT names.
        prism_df (pd.DataFrame): A DataFrame containing the failure diagnostics
            data queried from the PRISM database.

    Returns:
        dict: A dictionary containing the validation results, with the
        following keys:
        - "summary" (pd.DataFrame): A summary of matches, mismatches, and
          prescriptive checks per TDT.
        - "matches" (pd.DataFrame): A DataFrame of records that match perfectly.
        - "mismatches" (dict[str, pd.DataFrame]): A dictionary of DataFrames for
          different mismatch types.
        - "all_entries" (pd.DataFrame): A DataFrame showing the full outer join
          between TDT and PRISM data.
        - "prescriptive" (pd.DataFrame): A DataFrame with the unique failure
          descriptions and next steps from PRISM.
    """
    prism_df = prism_df.copy()
    all_matches = []
    summary_data = []
    all_entries_dfs = []  # To store each model's merged df

    # 1. Rename and clean PRISM columns
    prism_df.rename(columns={
        "FORM NAME": "TDT",
        "METRIC NAME": "METRIC_NAME",
        "FAILURE MODE": "FAILURE_MODE",
        "FAILURE DESCRIPTION": "FAILURE_DESCRIPTION",
        "NEXT STEPS": "NEXT_STEPS",
        "DIRECTION": "DIRECTION",
        "WEIGHT": "WEIGHT"
    }, inplace=True)
    
    if 'METRIC_NAME' in prism_df.columns:
        prism_df['METRIC_NAME'] = prism_df['METRIC_NAME'].str.replace('AP-TVI-', '', regex=False)
    if 'DIRECTION' in prism_df.columns:
        prism_df['DIRECTION'] = prism_df['DIRECTION'].apply(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    # --- NEW: Extract Prescriptive Data (No impact to validation) ---
    # We grab unique failure mode info from the DB source
    prescriptive_cols = ['TDT', 'FAILURE_MODE', 'FAILURE_DESCRIPTION', 'NEXT_STEPS']
    # Ensure columns exist before selecting
    existing_prescriptive_cols = [c for c in prescriptive_cols if c in prism_df.columns]
    prescriptive_df = prism_df[existing_prescriptive_cols].drop_duplicates().sort_values(by=['TDT', 'FAILURE_MODE'])
    
    # --- FILTER: Ensure we only show Prescriptive info for TDTs in the uploaded files ---
    valid_tdts = list(tdt_dfs.keys())
    if 'TDT' in prescriptive_df.columns:
        prescriptive_df = prescriptive_df[prescriptive_df['TDT'].isin(valid_tdts)]

    # 2. Setup for comparison
    columns_to_compare = ['DIRECTION', 'WEIGHT']
    mismatches_by_column = {col: [] for col in columns_to_compare}
    mismatches_by_column['Missing_in_PRISM'] = []
    mismatches_by_column['Missing_in_TDT'] = []
    join_keys = ['FAILURE_MODE', 'METRIC_NAME']

    # 3. Iterate through each TDT and compare
    for tdt_name, excel_df in tdt_dfs.items():
        prism_sub_df = prism_df[prism_df["TDT"] == tdt_name].copy() if "TDT" in prism_df.columns else pd.DataFrame()

        # --- Logic for Summary Table Improvements ---
        # A. Failure Mode Count (PRISM Side)
        # Get unique failure modes in PRISM for this TDT
        prism_fms = prism_sub_df[['FAILURE_MODE', 'FAILURE_DESCRIPTION', 'NEXT_STEPS']].drop_duplicates() if not prism_sub_df.empty else pd.DataFrame(columns=['FAILURE_MODE', 'FAILURE_DESCRIPTION', 'NEXT_STEPS'])
        fm_count = len(prism_fms)

        # B. Prescriptive Check (Description & Next Steps)
        prescriptive_check = "N/A"
        if fm_count > 0:
            # Check for nulls or empty strings in Description or Next Steps
            # We use fillna('') to handle NaNs, then strip checks to catch empty strings
            desc_missing = prism_fms['FAILURE_DESCRIPTION'].fillna('').astype(str).str.strip() == ''
            steps_missing = prism_fms['NEXT_STEPS'].fillna('').astype(str).str.strip() == ''
            
            if desc_missing.any() or steps_missing.any():
                prescriptive_check = "❌"
            else:
                prescriptive_check = "✅"

        # --- Existing Validation Logic ---
        merged_df = pd.merge(
            excel_df.drop_duplicates(subset=join_keys),
            prism_sub_df.drop_duplicates(subset=join_keys),
            on=join_keys,
            how='outer',
            suffixes=('_TDT', '_PRISM'),
            indicator=True
        )

        # Store the full merged df for the "All Entries" table
        merged_with_tdt = merged_df.copy()
        merged_with_tdt['TDT'] = tdt_name
        all_entries_dfs.append(merged_with_tdt)

        # Convert weight columns to numeric for proper comparison
        if 'WEIGHT_TDT' in merged_df.columns:
            merged_df['WEIGHT_TDT'] = pd.to_numeric(merged_df['WEIGHT_TDT'], errors='coerce')
        if 'WEIGHT_PRISM' in merged_df.columns:
            merged_df['WEIGHT_PRISM'] = pd.to_numeric(merged_df['WEIGHT_PRISM'], errors='coerce')

        # --- Identify Perfect Matches ---
        match_mask = (merged_df['_merge'] == 'both')
        for col in columns_to_compare:
            col_tdt = merged_df.get(f"{col}_TDT")
            col_prism = merged_df.get(f"{col}_PRISM")
            if col_tdt is not None and col_prism is not None:
                match_mask &= (col_tdt == col_prism) | (col_tdt.isna() & col_prism.isna())
        
        match_rows = merged_df[match_mask].copy()
        if not match_rows.empty:
            match_rows['TDT'] = tdt_name
            all_matches.append(match_rows)

        # --- Identify Mismatches by Specific Column ---
        for col in columns_to_compare:
            col_tdt = merged_df.get(f"{col}_TDT")
            col_prism = merged_df.get(f"{col}_PRISM")
            if col_tdt is not None and col_prism is not None:
                mismatch_condition = (col_tdt != col_prism) & ~(col_tdt.isna() & col_prism.isna())
                col_mismatch_mask = (merged_df['_merge'] == 'both') & mismatch_condition

                if col_mismatch_mask.any():
                    mismatch_subset = merged_df.loc[col_mismatch_mask, join_keys + [f'{col}_TDT', f'{col}_PRISM']].copy()
                    mismatch_subset.rename(columns={f'{col}_TDT': 'TDT_Value', f'{col}_PRISM': 'PRISM_Value'}, inplace=True)
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
            "Failure Mode Count": fm_count,
            "Prescriptive Check": prescriptive_check,
            "Match Count": len(match_rows),
            "Mismatch Count": total_mismatch_count,
            "Total TDT Records": len(excel_df.drop_duplicates(subset=join_keys))
        })

    # 4. Create Final DataFrames and Dictionary
    summary_df = pd.DataFrame(summary_data)
    
    # Ensure column order if summary_df is not empty
    if not summary_df.empty:
        desired_order = ["TDT", "Failure Mode Count", "Prescriptive Check", "Match Count", "Mismatch Count", "Total TDT Records"]
        # Filter cols that actually exist
        cols = [c for c in desired_order if c in summary_df.columns]
        summary_df = summary_df[cols]

    all_entries_df = pd.concat(all_entries_dfs, ignore_index=True) if all_entries_dfs else pd.DataFrame()
    
    # Format WEIGHT columns to remove trailing '.0'
    if not all_entries_df.empty:
        for col in ['WEIGHT_TDT', 'WEIGHT_PRISM']:
            if col in all_entries_df.columns:
                # Using .astype(str) to handle mixed types and replacing .0 for whole numbers
                all_entries_df[col] = all_entries_df[col].astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', pd.NA)

    # Reorder columns for the 'All Entries' table
    if not all_entries_df.empty:
        # Define the ideal column order
        col_order = ['TDT'] + join_keys
        for col in columns_to_compare:
            col_order.extend([f'{col}_TDT', f'{col}_PRISM'])

        # Get existing columns in the ideal order and then the rest
        existing_cols_in_order = [c for c in col_order if c in all_entries_df.columns]

        # Combine to get the final order
        final_order = existing_cols_in_order
        all_entries_df = all_entries_df[final_order]

    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    if not matches_df.empty:
        # Define the ideal column order
        col_order = ['TDT'] + join_keys
        for col in columns_to_compare:
            col_order.extend([f'{col}_TDT', f'{col}_PRISM'])
    
    # Format WEIGHT columns to remove trailing '.0'
    if not matches_df.empty:
        for col in ['WEIGHT_TDT', 'WEIGHT_PRISM']:
            if col in matches_df.columns:
                # Using .astype(str) to handle mixed types and replacing .0 for whole numbers
                matches_df[col] = matches_df[col].astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', pd.NA)

        # Get existing columns in the ideal order and then the rest
        existing_cols_in_order = [c for c in col_order if c in matches_df.columns]

        # Combine to get the final order
        final_order = existing_cols_in_order
        matches_df = matches_df[final_order]

    final_mismatches_dict = {}
    for mismatch_type, df_list in mismatches_by_column.items():
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            col_order = ['TDT'] + join_keys
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
        "all_entries": all_entries_df,
        "prescriptive": prescriptive_df
    }