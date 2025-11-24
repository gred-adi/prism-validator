import pandas as pd
import numpy as np

def validate_data(survey_df, tdt_df, prism_df, prism_calc_df=None):
    """Performs metric and calculation validation between TDT and PRISM data.

    This function is divided into two parts:
    1.  **Standard Metric Validation**: Compares the `Point Type` for each metric
        between the TDT and PRISM data on a per-TDT basis.
    2.  **Calculation Validation**: Extracts and displays details for `PRiSM Calc`
        points from both the TDT and PRISM data sources.

    Args:
        survey_df (pd.DataFrame): The full consolidated survey DataFrame, used
                                  for the calculation validation part.
        tdt_df (pd.DataFrame): The parsed TDT DataFrame specific to metric validation.
        prism_df (pd.DataFrame): The DataFrame with metric data queried from PRISM.
        prism_calc_df (pd.DataFrame, optional): A DataFrame with detailed
            calculation data from PRISM. Defaults to None.

    Returns:
        dict: A dictionary containing the validation results, including keys
        for "summary", "matches", "mismatches", "all_entries",
        "calculations_tdt", and "calculations_prism". Returns an empty dict
        if the input `tdt_df` is empty.
    """
    all_mismatches = []
    all_matches = []
    summary_data = []
    all_entries_dfs = []

    if tdt_df.empty:
        return {}

    # --- PREPARATION: Rename TDT Columns to match internal logic ---
    # The parser returns raw names ('Metric', 'Point Type'). We rename them here 
    # to standard keys ('METRIC_NAME', 'POINT_TYPE_TDT') used in the logic.
    tdt_clean = tdt_df.rename(columns={
        'Metric': 'METRIC_NAME',
        'Point Type': 'POINT_TYPE_TDT',
        'Function': 'FUNCTION_TDT'
    })
    
    # Ensure TDT column exists (fallback for older files)
    if 'TDT' not in tdt_clean.columns:
        if 'Model' in tdt_clean.columns:
            tdt_clean['TDT'] = tdt_clean['Model']
        else:
            tdt_clean['TDT'] = 'Unknown'

    # --- PREPARATION: Clean PRISM Columns ---
    prism_clean = prism_df.rename(columns={
        "FORM NAME": "TDT",
        "METRIC NAME": "METRIC_NAME",
        "POINT TYPE": "POINT_TYPE_PRISM"
    }).copy()
    
    if 'METRIC_NAME' in prism_clean.columns:
        prism_clean['METRIC_NAME'] = prism_clean['METRIC_NAME'].str.replace('AP-TVI-', '', regex=False)
    if 'POINT_TYPE_PRISM' in prism_clean.columns:
        prism_clean['POINT_TYPE_PRISM'] = prism_clean['POINT_TYPE_PRISM'].str.title().str.replace('Prism Calc', 'PRiSM Calc', regex=False)

    # --- PART 1: Standard Metric Validation (Group by TDT) ---
    # We iterate through unique TDTs to replicate the original "per-TDT" validation logic
    unique_tdts = tdt_clean['TDT'].dropna().unique()

    for tdt_name in unique_tdts:
        # Subset data
        sub_tdt = tdt_clean[tdt_clean['TDT'] == tdt_name]
        sub_prism = prism_clean[prism_clean['TDT'] == tdt_name]

        # Merge
        merged_df = pd.merge(
            sub_tdt.drop_duplicates(subset=['METRIC_NAME']),
            sub_prism.drop_duplicates(subset=['METRIC_NAME']),
            on='METRIC_NAME',
            how='outer',
            suffixes=('_TDT', '_PRISM'),
            indicator=True
        )

        # Handle TDT column in merged result (fill from name if missing due to merge)
        merged_df['TDT'] = tdt_name
        
        # Fix suffix overlap issues if TDT/PRISM columns collide strangely
        if 'POINT_TYPE_PRISM_PRISM' in merged_df.columns:
            merged_df.rename(columns={'POINT_TYPE_PRISM_PRISM': 'POINT_TYPE_PRISM'}, inplace=True)
        if 'POINT_TYPE_TDT_TDT' in merged_df.columns:
            merged_df.rename(columns={'POINT_TYPE_TDT_TDT': 'POINT_TYPE_TDT'}, inplace=True)

        all_entries_dfs.append(merged_df)

        # Identify Discrepancies
        # Safe access to columns (handle missing if left_only/right_only)
        pt_tdt = merged_df.get('POINT_TYPE_TDT', pd.Series([None]*len(merged_df)))
        pt_prism = merged_df.get('POINT_TYPE_PRISM', pd.Series([None]*len(merged_df)))

        data_mismatch = (merged_df['_merge'] == 'both') & (pt_tdt != pt_prism)
        missing_in_prism = merged_df['_merge'] == 'left_only'
        missing_in_tdt = merged_df['_merge'] == 'right_only'
        
        mismatch_rows = merged_df[data_mismatch | missing_in_prism | missing_in_tdt].copy()
        match_rows = merged_df[(merged_df['_merge'] == 'both') & ~data_mismatch].copy()

        # Summary
        summary_data.append({
            "TDT": tdt_name,
            "Match Count": len(match_rows),
            "Mismatch Count": len(mismatch_rows),
            "Total TDT Records": len(sub_tdt.drop_duplicates(subset=['METRIC_NAME']))
        })

        # Mismatches Detail
        if not mismatch_rows.empty:
            mismatch_rows['Status'] = 'Point Type Mismatch'
            mismatch_rows.loc[missing_in_prism, 'Status'] = 'Missing in PRISM'
            mismatch_rows.loc[missing_in_tdt, 'Status'] = 'Missing in TDT'
            # Ensure columns exist
            if 'POINT_TYPE_TDT' not in mismatch_rows: mismatch_rows['POINT_TYPE_TDT'] = None
            if 'POINT_TYPE_PRISM' not in mismatch_rows: mismatch_rows['POINT_TYPE_PRISM'] = None
            all_mismatches.append(mismatch_rows)

        # Matches Detail
        if not match_rows.empty:
            all_matches.append(match_rows)

    # --- Construct Final Tables for Part 1 ---
    summary_df = pd.DataFrame(summary_data)
    
    all_entries_df = pd.concat(all_entries_dfs, ignore_index=True) if all_entries_dfs else pd.DataFrame()
    if not all_entries_df.empty:
        cols = ['TDT', 'METRIC_NAME', 'POINT_TYPE_TDT', 'POINT_TYPE_PRISM']
        # Keep only available columns + others
        final_cols = [c for c in cols if c in all_entries_df.columns]
        all_entries_df = all_entries_df[final_cols]

    mismatches_df = pd.concat(all_mismatches, ignore_index=True) if all_mismatches else pd.DataFrame()
    if not mismatches_df.empty:
        cols = ['TDT', 'METRIC_NAME', 'Status', 'POINT_TYPE_TDT', 'POINT_TYPE_PRISM']
        final_cols = [c for c in cols if c in mismatches_df.columns]
        mismatches_df = mismatches_df[final_cols].fillna('N/A')

    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    if not matches_df.empty:
        cols = ['TDT', 'METRIC_NAME', 'POINT_TYPE_TDT', 'POINT_TYPE_PRISM']
        final_cols = [c for c in cols if c in matches_df.columns]
        matches_df = matches_df[final_cols]

    # --- PART 2: Calculation Validation ---
    # Table 1: TDT Calculations
    # Logic replicated from tdt_validations/calculation_validation/validator.py
    # 1. Filter for 'PRiSM Calc' points
    calc_tdt = pd.DataFrame()
    if 'Point Type' in survey_df.columns:
        prism_calc_raw = survey_df[survey_df['Point Type'] == 'PRiSM Calc'].copy()
        
        if not prism_calc_raw.empty:
            # 2. Define the columns that should NOT all be blank
            calc_cols = [
                'Calc Point Type',
                'Calculation Description',
                'Pseudo Code',
                'Language',
                'Input Point',
                'PRiSM Code'
            ]
            
            # 3. Ensure all columns exist, fill missing ones with NaN
            for col in calc_cols:
                if col not in prism_calc_raw.columns:
                    prism_calc_raw[col] = np.nan

            # 4. Deduplicate by TDT and Metric, keeping the first instance
            calc_tdt = prism_calc_raw.drop_duplicates(subset=['TDT', 'Metric']).copy()
            
            # 5. Find rows where ALL calc columns are null/NaN
            all_blank_mask = calc_tdt[calc_cols].isnull().all(axis=1)
            
            # 6. Create the 'Issue' column based on the mask
            calc_tdt['Issue'] = np.where(all_blank_mask, "Missing all calculation details", "✅")
    if not calc_tdt.empty:
            cols_to_show = [
                'TDT', 'Metric', 'Point Type', 'Calculation Description',
                'Pseudo Code', 'Language', 'Input Point', 'PRiSM Code'
            ]
            # Filter to ensure columns exist
            final_cols = [c for c in cols_to_show if c in calc_tdt.columns]
            calc_tdt = calc_tdt[final_cols]
    
    # Table 2: PRISM Calculations (Passed directly from query result)
    calc_prism = pd.DataFrame()
    if prism_calc_df is not None and not prism_calc_df.empty:
        temp_df = prism_calc_df.copy()

        # 1. Filter rows based on relevant TDTs (found in the survey/summary)
        # 'FORM NAME' corresponds to TDT in the PRISM query result
        if 'FORM NAME' in temp_df.columns:
             temp_df = temp_df[temp_df['FORM NAME'].isin(unique_tdts)]

        # 2. Rename columns
        rename_map = {
            "FORM NAME": "TDT",
            "METRIC NAME": "Metric",
            "POINT TYPE": "Point Type",
            "CALC_LOGIC": "Calculation Logic",
            "CALC_VARIABLES_NAMES": "Calculation Variables"
        }
        temp_df = temp_df.rename(columns=rename_map)

        # 3. Filter columns (only show those that were renamed)
        # Use a list comprehension to maintain the order specified in the map
        cols_to_keep = [new_name for old_name, new_name in rename_map.items() if new_name in temp_df.columns]
        calc_prism = temp_df[cols_to_keep]

    return {
        "summary": summary_df,
        "matches": matches_df,
        "mismatches": mismatches_df,
        "all_entries": all_entries_df,
        "calculations_tdt": calc_tdt,
        "calculations_prism": calc_prism
    }