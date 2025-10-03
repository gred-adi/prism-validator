import pandas as pd

def validate_data(tdt_dfs, prism_df):
    """
    Performs a robust comparison between TDT and PRISM dataframes and
    returns a dictionary of results.
    """
    all_mismatches = []
    all_matches = []
    summary_data = []
    all_entries_dfs = []  # To store each model's merged df

    # 1. Rename and clean PRISM columns
    prism_df.rename(columns={
        "FORM NAME": "TDT",
        "METRIC NAME": "METRIC_NAME",
        "POINT TYPE": "POINT_TYPE_PRISM"
    }, inplace=True)
    if 'METRIC_NAME' in prism_df.columns:
        prism_df['METRIC_NAME'] = prism_df['METRIC_NAME'].str.replace('AP-TVI-', '', regex=False)
    if 'POINT_TYPE_PRISM' in prism_df.columns:
        prism_df['POINT_TYPE_PRISM'] = prism_df['POINT_TYPE_PRISM'].str.title().str.replace('Prism Calc', 'PRiSM Calc', regex=False)

    # 2. Iterate through each TDT and compare
    for tdt_name, excel_df in tdt_dfs.items():
        prism_sub_df = prism_df[prism_df["TDT"] == tdt_name].copy()

        merged_df = pd.merge(
            excel_df.drop_duplicates(subset=['METRIC_NAME']),
            prism_sub_df.drop_duplicates(subset=['METRIC_NAME']),
            on='METRIC_NAME',
            how='outer',
            suffixes=('_TDT', '_PRISM'),
            indicator=True
        )

        # Store the full merged df for the "All Entries" table
        merged_with_tdt = merged_df.copy()
        merged_with_tdt['TDT'] = tdt_name
        all_entries_dfs.append(merged_with_tdt)

        # --- Identify Discrepancies using Vectorized Operations ---
        data_mismatch = (merged_df['_merge'] == 'both') & (merged_df['POINT_TYPE_TDT'] != merged_df['POINT_TYPE_PRISM'])
        missing_in_prism = merged_df['_merge'] == 'left_only'
        missing_in_tdt = merged_df['_merge'] == 'right_only'
        mismatch_rows = merged_df[data_mismatch | missing_in_prism | missing_in_tdt].copy()
        
        # --- Identify Matches ---
        match_rows = merged_df[(merged_df['_merge'] == 'both') & ~data_mismatch].copy()

        # --- Collect Summary Data for this TDT ---
        summary_data.append({
            "TDT": tdt_name,
            "Match Count": len(match_rows),
            "Mismatch Count": len(mismatch_rows),
            "Total TDT Records": len(excel_df.drop_duplicates(subset=['METRIC_NAME']))
        })

        # --- Process Mismatches for Detailed Table ---
        if not mismatch_rows.empty:
            mismatch_rows['Status'] = 'Point Type Mismatch'
            mismatch_rows.loc[missing_in_prism, 'Status'] = 'Missing in PRISM'
            mismatch_rows.loc[missing_in_tdt, 'Status'] = 'Missing in TDT (Excel)'
            mismatch_rows['TDT'] = tdt_name
            all_mismatches.append(mismatch_rows)

        # --- Process Matches for Detailed Table ---
        if not match_rows.empty:
            match_rows['TDT'] = tdt_name
            all_matches.append(match_rows)

    # 3. Create final DataFrames
    summary_df = pd.DataFrame(summary_data)
    all_entries_df = pd.concat(all_entries_dfs, ignore_index=True) if all_entries_dfs else pd.DataFrame()

    mismatches_df = pd.concat(all_mismatches, ignore_index=True) if all_mismatches else pd.DataFrame()
    if not mismatches_df.empty:
        mismatches_df = mismatches_df[[
            'TDT', 'METRIC_NAME', 'Status', 
            'POINT_TYPE_TDT', 'POINT_TYPE_PRISM'
        ]].fillna('N/A')

    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    if not matches_df.empty:
        matches_df = matches_df[[
            'TDT', 'METRIC_NAME', 'POINT_TYPE_TDT'
        ]].rename(columns={'POINT_TYPE_TDT': 'POINT_TYPE'})
    
    return {
        "summary": summary_df,
        "matches": matches_df,
        "mismatches": mismatches_df,
        "all_entries": all_entries_df
    }