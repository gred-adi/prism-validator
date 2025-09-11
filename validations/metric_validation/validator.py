import pandas as pd

def validate_data(tdt_dfs, prism_df):
    """
    Performs a robust comparison between TDT and PRISM dataframes.

    Returns three dataframes:
    1. summary_df: Counts of matches and mismatches per TDT.
    2. matches_df: Records that match perfectly.
    3. mismatches_df: Records with discrepancies or those missing from either source.
    """
    all_mismatches = []
    all_matches = []
    summary_data = [] # To store summary counts

    # Standardize PRISM column names to match Excel processing output
    prism_df.rename(columns={
        "FORM NAME": "TDT",
        "METRIC NAME": "METRIC_NAME",
        "POINT TYPE": "POINT_TYPE_PRISM"
    }, inplace=True)

    # Clean the PRISM METRIC_NAME by removing the specified prefix
    if 'METRIC_NAME' in prism_df.columns:
        prism_df['METRIC_NAME'] = prism_df['METRIC_NAME'].str.replace('AP-TVI-', '', regex=False)
    
    if 'POINT_TYPE_PRISM' in prism_df.columns:
        # Note: .str.replace is used on the Series for consistency
        prism_df['POINT_TYPE_PRISM'] = prism_df['POINT_TYPE_PRISM'].str.title().str.replace('Prism Calc', 'PRiSM Calc', regex=False)


    for tdt_name, excel_df in tdt_dfs.items():
        # Filter PRISM data for the current TDT
        prism_sub_df = prism_df[prism_df["TDT"] == tdt_name].copy()

        # Perform an outer merge to find all matches and differences.
        merged_df = pd.merge(
            excel_df.drop_duplicates(subset=['METRIC_NAME']),
            prism_sub_df.drop_duplicates(subset=['METRIC_NAME']),
            on='METRIC_NAME',
            how='outer',
            suffixes=('_TDT', '_PRISM'),
            indicator=True
        )

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
            mismatch_rows['TDT'] = tdt_name # Add TDT name for context
            all_mismatches.append(mismatch_rows)

        # --- Process Matches for Detailed Table ---
        if not match_rows.empty:
            match_rows['TDT'] = tdt_name
            all_matches.append(match_rows)


    # --- Create final DataFrames ---
    summary_df = pd.DataFrame(summary_data)

    if not all_mismatches:
        mismatches_df = pd.DataFrame()
    else:
        mismatches_df = pd.concat(all_mismatches, ignore_index=True)
        # Clean up the final dataframe for display, focusing on the compared columns
        mismatches_df = mismatches_df[[
            'TDT', 'METRIC_NAME', 'Status', 
            'POINT_TYPE_TDT', 'POINT_TYPE_PRISM'
        ]].fillna('N/A')

    if not all_matches:
        matches_df = pd.DataFrame()
    else:
        matches_df = pd.concat(all_matches, ignore_index=True)
         # Clean up the matches dataframe for display
        matches_df = matches_df[[
            'TDT', 'METRIC_NAME', 'POINT_TYPE_TDT'
        ]].rename(columns={
            'POINT_TYPE_TDT': 'POINT_TYPE'
        })
    
    return summary_df, matches_df, mismatches_df

# Placeholder for PDF report generation.
def generate_pdf_report(matches_df, mismatches_df):
    """
    (Placeholder) Generates a PDF report from the validation results.
    This functionality can be implemented in the future.
    """
    # To prevent errors if this function were accidentally called,
    # it returns empty bytes. The UI button is disabled.
    print("NOTE: PDF generation is a placeholder and is not implemented.")
    return b""