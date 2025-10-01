import pandas as pd

def validate_data(tdt_dfs, prism_df):
    """
    Compares TDT and PRISM data for metric configurations.

    This function performs a detailed comparison between metric data parsed from
    TDT files and the corresponding data queried from the PRISM database. It
    iterates through each TDT, using a pandas merge to identify matches,
    mismatches, and missing records based on the 'POINT_TYPE'.

    The comparison logic identifies:
    - **Matches:** Records where 'POINT_TYPE' is identical in both sources.
    - **Mismatches:** Records found in both sources but with a different
      'POINT_TYPE'.
    - **Missing Records:** Records found in one source but not the other.

    Args:
        tdt_dfs (dict[str, pd.DataFrame]): A dictionary where keys are TDT names
            and values are DataFrames of parsed metric data from the TDT files.
        prism_df (pd.DataFrame): A DataFrame containing the corresponding metric
            data queried from the PRISM database.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing three
        elements:
        1. summary_df (pd.DataFrame): A DataFrame summarizing the validation
           results per TDT, including match and mismatch counts.
        2. matches_df (pd.DataFrame): A DataFrame of all records that matched
           perfectly between the TDT and PRISM data.
        3. mismatches_df (pd.DataFrame): A single DataFrame containing all
           discrepancies, with a 'Status' column indicating the type of
           mismatch (e.g., 'Point Type Mismatch', 'Missing in PRISM').
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
        mismatches_df = pd.concat(all_mismatches, ignore_index=True) if all_mismatches else pd.DataFrame()
        # Clean up the final dataframe for display, focusing on the compared columns
        mismatches_df = mismatches_df[[
            'TDT', 'METRIC_NAME', 'Status', 
            'POINT_TYPE_TDT', 'POINT_TYPE_PRISM'
        ]].fillna('N/A')

    if not all_matches:
        matches_df = pd.DataFrame()
    else:
        matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
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

    This function is intended for future implementation of a PDF reporting
    feature. Currently, it serves as a placeholder and does not generate a
    report.

    Args:
        matches_df (pd.DataFrame): DataFrame containing the matched records.
        mismatches_df (pd.DataFrame): DataFrame containing the mismatched records.

    Returns:
        bytes: An empty byte string, as the function is not implemented.
    """
    # To prevent errors if this function were accidentally called,
    # it returns empty bytes. The UI button is disabled.
    print("NOTE: PDF generation is a placeholder and is not implemented.")
    return b""