import pandas as pd
import streamlit as st

@st.cache_data
def validate_data(tdt_dfs, prism_df):
    """
    Performs the comparison logic for the 'Metric Mapping' section.
    Compares POINT NAME, POINT DESCRIPTION, FUNCTION, and POINT TYPE.
    """
    all_mismatches = []
    all_matches = []
    summary_data = []

    # Rename PRISM columns to be consistent
    prism_df.rename(columns={
        "FORM NAME": "TDT",
        "METRIC NAME": "METRIC_NAME",
        "POINT NAME": "POINT_NAME",
        "POINT DESCRIPTION": "POINT_DESCRIPTION",
        "FUNCTION": "FUNCTION",
        "POINT TYPE": "POINT_TYPE"
    }, inplace=True)

    # Clean the PRISM METRIC_NAME by removing the specified prefix
    if 'METRIC_NAME' in prism_df.columns:
        prism_df['METRIC_NAME'] = prism_df['METRIC_NAME'].str.replace('AP-TVI-', '', regex=False)
    
    if 'POINT_TYPE' in prism_df.columns:
        # Note: .str.replace is used on the Series for consistency
        prism_df['POINT_TYPE'] = prism_df['POINT_TYPE'].str.title().str.replace('Prism Calc', 'PRiSM Calc', regex=False)
    
    if 'FUNCTION' in prism_df.columns:
        # Note: .str.replace is used on the Series for consistency
        prism_df['FUNCTION'] = prism_df['FUNCTION'].str.title().str.replace('NON-MODELED', 'Not Modeled', regex=False)

    # Columns to compare between TDT (Excel) and PRISM (SQL)
    columns_to_compare = ['POINT_NAME', 'POINT_DESCRIPTION', 'FUNCTION', 'POINT_TYPE']

    for tdt_name, excel_df in tdt_dfs.items():
        prism_sub_df = prism_df[prism_df["TDT"] == tdt_name].copy()

        # Merge dataframes on the metric name
        merged_df = pd.merge(
            excel_df.drop_duplicates(subset=['METRIC_NAME']),
            prism_sub_df.drop_duplicates(subset=['METRIC_NAME']),
            on='METRIC_NAME',
            how='outer',
            suffixes=('_TDT', '_PRISM'),
            indicator=True
        )

        # --- Identify Discrepancies ---
        mismatch_conditions = []
        for col in columns_to_compare:
            col_tdt = f"{col}_TDT"
            col_prism = f"{col}_PRISM"
            mismatch_conditions.append(
                merged_df[col_tdt].astype(str).str.upper() != merged_df[col_prism].astype(str).str.upper()
            )

        data_mismatch = (merged_df['_merge'] == 'both') & (pd.concat(mismatch_conditions, axis=1).any(axis=1))
        missing_in_prism = merged_df['_merge'] == 'left_only'
        missing_in_tdt = merged_df['_merge'] == 'right_only'
        
        mismatch_rows = merged_df[data_mismatch | missing_in_prism | missing_in_tdt].copy()
        match_rows = merged_df[(merged_df['_merge'] == 'both') & ~data_mismatch].copy()

        # --- Collect Summary Data ---
        summary_data.append({
            "MODEL": tdt_name, "Match Count": len(match_rows), "Mismatch Count": len(mismatch_rows),
            "Total Metric Records": len(excel_df.drop_duplicates(subset=['METRIC_NAME']))
        })

        # --- Process Detailed Mismatch Data ---
        if not mismatch_rows.empty:
            mismatch_rows['Status'] = 'Data Mismatch'
            mismatch_rows.loc[missing_in_prism, 'Status'] = 'Missing in PRISM'
            mismatch_rows.loc[missing_in_tdt, 'Status'] = 'Missing in TDT (Excel)'
            mismatch_rows['TDT'] = tdt_name
            all_mismatches.append(mismatch_rows)

        # --- Process Detailed Match Data ---
        if not match_rows.empty:
            match_rows['TDT'] = tdt_name
            all_matches.append(match_rows)

    # --- Create Final DataFrames ---
    summary_df = pd.DataFrame(summary_data)
    mismatches_df = pd.concat(all_mismatches, ignore_index=True) if all_mismatches else pd.DataFrame()
    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

    return summary_df, matches_df, mismatches_df
