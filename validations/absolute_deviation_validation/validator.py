import pandas as pd
import streamlit as st

@st.cache_data
def validate_data(model_dfs, prism_df):
    """
    Compares TDT and PRISM data for absolute deviation thresholds.

    This function executes the core validation logic by comparing the parsed
    Excel data (TDT) against the data queried from the PRISM database. It
    iterates through each model, performing a detailed comparison on key metrics.

    The comparison identifies:
    - **Perfect Matches:** Records where all threshold values are identical.
    - **Column-Specific Mismatches:** Records that exist in both sources but
      have differing values for one or more threshold columns.
    - **Missing Records:** Records present in one source but not the other.

    The function is cached with `@st.cache_data` to optimize performance by
    avoiding re-computation when inputs have not changed.

    Args:
        model_dfs (dict[str, pd.DataFrame]): A dictionary where keys are model
            names and values are DataFrames containing the parsed TDT data
            for absolute deviation thresholds.
        prism_df (pd.DataFrame): A DataFrame containing the corresponding
            threshold data queried from the PRISM database.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]: A tuple
        containing three elements:
        1. summary_df (pd.DataFrame): A DataFrame summarizing the validation
           results per model, including match and mismatch counts.
        2. matches_df (pd.DataFrame): A DataFrame containing all records that
           matched perfectly between the TDT and PRISM data.
        3. mismatches_dict (dict[str, pd.DataFrame]): A dictionary where each
           key is a type of mismatch (e.g., 'HIGH ALERT', 'Missing_in_PRISM')
           and the value is a DataFrame of the corresponding mismatched records.
    """
    prism_df = prism_df.copy()
    all_matches = []
    summary_data = []

    # 1. Rename and clean PRISM columns
    prism_df.rename(columns={
        "FORM NAME": "MODEL",
        "METRIC NAME": "METRIC_NAME",
        # Threshold columns match Excel, no rename needed if they are identical
    }, inplace=True)

    # Remove all rows where METRIC_NAME is NaN (null).
    prism_df = prism_df.dropna(subset=["METRIC_NAME"])

    # Dictionary to hold lists of mismatch dataframes
    columns_to_compare = ['HIGH ALERT', 'HIGH WARNING', 'LOW WARNING', 'LOW ALERT']
    mismatches_by_column = {col: [] for col in columns_to_compare}
    mismatches_by_column['Missing_in_PRISM'] = []
    mismatches_by_column['Missing_in_TDT'] = []
    
    join_keys = ['METRIC_NAME']

    for model_name, excel_df in model_dfs.items():
        prism_sub_df = prism_df[prism_df["MODEL"] == model_name].copy()

        merged_df = pd.merge(
            excel_df.drop_duplicates(subset=join_keys),
            prism_sub_df.drop_duplicates(subset=join_keys),
            on=join_keys,
            how='outer',
            suffixes=('_TDT', '_PRISM'),
            indicator=True
        )

        # --- Identify Perfect Matches ---
        match_mask = (merged_df['_merge'] == 'both')
        for col in columns_to_compare:
            # Compare as numbers, handling NaNs
            match_mask &= (pd.to_numeric(merged_df[f'{col}_TDT'], errors='coerce') == pd.to_numeric(merged_df[f'{col}_PRISM'], errors='coerce'))
        
        match_rows = merged_df[match_mask].copy()
        if not match_rows.empty:
            match_rows['MODEL'] = model_name
            all_matches.append(match_rows)

        # --- Identify Mismatches by Specific Column ---
        for col in columns_to_compare:
            col_mismatch_mask = (merged_df['_merge'] == 'both') & \
                                (pd.to_numeric(merged_df[f"{col}_TDT"], errors='coerce') != pd.to_numeric(merged_df[f"{col}_PRISM"], errors='coerce'))
            
            if col_mismatch_mask.any():
                mismatch_subset = merged_df.loc[col_mismatch_mask, ['METRIC_NAME', f'{col}_TDT', f'{col}_PRISM']].copy()
                mismatch_subset['MODEL'] = model_name
                mismatches_by_column[col].append(mismatch_subset)

        # --- Identify Records Missing from a Source ---
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
            "MODEL": model_name,
            "Match Count": len(match_rows),
            "Mismatch Count": total_mismatch_count,
            "Total TDT Records": len(excel_df.drop_duplicates(subset=join_keys))
        })

    # --- Create Final DataFrames and Dictionary ---
    summary_df = pd.DataFrame(summary_data)
    
    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    if not matches_df.empty:
        col_order = ['MODEL'] + join_keys
        for col in columns_to_compare: col_order.extend([f'{col}_TDT', f'{col}_PRISM'])
        matches_df = matches_df[[c for c in col_order if c in matches_df.columns]]

    final_mismatches_dict = {}
    for mismatch_type, df_list in mismatches_by_column.items():
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            col_order = ['MODEL'] + join_keys
            if mismatch_type in columns_to_compare:
                col_order.extend([f'{mismatch_type}_TDT', f'{mismatch_type}_PRISM'])
            else:
                tdt_cols = sorted([c for c in df.columns if '_TDT' in c])
                prism_cols = sorted([c for c in df.columns if '_PRISM' in c])
                col_order.extend(tdt_cols + prism_cols)
            final_mismatches_dict[mismatch_type] = df[[c for c in col_order if c in df.columns]]
        else:
            final_mismatches_dict[mismatch_type] = pd.DataFrame()

    return summary_df, matches_df, final_mismatches_dict