import pandas as pd
import streamlit as st

@st.cache_data
def validate_data(model_dfs, prism_df):
    """
    Compares TDT and PRISM data for metric mapping configurations.

    This function performs the core validation for metric mappings by comparing
    parsed TDT data against PRISM database data for each model. It joins the
    data on 'METRIC_NAME' to align records from both sources.

    The comparison logic includes special handling for 'PRiSM Calc' point types,
    where certain mismatches are ignored because the calculation is defined
    within PRISM itself.

    The validation identifies:
    - **Matches:** Records where all compared columns are identical. The 'PRiSM
      Calc' type in the TDT will override most checks, except for 'POINT_UNIT'.
    - **Mismatches:** Records in both sources with differing values, categorized
      by the specific column that has the mismatch.
    - **Missing Records:** Records present in one source but not the other.

    The function is cached with `@st.cache_data` for performance.

    Args:
        model_dfs (dict[str, pd.DataFrame]): A dictionary where keys are model
            names and values are DataFrames of parsed metric mapping data from the TDT.
        prism_df (pd.DataFrame): A DataFrame containing the corresponding
            metric mapping data queried from the PRISM database.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]: A tuple
        containing three elements:
        1. summary_df (pd.DataFrame): A DataFrame summarizing results per model,
           including match and mismatch counts.
        2. matches_df (pd.DataFrame): A DataFrame of all records that matched.
        3. mismatches_dict (dict[str, pd.DataFrame]): A dictionary where keys
           are mismatch types (e.g., 'POINT_NAME', 'Missing_in_PRISM') and
           values are DataFrames of the corresponding mismatched records.
    """
    # Create a copy to avoid modifying the cached dataframe.
    prism_df = prism_df.copy()

    all_matches = []
    summary_data = []
    
    # 1. Rename PRISM columns for consistency first.
    prism_df.rename(columns={
        "FORM NAME": 'MODEL',
        "METRIC NAME": "METRIC_NAME",
        "POINT NAME": "POINT_NAME",
        "POINT DESCRIPTION": "POINT_DESCRIPTION",
        "FUNCTION": "FUNCTION",
        "POINT TYPE": "POINT_TYPE",
        "POINT UNIT": "POINT_UNIT"
    }, inplace=True)

    # 2. Apply data transformations on the newly renamed columns.
    if 'METRIC_NAME' in prism_df.columns:
        prism_df['METRIC_NAME'] = prism_df['METRIC_NAME'].str.replace('AP-TVI-', '', regex=False)
    
    if 'POINT_TYPE' in prism_df.columns:
        prism_df['POINT_TYPE'] = prism_df['POINT_TYPE'].str.title().str.replace('Prism Calc', 'PRiSM Calc', regex=False)
    
    if 'FUNCTION' in prism_df.columns:
        prism_df['FUNCTION'] = prism_df['FUNCTION'].str.title().str.replace('Non-Modeled', 'Not Modeled', regex=False)


    columns_to_compare = ['POINT_NAME', 'POINT_DESCRIPTION', 'FUNCTION', 'POINT_TYPE', 'POINT_UNIT']
    
    # Dictionary to hold lists of mismatch dataframes
    mismatches_by_column = {col: [] for col in columns_to_compare}
    mismatches_by_column['Missing_in_PRISM'] = []
    mismatches_by_column['Missing_in_TDT'] = []

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

        # --- Identify Perfect Matches ---
        match_mask = (merged_df['_merge'] == 'both')
        for col in columns_to_compare:
            # Condition 1: The values in the column match (case-insensitive).
            values_match = (merged_df[f"{col}_TDT"].astype(str).str.upper() == merged_df[f"{col}_PRISM"].astype(str).str.upper())
            
            # Condition 2: The TDT point type is 'PRiSM Calc', which generally overrides the check for a match.
            is_prism_calc_override = (merged_df['POINT_TYPE_TDT'] == 'PRiSM Calc')

            # EXCEPTION: The override does not apply if the current column is 'POINT_UNIT'.
            # 'POINT_UNIT' must always match, regardless of the point type.
            if col == 'POINT_UNIT':
                condition = values_match
            else:
                condition = values_match | is_prism_calc_override
            
            match_mask &= condition
        
        match_rows = merged_df[match_mask].copy()
        if not match_rows.empty:
            match_rows['MODEL'] = model_name
            all_matches.append(match_rows)

        # --- Identify Mismatches by Specific Column ---
        for col in columns_to_compare:
            # A base mismatch occurs if values are different (case-insensitive).
            values_mismatch = (merged_df[f"{col}_TDT"].astype(str).str.upper() != merged_df[f"{col}_PRISM"].astype(str).str.upper())
            
            col_mismatch_mask = (merged_df['_merge'] == 'both') & values_mismatch
            
            # EXCEPTION: For columns other than 'POINT_UNIT', a mismatch is ignored if the TDT point type is 'PRiSM Calc'.
            # 'POINT_UNIT' mismatches are always flagged.
            if col != 'POINT_UNIT':
                col_mismatch_mask &= (merged_df['POINT_TYPE_TDT'] != 'PRiSM Calc')
            
            if col_mismatch_mask.any():
                mismatch_subset = merged_df.loc[col_mismatch_mask, ['METRIC_NAME', f'{col}_TDT', f'{col}_PRISM']].copy()
                mismatch_subset.rename(columns={f'{col}_TDT': 'TDT_Value', f'{col}_PRISM': 'PRISM_Value'}, inplace=True)
                mismatch_subset['MODEL'] = model_name
                # Reorder columns
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

    # --- Create Final DataFrames and Dictionary ---
    summary_df = pd.DataFrame(summary_data)
    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    
    if not matches_df.empty:
        new_column_order = ['MODEL', 'METRIC_NAME']
        for col in columns_to_compare:
            new_column_order.append(f"{col}_TDT")
            new_column_order.append(f"{col}_PRISM")
        final_columns = [col for col in new_column_order if col in matches_df.columns]
        matches_df = matches_df[final_columns]

    final_mismatches_dict = {}
    for mismatch_type, df_list in mismatches_by_column.items():
        if df_list:
            final_mismatches_dict[mismatch_type] = pd.concat(df_list, ignore_index=True)
        else:
            final_mismatches_dict[mismatch_type] = pd.DataFrame()

    # Reorder columns for each type of mismatch
    final_mismatches_dict = {}
    for mismatch_type, df_list in mismatches_by_column.items():
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            # Define the ideal column order for each mismatch type
            col_order = ['MODEL', 'METRIC_NAME']    
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