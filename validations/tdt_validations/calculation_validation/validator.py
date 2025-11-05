import pandas as pd
import streamlit as st
import numpy as np

@st.cache_data
def validate_calculation(survey_df: pd.DataFrame) -> dict:
    """
    Validates 'Calculation' related columns for metrics where 'Point Type' is 'PRiSM Calc'.
    This check is grouped by TDT, as calc points are identical across models in a TDT.
    
    Checks:
    - Finds all 'PRiSM Calc' points (deduplicated by TDT/Metric).
    - Tags them as "✅" or "Missing all calculation details".
      
    Returns:
        A dictionary containing a summary DataFrame (of statuses) and a
        details DataFrame (of all unique PRiSM Calc points).
    """
    if survey_df is None or survey_df.empty:
        return {"summary": pd.DataFrame(), "details": pd.DataFrame()}

    # 1. Filter for only 'PRiSM Calc' rows
    prism_calc_df = survey_df[survey_df['Point Type'] == 'PRiSM Calc'].copy()

    if prism_calc_df.empty:
        # Return empty frames, but the UI will catch this
        return {"summary": pd.DataFrame(), "details": pd.DataFrame()}

    # 2. Define the columns that should NOT all be blank
    calc_cols = [
        'Calc Point Type',
        'Calculation Description',
        'Pseudo Code',
        'Language',
        'Input Point',
        'PRiSM Code'
    ]
    
    # 3. Ensure all columns exist, fill missing ones with NaN for the check
    for col in calc_cols:
        if col not in prism_calc_df.columns:
            prism_calc_df[col] = np.nan
            
    # 4. Deduplicate by TDT and Metric, keeping the first instance's data
    deduped_calc_df = prism_calc_df.drop_duplicates(subset=['TDT', 'Metric']).copy()

    if deduped_calc_df.empty:
        # Should not happen if prism_calc_df was not empty, but good check
        return {"summary": pd.DataFrame(), "details": pd.DataFrame()}

    # 5. Find rows where ALL of these columns are null/NaN
    all_blank_mask = deduped_calc_df[calc_cols].isnull().all(axis=1)
    
    # 6. Create the 'Issue' column based on the mask
    deduped_calc_df['Issue'] = np.where(all_blank_mask, "Missing all calculation details", "✅")
    
    # 7. The 'details' df is *all* unique PRiSM Calc points, now with an Issue status
    details_df = deduped_calc_df
    
    # 8. Create Summary based on the 'Issue' status, grouped by TDT only
    summary_df = (
        details_df.groupby(['TDT', 'Issue'])
        .size()
        .to_frame('Count')
        .reset_index()
    )

    return {
        "summary": summary_df,
        "details": details_df
    }