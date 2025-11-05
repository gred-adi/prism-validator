import pandas as pd
import streamlit as st
import numpy as np

@st.cache_data
def validate_point_survey(survey_df: pd.DataFrame) -> dict:
    """
    Validates the 'Consolidated Point Survey' DataFrame for internal consistency.
    
    Checks (for non-PRiSM Calc points):
    1.  Duplicates: 'Metric', 'KKS Point Name', etc.
    2.  Blanks: 'Point Type', 'KKS Point Name', 'Unit', etc.
    
    Returns:
        A dictionary containing a summary DataFrame (with '✅' for OK) and a 
        details DataFrame (with *all* non-PRiSM Calc points).
    """
    if survey_df is None or survey_df.empty:
        return {
            "summary": pd.DataFrame(),
            "details": pd.DataFrame()
        }

    # 1. Filter for all points *except* PRiSM Calc
    details_df = survey_df[survey_df['Point Type'] != 'PRiSM Calc'].copy()
    if details_df.empty:
        return {"summary": pd.DataFrame(), "details": pd.DataFrame()}

    # 2. Initialize issue map
    issue_map = {}

    # 3. --- Run Duplicate Checks ---
    cols_for_duplicate_check = [
        'Metric', 
        'KKS Point Name', 
        'DCS Description', 
        'Canary Point Name', 
        'Canary Description'
    ]
    
    for col in cols_for_duplicate_check:
        if col in details_df.columns:
            # Find duplicates based on TDT, Model, and the specific column
            duplicate_mask = details_df.duplicated(subset=['TDT', 'Model', col], keep=False) & details_df[col].notna()
            indices = details_df[duplicate_mask].index
            
            # Map the issue to the original index
            for idx in indices:
                issue_map.setdefault(idx, []).append(f"Duplicate {col}")

    # 4. --- Run Blank Field Checks ---
    cols_for_blank_check = [
        'Point Type', 
        'KKS Point Name', 
        'DCS Description', 
        'Canary Point Name', 
        'Canary Description', 
        'Unit'
    ]
    
    for col in cols_for_blank_check:
        if col in details_df.columns:
            # Find rows where the column is null, NaN, or an empty string
            blank_mask = pd.isna(details_df[col]) | (details_df[col].astype(str).str.strip() == '')
            indices = details_df[blank_mask].index
            for idx in indices:
                issue_map.setdefault(idx, []).append(f"Blank {col}")
        else:
            # If the column doesn't even exist, flag all rows
            indices = details_df.index
            for idx in indices:
                issue_map.setdefault(idx, []).append(f"Missing Col: {col}")


    # 5. Create the 'Issue' column
    # Map the dictionary of issue lists to the dataframe's index
    issue_series = pd.Series(issue_map).map(', '.join)
    details_df['Issue'] = details_df.index.map(issue_series).fillna("✅")
    
    # 6. Create the summary table
    summary_df = (
        details_df.groupby(['TDT', 'Model', 'Issue'])
        .size()
        .to_frame('Count')
        .reset_index()
    )

    return {
        "summary": summary_df,
        "details": details_df
    }