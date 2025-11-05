import pandas as pd
import streamlit as st

@st.cache_data
def validate_point_survey(survey_df: pd.DataFrame) -> dict:
    """
    Validates the 'Consolidated Point Survey' DataFrame for internal consistency,
    specifically checking for duplicates in key identifier columns.
    
    Checks for duplicates (within the same TDT/Model) in:
    - Metric
    - KKS Point Name
    - DCS Description
    - Canary Point Name
    - Canary Description
    
    This validation EXCLUDES metrics where 'Point Type' is 'PRiSM Calc'.
    
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

    # Columns to check for duplicates
    columns_to_check = [
        'Metric', 
        'KKS Point Name', 
        'DCS Description', 
        'Canary Point Name', 
        'Canary Description'
    ]
    
    # 2. Find all duplicate indices and map them to an issue string
    issue_map = {}
    for col in columns_to_check:
        if col in details_df.columns:
            # Find duplicates based on TDT, Model, and the specific column
            duplicate_mask = details_df.duplicated(subset=['TDT', 'Model', col], keep=False) & details_df[col].notna()
            indices = details_df[duplicate_mask].index
            
            # Map the issue to the original index
            for idx in indices:
                issue_map.setdefault(idx, []).append(f"Duplicate {col}")

    # 3. Create the 'Issue' column
    # Map the dictionary of issue lists to the dataframe's index
    issue_series = pd.Series(issue_map).map(', '.join)
    details_df['Issue'] = details_df.index.map(issue_series).fillna("✅")
    
    # 4. Create the summary table
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