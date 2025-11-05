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
        A dictionary containing a summary DataFrame and a details DataFrame.
    """
    if survey_df is None or survey_df.empty:
        return {
            "summary": pd.DataFrame(columns=["TDT", "Model", "Duplicate Type", "Count"]),
            "details": pd.DataFrame()
        }

    # --- NEW: Filter out 'PRiSM Calc' points before validation ---
    validation_df = survey_df[survey_df['Point Type'] != 'PRiSM Calc'].copy()

    # Columns to check for duplicates
    # We ignore 'nan' or None values, as those aren't "duplicates"
    columns_to_check = [
        'Metric', 
        'KKS Point Name', 
        'DCS Description', 
        'Canary Point Name', 
        'Canary Description'
    ]
    
    all_duplicates_dfs = []

    for col in columns_to_check:
        if col in validation_df.columns:
            # Find duplicates based on TDT, Model, and the specific column
            # Use 'validation_df' (the filtered df) for the check
            duplicates = validation_df[validation_df.duplicated(subset=['TDT', 'Model', col], keep=False)].dropna(subset=[col]).copy()
            
            if not duplicates.empty:
                duplicates['Issue'] = f"Duplicate {col}"
                all_duplicates_dfs.append(duplicates)

    if not all_duplicates_dfs:
        # No duplicates found at all
        return {
            "summary": pd.DataFrame(columns=["TDT", "Model", "Issue", "Count"]),
            "details": pd.DataFrame()
        }

    # Combine all found duplicates
    details_df = pd.concat(all_duplicates_dfs, ignore_index=True)
    
    # A single row might be a duplicate for multiple reasons (e.g., same Metric AND same Canary Point Name)
    # We group by the row's original index to consolidate issues, then drop duplicates
    details_df = (
        details_df.groupby(level=0)
        .agg({
            **{col: 'first' for col in validation_df.columns if col in details_df.columns},
            'Issue': lambda x: ', '.join(x.unique())
        })
        .reset_index(drop=True)
        .drop_duplicates()
        .sort_values(by=['TDT', 'Model', 'Issue'] + columns_to_check)
    )

    # Create the summary table
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