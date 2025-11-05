import pandas as pd
import streamlit as st
import numpy as np

@st.cache_data
def validate_attribute(survey_df: pd.DataFrame, diag_df: pd.DataFrame) -> dict:
    """
    Validates 'Attribute' and 'Function' related columns from the survey DataFrame.
    It deduplicates all points by TDT/Metric first.

    Returns a dictionary with two reports:
    1.  'function_validation': Checks logic for 'Operational State', 'Not Modeled',
        'Fault Detection', and 'Filter Info Incomplete'. Returns all rows.
    2.  'filter_audit': A simple report listing all metrics with active (complete) filters.
    """
    if survey_df is None or survey_df.empty:
        return {}

    # 1. Deduplicate survey_df by TDT and Metric
    deduped_df = survey_df.drop_duplicates(subset=['TDT', 'Metric']).copy()

    # 2. Get diagnostic counts
    diag_counts = pd.DataFrame()
    if diag_df is not None and not diag_df.empty and 'Metric' in diag_df.columns:
        diag_counts = diag_df.groupby(['TDT', 'Metric']).size().to_frame('Diag_Count').reset_index()
    
    # 3. Merge diag_counts into the deduplicated survey data
    if not diag_counts.empty:
        details_df = pd.merge(deduped_df, diag_counts, on=['TDT', 'Metric'], how='left')
    else:
        details_df = deduped_df.copy()
        details_df['Diag_Count'] = 0 # Ensure column exists

    details_df['Diag_Count'] = details_df['Diag_Count'].fillna(0).astype(int)
    
    # --- Ensure Filter columns exist before checking them ---
    if 'Filter Condition' not in details_df.columns:
        details_df['Filter Condition'] = np.nan
    if 'Filter Value' not in details_df.columns:
        details_df['Filter Value'] = np.nan

    # 4. Run Validation Rules to find issues
    
    # Rule 1: Operational State must be constrained
    op_state_issue = (details_df['Function'] == 'Operational State') & (details_df['Constraint'] != 'Yes')
    
    # Rule 2: 'Not Modeled' should not be in diagnostics
    not_modeled_issue = (details_df['Function'] == 'Not Modeled') & (details_df['Diag_Count'] > 0)
    
    # Rule 3: Modeled points ('Op State' or 'Fault Detection') should be in diagnostics
    modeled_issue = (details_df['Function'].isin(['Operational State', 'Fault Detection'])) & (details_df['Diag_Count'] == 0)

    # --- Rule 4: Incomplete Filter Info (XOR logic) ---
    cond_notnull = details_df['Filter Condition'].notnull()
    val_notnull = details_df['Filter Value'].notnull()
    filter_incomplete_issue = (cond_notnull & ~val_notnull) | (~cond_notnull & val_notnull)

    # 5. Combine issues into a single "Issue" column
    issues_list = []
    issues_list.append(pd.Series(np.where(op_state_issue, "Op. State not constrained", pd.NA), index=details_df.index))
    issues_list.append(pd.Series(np.where(not_modeled_issue, "'Not Modeled' in Diagnostics", pd.NA), index=details_df.index))
    issues_list.append(pd.Series(np.where(modeled_issue, "'Modeled' not in Diagnostics", pd.NA), index=details_df.index))
    issues_list.append(pd.Series(np.where(filter_incomplete_issue, "Filter Info Incomplete", pd.NA), index=details_df.index))

    issue_df = pd.concat(issues_list, axis=1)
    
    details_df['Issue'] = issue_df.apply(lambda x: ', '.join(x.dropna()), axis=1)
    
    # --- MODIFICATION: Set blank issues to '✅' ---
    details_df['Issue'] = details_df['Issue'].replace('', '✅')
    function_details_df = details_df # Use all rows

    # 6. Create the Function Validation Summary
    function_summary = (
        function_details_df.groupby(['TDT', 'Issue'])
        .size()
        .to_frame('Count')
        .reset_index()
    )
    
    # 7. Create the Filter Audit report (unchanged)
    filter_df = details_df[
        details_df['Filter Condition'].notnull() & details_df['Filter Value'].notnull()
    ].copy()
    
    filter_summary = (
        filter_df.groupby('TDT')
        .size()
        .to_frame('Filter_Count')
        .reset_index()
    )
    
    # 8. Return the final dictionary of reports
    return {
        "function_validation": {
            "summary": function_summary,
            "details": function_details_df
        },
        "filter_audit": {
            "summary": filter_summary,
            "details": filter_df
        }
    }