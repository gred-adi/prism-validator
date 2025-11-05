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
        and 'Fault Detection' against 'Constraint' and 'diag_df' usage.
    2.  'filter_audit': A simple report listing all metrics with active filters.
    """
    if survey_df is None or survey_df.empty:
        return {}

    # 1. Deduplicate survey_df by TDT and Metric
    # TDT/Metric is the key for attributes and function logic
    deduped_df = survey_df.drop_duplicates(subset=['TDT', 'Metric']).copy()

    # 2. Get diagnostic counts (Validation 3 Prep)
    # We use the raw diag_df from the session state
    diag_counts = pd.DataFrame()
    if diag_df is not None and not diag_df.empty and 'Metric' in diag_df.columns:
        diag_counts = diag_df.groupby(['TDT', 'Metric']).size().to_frame('Diag_Count').reset_index()
    
    # 3. Merge diag_counts into the deduplicated survey data
    if not diag_counts.empty:
        details_df = pd.merge(deduped_df, diag_counts, on=['TDT', 'Metric'], how='left')
    else:
        details_df = deduped_df.copy()
        details_df['Diag_Count'] = 0 # Ensure column exists even if diag_df is empty

    details_df['Diag_Count'] = details_df['Diag_Count'].fillna(0).astype(int)

    # 4. Run Validation Rules to find issues
    
    # Rule 1: Operational State must be constrained
    op_state_issue = (details_df['Function'] == 'Operational State') & (details_df['Constraint'] != 'Yes')
    
    # Rule 2: 'Not Modeled' should not be in diagnostics
    not_modeled_issue = (details_df['Function'] == 'Not Modeled') & (details_df['Diag_Count'] > 0)
    
    # Rule 3: Modeled points ('Op State' or 'Fault Detection') should be in diagnostics
    modeled_issue = (details_df['Function'].isin(['Operational State', 'Fault Detection'])) & (details_df['Diag_Count'] == 0)

    # 5. Combine issues into a single "Issue" column
    # We create a list of pd.Series, which we'll concat and join
    issues_list = []
    issues_list.append(pd.Series(np.where(op_state_issue, "Op. State not constrained", pd.NA), index=details_df.index))
    issues_list.append(pd.Series(np.where(not_modeled_issue, "'Not Modeled' in Diag.", pd.NA), index=details_df.index))
    issues_list.append(pd.Series(np.where(modeled_issue, "'Modeled' not in Diag.", pd.NA), index=details_df.index))

    # Concatenate all issue series horizontally
    issue_df = pd.concat(issues_list, axis=1)
    
    # Join the non-null issues with a comma
    details_df['Issue'] = issue_df.apply(lambda x: ', '.join(x.dropna()), axis=1)
    
    # Any row with no issues becomes 'OK'
    details_df['Issue'] = details_df['Issue'].replace('', 'OK')

    # 6. Create the Function Validation Summary
    function_summary = (
        details_df.groupby(['TDT', 'Issue'])
        .size()
        .to_frame('Count')
        .reset_index()
    )
    
    # 7. Create the Filter Audit report
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
            "details": details_df
        },
        "filter_audit": {
            "summary": filter_summary,
            "details": filter_df
        }
    }