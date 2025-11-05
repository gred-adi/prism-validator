import pandas as pd
import streamlit as st
import numpy as np

@st.cache_data
def validate_diagnostics(diag_df: pd.DataFrame) -> dict:
    """
    Validates the 'Consolidated Failure Diagnostic' DataFrame.
    
    Checks:
    1.  Reports the 'Enabled' status of the failure mode.
    2.  Counts metrics per TDT/Failure Mode.
    3.  Sums 'Weighting' per TDT/Failure Mode and checks if it equals 100.
    
    Returns:
        A dictionary containing:
        - 'summary': A DataFrame with counts, enabled status, and weight sums.
        - 'details': The raw diag_df for filtering.
    """
    if diag_df is None or diag_df.empty:
        return {"summary": pd.DataFrame(), "details": pd.DataFrame()}

    # Ensure 'Weighting' column is numeric, coercing errors
    diag_df['Weighting'] = pd.to_numeric(diag_df['Weighting'], errors='coerce').fillna(0)
    
    # Ensure 'Failure Mode Enabled' column exists
    if 'Failure Mode Enabled' not in diag_df.columns:
         diag_df['Failure Mode Enabled'] = 'N/A' # Add placeholder if missing

    # 1. Create the Summary table
    summary_df = (
        diag_df.groupby(['TDT', 'Failure Mode'])
        .agg(
            Enabled=('Failure Mode Enabled', 'first'), # Get the enabled status
            Metric_Count=('Metric', 'size'),
            Total_Weight=('Weighting', 'sum')
        )
        .reset_index()
    )

    # 2. Add the 'Issue' column for weight validation
    # A sum of 0 is common (if no weights are used), so we only flag non-100 sums.
    summary_df['Issue'] = np.where(
        (summary_df['Total_Weight'] != 100) & (summary_df['Total_Weight'] != 0),
        "Weight Sum != 100",
        "OK"
    )
    
    # Re-order columns for better presentation
    summary_df = summary_df[['TDT', 'Failure Mode', 'Enabled', 'Metric_Count', 'Total_Weight', 'Issue']]

    # The 'details' df is just the raw, cleaned dataframe for filtering
    return {
        "summary": summary_df,
        "details": diag_df
    }