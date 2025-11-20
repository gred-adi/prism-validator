"""
This module contains helper functions for styling Pandas DataFrames in Streamlit.
"""

import pandas as pd

def highlight_diff(data, color='background-color: #FFCCCB'):
    """
    Highlights cells in PRISM columns that do not match their corresponding TDT column.
    Returns a DataFrame of styles.
    """
    attr = f'{color}'
    # Create a new DataFrame of the same shape as `data` to store styles, initialized with empty strings
    style_df = pd.DataFrame('', index=data.index, columns=data.columns)

    # Find pairs of TDT/PRISM columns to compare
    prism_cols = [c for c in data.columns if c.endswith('_PRISM')]

    for p_col in prism_cols:
        t_col = p_col.replace('_PRISM', '_TDT')
        if t_col in data.columns:
            # Using .astype(str) for robust comparison across dtypes and NaNs
            is_mismatch = data[p_col].astype(str) != data[t_col].astype(str)
            # Apply the style attribute to the PRISM column where there is a mismatch
            style_df.loc[is_mismatch, p_col] = attr

    return style_df

def highlight_issue_rows(row):
    """
    Applies a style to the entire row if the 'Issue' column is not '✅'.
    """
    style = 'background-color: #FFCCCB' if (pd.notna(row.get('Issue')) and row.get('Issue') != '✅') else ''
    return [style] * len(row)

def highlight_issue_cells(row, issue_to_col_map, current_issue):
    """
    Applies a style to cells that are flagged as duplicates based on the 'Issue' column.
    """
    styles = [''] * len(row)

    # Get the column(s) to highlight for this specific issue
    cols_to_highlight = issue_to_col_map.get(current_issue)

    if cols_to_highlight:
        # col_names can be a single string or a list of strings
        if isinstance(cols_to_highlight, str):
            cols_to_highlight = [cols_to_highlight]

        for col_name in cols_to_highlight:
            try:
                col_index = list(row.index).index(col_name)
                styles[col_index] = 'background-color: #FFCCCB' # Light red
            except ValueError:
                pass # Column not in the final view, so we skip

    return styles
