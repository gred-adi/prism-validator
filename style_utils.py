"""
This module contains helper functions for styling Pandas DataFrames in Streamlit.
"""

import pandas as pd

def highlight_diff(data, color='background-color: #FFCCCB'):
    """Highlights cells in PRISM columns that do not match their TDT counterpart.

    This function iterates through the columns of a DataFrame, identifying pairs
    of columns ending in '_TDT' and '_PRISM'. It compares the values in these
    pairs and applies a specified background color to the '_PRISM' cell if its
    value does not match the '_TDT' cell's value.

    Args:
        data (pd.DataFrame): The DataFrame to be styled. It is expected to
                             contain columns with '_TDT' and '_PRISM' suffixes.
        color (str, optional): The CSS style to apply to mismatched cells.
                               Defaults to a light red background.

    Returns:
        pd.DataFrame: A DataFrame of the same shape as `data`, containing the
        CSS styles to be applied to each cell.
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
    """Applies a style to an entire row based on the 'Issue' column.

    This function is intended to be used with `DataFrame.style.apply` along
    `axis=1`. It checks if a row contains an 'Issue' column and if the value
    in that column is not the success symbol ('✅'). If these conditions are
    met, it returns a list of style strings to apply a background color to
    every cell in that row.

    Args:
        row (pd.Series): A row of a DataFrame.

    Returns:
        list[str]: A list of CSS style strings, one for each cell in the row.
    """
    style = 'background-color: #FFCCCB' if (pd.notna(row.get('Issue')) and row.get('Issue') != '✅') else ''
    return [style] * len(row)

def highlight_issue_cells(row, issue_to_col_map, current_issue):
    """Applies a style to specific cells in a row based on an issue mapping.

    This function highlights one or more cells in a row if the `current_issue`
    matches a key in the `issue_to_col_map`. The map determines which columns
    should be highlighted for a given issue.

    Args:
        row (pd.Series): The row of the DataFrame being styled.
        issue_to_col_map (dict): A dictionary mapping issue strings to the
                                 column name (or list of column names) to be
                                 highlighted.
        current_issue (str): The specific issue to check for in this styling
                             application.

    Returns:
        list[str]: A list of CSS style strings for each cell in the row.
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
