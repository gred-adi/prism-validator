import pandas as pd
import streamlit as st
import os

# --- Import the new validator ---
from validations.tdt_validations.point_survey_validation.validator import validate_point_survey
# (other imports will go here as you build them)
# from validations.tdt_validations.calculation_validation.validator import validate_calculation
# from validations.tdt_validations.attribute_validation.validator import validate_attribute
# from validations.tdt_validations.diagnostics_validation.validator import validate_diagnostics
# from validations.tdt_validations.prescriptive_validation.validator import validate_prescriptive

# --- Page Configuration ---
st.set_page_config(page_title="TDT Validator", layout="wide")

# --- Initialize Session State ---
if 'validation_states' not in st.session_state:
    st.session_state.validation_states = {} # Ensure main state exists
if "tdt_point_survey" not in st.session_state.validation_states:
    st.session_state.validation_states["tdt_point_survey"] = {"results": None}
# (other validation states)


# --- Helper function to highlight specific cells ---
def highlight_duplicate_cells(row):
    """
    Applies a style to cells that are flagged as duplicates based on the 'Issue' column.
    The 'Issue' column itself is not highlighted as it will be hidden.
    """
    # Mapping from issue string (in 'Issue' col) to the column name
    issue_to_col_map = {
        'Duplicate Metric': 'Metric',
        'Duplicate KKS Point Name': 'KKS Point Name',
        'Duplicate DCS Description': 'DCS Description',
        'Duplicate Canary Point Name': 'Canary Point Name',
        'Duplicate Canary Description': 'Canary Description'
    }
    
    # The 'row' object is a pandas Series. We get the 'Issue' value from it.
    issues = str(row.get('Issue', ''))
    styles = [''] * len(row) # Default style (no highlight)
    
    # Highlight the specific column that has the duplicate
    for issue_key, col_name in issue_to_col_map.items():
        if issue_key in issues:
            try:
                # Find the *position* (index) of the column name in the row's index
                col_index = list(row.index).index(col_name)
                # Apply style to that position
                styles[col_index] = 'background-color: #FFCCCB' # Light red
            except ValueError:
                pass # Column not in the final view, so we skip
                
    return styles

# --- UPDATED: Reusable Helper Function for TDT Validations ---
def display_duplicate_results(results_dict, key_prefix):
    """
    Displays TDT validation results with a summary, filters, and
    SEPARATE sub-tables for each issue type, highlighting specific cells
    and hiding the 'Issue' column.
    """
    if not results_dict or results_dict.get('summary', pd.DataFrame()).empty:
        st.info("Run the validation to see the results. No issues found.")
        return

    summary_df = results_dict['summary']
    details_df = results_dict['details'] # This df still contains the 'Issue' column

    st.subheader("Validation Summary")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Validation Details")
    
    # --- Create Filters ---
    col1, col2 = st.columns(2)
    with col1:
        tdt_options = ['All'] + sorted(summary_df['TDT'].unique().tolist())
        tdt_filter = st.selectbox(
            "Filter by TDT",
            options=tdt_options,
            key=f"{key_prefix}_tdt_filter"
        )
    
    with col2:
        if tdt_filter == 'All':
            model_options = ['All']
        else:
            model_options = ['All'] + sorted(summary_df[summary_df['TDT'] == tdt_filter]['Model'].unique().tolist())
        
        model_filter = st.selectbox(
            "Filter by Model",
            options=model_options,
            key=f"{key_prefix}_model_filter",
            disabled=(tdt_filter == 'All')
        )

    # --- Filter Data based on dropdowns ---
    details_to_show = details_df.copy()
    if tdt_filter != 'All':
        details_to_show = details_to_show[details_to_show['TDT'] == tdt_filter]
    if model_filter != 'All':
        details_to_show = details_to_show[details_to_show['Model'] == model_filter]

    if details_to_show.empty:
        st.info("No duplicate details match your filter.")
        return

    # --- UPDATED: Split into sub-tables by issue ---
    
    # Get a list of all unique issues present in the *filtered* data
    all_issues = set()
    details_to_show['Issue'].str.split(', ').apply(all_issues.update)
    unique_issues = sorted(list(all_issues))

    if not unique_issues:
        st.info("No issues found in the filtered data.")
        return

    st.markdown(f"**Showing {len(details_to_show)} total duplicate entries across {len(unique_issues)} issue type(s):**")

    # Define the columns you want to show, INCLUDING 'Issue' for the styler
    cols_to_display = [
        'TDT', 'Model', 'Metric', 'Point Type', 'KKS Point Name', 
        'DCS Description', 'Canary Point Name', 'Canary Description', 'Unit', 'Issue'
    ]
    # Filter the list to only columns that actually exist
    existing_cols_to_display = [col for col in cols_to_display if col in details_to_show.columns]

    # Loop through each unique issue and create a table for it
    for issue in unique_issues:
        st.markdown(f"#### {issue}")
        
        # Find all rows that contain this specific issue
        issue_df = details_to_show[details_to_show['Issue'].str.contains(issue, na=False)].copy()
        
        # Filter the columns for display (still includes 'Issue' at this point)
        issue_df_filtered = issue_df[existing_cols_to_display]
        
        if issue_df_filtered.empty:
            st.warning(f"No data for issue: {issue}") # Should not happen, but good safety check
        else:
            # 1. Apply the cell-specific styler
            styler = issue_df_filtered.style.apply(highlight_duplicate_cells, axis=1)
            
            # 2. HIDE the 'Issue' column from the final table
            styler = styler.hide(subset=['Issue'], axis=1)
            
            # 3. Render the styled and hidden dataframe
            st.dataframe(
                styler,
                use_container_width=True
            )


# --- Main Page UI ---
st.title("TDT Configuration Validator")
st.markdown("Select a validation type from the tabs below to check the TDT contents for correctness and completeness.")

tab_list = [
    "TDT Consolidation Overview",
    "Point Survey Validation",
    "Calculation Validation",
    "Attribute Validation",
    "Diagnostics Validation",
    "Prescriptive Validation"
]
tabs = st.tabs(tab_list)

# --- Tab 0: TDT Consolidation Overview ---
with tabs[0]:
    st.header("TDT Consolidation Overview")
    st.markdown("This tab shows the consolidated data loaded from the TDT files on the **Home** page.")
    
    if 'overview_df' not in st.session_state or st.session_state.overview_df is None:
        st.warning("Please go to the **Home** page, upload your TDT files, and click 'Generate & Load Files' to see the overview.")
    else:
        st.subheader("Consolidated TDTs and Models")
        st.dataframe(st.session_state.overview_df, use_container_width=True)

        st.subheader("Full Consolidated Survey Data")
        st.dataframe(st.session_state.survey_df, use_container_width=True)

        st.subheader("Full Consolidated Diagnostics Data")
        st.dataframe(st.session_state.diag_df, use_container_width=True)

# --- Tab 1: Point Survey Validation ---
with tabs[1]:
    st.header("Point Survey Validation")
    st.markdown("Checks for duplicate entries (within the same TDT/Model) in key columns: `Metric`, `KKS Point Name`, `DCS Description`, `Canary Point Name`, and `Canary Description`. This check ignores metrics where `Point Type` is `PRiSM Calc`.")
    
    prerequisites_met = st.session_state.get('survey_df') is not None
    if not prerequisites_met:
        st.warning("Please load TDT files on the **Home** page first.")
    
    if st.button("Run Point Survey Validation", key="run_point_survey_val", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            try:
                # --- This is the new, active logic ---
                results = validate_point_survey(st.session_state.survey_df)
                st.session_state.validation_states["tdt_point_survey"]["results"] = results
                st.success("Point Survey Validation complete!")
            except Exception as e: 
                st.error(f"An error occurred: {e}")
                st.exception(e) # Provides a full traceback for debugging
    
    # --- Display results using the updated function ---
    display_duplicate_results(
        st.session_state.validation_states["tdt_point_survey"].get("results"), 
        "tdt_point_survey"
    )

# --- Tab 2: Calculation Validation (Placeholder) ---
with tabs[2]:
    st.header("Calculation Validation")
    st.markdown("Checks for logic in the **Calculation** sheets, such as missing code for 'PRiSM Calc' points.")
    prerequisites_met = st.session_state.get('survey_df') is not None
    if not prerequisites_met:
        st.warning("Please load TDT files on the **Home** page first.")
    
    if st.button("Run Calculation Validation", key="run_calc_val", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            st.info("Validation logic for 'Calculation' is not yet implemented.")
            
    # (Display results logic will go here)

# --- Tab 3: Attribute Validation (Placeholder) ---
with tabs[3]:
    st.header("Attribute Validation")
    st.markdown("Checks for logic in the **Attribute** sheets, such as filters missing values or invalid function types.")
    prerequisites_met = st.session_state.get('survey_df') is not None
    if not prerequisites_met:
        st.warning("Please load TDT files on the **Home** page first.")
    
    if st.button("Run Attribute Validation", key="run_attr_val", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            st.info("Validation logic for 'Attribute' is not yet implemented.")
            
    # (Display results logic will go here)

# --- Tab 4: Diagnostics Validation (Placeholder) ---
with tabs[4]:
    st.header("Diagnostics Validation")
    st.markdown("Checks for logic in the **Diagnostic** sheets, such as enabled failure modes missing directions or weights.")
    prerequisites_met = st.session_state.get('diag_df') is not None
    if not prerequisites_met:
        st.warning("Please load TDT files on the **Home** page first.")
    
    if st.button("Run Diagnostics Validation", key="run_diag_val", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            st.info("Validation logic for 'Diagnostics' is not yet implemented.")
            
    # (Display results logic will go here)

# --- Tab 5: Prescriptive Validation (Placeholder) ---
with tabs[5]:
    st.header("Prescriptive Validation")
    st.markdown("Checks for logic in the **Prescriptive** sheets (if they exist).")
    prerequisites_met = st.session_state.get('survey_df') is not None # Or another df
    if not prerequisites_met:
        st.warning("Please load TDT files on the **Home** page first.")
    
    if st.button("Run Prescriptive Validation", key="run_presc_val", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            st.info("Validation logic for 'Prescriptive' is not yet implemented.")
            
    # (Display results logic will go here)