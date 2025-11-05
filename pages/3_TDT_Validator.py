import pandas as pd
import streamlit as st
import os

# --- Import the new validator ---
from validations.tdt_validations.point_survey_validation.validator import validate_point_survey
from validations.tdt_validations.calculation_validation.validator import validate_calculation
from validations.tdt_validations.attribute_validation.validator import validate_attribute
from validations.tdt_validations.diagnostics_validation.validator import validate_diagnostics
# (other imports will go here as you build them)
# from validations.tdt_validations.prescriptive_validation.validator import validate_prescriptive

# --- Page Configuration ---
st.set_page_config(page_title="TDT Validator", layout="wide")

# --- Initialize Session State ---
if 'validation_states' not in st.session_state:
    st.session_state.validation_states = {} # Ensure main state exists
if "tdt_point_survey" not in st.session_state.validation_states:
    st.session_state.validation_states["tdt_point_survey"] = {"results": None}
if "tdt_calculation" not in st.session_state.validation_states:
    st.session_state.validation_states["tdt_calculation"] = {"results": None}
if "tdt_attribute" not in st.session_state.validation_states:
    st.session_state.validation_states["tdt_attribute"] = {"results": None}
if "tdt_diagnostics" not in st.session_state.validation_states:
    st.session_state.validation_states["tdt_diagnostics"] = {"results": None}


# --- Helper function to highlight specific cells ---
def highlight_duplicate_cells(row):
    """
    Applies a style to cells that are flagged as duplicates based on the 'Issue' column.
    """
    issue_to_col_map = {
        'Duplicate Metric': 'Metric',
        'Duplicate KKS Point Name': 'KKS Point Name',
        'Duplicate DCS Description': 'DCS Description',
        'Duplicate Canary Point Name': 'Canary Point Name',
        'Duplicate Canary Description': 'Canary Description'
    }
    issues = str(row.get('Issue', ''))
    styles = [''] * len(row)
    for issue_key, col_name in issue_to_col_map.items():
        if issue_key in issues:
            try:
                col_index = list(row.index).index(col_name)
                styles[col_index] = 'background-color: #FFCCCB' # Light red
            except ValueError:
                pass 
    return styles

# --- Helper function to highlight rows with any issue ---
def highlight_issue_rows(row):
    """
    Applies a style to the entire row if the 'Issue' column is not 'OK'.
    """
    style = 'background-color: #FFCCCB' if (pd.notna(row.get('Issue')) and row.get('Issue') != 'OK') else ''
    return [style] * len(row)

# --- Reusable Helper Function for Duplicate results ---
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
    details_df = results_dict['details'] 

    st.subheader("Validation Summary")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Validation Details")
    
    col1, col2 = st.columns(2)
    with col1:
        tdt_options = ['All'] + sorted(summary_df['TDT'].unique().tolist())
        tdt_filter = st.selectbox( "Filter by TDT", options=tdt_options, key=f"{key_prefix}_tdt_filter" )
    
    with col2:
        if tdt_filter == 'All':
            model_options = ['All']
        else:
            model_options = ['All'] + sorted(summary_df[summary_df['TDT'] == tdt_filter]['Model'].unique().tolist())
        model_filter = st.selectbox( "Filter by Model", options=model_options, key=f"{key_prefix}_model_filter", disabled=(tdt_filter == 'All') )

    details_to_show = details_df.copy()
    if tdt_filter != 'All':
        details_to_show = details_to_show[details_to_show['TDT'] == tdt_filter]
    if model_filter != 'All':
        details_to_show = details_to_show[details_to_show['Model'] == model_filter]

    if details_to_show.empty:
        st.info("No duplicate details match your filter.")
        return

    all_issues = set()
    details_to_show['Issue'].str.split(', ').apply(all_issues.update)
    unique_issues = sorted(list(all_issues))

    if not unique_issues:
        st.info("No issues found in the filtered data.")
        return

    st.markdown(f"**Showing {len(details_to_show)} total duplicate entries across {len(unique_issues)} issue type(s):**")

    cols_to_display = [
        'TDT', 'Model', 'Metric', 'Point Type', 'KKS Point Name', 
        'DCS Description', 'Canary Point Name', 'Canary Description', 'Unit', 'Issue'
    ]
    existing_cols_to_display = [col for col in cols_to_display if col in details_to_show.columns]

    for issue in unique_issues:
        st.markdown(f"#### {issue}")
        issue_df = details_to_show[details_to_show['Issue'].str.contains(issue, na=False)].copy()
        issue_df_filtered = issue_df[existing_cols_to_display]
        
        if not issue_df_filtered.empty:
            styler = issue_df_filtered.style.apply(highlight_duplicate_cells, axis=1)
            styler = styler.hide(subset=['Issue'], axis=1)
            st.dataframe(styler, use_container_width=True)

# --- Generic Helper for Simple Results (TDT-only filter) ---
def display_simple_results(results_dict, key_prefix, columns_to_show, show_summary=True, summary_info_msg="No items found to summarize.", details_info_msg="No items were found in the TDTs."):
    """
    Displays TDT validation results with a summary, TDT filter,
    and a single details table. Highlights rows with issues.
    """
    if not results_dict:
        st.info("Run the validation to see the results.")
        return

    summary_df = results_dict.get('summary', pd.DataFrame())
    details_df = results_dict.get('details', pd.DataFrame())

    if show_summary:
        st.subheader("Validation Summary")
        if summary_df.empty:
            st.info(summary_info_msg)
        else:
            st.dataframe(summary_df, use_container_width=True)

    st.subheader("Validation Details")
    
    if details_df.empty:
        st.info(details_info_msg)
        return
        
    tdt_options = ['All'] + sorted(details_df['TDT'].unique().tolist())
    tdt_filter = st.selectbox("Filter by TDT", options=tdt_options, key=f"{key_prefix}_tdt_filter")

    # Filter Data
    details_to_show = details_df.copy()
    if tdt_filter != 'All':
        details_to_show = details_to_show[details_to_show['TDT'] == tdt_filter]

    if details_to_show.empty:
        st.info("No details match your filter.")
        return

    # 1. Get the list of columns that will *actually* be displayed
    existing_cols_to_display = [col for col in columns_to_show if col in details_to_show.columns]
    
    # 2. Get the *full* list of columns needed for the styler
    styler_cols = existing_cols_to_display + ['Issue']
    styler_cols = list(dict.fromkeys(styler_cols))
    
    # 3. Filter the dataframe to *only* the columns needed for the styler
    existing_styler_cols = [col for col in styler_cols if col in details_to_show.columns]
    details_for_styler = details_to_show[existing_styler_cols]

    if details_for_styler.empty:
        st.info("No data to display.")
        return

    # 4. Apply the row highlighter
    styler = details_for_styler.style.apply(highlight_issue_rows, axis=1)
    
    # 5. Hide the 'Issue' column (if it exists)
    if 'Issue' in details_for_styler.columns:
        styler = styler.hide(subset=['Issue'], axis=1)
        
    st.dataframe(styler, use_container_width=True)


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
                results = validate_point_survey(st.session_state.survey_df)
                st.session_state.validation_states["tdt_point_survey"]["results"] = results
                st.success("Point Survey Validation complete!")
            except Exception as e: 
                st.error(f"An error occurred: {e}")
                st.exception(e) 
    
    display_duplicate_results(
        st.session_state.validation_states["tdt_point_survey"].get("results"), 
        "tdt_point_survey"
    )

# --- Tab 2: Calculation Validation ---
with tabs[2]:
    st.header("Calculation Validation")
    st.markdown("Checks all `PRiSM Calc` metrics, grouped by TDT, and identifies any that are missing all calculation-specific fields (`Calc Point Type`, `Calculation Description`, etc.).")
    
    prerequisites_met = st.session_state.get('survey_df') is not None
    if not prerequisites_met:
        st.warning("Please load TDT files on the **Home** page first.")
    
    if st.button("Run Calculation Validation", key="run_calc_val", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            try:
                results = validate_calculation(st.session_state.survey_df)
                st.session_state.validation_states["tdt_calculation"]["results"] = results
                st.success("Calculation Validation complete!")
            except Exception as e: 
                st.error(f"An error occurred: {e}")
                st.exception(e)
            
    calc_cols_to_show = [
        'TDT', 'Metric', 'Point Type', 'Calc Point Type', 'Calculation Description', 
        'Pseudo Code', 'Language', 'Input Point', 'PRiSM Code'
    ]
    display_simple_results(
        st.session_state.validation_states["tdt_calculation"].get("results"), 
        "tdt_calculation",
        calc_cols_to_show,
        summary_info_msg="No 'PRiSM Calc' points found to summarize.",
        details_info_msg="No 'PRiSM Calc' points were found in the TDTs."
    )

# --- Tab 3: Attribute Validation ---
with tabs[3]:
    st.header("Attribute Validation")
    st.markdown("Checks logic for `Function`, `Constraint`, and `Diagnostic` usage. Also checks for incomplete `Filter` definitions and provides an audit of all active filters.")
    
    prerequisites_met = (st.session_state.get('survey_df') is not None) and (st.session_state.get('diag_df') is not None)
    if not prerequisites_met:
        st.warning("Please load TDT files on the **Home** page first (this check requires both Survey and Diagnostic data).")
    
    if st.button("Run Attribute Validation", key="run_attr_val", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            try:
                results = validate_attribute(st.session_state.survey_df, st.session_state.diag_df)
                st.session_state.validation_states["tdt_attribute"]["results"] = results
                st.success("Attribute Validation complete!")
            except Exception as e: 
                st.error(f"An error occurred: {e}")
                st.exception(e)
            
    results_data = st.session_state.validation_states["tdt_attribute"].get("results")
    
    if results_data:
        # --- Report 1: Function & Diagnostic Validation ---
        st.markdown("---")
        st.header("Function & Diagnostic Usage Validation")
        st.markdown("Audits all metrics (grouped by TDT) for correct `Function`, `Constraint`, `Diagnostic` usage, and `Filter` completeness. Rows in red indicate a logical conflict.")
        
        function_results = results_data.get("function_validation")
        function_cols_to_show = [
            'TDT', 'Metric', 'Function', 'Constraint', 'Diag_Count', 'Filter Condition', 'Filter Value'
        ]
        display_simple_results(
            function_results, 
            "tdt_attr_func",
            function_cols_to_show,
            summary_info_msg="No metrics found to summarize.",
            details_info_msg="No metrics were found in the TDTs."
        )

        # --- Report 2: Filter Audit ---
        st.markdown("---")
        st.header("Filter Audit")
        st.markdown("Lists all metrics (grouped by TDT) that have **complete** `Filter Condition` and `Filter Value` defined.")
        
        filter_results = results_data.get("filter_audit")
        filter_cols_to_show = [
            'TDT', 'Metric', 'Function', 'Filter Condition', 'Filter Value'
        ]
        display_simple_results(
            filter_results, 
            "tdt_attr_filter",
            filter_cols_to_show,
            show_summary=False,
            details_info_msg="No metrics with *complete* filters were found."
        )
    
    elif st.session_state.validation_states["tdt_attribute"].get("results") is None:
        st.info("Click 'Run Attribute Validation' to see the reports.")


# --- Tab 4: Diagnostics Validation (UPDATED) ---
with tabs[4]:
    st.header("Diagnostics Validation")
    st.markdown("Provides a summary of Failure Modes and a drill-down to inspect metric weights and directions.")
    
    prerequisites_met = st.session_state.get('diag_df') is not None
    if not prerequisites_met:
        st.warning("Please load TDT files on the **Home** page first.")
    
    if st.button("Run Diagnostics Validation", key="run_diag_val", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            try:
                results = validate_diagnostics(st.session_state.diag_df)
                st.session_state.validation_states["tdt_diagnostics"]["results"] = results
                st.success("Diagnostics Validation complete!")
            except Exception as e: 
                st.error(f"An error occurred: {e}")
                st.exception(e)

    # --- Display Logic for Diagnostics ---
    results_data = st.session_state.validation_states["tdt_diagnostics"].get("results")
    
    if results_data and not results_data.get('summary', pd.DataFrame()).empty:
        summary_df = results_data['summary']
        details_df = results_data['details']

        # --- 1. TDT Overview (Failure Mode Count per TDT) ---
        st.subheader("TDT Overview")
        overview_df = (
            summary_df.groupby('TDT')['Failure Mode']
            .nunique()
            .to_frame('Unique Failure Mode Count')
            .reset_index()
        )
        st.dataframe(overview_df, use_container_width=True)

        # --- 2. Failure Mode Summary (Metric Count & Weight Check) ---
        st.subheader("Failure Mode Summary")
        st.markdown("Shows metric count and total weight per failure mode. Rows in red have a `Total_Weight` that is not 100 (or 0).")
        
        # --- NEW TDT FILTER FOR SUMMARY ---
        summary_tdt_options = ['All'] + sorted(summary_df['TDT'].unique().tolist())
        summary_tdt_filter = st.selectbox(
            "Filter Summary by TDT",
            options=summary_tdt_options,
            key="diag_summary_tdt_filter"
        )

        summary_to_show = summary_df.copy()
        if summary_tdt_filter != 'All':
            summary_to_show = summary_to_show[summary_to_show['TDT'] == summary_tdt_filter]
        
        st.dataframe(
            summary_to_show.style.apply(highlight_issue_rows, axis=1),
            use_container_width=True
        )

        # --- 3. Details Drill-Down ---
        st.subheader("Details Drill-Down")
        
        # --- UPDATED: Use the summary TDT filter ---
        if summary_tdt_filter == 'All':
            st.info("Select a TDT from the 'Filter Summary by TDT' dropdown above to see failure modes.")
        else:
            # Create Failure Mode filter based on the *summary TDT filter*
            fm_options = sorted(
                details_df[details_df['TDT'] == summary_tdt_filter]['Failure Mode'].unique().tolist()
            )
            fm_filter = st.selectbox(
                "Filter by Failure Mode", 
                options=fm_options, 
                key="diag_fm_filter"
            )
        
            if fm_filter: # Check if a failure mode is selected
                # Filter the details dataframe
                filtered_details = details_df[
                    (details_df['TDT'] == summary_tdt_filter) &
                    (details_df['Failure Mode'] == fm_filter)
                ]
                
                # Display the relevant columns
                st.dataframe(
                    filtered_details[['Metric', 'Direction', 'Weighting']],
                    use_container_width=True
                )
            else:
                st.info(f"No failure modes found for TDT: '{summary_tdt_filter}'.")

    elif st.session_state.validation_states["tdt_diagnostics"].get("results") is None:
        st.info("Click 'Run Diagnostics Validation' to see the reports.")
    else:
        st.info("No diagnostic data was found in the TDTs.")


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