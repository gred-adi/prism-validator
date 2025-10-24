import pandas as pd
import streamlit as st
from db_utils import PrismDB
import os

# --- Import validation-specific modules ---
from validations.metric_validation.query import get_query as get_metric_query
from validations.metric_validation.parser import parse_excel as parse_metric_excel
from validations.metric_validation.validator import validate_data as validate_metric_data

from validations.metric_mapping_validation.query import get_query as get_metric_mapping_query
from validations.metric_mapping_validation.parser import parse_excel as parse_metric_mapping_excel
from validations.metric_mapping_validation.validator import validate_data as validate_metric_mapping_data

from validations.failure_diagnostics_validation.query import get_query as get_failure_diag_query
from validations.failure_diagnostics_validation.parser import parse_excel as parse_failure_diag_excel
from validations.failure_diagnostics_validation.validator import validate_data as validate_failure_diag_data

from validations.filter_validation.query import get_query as get_filter_query
from validations.filter_validation.parser import parse_excel as parse_filter_excel
from validations.filter_validation.validator import validate_data as validate_filter_data

from validations.absolute_deviation_validation.query import get_query as get_abs_dev_query
from validations.absolute_deviation_validation.parser import parse_excel as parse_abs_dev_excel
from validations.absolute_deviation_validation.validator import validate_data as validate_abs_dev_data

from validations.model_deployment_config.query import get_query as get_model_deployment_query

# from file_generator import generate_files_from_folder, convert_dfs_to_excel_bytes

# --- Import the file generator module ---
# from file_generator import generate_files_from_folder

# --- Page Configuration ---
st.set_page_config(page_title="PRISM Config Validator", layout="wide")

# --- Initialize Session State ---
if 'db' not in st.session_state: st.session_state.db = None
if 'tdt_folder_path' not in st.session_state: st.session_state.tdt_folder_path = ""
# NEW: States for raw DataFrames and overview
if 'survey_df' not in st.session_state: st.session_state.survey_df = None
if 'diag_df' not in st.session_state: st.session_state.diag_df = None
if 'overview_df' not in st.session_state: st.session_state.overview_df = None
# States for file bytes (for parsers and downloads)
if 'uploaded_survey_file' not in st.session_state: st.session_state.uploaded_survey_file = None
if 'uploaded_diag_file' not in st.session_state: st.session_state.uploaded_diag_file = None
if 'uploaded_stats_file' not in st.session_state: st.session_state.uploaded_stats_file = None
# States for validation results
if 'validation_states' not in st.session_state:
    st.session_state.validation_states = {
        "metric_validation": {"results": None}, "metric_mapping": {"results": None},
        "failure_diagnostics": {"results": None}, "filter_validation": {"results": None},
        "absolute_deviation": {"results": None}, "model_deployment_config": {"results": None}
    }

# --- Reusable Helper Functions ---
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

def display_results(results, key_prefix, filter_column_name):
    """
    Generic function to display validation results in a consistent format.
    Includes a summary table, a filter, and detailed tables for matches,
    mismatches, and (conditionally) all entries.
    """
    if not results or results.get('summary', pd.DataFrame()).empty:
        st.info("Run the validation to see the results.")
        return

    st.subheader("Validation Summary")
    st.dataframe(results['summary'], use_container_width=True)

    st.subheader("Validation Details")
    # Ensure the filter column exists before proceeding
    if filter_column_name not in results['summary'].columns:
        st.error(f"Cannot generate filter; expected column '{filter_column_name}' not in summary.")
        return

    # --- Create Filter Dropdown ---
    filter_options = ['All'] + results['summary'][filter_column_name].tolist()
    tdt_filter = st.selectbox(
        f"Filter by {filter_column_name}",
        options=filter_options,
        key=f"tdt_filter_{key_prefix}"
    )

    # --- Conditionally Display "All Entries" Table ---
    all_entries_df = results.get('all_entries')
    if tdt_filter != 'All' and all_entries_df is not None and not all_entries_df.empty:
        st.markdown("#### All Entries (Filtered)")
        # Ensure the filter column exists in the all_entries dataframe
        if filter_column_name in all_entries_df.columns:
            all_entries_to_show = all_entries_df[all_entries_df[filter_column_name] == tdt_filter].copy()
            # Optional: Add a column to show match status
            if '_merge' in all_entries_to_show.columns:
                all_entries_to_show['Status'] = all_entries_to_show['_merge'].replace({
                    'both': 'Match/Mismatch',
                    'left_only': 'Missing in PRISM',
                    'right_only': 'Missing in TDT'
                })

            # --- NEW: Drop specified columns ---
            cols_to_drop = ['_merge', 'Status', 'FORM_ID', 'METRIC_ID', 'FORM ID', 'METRIC ID', 'POINT NAME', 'FUNCTION', 'THRESHOLD TYPE']
            all_entries_to_show = all_entries_to_show.drop(columns=[col for col in cols_to_drop if col in all_entries_to_show.columns])

            st.dataframe(all_entries_to_show.style.apply(highlight_diff, axis=None), use_container_width=True)
            st.metric("Total Entries Shown", len(all_entries_to_show))
        else:
            st.warning(f"Filter column '{filter_column_name}' not found in the 'All Entries' table.")


    # --- Display Matches Table ---
    st.markdown("#### Matches")
    matches_to_show = results.get('matches', pd.DataFrame())
    if tdt_filter != 'All' and filter_column_name in matches_to_show.columns:
        matches_to_show = matches_to_show[matches_to_show[filter_column_name] == tdt_filter]

    # --- NEW: Drop specified columns ---
    cols_to_drop = ['_merge', 'Status', 'FORM_ID', 'METRIC_ID', 'FORM ID', 'METRIC ID', 'POINT NAME', 'FUNCTION', 'THRESHOLD TYPE']
    matches_to_show = matches_to_show.drop(columns=[col for col in cols_to_drop if col in matches_to_show.columns])

    st.dataframe(matches_to_show, use_container_width=True)
    st.metric("Total Matches Shown", len(matches_to_show))

    # --- Display Mismatches Tables ---
    st.markdown("#### Mismatches")
    mismatches_data = results.get('mismatches', {})
    total_mismatches_shown = 0

    if isinstance(mismatches_data, dict):
        for mismatch_type, mismatch_df in mismatches_data.items():
            if not mismatch_df.empty:
                st.markdown(f"##### {mismatch_type.replace('_', ' ').title()}")
                mismatches_to_show = mismatch_df
                if tdt_filter != 'All' and filter_column_name in mismatches_to_show.columns:
                    mismatches_to_show = mismatches_to_show[mismatches_to_show[filter_column_name] == tdt_filter]
                st.dataframe(mismatches_to_show, use_container_width=True)
                total_mismatches_shown += len(mismatches_to_show)
    elif isinstance(mismatches_data, pd.DataFrame) and not mismatches_data.empty:
        mismatches_to_show = mismatches_data
        if tdt_filter != 'All' and filter_column_name in mismatches_to_show.columns:
            mismatches_to_show = mismatches_to_show[mismatches_to_show[filter_column_name] == tdt_filter]
        st.dataframe(mismatches_to_show, use_container_width=True)
        total_mismatches_shown = len(mismatches_to_show)

    st.metric("Total Mismatches Shown", total_mismatches_shown)

# --- Sidebar UI ---
with st.sidebar:
    st.header("1. Database Connection")
    db_host = st.text_input("Host", value=st.secrets.get("db", {}).get("host", ""))
    db_name = st.text_input("Database", value=st.secrets.get("db", {}).get("database", ""))
    db_user = st.text_input("User", value=st.secrets.get("db", {}).get("user", ""))
    db_pass = st.text_input("Password", type="password", value=st.secrets.get("db", {}).get("password", ""))
    if st.button("Connect to Database"):
        # (DB connection logic is unchanged)
        with st.spinner("Connecting..."):
            try:
                st.session_state.db = PrismDB(db_host, db_name, db_user, db_pass)
                st.session_state.db.test_connection()
                st.success("✅ Connection successful!")
            except Exception as e:
                st.session_state.db = None
                st.error(f"❌ Connection failed: {e}")
    if st.session_state.db: st.success("Database is connected.")
    else: st.warning("Database is not connected.")

    st.markdown("---")

    st.header("2. Upload Statistics File")
    uploaded_stats = st.file_uploader("Upload Consolidated Statistics File", type=["xlsx"], key="stats_uploader")
    if uploaded_stats: st.session_state.uploaded_stats_file = uploaded_stats

    # --- UPDATED: Optional Downloads Section ---
    st.markdown("---")
    st.header("Download Generated Files")
    st.download_button(
        label="Download Survey File",
        data=st.session_state.uploaded_survey_file or b"",
        file_name="Consolidated_Point_Survey.xlsx",
        mime="application/vnd.ms-excel",
        disabled=st.session_state.uploaded_survey_file is None
    )
    st.download_button(
        label="Download Diagnostics File",
        data=st.session_state.uploaded_diag_file or b"",
        file_name="Consolidated_Failure_Diagnostics.xlsx",
        mime="application/vnd.ms-excel",
        disabled=st.session_state.uploaded_diag_file is None
    )

# --- Main Page UI ---
st.title("PRISM Configuration Validator")
st.markdown("Select a validation type from the tabs below.")

tab_list = [
    "Consolidation Overview", # NEW TAB
    "Metric Validation (Template)",
    "Metric Mapping Validation (Project)",
    "Filter Validation (Project)",
    "Failure Diagnostics Validation (Template)",
    "Absolute Deviation Validation",
    "Model Deployment Config"
]
tabs = st.tabs(tab_list)

# --- NEW Tab 0: Consolidation Overview ---
with tabs[0]:
    st.header("TDT Consolidation Overview")
    if st.session_state.overview_df is None:
        st.info("Select a TDT folder and click 'Generate & Load Files' in the sidebar to see the consolidation overview.")
    else:
        st.subheader("Consolidated TDTs and Models")
        st.dataframe(st.session_state.overview_df, use_container_width=True)

        st.subheader("Full Consolidated Survey Data")
        st.dataframe(st.session_state.survey_df, use_container_width=True)

        st.subheader("Full Consolidated Diagnostics Data")
        st.dataframe(st.session_state.diag_df, use_container_width=True)

with tabs[1]:
    st.header("Metric Validation (TDT vs PRISM Template)")
    prerequisites_met = st.session_state.db and st.session_state.uploaded_survey_file
    if not prerequisites_met: st.warning("Please connect to DB and generate the 'Consolidated Survey File'.")
    if st.button("Run Metric Validation", key="run_metric_validation", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            try:
                prism_df = st.session_state.db.run_query(get_metric_query())
                tdt_dfs = parse_metric_excel(st.session_state.uploaded_survey_file)
                # The validator now returns a dictionary
                results = validate_metric_data(tdt_dfs, prism_df)
                st.session_state.validation_states["metric_validation"]["results"] = results
            except Exception as e: st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["metric_validation"]["results"], "metric_val", "TDT")

with tabs[2]:
    st.header("Metric Mapping Validation (TDT vs PRISM Project)")
    prerequisites_met = st.session_state.db and st.session_state.uploaded_survey_file
    if not prerequisites_met: st.warning("Please connect to DB and generate the 'Consolidated Survey File'.")
    if st.button("Run Metric Mapping Validation", key="run_metric_mapping", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            try:
                prism_df = st.session_state.db.run_query(get_metric_mapping_query())
                model_dfs = parse_metric_mapping_excel(st.session_state.uploaded_survey_file)
                # The validator now returns a dictionary
                results = validate_metric_mapping_data(model_dfs, prism_df)
                st.session_state.validation_states["metric_mapping"]["results"] = results
            except Exception as e: st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["metric_mapping"]["results"], "metric_map", "MODEL")

with tabs[3]:
    st.header("Filter Validation (TDT vs PRISM Project)")
    prerequisites_met = st.session_state.db and st.session_state.uploaded_survey_file
    if not prerequisites_met: st.warning("Please connect to DB and generate the 'Consolidated Survey File'.")
    if st.button("Run Filter Validation", key="run_filter_validation", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            try:
                prism_df = st.session_state.db.run_query(get_filter_query())
                model_dfs = parse_filter_excel(st.session_state.uploaded_survey_file)
                # The validator now returns a dictionary
                results = validate_filter_data(model_dfs, prism_df)
                st.session_state.validation_states["filter_validation"]["results"] = results
            except Exception as e: st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["filter_validation"]["results"], "filter_val", "MODEL")

with tabs[4]:
    st.header("Failure Diagnostics Validation (TDT vs PRISM Template)")
    prerequisites_met = st.session_state.db and st.session_state.uploaded_diag_file
    if not prerequisites_met: st.warning("Please connect to DB and generate the 'Consolidated Failure Diagnostics File'.")
    if st.button("Run Failure Diagnostics Validation", key="run_failure_diagnostics", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            try:
                prism_df = st.session_state.db.run_query(get_failure_diag_query())
                tdt_dfs = parse_failure_diag_excel(st.session_state.uploaded_diag_file)
                # The validator now returns a dictionary
                results = validate_failure_diag_data(tdt_dfs, prism_df)
                st.session_state.validation_states["failure_diagnostics"]["results"] = results
            except Exception as e: st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["failure_diagnostics"]["results"], "failure_diag", "TDT")

with tabs[5]:
    st.header("Absolute Deviation Validation")
    prerequisites_met = st.session_state.db and st.session_state.uploaded_stats_file
    if not prerequisites_met: st.warning("Please connect to DB and upload the 'Consolidated Statistics File'.")
    if st.button("Run Absolute Deviation Validation", key="run_abs_dev_validation", disabled=not prerequisites_met):
        with st.spinner('Running...'):
            try:
                prism_df = st.session_state.db.run_query(get_abs_dev_query())
                model_dfs = parse_abs_dev_excel(st.session_state.uploaded_stats_file)
                # The validator now returns a dictionary
                results = validate_abs_dev_data(model_dfs, prism_df)
                st.session_state.validation_states["absolute_deviation"]["results"] = results
            except Exception as e: st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["absolute_deviation"]["results"], "abs_dev", "MODEL")

with tabs[6]:
    # (Tab 6 logic is unchanged)
    st.header("Model Deployment Configuration")
    prereqs_met = st.session_state.db
    if not prereqs_met: st.warning("Please connect to the database to fetch deployment configurations.")
    st.subheader("1. Enter Asset Descriptions")
    assets_input = st.text_area("Enter a list of Asset Descriptions, one per line.", height=150, disabled=not prereqs_met, value="AP-TVI-U1-BOP\nAP-TVI-U2-BOP\nAP-TVI-BOP")
    if st.button("Fetch Configuration", key="fetch_deployment_config", disabled=not prereqs_met):
        asset_list = [asset.strip() for asset in assets_input.split('\n') if asset.strip()]
        if not asset_list: st.error("Please enter at least one Asset Description.")
        else:
            with st.spinner("Fetching configuration..."):
                try:
                    query = get_model_deployment_query(asset_list)
                    config_df = st.session_state.db.run_query(query)
                    st.session_state.validation_states["model_deployment_config"]["results"] = config_df
                except Exception as e: st.error(f"An error occurred: {e}")
    st.subheader("2. Deployment Configuration Results")
    results_df = st.session_state.validation_states["model_deployment_config"]["results"]
    if results_df is not None: st.dataframe(results_df, use_container_width=True)
    else: st.info("Enter Asset Descriptions and click 'Fetch Configuration' to see results.")