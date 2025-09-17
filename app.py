import pandas as pd
import streamlit as st
from db_utils import PrismDB

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


# --- Page Configuration ---
st.set_page_config(page_title="PRISM Config Validator", layout="wide")

# --- Initialize Session State (Unchanged) ---
if 'validation_states' not in st.session_state:
    st.session_state.validation_states = {
        "metric_validation": {"results": None},
        "metric_mapping": {"results": None},
        "failure_diagnostics": {"results": None},
        "filter_validation": {"results": None},
        "absolute_deviation": {"results": None}
    }
if 'db' not in st.session_state:
    st.session_state.db = None
if 'uploaded_survey_file' not in st.session_state:
    st.session_state.uploaded_survey_file = None
if 'uploaded_diag_file' not in st.session_state:
    st.session_state.uploaded_diag_file = None
if 'uploaded_stats_file' not in st.session_state:
    st.session_state.uploaded_stats_file = None

# --- Reusable Helper Functions (Unchanged) ---
def display_results(results, key_prefix, filter_column_name):
    if not results or results['summary'].empty:
        st.info("Run the validation to see the results.")
        return
    st.subheader("Validation Summary")
    st.dataframe(results['summary'], use_container_width=True)
    st.subheader("Validation Details")
    if filter_column_name not in results['summary'].columns:
        st.error(f"Cannot generate filter; expected column '{filter_column_name}' not in summary.")
        return
    filter_options = ['All'] + results['summary'][filter_column_name].tolist()
    tdt_filter = st.selectbox(f"Filter by {filter_column_name}", options=filter_options, key=f"tdt_filter_{key_prefix}")
    st.markdown("#### Matches")
    matches_to_show = results.get('matches', pd.DataFrame())
    if tdt_filter != 'All' and filter_column_name in matches_to_show.columns:
        matches_to_show = matches_to_show[matches_to_show[filter_column_name] == tdt_filter]
    st.dataframe(matches_to_show, use_container_width=True)
    st.metric("Total Matches Shown", len(matches_to_show))
    st.markdown("#### Mismatches")
    mismatches_data = results.get('mismatches', {})
    if isinstance(mismatches_data, dict):
        total_mismatches_shown = 0
        for mismatch_type, mismatch_df in mismatches_data.items():
            if not mismatch_df.empty:
                st.markdown(f"##### {mismatch_type.replace('_', ' ').title()}")
                mismatches_to_show = mismatch_df
                if tdt_filter != 'All' and filter_column_name in mismatches_to_show.columns:
                    mismatches_to_show = mismatches_to_show[mismatches_to_show[filter_column_name] == tdt_filter]
                st.dataframe(mismatches_to_show, use_container_width=True)
                total_mismatches_shown += len(mismatches_to_show)
        st.metric("Total Mismatches Shown", total_mismatches_shown)
    else:
        mismatches_to_show = mismatches_data
        if tdt_filter != 'All' and filter_column_name in mismatches_to_show.columns:
            mismatches_to_show = mismatches_to_show[mismatches_to_show[filter_column_name] == tdt_filter]
        st.dataframe(mismatches_to_show, use_container_width=True)
        st.metric("Total Mismatches Shown", len(mismatches_to_show))

# --- Sidebar UI (Unchanged) ---
with st.sidebar:
    st.header("1. Database Connection")
    db_host = st.text_input("Host", value=st.secrets.get("db", {}).get("host", ""))
    db_name = st.text_input("Database", value=st.secrets.get("db", {}).get("database", ""))
    db_user = st.text_input("User", value=st.secrets.get("db", {}).get("user", ""))
    db_pass = st.text_input("Password", type="password", value=st.secrets.get("db", {}).get("password", ""))
    if st.button("Connect to Database"):
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
    st.header("2. Upload Reference Files")
    uploaded_survey = st.file_uploader("Upload Consolidated Survey File", type=["xlsx"], key="survey_uploader")
    if uploaded_survey:
        st.session_state.uploaded_survey_file = uploaded_survey

    uploaded_diag = st.file_uploader("Upload Consolidated Failure Diagnostics File", type=["xlsx"], key="diag_uploader")
    if uploaded_diag:
        st.session_state.uploaded_diag_file = uploaded_diag

    uploaded_stats = st.file_uploader("Upload Consolidated Statistics File", type=["xlsx"], key="stats_uploader")
    if uploaded_stats:
        st.session_state.uploaded_stats_file = uploaded_stats

# --- Main Page UI ---
st.title("PRISM Configuration Validator")
st.markdown("Select a validation type from the tabs below. Each tab will indicate which files are required.")

# NEW: Reordered tabs for a more logical workflow
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Metric Validation (Template)",
    "Metric Mapping Validation (Project)",
    "Filter Validation (Project)",
    "Failure Diagnostics Validation (Template)",
    "Absolute Deviation Validation (Project)"
])

# --- Tab 1: Metric Validation ---
with tab1:
    st.header("Metric Validation (TDT vs PRISM Template)")
    # NEW: Per-tab prerequisite check
    prerequisites_met_tab1 = st.session_state.db and st.session_state.uploaded_survey_file
    if not prerequisites_met_tab1:
        st.warning("Please connect to the database and upload the 'Consolidated Survey File' to run this validation.")
    
    if st.button("Run Metric Validation", key="run_metric_validation", disabled=not prerequisites_met_tab1):
        with st.spinner('Running metric validation...'):
            try:
                prism_df = st.session_state.db.run_query(get_metric_query())
                tdt_dfs = parse_metric_excel(st.session_state.uploaded_survey_file)
                summary, matches, mismatches = validate_metric_data(tdt_dfs, prism_df)
                st.session_state.validation_states["metric_validation"]["results"] = {'summary': summary, 'matches': matches, 'mismatches': mismatches}
            except Exception as e: st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["metric_validation"]["results"], "metric_val", "TDT")

# --- Tab 2: Metric Mapping Validation ---
with tab2:
    st.header("Metric Mapping Validation (TDT vs PRISM Project)")
    prerequisites_met_tab2 = st.session_state.db and st.session_state.uploaded_survey_file
    if not prerequisites_met_tab2:
        st.warning("Please connect to the database and upload the 'Consolidated Survey File' to run this validation.")

    if st.button("Run Metric Mapping Validation", key="run_metric_mapping", disabled=not prerequisites_met_tab2):
        with st.spinner('Running metric mapping validation...'):
            try:
                prism_df = st.session_state.db.run_query(get_metric_mapping_query())
                model_dfs = parse_metric_mapping_excel(st.session_state.uploaded_survey_file)
                summary, matches, mismatches = validate_metric_mapping_data(model_dfs, prism_df)
                st.session_state.validation_states["metric_mapping"]["results"] = {'summary': summary, 'matches': matches, 'mismatches': mismatches}
            except Exception as e: st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["metric_mapping"]["results"], "metric_map", "MODEL")

# --- Tab 3: Filter Validation ---
with tab3:
    st.header("Filter Validation (TDT vs PRISM Project)")
    prerequisites_met_tab3 = st.session_state.db and st.session_state.uploaded_survey_file
    if not prerequisites_met_tab3:
        st.warning("Please connect to the database and upload the 'Consolidated Survey File' to run this validation.")

    if st.button("Run Filter Validation", key="run_filter_validation", disabled=not prerequisites_met_tab3):
        with st.spinner('Running filter validation...'):
            try:
                prism_df = st.session_state.db.run_query(get_filter_query())
                model_dfs = parse_filter_excel(st.session_state.uploaded_survey_file)
                summary, matches, mismatches = validate_filter_data(model_dfs, prism_df)
                st.session_state.validation_states["filter_validation"]["results"] = {'summary': summary, 'matches': matches, 'mismatches': mismatches}
            except Exception as e: st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["filter_validation"]["results"], "filter_val", "MODEL")

# --- Tab 4: Failure Diagnostics Validation ---
with tab4:
    st.header("Failure Diagnostics Validation (TDT vs PRISM Template)")
    prerequisites_met_tab4 = st.session_state.db and st.session_state.uploaded_diag_file
    if not prerequisites_met_tab4:
        st.warning("Please connect to the database and upload the 'Consolidated Failure Diagnostics File' to run this validation.")

    if st.button("Run Failure Diagnostics Validation", key="run_failure_diagnostics", disabled=not prerequisites_met_tab4):
        with st.spinner('Running failure diagnostics validation...'):
            try:
                prism_df = st.session_state.db.run_query(get_failure_diag_query())
                tdt_dfs = parse_failure_diag_excel(st.session_state.uploaded_diag_file)
                summary, matches, mismatches = validate_failure_diag_data(tdt_dfs, prism_df)
                st.session_state.validation_states["failure_diagnostics"]["results"] = {'summary': summary, 'matches': matches, 'mismatches': mismatches}
            except Exception as e: st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["failure_diagnostics"]["results"], "failure_diag", "TDT")

# --- Tab 5: Absolute Deviation Validation ---
with tab5:
    st.header("Absolute Deviation Validation (Project)")
    prerequisites_met_tab5 = st.session_state.db and st.session_state.uploaded_stats_file
    if not prerequisites_met_tab5:
        st.warning("Please connect to the database and upload the 'Consolidated Statistics File' to run this validation.")
    
    if st.button("Run Absolute Deviation Validation", key="run_abs_dev_validation", disabled=not prerequisites_met_tab5):
        with st.spinner('Running absolute deviation validation...'):
            try:
                prism_df = st.session_state.db.run_query(get_abs_dev_query())
                model_dfs = parse_abs_dev_excel(st.session_state.uploaded_stats_file)
                summary, matches, mismatches = validate_abs_dev_data(model_dfs, prism_df)
                st.session_state.validation_states["absolute_deviation"]["results"] = {'summary': summary, 'matches': matches, 'mismatches': mismatches}
            except Exception as e: st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["absolute_deviation"]["results"], "abs_dev", "MODEL")