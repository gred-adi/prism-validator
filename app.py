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

# --- Page Configuration ---
st.set_page_config(page_title="PRISM Config Validator", layout="wide")

# --- Initialize Session State ---
if 'validation_states' not in st.session_state:
    st.session_state.validation_states = {
        "metric_validation": {"results": None},
        "metric_mapping": {"results": None},
        "failure_diagnostics": {"results": None},
        "filter_validation": {"results": None}
    }
if 'db' not in st.session_state:
    st.session_state.db = None
if 'uploaded_file_state' not in st.session_state:
    st.session_state.uploaded_file_state = None

# --- Reusable Helper Functions ---
def display_results(results, key_prefix, filter_column_name):
    """
    Helper function to display results.
    Now handles both single DataFrame and dictionary of DataFrames for mismatches.
    """
    if not results or results['summary'].empty:
        st.info("Run the validation to see the results.")
        return

    st.subheader("Validation Summary")
    st.dataframe(results['summary'], use_container_width=True)
    st.subheader("Validation Details")

    if filter_column_name not in results['summary'].columns:
        st.error(f"Cannot generate filter because the expected column '{filter_column_name}' was not found in the summary.")
        return

    filter_options = ['All'] + results['summary'][filter_column_name].tolist()
    
    tdt_filter = st.selectbox(
        f"Filter by {filter_column_name}",
        options=filter_options,
        key=f"tdt_filter_{key_prefix}"
    )

    # --- Display Matches ---
    st.markdown("#### Matches")
    matches_to_show = results.get('matches', pd.DataFrame())
    if tdt_filter != 'All' and filter_column_name in matches_to_show.columns:
        matches_to_show = matches_to_show[matches_to_show[filter_column_name] == tdt_filter]
    st.dataframe(matches_to_show, use_container_width=True)
    st.metric("Total Matches Shown", len(matches_to_show))

    # --- Display Mismatches ---
    st.markdown("#### Mismatches")
    mismatches_data = results.get('mismatches', {})
    
    # FIX: Check if mismatches is a dictionary (new structure) or a DataFrame (old structure)
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
    else: # Fallback for old structure (single DataFrame)
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

    if st.session_state.db:
        st.success("Database is connected.")
    else:
        st.warning("Database is not connected.")

    st.markdown("---")
    st.header("2. Upload Reference File")
    uploaded_file = st.file_uploader("Upload Excel Reference", type=["xlsx"])
    if uploaded_file:
        st.session_state.uploaded_file_state = uploaded_file

# --- Main Page UI ---
st.title("PRISM Configuration Validator")
st.markdown("Select a validation type from the tabs below and click 'Run Validation'.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Metric Validation (Template)",
    "Metric Mapping Validation (Project)",
    "Failure Diagnostics Validation (Template)",
    "Filter Validation (Project)"
])

# --- Tab 1: Metric Validation ---
with tab1:
    st.header("Metric Validation (TDT vs PRISM Template)")
    if st.button("Run Metric Validation", key="run_metric_validation"):
        if not st.session_state.db or not st.session_state.uploaded_file_state:
            st.warning("Please connect to the database and upload a file first.")
        else:
            with st.spinner('Running metric validation...'):
                try:
                    prism_df = st.session_state.db.run_query(get_metric_query())
                    tdt_dfs = parse_metric_excel(st.session_state.uploaded_file_state)
                    summary, matches, mismatches = validate_metric_data(tdt_dfs, prism_df)
                    st.session_state.validation_states["metric_validation"]["results"] = {
                        'summary': summary, 'matches': matches, 'mismatches': mismatches
                    }
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["metric_validation"]["results"], "metric_val", "TDT")

# --- Tab 2: Metric Mapping Validation ---
with tab2:
    st.header("Metric Mapping Validation (TDT vs PRISM Project)")
    if st.button("Run Metric Mapping Validation", key="run_metric_mapping"):
        if not st.session_state.db or not st.session_state.uploaded_file_state:
            st.warning("Please connect to the database and upload a file first.")
        else:
            with st.spinner('Running metric mapping validation...'):
                try:
                    prism_df = st.session_state.db.run_query(get_metric_mapping_query())
                    model_dfs = parse_metric_mapping_excel(st.session_state.uploaded_file_state)
                    summary, matches, mismatches = validate_metric_mapping_data(model_dfs, prism_df)
                    st.session_state.validation_states["metric_mapping"]["results"] = {
                        'summary': summary, 'matches': matches, 'mismatches': mismatches
                    }
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    display_results(st.session_state.validation_states["metric_mapping"]["results"], "metric_map", "Model")

# --- Tab 3: Failure Diagnostics Validation (Placeholder) ---
with tab3:
    st.header("Failure Diagnostics Validation (TDT vs PRISM Template)")
    st.info("This section is under construction.")
    if st.button("Run Failure Diagnostics Validation", key="run_failure_diagnostics", disabled=True):
        pass

# --- Tab 4: Filter Validation (Placeholder) ---
with tab4:
    st.header("Filter Validation (TDT vs PRISM Project)")
    st.info("This section is under construction.")
    if st.button("Run Filter Validation", key="run_filter_validation", disabled=True):
        pass