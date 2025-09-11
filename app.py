import streamlit as st
from db_utils import PrismDB

# --- Import validation-specific modules ---
# This new structure makes it clear which logic belongs to which section
from validations.metric_validation.query import get_query as get_metric_query
from validations.metric_validation.parser import parse_excel as parse_metric_excel
from validations.metric_validation.validator import validate_data as validate_metric_data

# (You will add imports for the other sections here as you build them)
# from validations.metric_mapping_validation.query import get_query as get_metric_mapping_query
# ... and so on

# --- Page Configuration ---
st.set_page_config(page_title="PRISM Config Validator", layout="wide")

# --- Initialize Session State ---
# We'll use a dictionary to hold the state for each validation type
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
def display_results(results):
    """Helper function to display the validation results UI."""
    if not results:
        st.info("Run the validation to see the results.")
        return

    st.subheader("Validation Summary")
    st.dataframe(results['summary'], use_container_width=True)
    st.subheader("Validation Details")
    tdt_list = ['All'] + results['summary']['TDT'].tolist()

    match_filter = st.selectbox("Filter by TDT", options=tdt_list, key=f"match_filter_{results['summary']['TDT'].iloc[0]}")

    st.markdown("#### Matches")
    matches_to_show = results['matches']
    if match_filter != 'All':
        matches_to_show = results['matches'][results['matches']['TDT'] == match_filter]
    st.dataframe(matches_to_show, use_container_width=True)
    st.metric("Total Matches", len(matches_to_show))

    st.markdown("#### Mismatches")
    mismatches_to_show = results['mismatches']
    if match_filter != 'All':
        mismatches_to_show = results['mismatches'][results['mismatches']['TDT'] == match_filter]
    st.dataframe(mismatches_to_show, use_container_width=True)
    st.metric("Total Mismatches", len(mismatches_to_show))

# --- Sidebar UI (Remains mostly the same) ---
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

# --- Create Tabs ---
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
        if not st.session_state.db:
            st.error("Please connect to the database first.")
        elif st.session_state.uploaded_file_state is None:
            st.error("Please upload the Excel reference file first.")
        else:
            with st.spinner('Running metric validation...'):
                try:
                    prism_df = st.session_state.db.run_query(get_metric_query())
                    tdt_dfs = parse_metric_excel(st.session_state.uploaded_file_state)
                    summary_df, matches_df, mismatches_df = validate_metric_data(tdt_dfs, prism_df)
                    st.session_state.validation_states["metric_validation"]["results"] = {
                        'summary': summary_df, 'matches': matches_df, 'mismatches': mismatches_df
                    }
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Display results for this tab
    display_results(st.session_state.validation_states["metric_validation"]["results"])


# --- Tab 2: Metric Mapping Validation (Placeholder) ---
with tab2:
    st.header("Metric Mapping Validation (TDT vs PRISM Project)")
    st.info("This section is under construction.")
    if st.button("Run Metric Mapping Validation", key="run_metric_mapping", disabled=True):
        # This is where you would call your metric_mapping_validation logic
        pass
    # display_results(...) for this section


# --- Tab 3: Failure Diagnostics Validation (Placeholder) ---
with tab3:
    st.header("Failure Diagnostics Validation (TDT vs PRISM Template)")
    st.info("This section is under construction.")
    if st.button("Run Failure Diagnostics Validation", key="run_failure_diagnostics", disabled=True):
        pass
    # display_results(...) for this section


# --- Tab 4: Filter Validation (Placeholder) ---
with tab4:
    st.header("Filter Validation (TDT vs PRISM Project)")
    st.info("This section is under construction.")
    if st.button("Run Filter Validation", key="run_filter_validation", disabled=True):
        pass
    # display_results(...) for this section