import streamlit as st
import pandas as pd
from db_utils import PrismDB  # Use the new PrismDB class
from excel_utils import process_excel_reference
from validation_utils import validate_data, generate_pdf_report # Re-added PDF for completeness

# --- Page Configuration ---
st.set_page_config(page_title="PRISM Config Validator", layout="wide")

# --- Initialize Session State ---
if 'db' not in st.session_state:
    st.session_state.db = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'uploaded_file_state' not in st.session_state:
    st.session_state.uploaded_file_state = None

# --- Sidebar UI ---
with st.sidebar:
    st.header("Database Connection")

    db_host = st.text_input("Host", value=st.secrets.get("db", {}).get("host", ""))
    db_name = st.text_input("Database", value=st.secrets.get("db", {}).get("database", ""))
    db_user = st.text_input("User", value=st.secrets.get("db", {}).get("user", ""))
    db_pass = st.text_input("Password", type="password", value=st.secrets.get("db", {}).get("password", ""))

    if st.button("Connect to Database"):
        with st.spinner("Connecting..."):
            try:
                # Instantiate the PrismDB class and store it in session state
                st.session_state.db = PrismDB(db_host, db_name, db_user, db_pass)
                # Test the connection
                st.session_state.db.test_connection()
                st.success("✅ Connection successful!")
            except Exception as e:
                st.session_state.db = None
                st.error(f"❌ Connection failed: {e}")

    # Display connection status based on the session state object
    if st.session_state.db:
        st.success("Database is connected.")
    else:
        st.warning("Database is not connected.")

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload Excel Reference", type=["xlsx"])
    if uploaded_file:
        st.session_state.uploaded_file_state = uploaded_file

# --- Main Page UI ---
st.title("PRISM Configuration Validator")
st.markdown("This tool validates PRISM configurations against a reference Excel file.")

# Define the SQL query
sql_query = """ 
SELECT
    x.[FORM ID], x.[FORM NAME], x.[METRIC ID], x.[METRIC NAME],
    x.[FUNCTION], x.[POINT TYPE], x.[THRESHOLD TYPE]
FROM dbo.SINE_TDT_REV x
WHERE x.[FORM TYPE] = 'TEMPLATE'
    AND x.[THRESHOLD TYPE] = 'Input Signal'
    AND x.[FORM NAME] LIKE '%TVI%'
"""

if st.button("Run Validation", type="primary"):
    if not st.session_state.db:
        st.error("Please connect to the database first.")
    elif st.session_state.uploaded_file_state is None:
        st.error("Please upload the Excel reference file first.")
    else:
        with st.spinner('Running validation... This may take a moment.'):
            try:
                # Use the connection object from session state to run the query
                prism_df = st.session_state.db.run_query(sql_query)
                tdt_dfs = process_excel_reference(st.session_state.uploaded_file_state)
                
                summary_df, matches_df, mismatches_df = validate_data(tdt_dfs, prism_df)
                
                st.session_state.validation_results = {
                    'summary': summary_df,
                    'matches': matches_df,
                    'mismatches': mismatches_df
                }
                # No need for rerun(), Streamlit's flow handles the update
            except Exception as e:
                st.error(f"An error occurred during validation: {e}")

# --- Display Validation Results ---
if st.session_state.validation_results:
    results = st.session_state.validation_results
    
    st.subheader("Validation Summary")
    st.dataframe(results['summary'], use_container_width=True)

    st.subheader("Validation Details")
    
    tdt_list = ['All'] + results['summary']['TDT'].tolist()

    st.markdown("#### Matches")
    match_filter = st.selectbox("Filter by TDT (Matches)", options=tdt_list, key="match_filter")
    
    matches_to_show = results['matches']
    if match_filter != 'All':
        matches_to_show = results['matches'][results['matches']['TDT'] == match_filter]
    
    st.dataframe(matches_to_show, use_container_width=True)
    st.metric("Total Matches", len(results['matches']))

    st.markdown("#### Mismatches")
    mismatch_filter = st.selectbox("Filter by TDT (Mismatches)", options=tdt_list, key="mismatch_filter")
    
    mismatches_to_show = results['mismatches']
    if mismatch_filter != 'All':
        mismatches_to_show = results['mismatches'][results['mismatches']['TDT'] == mismatch_filter]

    st.dataframe(mismatches_to_show, use_container_width=True)
    st.metric("Total Mismatches", len(results['mismatches']))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        csv_data = results['mismatches'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Mismatches as CSV",
            data=csv_data,
            file_name="mismatch_report.csv",
            mime="text/csv",
            key='download-csv'
        )
    with col2:
        st.button("📄 Generate PDF Report", disabled=True)

