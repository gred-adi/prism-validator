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
    FORM.ProjectID AS [FORM ID],
    FORM.Name AS [FORM NAME],
    FORM_METRIC.PointTypeMetricID AS [METRIC ID],
    FORM_METRIC.Description AS [METRIC NAME],
    CASE
        WHEN FORM_POINTS.ConstrainedPt = 1 THEN 'OPERATIONAL STATE'
        WHEN COUNT(FAULT_DETAIL.UpDownValue) >= 1 THEN 'FAULT DETECTION'
        ELSE 'NON-MODELED'
    END AS [FUNCTION],
    CASE
        WHEN FORM_POINTS_SYS.DigitalGroupID IS NOT NULL THEN 'DIGITAL'
        WHEN FORM_POINTS_CALC.PointCalcID IS NOT NULL THEN 'PRISM CALC'
        ELSE 'ANALOG'
    END AS [POINT TYPE],
    'Input Signal' AS [THRESHOLD TYPE] -- This is fixed by the WHERE clause
FROM
    prismdb.dbo.Projects FORM
    LEFT JOIN prismdb.dbo.Assets ASSET ON FORM.AssetID = ASSET.AssetID
    LEFT JOIN prismdb.dbo.Projects PARENT ON FORM.ParentTemplateID = PARENT.ProjectID
    LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS ON FORM.ProjectID = FORM_POINTS.ProjectID AND FORM_POINTS.PointTypeID IN (1, 2)
    LEFT JOIN prismdb.dbo.PointTypeMetric FORM_METRIC ON FORM_POINTS.PointTypeMetricID = FORM_METRIC.PointTypeMetricID
    LEFT JOIN prismdb.dbo.SystemPoints FORM_POINTS_SYS ON FORM_POINTS.SystemPointId = FORM_POINTS_SYS.Id
    LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS_DETAIL ON FORM_POINTS.ProjectID = FORM_POINTS_DETAIL.ProjectID
        AND FORM_POINTS.OrderIndex = FORM_POINTS_DETAIL.OrderIndex
        AND ((FORM_POINTS.PointTypeID = 1 AND FORM_POINTS_DETAIL.PointTypeID <> 2) OR (FORM_POINTS.PointTypeID = 2 AND FORM_POINTS_DETAIL.PointTypeID = 2))
    LEFT JOIN prismdb.dbo.PointCalc FORM_POINTS_CALC ON FORM_POINTS.ProjectPointID = FORM_POINTS_CALC.ProjectPointID
    LEFT JOIN prismdb.dbo.FaultDiagnostic FAULT ON (FORM.PROJECTTYPEID = 2 AND FORM.ProjectID = FAULT.TemplateID)
    LEFT JOIN prismdb.dbo.FaultSignatureDev FAULT_DETAIL ON FAULT.FaultDiagnosticID = FAULT_DETAIL.FaultDiagnosticID AND FORM_POINTS.PointTypeMetricID = FAULT_DETAIL.PointTypeMetricID
WHERE
    FORM.PROJECTTYPEID = 2 -- This corresponds to [FORM TYPE] = 'TEMPLATE'
    AND FORM_POINTS_DETAIL.PointTypeID = 1 -- This corresponds to [THRESHOLD TYPE] = 'Input Signal'
    AND FORM.Name LIKE '%TVI%'
GROUP BY
    FORM.ProjectID,
    FORM.Name,
    FORM_METRIC.PointTypeMetricID,
    FORM_METRIC.Description,
    FORM_POINTS.ConstrainedPt,
    FORM_POINTS_SYS.DigitalGroupID,
    FORM_POINTS_CALC.PointCalcID
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

