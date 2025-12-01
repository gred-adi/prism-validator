"""
Main entry point for the Streamlit application.

This script sets up the multipage navigation for the PRISM Configuration Validator tool
using Streamlit's `st.navigation` feature. It defines the pages and
their corresponding icons, titles, and script paths.

It also hosts the global Sidebar logic for generating TDT reference files,
ensuring that `survey_df` and `diag_df` are available to all modules.
"""
import streamlit as st
import os
from file_generator import generate_files_from_uploads, convert_dfs_to_excel_bytes

# --- Global Sidebar Logic ---
with st.sidebar:
    st.title("Global Settings")

    # --- 1. Root Folder Configuration ---
    st.header("📂 Output Configuration")
    
    # Initialize base_path in session state if not present
    if 'base_path' not in st.session_state:
        st.session_state.base_path = os.getcwd()

    # Text input for root folder
    st.session_state.base_path = st.text_input(
        "Root Output Folder", 
        value=st.session_state.base_path,
        help="All generated datasets and reports will be saved within this folder."
    )
    
    # Optional: Display current status
    if os.path.exists(st.session_state.base_path):
        st.caption(f"✅ Path exists")
    else:
        st.caption(f"⚠️ Path does not exist (will be created)")

    st.divider()

    # --- 2. TDT File Upload ---
    st.header("📤 Upload TDT Files")
    st.info("Upload your TDT files here to make the data available to all modules in the toolkit.")

    uploaded_files = st.file_uploader(
        "Upload TDT Excel files",
        type=["xlsx"],
        accept_multiple_files=True,
        key="sidebar_tdt_uploader"
    )

    if uploaded_files:
        if st.button("Generate & Load Files", key="sidebar_generate_btn"):
            with st.spinner("Generating reference files..."):
                try:
                    # Generate DFs
                    s_df, d_df = generate_files_from_uploads(uploaded_files)
                    st.session_state.survey_df = s_df
                    st.session_state.diag_df = d_df
                    
                    # Create overview
                    st.session_state.overview_df = s_df[['TDT', 'Model']].drop_duplicates().sort_values(by=['TDT', 'Model']).reset_index(drop=True)
                    
                    # Convert to bytes for parsers/downloads
                    survey_bytes, diag_bytes = convert_dfs_to_excel_bytes(s_df, d_df)
                    st.session_state.uploaded_survey_file = survey_bytes
                    st.session_state.uploaded_diag_file = diag_bytes
                    
                    st.success(f"Successfully loaded {len(st.session_state.overview_df)} models!")
                except Exception as e:
                    st.error(f"File generation failed: {e}")

    # Show status if loaded
    if st.session_state.get('survey_df') is not None:
        st.divider()
        num_tdts = st.session_state.overview_df['TDT'].nunique()
        num_models = len(st.session_state.overview_df)
        st.success(f"✅ TDT Data Loaded\n\n**TDTs:** {num_tdts}\n\n**Models:** {num_models}")
    
    st.markdown("---")

# --- Navigation Setup ---
pages = {
    "Home": [st.Page("pages/Home.py", title="Home", icon="🏠")],
    "TDT Config Validator": [st.Page("pages/3_TDT_Validator.py", title="TDT Config Validator", icon="☑️")],
    "PRISM Config Validator": [st.Page("pages/1_PRISM_Config_Validator.py", title="PRISM Config Validator", icon="✅")],
    "Canary Historian Downloader": [st.Page("pages/2_Canary_Historian_Downloader.py", title="Canary Historian Downloader", icon="⬇️")],
    "Model Development Tools": [
        st.Page("pages/2_Data_Cleansing.py", title="Data Cleansing", icon="1️⃣"),
        st.Page("pages/3_Holdout_Splitting.py", title="Holdout Splitting", icon="2️⃣"),
        st.Page("pages/7_Outlier_Removal.py", title="Outlier Removal", icon="3️⃣"),
        st.Page("pages/4_Training_Validation_Splitting.py", title="Training-Validation Splitting", icon="4️⃣"),
    ],
    "Model Validation Tools": [
        st.Page("pages/5_Model_Accuracy.py", title="Model Accuracy", icon="🎯"),
        st.Page("pages/6_Model_FPR.py", title="Model FPR", icon="🔎"),
    ],
}

pg = st.navigation(pages)
pg.run()