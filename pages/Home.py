import streamlit as st
import os
from file_generator import generate_files_from_folder, convert_dfs_to_excel_bytes

st.set_page_config(page_title="Home", layout="wide")

st.title("Home")

st.markdown("Welcome to the PRISM Config Validator and Canary Historian Downloader!")

# --- Shared TDT Folder Selection ---
st.header("1. Generate Files from TDT Folder")

uploaded_files = st.file_uploader(
    "Upload TDT Excel files",
    type=["xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Generate & Load Files"):
        with st.spinner("Generating reference files..."):
            try:
                # Generate DFs
                s_df, d_df = generate_files_from_folder(uploaded_files)
                st.session_state.survey_df = s_df
                st.session_state.diag_df = d_df
                # Create overview
                st.session_state.overview_df = s_df[['TDT', 'Model']].drop_duplicates().sort_values(by=['TDT', 'Model']).reset_index(drop=True)
                # Convert to bytes for parsers
                survey_bytes, diag_bytes = convert_dfs_to_excel_bytes(s_df, d_df)
                st.session_state.uploaded_survey_file = survey_bytes
                st.session_state.uploaded_diag_file = diag_bytes
                st.success("Files generated successfully!")
            except Exception as e: st.error(f"File generation failed: {e}")
