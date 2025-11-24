"""
Main entry point for the Streamlit application.

This script sets up the multipage navigation for the PRISM Configuration Validator tool
using Streamlit's experimental `st.navigation` feature. It defines the pages and
their corresponding icons, titles, and script paths.
"""
import streamlit as st

pages = {
    "Home": [st.Page("pages/Home.py", title="Home", icon="🏠")],
    "TDT Validator": [st.Page("pages/3_TDT_Validator.py", title="TDT Validator", icon="☑️")],
    "PRISM Config Validator": [st.Page("pages/1_PRISM_Config_Validator.py", title="PRISM Config Validator", icon="✅")],
    "Canary Historian Downloader": [st.Page("pages/2_Canary_Historian_Downloader.py", title="Canary Historian Downloader", icon="⬇️")],
    "Model Development Tools": [
        st.Page("pages/2_Data_Cleansing.py", title="Data Cleansing", icon="🧹"),
        st.Page("pages/3_Holdout_Splitting.py", title="Holdout Splitting", icon="🔪"),
        st.Page("pages/7_Outlier_Removal.py", title="Outlier Removal", icon="🗑️"),
        st.Page("pages/4_Training_Validation_Splitting.py", title="Training-Validation Splitting", icon="🔬"),
    ],
    "Model Validation Tools": [
        st.Page("pages/5_Model_Accuracy.py", title="Model Accuracy", icon="🎯"),
        st.Page("pages/6_Model_FPR.py", title="Model FPR", icon="🔎"),
    ],
}

pg = st.navigation(pages)
pg.run()