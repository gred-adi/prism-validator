"""
This script defines the Home page of the Streamlit application.

The Home page serves as the landing page/dashboard, providing an overview
of the available modules and their purposes.
"""
import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

st.title("PRISM Dev/QA Toolkit")

st.markdown("""
Welcome to the **PRISM Development & QA Toolkit**. This application centralizes the workflows for validating Technical Design Templates (TDTs), auditing live PRISM configurations, and preparing data for model development.

### 👈 Getting Started
To begin, please **upload your TDT Excel files in the Global Settings Sidebar** on the left.
* This sidebar is accessible from **any page**.
* Uploading files there generates the reference data (`survey_df`) required by most modules below.
* You only need to upload them once per session.
""")

st.divider()

st.header("Available Modules")

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.subheader("🔍 Validation Tools")
    with st.container(border=True):
        st.markdown("#### ✅ PRISM Config Validator")
        st.markdown("Compares your offline **TDT files** against the live **PRISM Database**. Identifies discrepancies in metric configuration, mappings, filters, and diagnostics.")
    
    with st.container(border=True):
        st.markdown("#### ☑️ TDT Validator")
        st.markdown("Performs offline integrity checks on the TDT files themselves. Detects duplicates, missing mandatory fields, and logic errors within the Excel templates before you deploy.")

    with st.container(border=True):
        st.markdown("#### 🎯 Model Accuracy")
        st.markdown("Post-deployment validation tool to calculate model accuracy scores by comparing PRISM metrics against calculated values.")

    with st.container(border=True):
        st.markdown("#### 🔎 Model FPR")
        st.markdown("Generates False Positive Rate (FPR) reports for QA, analyzing model behavior against holdout data.")

with col2:
    st.subheader("🛠️ Development Tools")
    with st.container(border=True):
        st.markdown("#### ⬇️ Canary Historian Downloader")
        st.markdown("Automatically fetches historical time-series data from the Canary Historian using the tag definitions found in your TDTs.")

    with st.container(border=True):
        st.markdown("#### 1️⃣ Data Cleansing")
        st.markdown("Interactive wizard to clean raw datasets. Apply numeric and datetime filters to remove bad data or shutdown periods.")

    with st.container(border=True):
        st.markdown("#### 2️⃣ Holdout Splitting")
        st.markdown("Splits your cleaned dataset into **Training/Validation** and **Holdout** sets based on a time horizon.")

    with st.container(border=True):
        st.markdown("#### 4️⃣ Train-Val Splitting")
        st.markdown("Performs stratified splitting of your training data to ensure balanced coverage of operational states.")

st.markdown("---")
st.caption("PRISM Dev/QA Toolkit | v1.0")