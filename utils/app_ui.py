import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.title("Navigation")
        st.page_link("main.py", label="Home", icon="🏠", width="stretch")
        st.subheader("Model Development", divider="blue")
        st.page_link("pages/data_cleaning.py", label="Data Cleaning", icon="1️⃣")
        st.page_link("pages/split_holdout_dataset.py", label="Split Holdout Dataset", icon="2️⃣")
        st.page_link("pages/outlier_removal.py", label="Outlier Removal", icon="3️⃣")
        st.page_link("pages/train_validation_split.py", label="Train Validation Split", icon="4️⃣")
        st.subheader("Model Validation", divider="red")
        st.page_link("pages/calculate_accuracy.py", label="Calculate Accuracy", icon="🅰️")
        st.page_link("pages/model_qa.py", label="QA", icon="🅱️")

def get_model_info(auto_site_name="", auto_model_name="", auto_inclusive_dates=""):
    """
    Gets user inputs for site_name, utility_name, model_name, sprint_name, inclusive_dates
    and stores them in session_state
    """
    st.write("Please enter/confirm the Site Name, Utility Name, Sprint Name, and Inclusive Dates of the dataset.")
    site_name = st.text_input("Site Name (e.g., TVI/TSI)", value=auto_site_name)
    utility_name = st.text_input("Utility Name (e.g., BOP)")
    model_name = st.text_input("Model Name", value=auto_model_name)
    sprint_name = st.text_input("Sprint Name (e.g., Sprint_1)")
    inclusive_dates = st.text_input("Inclusive Dates YYYYMMDD (e.g., 20240101-20240601)", value=auto_inclusive_dates)

    st.write("Please make sure all entered data is correct before pressing \"Confirm\"")
    if st.button("Confirm"):
        if site_name == "" or utility_name == "" or model_name == "" or sprint_name == "" or inclusive_dates == "":
            st.error("Please ensure that all fields are filled in.")
        else:
            st.session_state.site_name = site_name
            st.session_state.utility_name = utility_name
            st.session_state.model_name = model_name
            st.session_state.sprint_name = sprint_name
            st.session_state.inclusive_dates = inclusive_dates