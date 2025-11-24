import streamlit as st

def render_sidebar():
    """
    Renders a custom sidebar navigation menu.
    
    Note: Since app.py uses st.navigation(), this creates a secondary menu 
    inside the sidebar. If you only want one menu, you can remove the calls 
    to render_sidebar() in your pages.
    """
    with st.sidebar:
        st.divider()  # Add a separator from the main app navigation
        st.subheader("Quick Access")
        
        # Fixed: Link to the actual file path defined in app.py
        st.page_link("pages/Home.py", label="Home", icon="🏠")
        
        st.caption("Model Development")
        # Fixed: Updated paths and icons to match app.py definitions
        st.page_link("pages/2_Data_Cleansing.py", label="Data Cleansing", icon="🧹")
        st.page_link("pages/3_Holdout_Splitting.py", label="Holdout Splitting", icon="🔪")
        st.page_link("pages/7_Outlier_Removal.py", label="Outlier Removal", icon="🗑️")
        st.page_link("pages/4_Training_Validation_Splitting.py", label="Training-Validation Splitting", icon="🔬")
        
        st.caption("Model Validation")
        # Fixed: Updated paths and icons to match app.py definitions
        st.page_link("pages/5_Model_Accuracy.py", label="Model Accuracy", icon="🎯")
        st.page_link("pages/6_Model_FPR.py", label="Model FPR", icon="🔎")

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