import streamlit as st

def render_sidebar():
    """
    Renders a custom sidebar navigation menu.
    
    This function should be called in pages that need this specific navigation menu.
    It links to the pages registered in app.py.
    """
    with st.sidebar:
        st.divider()  # Add a separator from the main app navigation
        st.subheader("Quick Access")
        
        # FIX 1: "main.py" does not exist in st.navigation. 
        # Changed to "pages/Home.py" which is registered as "Home" in app.py
        st.page_link("pages/Home.py", label="Home", icon="🏠")
        
        st.caption("Model Development")
        
        # FIX 2: Updated all filenames below to match the actual files in your 'pages/' folder
        # and the definitions in your app.py
        
        # Was "pages/data_cleaning.py"
        st.page_link("pages/2_Data_Cleansing.py", label="Data Cleansing", icon="1️⃣")
        
        # Was "pages/split_holdout_dataset.py"
        st.page_link("pages/3_Holdout_Splitting.py", label="Holdout Splitting", icon="2️⃣")
        
        # Was "pages/outlier_removal.py"
        st.page_link("pages/7_Outlier_Removal.py", label="Outlier Removal", icon="3️⃣")
        
        # Was "pages/train_validation_split.py"
        st.page_link("pages/4_Training_Validation_Splitting.py", label="Train Validation Split", icon="4️⃣")
        
        st.caption("Model Validation")
        
        # Was "pages/calculate_accuracy.py"
        st.page_link("pages/5_Model_Accuracy.py", label="Calculate Accuracy", icon="🅰️")
        
        # Was "pages/model_qa.py"
        st.page_link("pages/6_Model_FPR.py", label="QA", icon="🅱️")

def get_model_info(auto_site_name="", auto_model_name="", auto_inclusive_dates=""):
    """
    Gets user inputs for site_name, system_name, model_name, sprint_name, inclusive_dates
    and stores them in session_state
    """
    st.write("Please enter/confirm the Site Name, System Name, Sprint Name, and Inclusive Dates of the dataset.")

    site_name = st.text_input("Site Name", value=auto_site_name, help="e.g., TVI/TSI")
    system_name = st.text_input("System Name", help="e.g., BOP")
    model_name = st.text_input("Model Name", value=auto_model_name)
    sprint_name = st.text_input("Sprint Name", help="e.g., Sprint_1")
    inclusive_dates = st.text_input("Inclusive Dates YYYYMMDD", value=auto_inclusive_dates, help="e.g., 20240101-20240601")

    st.write("Please make sure all entered data is correct before pressing \"Confirm\"")
    if st.button("Confirm"):
        if site_name == "" or system_name == "" or model_name == "" or sprint_name == "" or inclusive_dates == "":
            st.error("Please ensure that all fields are filled in.")
        else:
            st.session_state.site_name = site_name
            st.session_state.system_name = system_name
            st.session_state.model_name = model_name
            st.session_state.sprint_name = sprint_name
            st.session_state.inclusive_dates = inclusive_dates