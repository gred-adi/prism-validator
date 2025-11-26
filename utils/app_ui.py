import streamlit as st

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