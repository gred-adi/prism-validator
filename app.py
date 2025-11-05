import streamlit as st

pages = {
    "Home": [st.Page("pages/Home.py", title="Home", icon="🏠")],
    "Apps": [
        st.Page("pages/3_TDT_Validator.py", title="TDT Validator", icon="☑️"),
        st.Page("pages/1_PRISM_Config_Validator.py", title="PRISM Config Validator", icon="✅"),
        st.Page("pages/2_Canary_Historian_Downloader.py", title="Canary Historian Downloader", icon="⬇️"),
    ],
}

pg = st.navigation(pages)
pg.run()