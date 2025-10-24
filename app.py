import streamlit as st
from st_pages import get_nav_from_toml

st.set_page_config(page_title="PRISM & Canary", layout="wide")

st.navigation(get_nav_from_toml())

st.title("PRISM & Canary")
st.markdown("Select an application from the sidebar to get started.")
