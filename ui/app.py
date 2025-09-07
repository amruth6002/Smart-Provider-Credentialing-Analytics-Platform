import os, sys

# Ensure project root is importable (parent of this file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
from src.engine import ProviderDQEngine
from src.nlu import parse_intent

# Only set page config if it hasn't been set already (to avoid conflicts with dashboard.py)
try:
    st.set_page_config(page_title="Provider Data Quality Chatbot", layout="wide")
except st.errors.StreamlitAPIException:
    # Page config has already been set, skip
    pass
st.title("Provider Data Quality Analytics & Chatbot")

if "engine" not in st.session_state:
    st.session_state.engine = ProviderDQEngine()
if "loaded" not in st.session_state:
    st.session_state.loaded = False

def save_temp(uploaded):
    if not uploaded:
        return None
    os.makedirs("./tmp", exist_ok=True)
    path = os.path.abspath(os.path.join("tmp", uploaded.name))
    with open(path, "wb") as f:
        f.write(uploaded.read())
    return path

with st.sidebar:
    st.header("Load Data")

    # Option A: Auto-load from local datasets folder
    st.subheader("Option A: Auto-load from datasets folder")
    default_dir = os.path.join(ROOT_DIR, "datasets", "Credentialing use case data")
    st.caption(f"Folder: {default_dir}")
    if st.button("Load from datasets folder"):
        roster_path = os.path.join(default_dir, "provider_roster_with_errors.csv")
        ny_path = os.path.join(default_dir, "ny_medical_license_database.csv")
        ca_path = os.path.join(default_dir, "ca_medical_license_database.csv")
        npi_path = os.path.join(default_dir, "mock_npi_registry.csv")

        missing = [p for p in [roster_path, ny_path, ca_path, npi_path] if not os.path.exists(p)]
        if missing:
            st.error("Missing files:\n" + "\n".join(missing))
        else:
            st.session_state.engine.load_files(roster_path, ny_path, ca_path, npi_path)
            st.session_state.loaded = True
            st.success("Data loaded from datasets folder.")

    st.markdown("---")

    # Option B: Upload files manually
    st.subheader("Option B: Upload CSVs")
    roster = st.file_uploader("Provider Roster (CSV)", type=["csv"], key="roster_upl")
    ny = st.file_uploader("NY License DB (CSV)", type=["csv"], key="ny_upl")
    ca = st.file_uploader("CA License DB (CSV)", type=["csv"], key="ca_upl")
    npi = st.file_uploader("Mock NPI Registry (CSV)", type=["csv"], key="npi_upl")
    if st.button("Load uploaded files"):
        if roster is None:
            st.error("Please upload the roster file.")
        else:
            r_path = save_temp(roster)
            ny_path = save_temp(ny)
            ca_path = save_temp(ca)
            npi_path = save_temp(npi)
            st.session_state.engine.load_files(r_path, ny_path, ca_path, npi_path)
            st.session_state.loaded = True
            st.success("Data loaded from uploads.")

st.markdown("Ask questions like:")
st.markdown("- How many providers have expired licenses in our network?")
st.markdown("- Show me all providers with phone number formatting issues")
st.markdown("- Which providers are missing NPI numbers?")
st.markdown("- Find potential duplicate provider records")
st.markdown("- What is our overall provider data quality score?")
st.markdown("- Show me a summary of all data quality problems by state")
st.markdown("- Generate a compliance report for expired licenses")
st.markdown("- Filter providers by license expiration date (next 60 days)")
st.markdown("- Show providers practicing in multiple states with single licenses")
st.markdown("- Export a list of providers requiring credential updates")

query = st.text_input("Your question", "")
run_clicked = st.button("Run")
if run_clicked and query:
    if not st.session_state.loaded:
        st.error("Please load data first (sidebar).")
    else:
        intent, params = parse_intent(query)
        st.write(f"Intent: {intent}  Params: {params}")
        res = st.session_state.engine.run_query(intent, params)
        if isinstance(res, pd.DataFrame):
            st.dataframe(res, use_container_width=True)
            csv = res.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name=f"{intent}.csv", mime="text/csv")
        else:
            # Scalar metric result
            st.metric("Result", res)

# Quick stats
if st.session_state.loaded:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expired Licenses", st.session_state.engine.count_expired())
    with col2:
        st.metric("Overall DQ Score", round(st.session_state.engine.get_quality_score(), 2))
    with col3:
        st.metric("Missing NPI", int(st.session_state.engine.list_missing_npi().shape[0]))
    with col4:
        st.metric("Phone Issues", int(st.session_state.engine.list_phone_issues().shape[0]))