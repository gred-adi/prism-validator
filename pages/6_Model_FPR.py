import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import io

# Utils
from utils.model_dev_utils import scan_folders_structure
from utils.constraint_utils import fetch_model_constraints
from utils.qa_plotting import (
    plot_omr_and_constraint_dynamic, 
    plot_omr_heatmap,
    generate_fpr_plots_memory,
    downsample_data
)
from utils.qa_data_processing import load_and_process_data
from utils.data_preparation import prepare_omr_data
from db_utils import PrismDB  # <--- Added Import

st.set_page_config(page_title="Model FPR", page_icon="🅱️", layout="wide")

# --- Session State ---
if 'fpr_step' not in st.session_state: st.session_state.fpr_step = 1
if 'fpr_scanned_df' not in st.session_state: st.session_state.fpr_scanned_df = None
if 'fpr_selected_model_row' not in st.session_state: st.session_state.fpr_selected_model_row = None
if 'fpr_constraints_df' not in st.session_state: st.session_state.fpr_constraints_df = None
if 'db' not in st.session_state: st.session_state.db = None # Reuse global db if available

# --- Sidebar: Database Connection ---
with st.sidebar:
    st.header("🌍 Database Connection")
    st.caption("Required to fetch active model constraints/filters.")
    
    # Use secrets if available, else empty
    secrets_db = st.secrets.get("db", {})
    db_host = st.text_input("Host", value=secrets_db.get("host", ""))
    db_name = st.text_input("Database", value=secrets_db.get("database", ""))
    db_user = st.text_input("User", value=secrets_db.get("user", ""))
    db_pass = st.text_input("Password", type="password", value=secrets_db.get("password", ""))
    
    if st.button("Connect to Database"):
        with st.spinner("Connecting..."):
            try:
                st.session_state.db = PrismDB(db_host, db_name, db_user, db_pass)
                st.session_state.db.test_connection()
                st.success("✅ Connection successful!")
            except Exception as e:
                st.session_state.db = None
                st.error(f"❌ Connection failed: {e}")
                
    if st.session_state.db:
        st.success("Database Connected")
    else:
        st.warning("Database Disconnected")

# --- Navigation ---
def next_step(): st.session_state.fpr_step += 1
def prev_step(): st.session_state.fpr_step -= 1

st.title("🔎 Model FPR & QA Wizard")

steps = ["1. Select Model", "2. Configure Constraints", "3. Generate Report"]
current = st.session_state.fpr_step
st.progress(current / len(steps), text=f"Step {current}: {steps[current-1]}")

# ==========================================
# STEP 1: SCAN & SELECT
# ==========================================
if current == 1:
    st.header("Step 1: Select Model")
    
    # Folder Path
    default_path = st.session_state.get('base_path', Path.cwd())
    root_folder = st.text_input("Root Folder Path", value=default_path)
    
    if st.button("🔍 Scan Models"):
        with st.spinner("Scanning..."):
            df = scan_folders_structure(root_folder)
            st.session_state.fpr_scanned_df = df
            
    if st.session_state.fpr_scanned_df is not None and not st.session_state.fpr_scanned_df.empty:
        df = st.session_state.fpr_scanned_df.copy()
        
        # Filters
        c1, c2, c3 = st.columns(3)
        with c1: 
            site_sel = st.multiselect("Site", df['Site'].unique())
        with c2: 
            sys_sel = st.multiselect("System", df['System'].unique())
        with c3:
            sprint_sel = st.multiselect("Sprint", df['Sprint'].unique())
            
        if site_sel: df = df[df['Site'].isin(site_sel)]
        if sys_sel: df = df[df['System'].isin(sys_sel)]
        if sprint_sel: df = df[df['Sprint'].isin(sprint_sel)]
        
        # Display logic: Only allow selecting ONE model for deep dive
        st.dataframe(
            df[['Site', 'System', 'Sprint', 'Model', 'Dataset Found']],
            use_container_width=True,
            hide_index=True
        )
        
        # Only show models where dataset was actually found
        valid_models_df = df[df['Dataset Found']]
        
        if valid_models_df.empty:
            st.warning("No models with valid 'dataset' folders found in selection.")
        else:
            model_names = valid_models_df['Model'].unique()
            selected_model_name = st.selectbox("Select Model to Process", model_names)
            
            if st.button("Proceed with Selected Model ➡️"):
                # Store the full row for the selected model
                row = valid_models_df[valid_models_df['Model'] == selected_model_name].iloc[0]
                st.session_state.fpr_selected_model_row = row
                next_step()
                st.rerun()

# ==========================================
# STEP 2: CONSTRAINTS
# ==========================================
elif current == 2:
    st.header("Step 2: Configure Constraints")
    model_row = st.session_state.fpr_selected_model_row
    model_name = model_row['Model']
    
    st.info(f"Configuring **{model_name}**. We will fetch active filters from PRISM to use as constraints.")
    
    # DB Connection Check
    if st.session_state.db is None:
        st.warning("⚠️ Database not connected. Cannot fetch constraints automatically.")
        st.markdown("Please connect in the sidebar, or manually add constraints below.")
    
    # Fetch Logic
    # We check if we need to fetch (either it's None, or we want to allow refetch)
    # Using a button for fetch gives user control
    if st.button("Fetch Constraints from DB") or st.session_state.fpr_constraints_df is None:
        if st.session_state.db:
            with st.spinner("Fetching constraints..."):
                constraints = fetch_model_constraints(st.session_state.db, model_name)
                st.session_state.fpr_constraints_df = constraints
        else:
            # Empty frame structure if no DB or first run
            if st.session_state.fpr_constraints_df is None:
                st.session_state.fpr_constraints_df = pd.DataFrame(columns=["Point Name", "Operator", "Value"])

    # Editable Table
    st.markdown("### Active Constraints (Model OFF conditions)")
    st.caption("Rows below define when the model is considered 'OFF'. You can add/edit rows here.")
    
    edited_constraints = st.data_editor(
        st.session_state.fpr_constraints_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Operator": st.column_config.SelectboxColumn("Operator", options=["<", ">", "==", "<=", ">="])
        }
    )
    st.session_state.fpr_constraints_df = edited_constraints
    
    col_b, col_n = st.columns([1, 5])
    with col_b: st.button("⬅️ Back", on_click=prev_step)
    with col_n: 
        if st.button("Confirm & Next ➡️", type="primary"):
            next_step()
            st.rerun()

# ==========================================
# STEP 3: GENERATE
# ==========================================
elif current == 3:
    st.header("Step 3: Generate FPR Report")
    model_row = st.session_state.fpr_selected_model_row
    constraints = st.session_state.fpr_constraints_df
    
    # UI for Plot Selection
    st.subheader("Report Configuration")
    c1, c2 = st.columns(2)
    with c1:
        inc_ts = st.checkbox("Include Time Series Plots", value=True)
        inc_dist = st.checkbox("Include Distribution Plots", value=True)
    with c2:
        inc_heat = st.checkbox("Include OMR Heatmap", value=True, help="Visualizes OMR intensity by Day of Week and Hour.")
        inc_fpr = st.checkbox("Include FPR Curve", value=True)

    if st.button("🚀 Generate Report", type="primary"):
        with st.spinner("Loading data and generating plots..."):
            try:
                # 1. Load Data (Using paths from Step 1 scan)
                dataset_path = Path(model_row['Dataset Path'])
                
                # Locate files safely
                raw_candidates = list(dataset_path.glob("*RAW.csv"))
                holdout_candidates = list(dataset_path.glob("*HOLDOUT.csv"))
                omr_candidates = list(dataset_path.glob("*OMR*WITHOUT-OUTLIER*")) # Flexible search for OMR
                
                if not raw_candidates or not holdout_candidates:
                    st.error("Missing required RAW or HOLDOUT CSV files in dataset folder.")
                    st.stop()
                    
                raw_file = raw_candidates[0]
                holdout_file = holdout_candidates[0]
                
                # Load Raw Data (for constraints check)
                constraint_cols = constraints['Point Name'].tolist() if not constraints.empty else []
                
                # Use load_and_process utility
                raw_df = load_and_process_data(str(raw_file), constraint_cols)
                
                # Load OMR Data
                if not omr_candidates:
                    st.error("Could not find OMR Validation file (e.g. OMR-VALIDATION-WITHOUT-OUTLIER).")
                    st.stop()
                    
                omr_cleaned = prepare_omr_data(str(omr_candidates[0]))
                
                # 2. Generate Plots (In-Memory for preview)
                st.subheader("Preview: Cleaned Validation OMR")
                
                # Downsample for UI
                raw_small = downsample_data(raw_df)
                omr_small = downsample_data(omr_cleaned)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                plot_omr_and_constraint_dynamic(
                    raw_small, omr_small, constraints, 
                    ax=ax, timestamp_col="timestamp"
                )
                st.pyplot(fig)
                
                if inc_heat:
                    st.subheader("Preview: Heatmap")
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    plot_omr_heatmap(omr_cleaned, ax=ax2, timestamp_col="timestamp")
                    st.pyplot(fig2)
                
                # (Actual PDF generation logic placeholders)
                st.success("Analysis complete. PDF Generation logic would execute here.")
                st.info("Note: Full PDF generation requires integrating `report_generator` with the new plot objects.")
                
            except Exception as e:
                st.error(f"Error during generation: {e}")
                st.warning("Please check file structure and column names in your CSVs.")

    st.button("Back", on_click=prev_step)