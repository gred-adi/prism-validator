import streamlit as st
import pandas as pd
from pathlib import Path
import shutil
import os

# Utils
from utils.model_dev_utils import scan_folders_structure
from utils.constraint_utils import fetch_model_constraints
from utils.qa_plotting import generate_report_plots
from utils.qa_ks_comparison import compare_data_distributions
from utils.qa_reporting import generate_qa_report_playwright
from db_utils import PrismDB

st.set_page_config(page_title="Model FPR", page_icon="🔎", layout="wide")

# --- Session State ---
if 'fpr_step' not in st.session_state: st.session_state.fpr_step = 1
if 'fpr_scanned_df' not in st.session_state: st.session_state.fpr_scanned_df = None
if 'fpr_selected_model_row' not in st.session_state: st.session_state.fpr_selected_model_row = None
if 'fpr_constraints_df' not in st.session_state: st.session_state.fpr_constraints_df = None
if 'db' not in st.session_state: st.session_state.db = None 

# --- Sidebar: Database Connection ---
with st.sidebar:
    st.header("🌍 Database Connection")
    st.caption("Required to fetch active model constraints/filters.")
    
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

st.title("🔎 Model FPR Wizard")
st.markdown("""
Generates comprehensive False Positive Rate (FPR) and Kolmogorov-Smirnov (KS) consistency reports for your models.
This tool analyzes the model's behavior on holdout data and validates the consistency between training/validation and holdout datasets.

**How to Use:**
1.  **Connect Database:** Connect to the PRISM database in the sidebar to automatically fetch active model constraints (filters).
2.  **Scan Directory:** Enter the path to your model's root folder. The tool will identify models with the required datasets (Raw, Holdout, and OMR files).
3.  **Select Model:** Choose a valid model from the list to proceed.
4.  **Configure Constraints:** Review the active constraints fetched from the database or add manual "Model OFF" conditions.
5.  **Generate:** The tool will calculate FPR statistics, generate distribution plots, and produce a detailed PDF QA report.
""")

steps = ["Select Model", "Configure Constraints", "Generate Report"]
current = st.session_state.fpr_step
st.progress(current / len(steps), text=f"Step {current}: {steps[current-1]}")

# ==========================================
# STEP 1: SCAN & SELECT
# ==========================================
if current == 1:
    st.header("Step 1: Select Model")
    
    default_path = st.session_state.get('base_path', os.getcwd())
    root_folder = st.text_input("Root Folder Path", value=default_path)
    
    if st.button("🔍 Scan Models"):
        with st.spinner("Scanning..."):
            df = scan_folders_structure(root_folder)
            st.session_state.fpr_scanned_df = df
            
    if st.session_state.fpr_scanned_df is not None and not st.session_state.fpr_scanned_df.empty:
        df = st.session_state.fpr_scanned_df.copy()
        
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
        
        # --- NEW: File Validation Columns ---
        # Helper to create status emoji
        def get_status(x):
            return "✅" if x else "❌"

        # Apply to file columns
        # Note: 'Raw File', 'Holdout File' etc. contain filenames if present, else None
        # So bool(x) works perfectly.
        df['Raw'] = df['Raw File'].apply(get_status)
        df['Holdout'] = df['Holdout File'].apply(get_status)
        df['OMR Cleaned'] = df['OMR Cleaned File'].apply(get_status)
        df['OMR Raw'] = df['OMR Raw File'].apply(get_status)
        df['OMR Holdout'] = df['OMR Holdout File'].apply(get_status)

        # Define column order for display
        display_cols = [
            'Site', 'System', 'Sprint', 'Model',
            'Raw', 'Raw File', 
            'Holdout', 'Holdout File',
            'OMR Cleaned', 'OMR Cleaned File',
            'OMR Raw', 'OMR Raw File',
            'OMR Holdout', 'OMR Holdout File'
        ]
        
        # Render
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Selection Logic
        valid_models_df = df[df['Dataset Found']]
        
        if valid_models_df.empty:
            st.warning("No models with valid 'dataset' folders found in selection.")
        else:
            model_names = valid_models_df['Model'].unique()
            selected_model_name = st.selectbox("Select Model to Process", model_names)
            
            if st.button("Proceed with Selected Model ➡️"):
                row = valid_models_df[valid_models_df['Model'] == selected_model_name].iloc[0]
                
                # Check if ALL required files exist for the selected model
                required_files = ['Raw File', 'Holdout File', 'OMR Cleaned File', 'OMR Raw File', 'OMR Holdout File']
                missing = [f for f in required_files if row[f] is None]
                
                if missing:
                    st.error(f"Cannot proceed. Missing required files: {', '.join(missing)}")
                else:
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
    
    if st.session_state.db is None:
        st.warning("⚠️ Database not connected. Cannot fetch constraints automatically.")
    
    if st.button("Fetch Constraints from DB") or st.session_state.fpr_constraints_df is None:
        if st.session_state.db:
            with st.spinner("Fetching constraints..."):
                constraints = fetch_model_constraints(st.session_state.db, model_name)
                st.session_state.fpr_constraints_df = constraints
        else:
            if st.session_state.fpr_constraints_df is None:
                # Initialize empty dataframe with correct structure
                st.session_state.fpr_constraints_df = pd.DataFrame(columns=["Metric Name", "Point Name", "Operator", "Value"])

    st.markdown("### Active Constraints (Model OFF conditions)")
    edited_constraints = st.data_editor(
        st.session_state.fpr_constraints_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Metric Name": st.column_config.TextColumn("Metric Name", disabled=True),
            "Point Name": st.column_config.TextColumn("Point Name", disabled=True),
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
    
    # Advanced Configuration
    with st.expander("Advanced Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            warning_thresh = st.number_input("Warning Threshold (%)", value=5.0)
            alert_thresh = st.number_input("Alert Threshold (%)", value=10.0)
        with col2:
            sub_ts_len = st.number_input("Sub-TS Length (min)", value=60)
            n_ts_above = st.number_input("N-Points Above Threshold", value=50)

    if st.button("🚀 Generate Report", type="primary"):
        # Use st.status container for better progress tracking
        with st.status("🚀 Starting Report Generation...", expanded=True) as status:
            try:
                # 1. Setup Paths
                st.write("📂 Setting up directories...")
                dataset_path = Path(model_row['Dataset Path'])
                base_dir = dataset_path.parent
                
                # Output Directories
                perf_dir = base_dir / "performance_assessment_report"
                fpr_dir = perf_dir / "FPR"
                ks_dir = perf_dir / "KS"
                report_dir = perf_dir / "report_document"
                
                # Create Dirs
                fpr_dir.mkdir(parents=True, exist_ok=True)
                ks_dir.mkdir(parents=True, exist_ok=True)
                report_dir.mkdir(parents=True, exist_ok=True)
                
                # 2. Identify Files (Already validated in Step 1)
                raw_file = Path(model_row['Full Path']) / "dataset" / model_row['Raw File']
                holdout_file = Path(model_row['Full Path']) / "dataset" / model_row['Holdout File']
                
                # OMR Filenames from Step 1 scan
                val_wo_outlier = model_row['OMR Cleaned File']
                val_w_outlier = model_row['OMR Raw File']
                holdout_omr = model_row['OMR Holdout File']

                # 3. Prepare Constraints Lists
                if not constraints.empty:
                    # We strictly use 'Point Name' for the actual logic as it matches the CSV header
                    c_cols = constraints['Point Name'].tolist()
                    c_ops = constraints['Operator'].tolist()
                    c_vals = constraints['Value'].tolist()
                else:
                    c_cols, c_ops, c_vals = [], [], []

                # 4. Generate FPR Plots & Stats (Saves to Disk)
                st.write("📊 Generating FPR Plots & Statistics...")
                generate_report_plots(
                    data_fpath=str(dataset_path),
                    fpr_fpath=str(fpr_dir),
                    model_name=model_row['Model'],
                    constraint_cols=c_cols,
                    condition_limits=c_vals,
                    operators=c_ops,
                    raw_data_fpath=str(raw_file),
                    holdout_fpath=str(holdout_file),
                    holdout_omr_fname=holdout_omr,
                    val_without_outlier_omr_fname=val_wo_outlier,
                    val_with_outlier_omr_fname=val_w_outlier,
                    warning_threshold=warning_thresh,
                    alert_threshold=alert_thresh,
                    sub_ts_length_in_minutes=sub_ts_len,
                    n_ts_above_threshold=n_ts_above
                )
                
                # Show generated FPR files
                fpr_files = [f.name for f in fpr_dir.iterdir() if f.suffix == '.jpg']
                st.caption(f"✅ Created {len(fpr_files)} FPR plots.")
                
                # 5. Generate KS Plots & Stats (Saves to Disk)
                st.write("📉 Generating KS Distribution Comparisons...")
                compare_data_distributions(
                    validation_fname=str(raw_file),
                    ks_file_path=str(ks_dir),
                    holdout_fpath=str(holdout_file),
                    description_row=0
                )
                
                # Show generated KS files
                ks_files = [f.name for f in ks_dir.iterdir() if f.suffix == '.jpg']
                st.caption(f"✅ Created {len(ks_files)} KS distribution plots.")
                
                # 6. Generate PDF Report (Playwright)
                st.write("📄 Rendering PDF Report...")
                
                # File paths expected by the generator
                success = generate_qa_report_playwright(
                    model_name=model_row['Model'],
                    fpr_file_path=str(fpr_dir),
                    ks_file_path=str(ks_dir),
                    report_file_path=str(report_dir),
                    fpr_stats_cleaned_omr_fpath=str(fpr_dir / "fpr_stats_cleaned_val_omr_df.yaml"),
                    fpr_stats_holdout_omr_fpath=str(fpr_dir / "fpr_stats_holdout_omr_df.yaml"),
                    fpr_stats_raw_omr_fpath=str(fpr_dir / "fpr_stats_raw_val_omr_df.yaml"),
                    fprp_stats_holdout_omr_fpath=str(fpr_dir / "fprp_stats_holdout_omr_df.yaml"),
                    data_stats_fpath=str(ks_dir / "data_stats.yaml"),
                    ks_df_fpath=str(ks_dir / "ks_results.csv"),
                    datasets_range_fpath=str(fpr_dir / "datasets_range.yaml")
                )
                
                if success:
                    pdf_path = report_dir / f"model_qa_report_{model_row['Model']}.pdf"
                    status.update(label="✅ Report Generation Complete!", state="complete", expanded=False)
                    
                    st.success(f"Report generated successfully!")
                    st.caption(f"Location: `{pdf_path}`")
                    
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name=pdf_path.name,
                            mime="application/pdf"
                        )
                else:
                    status.update(label="❌ PDF Generation Failed", state="error")
                    st.error("PDF Generation failed. Check logs.")
                
            except Exception as e:
                status.update(label="❌ Error Occurred", state="error")
                st.error(f"Error during processing: {e}")
                st.exception(e)

    st.button("Back", on_click=prev_step)