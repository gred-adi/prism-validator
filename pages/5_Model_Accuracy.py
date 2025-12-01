import streamlit as st
import pandas as pd
import os
from pathlib import Path
import time

from utils.model_val_utils import extract_numeric
from db_utils import PrismDB
from validations.prism_validations.metric_mapping_validation.query import get_query as get_metric_mapping_query

st.set_page_config(page_title="Calculate Accuracy", page_icon="🎯", layout="wide")

# --- Initialize Session State ---
if 'acc_step' not in st.session_state: st.session_state.acc_step = 1
if 'scanned_models_df' not in st.session_state: st.session_state.scanned_models_df = None
if 'accuracy_results' not in st.session_state: st.session_state.accuracy_results = None
if 'db' not in st.session_state: st.session_state.db = None

# --- Sidebar: Database Connection ---
with st.sidebar:
    st.header("🌍 Database Connection")
    st.caption("Required to fetch 'Included in Profile' status.")
    
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

# --- Helper Functions ---

def scan_folders(root_path):
    """
    Scans the folder structure: Root -> Site -> System -> Sprint -> Model
    Checks for .dat files existence in relative_deviation folder.
    """
    found_models = []
    root = Path(root_path)
    
    if not root.exists():
        return pd.DataFrame()

    # We expect strict depth: Site (1) -> System (2) -> Sprint (3) -> Model (4)
    try:
        # Level 1: Site
        for site_dir in [d for d in root.iterdir() if d.is_dir()]:
            # Level 2: System
            for system_dir in [d for d in site_dir.iterdir() if d.is_dir()]:
                # Level 3: Sprint
                for sprint_dir in [d for d in system_dir.iterdir() if d.is_dir()]:
                    # Level 4: Model
                    for model_dir in [d for d in sprint_dir.iterdir() if d.is_dir()]:
                        model_name = model_dir.name
                        
                        # Check 1: .dat file existence in relative_deviation folder
                        rel_dev_path = model_dir / "relative_deviation"
                        dat_file_path = rel_dev_path / f"{model_name}.dat"
                        
                        has_dat = dat_file_path.exists() and dat_file_path.is_file()
                        
                        found_models.append({
                            "Select": False,
                            "Site": site_dir.name,
                            "System": system_dir.name,
                            "Sprint": sprint_dir.name,
                            "Model": model_name,
                            "Dat File Found": has_dat,
                            "Dat Filename": dat_file_path.name if has_dat else None,
                            "Full Path": str(model_dir),
                            "Dat Path": str(dat_file_path) if has_dat else None
                        })
    except Exception as e:
        st.error(f"Error scanning directories: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(found_models)
    
    # Sort results
    if not df.empty:
        df = df.sort_values(by=['Site', 'System', 'Sprint', 'Model'])
        
    return df

def calculate_model_accuracy(row, prism_metrics_df):
    """
    Calculates accuracy for a single model row using PRISM DB 'Included in Profile' status.
    """
    model_name = row['Model']
    dat_path = row['Dat Path']
    
    # 1. Get Metrics from PRISM Query Result
    # Filter for this model
    if prism_metrics_df is None or prism_metrics_df.empty:
        return None, "PRISM metrics data is empty."

    # Column names match the query output: [FORM NAME], [METRIC NAME], [INCLUDED IN PROFILE]
    model_metrics_df = prism_metrics_df[prism_metrics_df['FORM NAME'] == model_name]
    
    if model_metrics_df.empty:
        return None, f"Model '{model_name}' not found in PRISM database."

    # Filter: INCLUDED IN PROFILE = 'YES'
    target_metrics = model_metrics_df[
        model_metrics_df['INCLUDED IN PROFILE'] == 'YES'
    ]['METRIC NAME'].unique().tolist()
    
    if not target_metrics:
        return None, "No metrics found with [INCLUDED IN PROFILE] = 'YES'."

    # 2. Read .dat file
    try:
        # Based on original logic: encoding UTF-16, tab delimiter
        df_data = pd.read_csv(dat_path, encoding="UTF-16", delimiter='\t')
    except Exception as e:
        return None, f"Error reading .dat file: {e}"

    # 3. Clean Column Names (Mapping Logic from original script)
    # This aligns the .dat file headers (e.g., "TI-101 (Temp)") with PRISM Metric Names ("TI-101")
    column_mapping = {}
    for col in df_data.columns:
        if "Virtual" in col:
            extracted_sensor_name = col.split('(Virtual')[0].strip()
        elif "Arkanghel" in col:
            extracted_sensor_name = col.split('(Arkanghel')[0].strip()
        else:
            extracted_sensor_name = col.split('(')[0].strip()
        column_mapping[col] = extracted_sensor_name

    df_data.rename(columns=column_mapping, inplace=True)

    # 4. Filter columns to target metrics
    # Intersect available columns with target metrics found in PRISM
    available_metrics = [m for m in target_metrics if m in df_data.columns]
    
    if not available_metrics:
        return None, f"None of the {len(target_metrics)} active profile metrics found in .dat file."

    # 5. Calculate Accuracy
    try:
        # Extract numeric values
        df_subset = df_data[available_metrics].apply(lambda col: col.apply(extract_numeric))
        
        results = []
        for col in df_subset.columns:
            mean_rel_dev = df_subset[col].mean()
            
            # Accuracy = (1 - abs(mean_rel_dev)) * 100
            if pd.notna(mean_rel_dev):
                acc_val = round((1 - abs(mean_rel_dev)) * 100, 2)
                avg_dev_val = round(abs(mean_rel_dev) * 100, 2)
                results.append({
                    'Metrics': col,
                    'Average - Relative Deviation (%)': avg_dev_val,
                    'Accuracy (%)': acc_val
                })
        
        if not results:
             return None, "No valid numeric data could be extracted for accuracy calculation."

        df_scores = pd.DataFrame(results)
        
        # 6. Save Result
        save_dir = Path(row['Full Path']) / "relative_deviation"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{model_name}_Accuracy.csv"
        df_scores.to_csv(save_path, index=False)
        
        avg_model_acc = df_scores['Accuracy (%)'].mean()
        
        return avg_model_acc, f"Saved to {save_path.name}"

    except Exception as e:
        return None, f"Calculation Error: {e}"

# --- Page Layout ---

st.title("🎯 Model Accuracy Wizard")

# Progress Indicator
steps = ["1. Scan & Select Models", "2. Process & Results"]
current_step = st.session_state.acc_step
st.progress(current_step / len(steps), text=f"Step {current_step}: {steps[current_step-1]}")

# ==========================================
# STEP 1: SCAN & SELECT
# ==========================================
if current_step == 1:
    st.header("Step 1: Scan Directory")
    st.markdown("""
    Select the root folder containing your model hierarchy. 
    The tool expects the following structure: `Root > Site > System > Sprint > Model > relative_deviation`.
    """)

    # 1. Folder Input
    default_path = st.session_state.get('base_path', os.getcwd())
    root_folder = st.text_input("Root Folder Path", value=default_path)

    # 2. Scan Action
    if st.button("🔍 Scan Folder Structure", type="primary"):
        with st.spinner("Scanning folders..."):
            df_models = scan_folders(root_folder)
            st.session_state.scanned_models_df = df_models
            
            if df_models.empty:
                st.warning("No models found. Please check the folder structure.")
            else:
                st.success(f"Found {len(df_models)} model folders.")

    # 3. Display & Selection
    if st.session_state.scanned_models_df is not None and not st.session_state.scanned_models_df.empty:
        st.subheader("Found Models")
        st.markdown("Select the models you wish to process. **Note:** Only models with a **.dat file** can be processed.")
        
        df = st.session_state.scanned_models_df.copy()
        
        # --- Filters ---
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            site_options = sorted(df['Site'].unique().tolist())
            selected_sites = st.multiselect("Filter by Site", site_options, default=site_options)
        with col_f2:
            system_options = sorted(df['System'].unique().tolist())
            selected_systems = st.multiselect("Filter by System", system_options, default=system_options)
        with col_f3:
            sprint_options = sorted(df['Sprint'].unique().tolist())
            selected_sprints = st.multiselect("Filter by Sprint", sprint_options, default=sprint_options)
            
        # Apply Filters
        filtered_df = df[
            (df['Site'].isin(selected_sites)) &
            (df['System'].isin(selected_systems)) &
            (df['Sprint'].isin(selected_sprints))
        ].copy()
        
        # Add visual indicators for the user
        filtered_df['Dat File Found'] = filtered_df['Dat File Found'].apply(lambda x: "✅" if x else "❌")
        
        column_config = {
            "Select": st.column_config.CheckboxColumn("Select", default=False),
            "Dat File Found": st.column_config.TextColumn("Dat File", width="small"),
            "Full Path": None, 
            "Dat Path": None
        }
        
        st.caption(f"Showing {len(filtered_df)} of {len(df)} models.")
        
        edited_df = st.data_editor(
            filtered_df,
            hide_index=True,
            column_config=column_config,
            disabled=["Site", "System", "Sprint", "Model", "Dat File Found", "Dat Filename"],
            use_container_width=True,
            key="acc_model_selector"
        )
        
        # --- Update selection in the MAIN dataframe ---
        if not edited_df.empty:
            # 1. Create a map of {FullPath: SelectedStatus} from the edited dataframe
            selection_map = dict(zip(edited_df['Full Path'], edited_df['Select']))
            
            # 2. Update the main df 'Select' column where Full Path matches
            st.session_state.scanned_models_df['Select'] = st.session_state.scanned_models_df.apply(
                lambda row: selection_map.get(row['Full Path'], row['Select']), axis=1
            )
        
        # 4. Proceed Button Logic
        current_main_df = st.session_state.scanned_models_df
        valid_mask = (
            current_main_df['Select'] & 
            current_main_df['Dat File Found']
        )
        selected_count = current_main_df['Select'].sum()
        valid_count = valid_mask.sum()
        
        if selected_count > 0:
            if valid_count < selected_count:
                st.warning(f"⚠️ {selected_count} selected, but only {valid_count} have .dat files.")
            
            if st.button(f"Proceed with {valid_count} Models ➡️", type="primary", disabled=(valid_count==0)):
                st.session_state.acc_step = 2
                st.rerun()

# ==========================================
# STEP 2: PROCESS
# ==========================================
elif current_step == 2:
    st.header("Step 2: Processing & Results")
    
    # 1. Check Previous Step Data
    if st.session_state.scanned_models_df is None:
        st.error("No data loaded. Please return to Step 1.")
        if st.button("Back to Step 1"):
            st.session_state.acc_step = 1
            st.rerun()
        st.stop()

    # 2. Check Database Connection (Critical for this step)
    if st.session_state.db is None:
        st.error("❌ **Database Disconnected.** We need to query PRISM to find which metrics are 'Included in Profile'. Please connect in the sidebar.")
        st.stop()
        
    df = st.session_state.scanned_models_df
    
    # Filter for processing (Must be Selected + Valid Dat File)
    to_process = df[df['Select'] & df['Dat File Found']].copy()
    
    st.info(f"Ready to calculate accuracy for **{len(to_process)}** models. This will fetch metric definitions from PRISM.")
    
    if st.button("🚀 Run Model Accuracy", type="primary"):
        
        # --- A. Batch Query Metrics ---
        selected_model_names = to_process['Model'].unique().tolist()
        prism_metrics_df = None
        
        with st.spinner("Fetching metric configurations from PRISM..."):
            try:
                # Use the Metric Mapping query which returns [FORM NAME], [METRIC NAME], [INCLUDED IN PROFILE]
                query = get_metric_mapping_query(selected_model_names)
                prism_metrics_df = st.session_state.db.run_query(query)
            except Exception as e:
                st.error(f"Failed to query database: {e}")
                st.stop()
        
        # --- B. Processing Loop ---
        results_log = []
        progress_bar = st.progress(0, text="Starting...")
        
        for i, (index, row) in enumerate(to_process.iterrows()):
            model_name = row['Model']
            progress_bar.progress((i) / len(to_process), text=f"Processing {model_name}...")
            
            acc_val, msg = calculate_model_accuracy(row, prism_metrics_df)
            
            status = "✅ Success" if acc_val is not None else "❌ Failed"
            
            results_log.append({
                "Site": row['Site'],
                "System": row['System'],
                "Sprint": row['Sprint'],
                "Model": model_name,
                "Status": status,
                "Avg Accuracy": f"{acc_val:.2f}%" if acc_val is not None else "N/A",
                "Details": msg
            })
            
        progress_bar.progress(1.0, text="Complete!")
        st.session_state.accuracy_results = pd.DataFrame(results_log)
        st.balloons()
        
    # Display Results
    if st.session_state.accuracy_results is not None:
        st.divider()
        st.subheader("Processing Summary")
        
        res_df = st.session_state.accuracy_results
        
        # Metrics
        success_count = len(res_df[res_df['Status'].str.contains("Success")])
        fail_count = len(res_df) - success_count
        
        c1, c2 = st.columns(2)
        c1.metric("Successful", success_count)
        c2.metric("Failed", fail_count)
        
        st.dataframe(res_df, use_container_width=True)
        
    st.markdown("---")
    if st.button("⬅️ Back to Selection"):
        st.session_state.acc_step = 1
        st.session_state.accuracy_results = None
        st.rerun()