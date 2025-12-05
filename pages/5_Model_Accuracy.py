import streamlit as st
import pandas as pd
import os
from pathlib import Path
import time
from playwright.sync_api import sync_playwright
from jinja2 import Environment
import io
import zipfile
from datetime import datetime
import sys
import asyncio

from utils.model_val_utils import extract_numeric
from db_utils import PrismDB
from validations.prism_validations.metric_mapping_validation.query import get_query as get_metric_mapping_query

st.set_page_config(page_title="Calculate Accuracy", page_icon="🎯", layout="wide")

st.title("🎯 Model Accuracy Wizard")
st.markdown("""
Calculates the accuracy of your deployed models by comparing the `.dat` file (from PRISM's relative deviation folder) against the model's active metrics in the database.

**How to Use:**
1.  **Connect Database:** Connect to the PRISM database in the sidebar to fetch active metric configurations.
2.  **Scan Directory:** Point the tool to your model root folder (containing the `relative_deviation` subfolders).
3.  **Process:** Select the models to analyze. The tool will calculate the accuracy score for each based on the formula: `(1 - abs(Mean Relative Deviation)) * 100`.
4.  **Report:** Download a summary CSV or a PDF report.
""")

# --- Initialize Session State ---
if 'acc_step' not in st.session_state: st.session_state.acc_step = 1
if 'scanned_models_df' not in st.session_state: st.session_state.scanned_models_df = None
if 'accuracy_results' not in st.session_state: st.session_state.accuracy_results = None
if 'accuracy_details' not in st.session_state: st.session_state.accuracy_details = {} # Store detailed DF for reports
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

def generate_accuracy_report(system_name, system_data):
    """
    Generates a PDF report for a specific System using Playwright/HTML.
    system_data is a list of dicts: {'Sprint':, 'Model':, 'Avg Accuracy':, 'Data': DataFrame}
    """
    # 1. Structure Data for Template
    # Group by Sprint
    system_data.sort(key=lambda x: (x['Sprint'], x['Model']))
    
    sprints_data = {}
    for item in system_data:
        s_name = item['Sprint']
        if s_name not in sprints_data:
            sprints_data[s_name] = []
        sprints_data[s_name].append(item)

    # 2. Define HTML Template
    template_str = """
    <html>
    <head>
        <style>
            body { font-family: "Helvetica", "Arial", sans-serif; margin: 40px; color: #333; }
            h1 { text-align: center; color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
            .meta { text-align: center; color: #7f8c8d; font-size: 0.9em; margin-bottom: 40px; font-style: italic; }
            
            .sprint-header { 
                background-color: #2980b9; 
                color: white; 
                padding: 8px 15px; 
                border-radius: 4px; 
                margin-top: 30px; 
                font-size: 1.2em;
                page-break-after: avoid; 
            }
            
            .model-section { 
                margin-left: 10px; 
                margin-top: 20px; 
                page-break-inside: avoid; 
                border: 1px solid #eee;
                padding: 15px;
                border-radius: 5px;
                background-color: #fcfcfc;
            }
            
            .model-title { 
                font-size: 1.1em; 
                font-weight: bold; 
                color: #2c3e50; 
                margin-bottom: 5px; 
            }
            
            .accuracy-score {
                font-weight: bold;
                color: #27ae60;
                margin-bottom: 10px;
                display: block;
            }

            table { 
                width: 100%; 
                border-collapse: collapse; 
                font-size: 0.85em; 
                margin-top: 10px;
                background-color: white;
            }
            
            th, td { 
                border: 1px solid #bdc3c7; 
                padding: 6px 8px; 
                text-align: left; 
            }
            
            th { 
                background-color: #ecf0f1; 
                font-weight: bold; 
                text-align: center; 
                color: #2c3e50;
            }
            
            td.num { text-align: center; }
            tr:nth-child(even) { background-color: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1>Model Accuracy Report: {{ system_name }}</h1>
        <div class="meta">Generated on: {{ generation_date }}</div>

        {% for sprint_name, models in sprints_data.items() %}
            <div class="sprint-container">
                <h2 class="sprint-header">Sprint: {{ sprint_name }}</h2>
                
                {% for model in models %}
                    <div class="model-section">
                        <div class="model-title">Model: {{ model['Model'] }}</div>
                        <span class="accuracy-score">Average Accuracy: {{ model['Avg Accuracy'] }}</span>
                        
                        <table>
                            <thead>
                                <tr>
                                    <th style="width: 60%;">Metric</th>
                                    <th style="width: 20%;">Avg Rel Dev (%)</th>
                                    <th style="width: 20%;">Accuracy (%)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for index, row in model['Data'].iterrows() %}
                                <tr>
                                    <td>{{ row['Metrics'] }}</td>
                                    <td class="num">{{ row['Average - Relative Deviation (%)'] }}</td>
                                    <td class="num">{{ row['Accuracy (%)'] }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                {% endfor %}
            </div>
        {% endfor %}
    </body>
    </html>
    """
    
    # 3. Render HTML
    env = Environment()
    template = env.from_string(template_str)
    html_content = template.render(
        system_name=system_name,
        generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        sprints_data=sprints_data
    )
    
    # 4. Generate PDF via Playwright
    # Windows fix for asyncio loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html_content)
            pdf_bytes = page.pdf(
                format="A4",
                margin={'top': '20mm', 'bottom': '20mm', 'left': '20mm', 'right': '20mm'},
                print_background=True
            )
            browser.close()
            return pdf_bytes
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

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
    Returns: (Avg Accuracy, Message, Detailed DataFrame)
    """
    model_name = row['Model']
    dat_path = row['Dat Path']
    
    # 1. Get Metrics from PRISM Query Result
    # Filter for this model
    if prism_metrics_df is None or prism_metrics_df.empty:
        return None, "PRISM metrics data is empty.", None

    # Column names match the query output: [FORM NAME], [METRIC NAME], [INCLUDED IN PROFILE]
    model_metrics_df = prism_metrics_df[prism_metrics_df['FORM NAME'] == model_name]
    
    if model_metrics_df.empty:
        return None, f"Model '{model_name}' not found in PRISM database.", None

    # Filter: INCLUDED IN PROFILE = 'YES'
    target_metrics = model_metrics_df[
        model_metrics_df['INCLUDED IN PROFILE'] == 'YES'
    ]['METRIC NAME'].unique().tolist()
    
    if not target_metrics:
        return None, "No metrics found with [INCLUDED IN PROFILE] = 'YES'.", None

    # 2. Read .dat file
    try:
        # Based on original logic: encoding UTF-16, tab delimiter
        df_data = pd.read_csv(dat_path, encoding="UTF-16", delimiter='\t')
    except Exception as e:
        return None, f"Error reading .dat file: {e}", None

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
        return None, f"None of the {len(target_metrics)} active profile metrics found in .dat file.", None

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
             return None, "No valid numeric data could be extracted for accuracy calculation.", None

        df_scores = pd.DataFrame(results)
        
        # 6. Save Result
        save_dir = Path(row['Full Path']) / "relative_deviation"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{model_name}_Accuracy.csv"
        df_scores.to_csv(save_path, index=False)
        
        avg_model_acc = df_scores['Accuracy (%)'].mean()
        
        return avg_model_acc, f"Saved to {save_path.name}", df_scores

    except Exception as e:
        return None, f"Calculation Error: {e}", None

# Progress Indicator
steps = ["Scan & Select Models", "Process & Results"]
current_step = st.session_state.acc_step
st.progress(current_step / len(steps), text=f"Step {current_step}: {steps[current_step-1]}")

# ==========================================
# STEP 1: SCAN & SELECT
# ==========================================
if current_step == 1:
    st.header("Step 1: Scan Directory and Select Model(s)")
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
        st.markdown("Select the models you wish to process. **Note:** Only models with a **.dat file** can be processed.")
        
        df = st.session_state.scanned_models_df.copy()
        
        # --- Filters ---
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            site_options = sorted(df['Site'].unique().tolist())
            selected_sites = st.multiselect("Site", site_options)
        with col_f2:
            system_options = sorted(df['System'].unique().tolist())
            selected_systems = st.multiselect("System", system_options)
        with col_f3:
            sprint_options = sorted(df['Sprint'].unique().tolist())
            selected_sprints = st.multiselect("Sprint", sprint_options)
            
        filtered_df = df.copy() 

        # Apply Filters (Only if selected)
        if selected_sites:
            filtered_df = df[df['Site'].isin(selected_sites)]
        if selected_systems:
            filtered_df = df[df['System'].isin(selected_systems)]
        if selected_sprints:
            filtered_df = df[df['Sprint'].isin(selected_sprints)]
        
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
        # We also need to store the detailed data for the report
        accuracy_details_store = [] # List of dicts: {System, Sprint, Model, Data(df), AvgAcc}
        
        progress_bar = st.progress(0, text="Starting...")
        
        for i, (index, row) in enumerate(to_process.iterrows()):
            model_name = row['Model']
            progress_bar.progress((i) / len(to_process), text=f"Processing {model_name}...")
            
            # calculate_model_accuracy now returns 3 values
            acc_val, msg, detailed_df = calculate_model_accuracy(row, prism_metrics_df)
            
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
            
            if detailed_df is not None:
                accuracy_details_store.append({
                    "System": row['System'],
                    "Sprint": row['Sprint'],
                    "Model": model_name,
                    "Avg Accuracy": f"{acc_val:.2f}%",
                    "Data": detailed_df
                })
            
        progress_bar.progress(1.0, text="Complete!")
        st.session_state.accuracy_results = pd.DataFrame(results_log)
        st.session_state.accuracy_details = accuracy_details_store
        st.balloons()
        
    # Display Results
    if st.session_state.accuracy_results is not None:
        st.divider()
        st.subheader("Processing Summary")
        
        res_df = st.session_state.accuracy_results
        details_store = st.session_state.accuracy_details
        
        # Metrics
        success_count = len(res_df[res_df['Status'].str.contains("Success")])
        fail_count = len(res_df) - success_count
        
        c1, c2 = st.columns(2)
        c1.metric("Successful", success_count)
        c2.metric("Failed", fail_count)
        
        st.dataframe(res_df, use_container_width=True)
        
        # Report Generation
        if details_store:
            st.divider()
            st.subheader("📄 Report Generation")
            
            # Group by System
            systems = list(set(d['System'] for d in details_store))
            
            # Generate PDFs
            pdfs_to_zip = []
            
            with st.spinner("Generating PDF reports..."):
                for sys_name in systems:
                    sys_data = [d for d in details_store if d['System'] == sys_name]
                    pdf_bytes = generate_accuracy_report(sys_name, sys_data)
                    
                    if pdf_bytes:
                        pdfs_to_zip.append({
                            "name": f"{sys_name}_Accuracy_Report.pdf",
                            "data": pdf_bytes
                        })
            
            # Download Logic
            if len(pdfs_to_zip) == 1:
                st.download_button(
                    label=f"📥 Download PDF ({systems[0]})",
                    data=pdfs_to_zip[0]['data'],
                    file_name=pdfs_to_zip[0]['name'],
                    mime="application/pdf"
                )
            elif len(pdfs_to_zip) > 1:
                # Zip multiple PDFs
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in pdfs_to_zip:
                        zf.writestr(p['name'], p['data'])
                
                st.download_button(
                    label="📥 Download All Reports (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name="Model_Accuracy_Reports.zip",
                    mime="application/zip"
                )
        
    st.markdown("---")
    if st.button("⬅️ Back to Selection"):
        st.session_state.acc_step = 1
        st.session_state.accuracy_results = None
        st.session_state.accuracy_details = {}
        st.rerun()