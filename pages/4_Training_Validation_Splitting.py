import streamlit as st
import pandas as pd
import gc
from pathlib import Path
from datetime import datetime

# Import utils
from utils.model_dev_utils import (
    cleaned_dataset_name_split, 
    read_prism_csv, 
    generate_tvs_visualizations,
    generate_tvs_report
)
from verstack.stratified_continuous_split import scsplit

st.set_page_config(page_title="Train Validation Split", page_icon="🔀", layout="wide")

st.title("🔀 Training-Validation Split Wizard")
st.markdown("""
This wizard helps you split your Training/Validation data into separate **Training** and **Validation** sets. 
It uses **stratified splitting** based on an Operational State metric to ensure both sets are representative of the model's operating range.

**How to Use:**
1.  **Load TDT Files:** Ensure your TDT files are processed in the Global Settings sidebar. Select the desired **TDT** and **Model**.
2.  **Set Output Folder:** On the Global Settings sidebar, specify the base output folder where cleaned datasets and reports will be saved.             
3.  **Upload Data:** Upload the `CLEANED_...WITH-OUTLIER` and `CLEANED_...WITHOUT-OUTLIER` files.
4.  **Configure:** Select the **Operational State** metric to stratify by and the desired split ratio (e.g., 80% Train, 20% Validation).
5.  **Split:** Run the split algorithm.
6.  **Visualize & Save:** Review distribution plots to verify balance and save the final datasets along with a PDF report.
""")

# --- Initialize Session State ---
if 'tvs_step' not in st.session_state: st.session_state.tvs_step = 1

# Data States
if 'tvs_cleaned_w_outlier' not in st.session_state: st.session_state.tvs_cleaned_w_outlier = None
if 'tvs_cleaned_wo_outlier' not in st.session_state: st.session_state.tvs_cleaned_wo_outlier = None
if 'tvs_header' not in st.session_state: st.session_state.tvs_header = None
if 'tvs_mapped_wo_outlier' not in st.session_state: st.session_state.tvs_mapped_wo_outlier = None
if 'tvs_mapped_w_outlier' not in st.session_state: st.session_state.tvs_mapped_w_outlier = None
if 'tvs_mapping_table' not in st.session_state: st.session_state.tvs_mapping_table = None

# Results
if 'ds_result_train' not in st.session_state: st.session_state.ds_result_train = None
if 'ds_result_val_wo' not in st.session_state: st.session_state.ds_result_val_wo = None
if 'ds_result_val_w' not in st.session_state: st.session_state.ds_result_val_w = None

# Metadata
if 'site_name' not in st.session_state: st.session_state.site_name = ""
if 'system_name' not in st.session_state: st.session_state.system_name = ""
if 'model_name' not in st.session_state: st.session_state.model_name = ""
if 'sprint_name' not in st.session_state: st.session_state.sprint_name = ""
if 'inclusive_dates' not in st.session_state: st.session_state.inclusive_dates = ""
if 'selected_tdt' not in st.session_state: st.session_state.selected_tdt = None
if 'selected_model_survey' not in st.session_state: st.session_state.selected_model_survey = None

# --- Helper Functions ---
def next_step(): st.session_state.tvs_step += 1
def prev_step(): st.session_state.tvs_step -= 1

def load_and_map_data(w_outlier_file, wo_outlier_file, survey_df, model_name):
    """Loads CSVs and maps column names using the TDT survey data."""
    try:
        # 1. Read Raw CSVs
        df_w_raw = pd.read_csv(w_outlier_file, index_col=False)
        df_wo_raw = pd.read_csv(wo_outlier_file, index_col=False)

        # 2. Extract Data & Header (Using utils function logic)
        # Note: read_prism_csv expects standard PRISM format (4 header rows)
        df_w_clean, header_w = read_prism_csv(df_w_raw)
        df_wo_clean, header_wo = read_prism_csv(df_wo_raw)

        # 3. Get Mapping from Survey
        # We look for the model in the survey data
        model_survey = survey_df[survey_df['Model'] == model_name]
        
        # Create map: Canary Point Name -> Metric
        # Handle cases where 'Canary Point Name' might be missing or named differently in survey
        point_col = 'Canary Point Name' if 'Canary Point Name' in model_survey.columns else 'Point Name'
        
        name_to_metric = pd.Series(
            model_survey['Metric'].values,
            index=model_survey[point_col]
        ).to_dict()

        # 4. Apply Mapping function and track mapping
        mapping_log = []

        def map_cols(df):
            new_cols = []
            for col in df.columns:
                if col == 'DATETIME':
                    new_cols.append(col)
                else:
                    # Try to find mapping
                    mapped = name_to_metric.get(col)
                    if mapped and str(mapped) != 'nan':
                        new_cols.append(str(mapped))
                        # Only log distinct mappings once
                        if not any(d['Original Column'] == col for d in mapping_log):
                            mapping_log.append({'Original Column': col, 'Mapped Metric': str(mapped)})
                    else:
                        new_cols.append(col)
                        if not any(d['Original Column'] == col for d in mapping_log):
                            mapping_log.append({'Original Column': col, 'Mapped Metric': 'Not Found (Kept Original)'})
            df.columns = new_cols
            return df

        df_w_mapped = map_cols(df_w_clean)
        # Reset log before second pass if you only want unique mappings from one file, 
        # OR keep it cumulative. Since columns should be identical, cumulative is fine or just rely on first.
        # We will just rely on the first map since columns should be identical.
        
        df_wo_mapped = map_cols(df_wo_clean)

        # Save to session
        st.session_state.tvs_header = header_wo # Save one header for export
        st.session_state.tvs_mapped_w_outlier = df_w_mapped
        st.session_state.tvs_mapped_wo_outlier = df_wo_mapped
        st.session_state.tvs_mapping_table = pd.DataFrame(mapping_log)
        
        return True

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return False

# Progress Indicator
steps = ["Upload & Map Data", "Configure & Split", "Visualize & Export"]
current = st.session_state.tvs_step
st.progress(current / len(steps), text=f"Step {current}: {steps[current-1]}")

# ==========================================
# STEP 1: UPLOAD & MAP
# ==========================================
if current == 1:
    st.header("Step 1: Upload Data & Select TDT")
    st.info("Upload the cleaned datasets. We will use the **TDT Survey Data** (loaded on Global Settings sidebar) to map point names to metrics and identify the Operational State.")

    # 1. TDT/Model Selection
    if st.session_state.survey_df is None:
        st.error("❌ TDT Data not found. Please go to the **Home** page and load your TDT files first.")
        st.stop()
    
    survey_df = st.session_state.survey_df
    tdt_list = sorted(survey_df['TDT'].unique())
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        sel_tdt = st.selectbox("Select TDT", tdt_list, index=0 if not st.session_state.selected_tdt else tdt_list.index(st.session_state.selected_tdt) if st.session_state.selected_tdt in tdt_list else 0)
        st.session_state.selected_tdt = sel_tdt
        
    with col_sel2:
        model_list = sorted(survey_df[survey_df['TDT'] == sel_tdt]['Model'].unique())
        sel_model = st.selectbox("Select Model", model_list, index=0 if not st.session_state.selected_model_survey else model_list.index(st.session_state.selected_model_survey) if st.session_state.selected_model_survey in model_list else 0)
        st.session_state.selected_model_survey = sel_model

    st.markdown("---")

    # 2. File Upload
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        outlier_file = st.file_uploader("Upload Cleaned Dataset **WITH OUTLIERS**", type=["csv"])
    with col_up2:
        no_outlier_file = st.file_uploader("Upload Cleaned Dataset **WITHOUT OUTLIERS**", type=["csv"])

    # --- VALIDATION LOGIC ---
    files_valid = False
    
    if outlier_file and no_outlier_file and sel_model:
        try:
            # Parse filenames (Expected format: CLEANED-{ModelName}-{Dates}-{Type}.csv)
            _, model_w, _ = cleaned_dataset_name_split(outlier_file.name)
            _, model_wo, _ = cleaned_dataset_name_split(no_outlier_file.name)
            
            # Validation Checks
            if model_w != model_wo:
                st.error(
                    f"⚠️ **File Mismatch**: The model names in the uploaded files do not match.\n\n"
                    f"- **With Outliers**: `{model_w}`\n"
                    f"- **Without Outliers**: `{model_wo}`"
                )
            elif model_w != sel_model:
                st.error(
                    f"⚠️ **Selection Mismatch**: The uploaded files (`{model_w}`) do not match the selected Model (`{sel_model}`)."
                )
            else:
                st.success(f"✅ **Validation Successful**: Files match the selected model `{sel_model}`.")
                files_valid = True
                
                # Auto-fill metadata if valid
                try:
                    site, _, dates = cleaned_dataset_name_split(outlier_file.name)
                    if not st.session_state.model_name: st.session_state.model_name = sel_model
                    if not st.session_state.site_name: st.session_state.site_name = site
                    if not st.session_state.inclusive_dates: st.session_state.inclusive_dates = dates
                except: pass

        except Exception as e:
            st.warning(f"Could not automatically validate filenames. Please ensure they follow the naming convention.\n\nError: {e}")
            # If we can't validate, we might still allow the user to proceed at their own risk, 
            # or strictly block. Blocking is safer for "Wizard" flows.
            files_valid = False

    # 3. Load Button (Disabled unless validation passes)
    if st.button("Load & Map Data", type="primary", disabled=not files_valid):
        if outlier_file and no_outlier_file and sel_model:
            with st.spinner("Mapping columns using TDT definition..."):
                success = load_and_map_data(outlier_file, no_outlier_file, survey_df, sel_model)
                # Note: We DO NOT call next_step() here immediately anymore.
                # We let the user review the mapping table below first.
                st.rerun()

    # 4. Mapping Review Section (Visible only if data mapped)
    if st.session_state.tvs_mapping_table is not None:
        st.markdown("### 📋 Mapping Review")
        st.info("Please review the column mappings below. Columns marked 'Not Found' will retain their original names.")
        
        with st.expander("View Column Mapping Table", expanded=True):
            st.dataframe(st.session_state.tvs_mapping_table, use_container_width=True)
            
        # Add a dedicated "Proceed" button
        if st.button("Confirm Mapping & Proceed ➡️", type="primary"):
            next_step()
            st.rerun()

# ==========================================
# STEP 2: CONFIGURE & SPLIT
# ==========================================
elif current == 2:
    st.header("Step 2: Configure Split")
    
    if st.session_state.tvs_mapped_wo_outlier is None:
        st.error("No data loaded.")
        st.button("Back", on_click=prev_step)
        st.stop()

    df_wo = st.session_state.tvs_mapped_wo_outlier
    df_w = st.session_state.tvs_mapped_w_outlier
    survey_df = st.session_state.survey_df
    model_name = st.session_state.selected_model_survey

    # 1. Identify Operational State
    # Look for 'Constraint' == 'Yes' or 'Function' == 'Operational State'
    model_survey = survey_df[survey_df['Model'] == model_name]
    
    # Try to find constrained point
    op_state_rows = model_survey[model_survey['Constraint'] == 'Yes']
    op_state_metric = None
    if not op_state_rows.empty:
        op_state_metric = op_state_rows['Metric'].iloc[0]
    else:
        # Fallback to function
        op_state_rows = model_survey[model_survey['Function'] == 'Operational State']
        if not op_state_rows.empty:
             op_state_metric = op_state_rows['Metric'].iloc[0]

    # UI for Op State
    st.subheader("Split Configuration")
    col_cfg1, col_cfg2 = st.columns(2)
    
    with col_cfg1:
        # Allow user to override Op State if detection failed or is wrong
        numeric_cols = [c for c in df_wo.columns if c != 'DATETIME']
        default_idx = numeric_cols.index(op_state_metric) if op_state_metric in numeric_cols else 0
        
        selected_op_state = st.selectbox(
            "Operational State (Stratification Variable)", 
            numeric_cols, 
            index=default_idx,
            help="The variable used to stratify the split (ensure balanced ranges)."
        )

    with col_cfg2:
        split_ratio = st.slider("Train Size ratio", 0.5, 0.9, 0.8, 0.05)
        test_ratio = 1.0 - split_ratio
        st.caption(f"Train: {int(split_ratio*100)}% | Validation: {int(test_ratio*100)}%")

    # 2. Identify Feature Subset (Fault Detection points)
    # Filter dataset to only include 'Fault Detection' and 'Op State' points defined in TDT
    relevant_metrics = model_survey[model_survey['Function'].isin(['Operational State', 'Fault Detection'])]['Metric'].tolist()
    # Add selected op state if missing
    if selected_op_state not in relevant_metrics:
        relevant_metrics.append(selected_op_state)
    
    # Filter dataset columns
    cols_to_keep = ['DATETIME'] + [c for c in df_wo.columns if c in relevant_metrics]
    
    st.info(f"Using **{len(cols_to_keep)-1}** modeled features found in TDT (out of {len(df_wo.columns)-1} total in CSV).")
    
    if st.button("🚀 Run Stratified Split", type="primary"):
        with st.spinner("Running Verstack Stratified Continuous Split..."):
            try:
                # Prepare dataset for split
                dataset_for_split = df_wo[cols_to_keep].copy()
                
                # Run SC Split
                ds_train, ds_validate = scsplit(
                    dataset_for_split,
                    stratify=dataset_for_split[selected_op_state],
                    test_size=test_ratio,
                    train_size=split_ratio,
                    continuous=True,
                    random_state=42 # Fixed seed for reproducibility
                )
                
                # Map back to full dataframes
                # Train = Subset of WO-Outlier matching timestamps
                # Val WO = Subset of WO-Outlier matching timestamps
                # Val W = Subset of W-Outlier NOT in Train timestamps
                
                train_dates = ds_train['DATETIME']
                val_dates = ds_validate['DATETIME']
                
                final_train = df_wo[df_wo['DATETIME'].isin(train_dates)].reset_index(drop=True)
                final_val_wo = df_wo[df_wo['DATETIME'].isin(val_dates)].reset_index(drop=True)
                
                # For Validation With Outlier, it is (All With Outlier) - (Train Data)
                # Note: We assume timestamps align perfectly.
                final_val_w = df_w[~df_w['DATETIME'].isin(train_dates)].reset_index(drop=True)
                
                # Save to state
                st.session_state.ds_result_train = final_train
                st.session_state.ds_result_val_wo = final_val_wo
                st.session_state.ds_result_val_w = final_val_w
                
                st.success("Split Complete!")
                
                # We deliberately do not proceed automatically.
                # The results will be shown in the persistent block below.

            except Exception as e:
                st.error(f"Split failed: {e}")

    # --- Persistent Result View & Proceed Button ---
    if st.session_state.ds_result_train is not None:
        st.divider()
        st.subheader("Split Results")
        
        final_train = st.session_state.ds_result_train
        final_val_wo = st.session_state.ds_result_val_wo
        final_val_w = st.session_state.ds_result_val_w
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Training Set", f"{len(final_train):,}")
        c2.metric("Validation Set (w/o Outlier)", f"{len(final_val_wo):,}")
        c3.metric("Validation Set (w/ Outlier)", f"{len(final_val_w):,}")
        
        # Proceed Button
        if st.button("Proceed to Visualize & Export ➡️", type="primary"):
            next_step()
            st.rerun()

    st.button("Back", on_click=prev_step)

# ==========================================
# STEP 3: VISUALIZE & EXPORT
# ==========================================
elif current == 3:
    st.header("Step 3: Visualize & Export")
    
    if st.session_state.ds_result_train is None:
        st.error("Split results missing.")
        st.button("Back", on_click=prev_step)
        st.stop()

    train_df = st.session_state.ds_result_train
    
    # 1. Metadata Configuration (Moved to end)
    st.subheader("Export Configuration")
    with st.container(border=True):
        mc1, mc2 = st.columns(2)
        with mc1:
            st.session_state.site_name = st.text_input("Site Name", value=st.session_state.site_name)
            st.session_state.system_name = st.text_input("System Name", value=st.session_state.system_name)
            st.session_state.model_name = st.text_input("Model Name", value=st.session_state.model_name)
        with mc2:
            st.session_state.sprint_name = st.text_input("Sprint Name", value=st.session_state.sprint_name)
            st.session_state.inclusive_dates = st.text_input("Inclusive Dates", value=st.session_state.inclusive_dates)

    # 2. Visualizations
    st.subheader("Training Set Visualization")
    
    # Multiselect for metrics (default to top 3 to be fast)
    numeric_cols = [c for c in train_df.columns if c != 'DATETIME']
    selected_viz_metrics = st.multiselect(
        "Select metrics to visualize (Line Plot + Histogram)", 
        numeric_cols,
        default=numeric_cols
    )
    
    show_viz = st.checkbox("Show Plots in App", value=False)
    
    if show_viz and selected_viz_metrics:
        st.markdown("---")
        # Pass the validation set (without outliers) for comparison plotting
        generate_tvs_visualizations(
            train_df, 
            selected_viz_metrics, 
            df_val=st.session_state.ds_result_val_wo
        )
        st.markdown("---")

    # 3. Export Action
    if st.button("💾 Save Files & Generate Report", type="primary"):
        if not all([st.session_state.site_name, st.session_state.system_name, st.session_state.sprint_name]):
            st.error("Please fill in all metadata fields.")
        else:
            with st.spinner("Saving files and generating report..."):
                # Paths
                # USE GLOBAL BASE PATH
                base_path = Path(st.session_state.get('base_path', Path.cwd()))
                folder_path = base_path / st.session_state.site_name / st.session_state.system_name / st.session_state.sprint_name / st.session_state.model_name / "data_splitting"
                folder_path.mkdir(parents=True, exist_ok=True)
                
                # Naming Convention
                prefix = f"{st.session_state.model_name}-{st.session_state.inclusive_dates}"
                f_train = folder_path / f"TRAINING-{prefix}-WITHOUT-OUTLIER.csv"
                f_val_w = folder_path / f"VALIDATION-{prefix}-WITH-OUTLIER.csv"
                f_val_wo = folder_path / f"VALIDATION-{prefix}-WITHOUT-OUTLIER.csv"
                f_report = folder_path / f"{prefix}-TRAIN-VAL-SPLIT-REPORT.pdf"

                # Prepare Data (Attach Header)
                header = st.session_state.tvs_header
                
                # Helper to save
                def save_with_header(df, path):
                    # Temporarily adjust columns to match header to allow concat
                    df_to_save = df.copy()
                    df_to_save.columns = header.columns 
                    final_df = pd.concat([header, df_to_save], ignore_index=True)
                    final_df.to_csv(path, index=False)

                save_with_header(train_df, f_train)
                save_with_header(st.session_state.ds_result_val_w, f_val_w)
                save_with_header(st.session_state.ds_result_val_wo, f_val_wo)

                # Report Generation
                # Generate plots for *all* selected metrics (or just selected) for the PDF
                # If none selected, maybe auto-select top 10?
                metrics_for_report = selected_viz_metrics if selected_viz_metrics else numeric_cols[:10]
                
                # Generate images in memory
                # Include the validation set here as well for the PDF report
                plot_images = generate_tvs_visualizations(
                    train_df, 
                    metrics_for_report, 
                    df_val=st.session_state.ds_result_val_wo,
                    display_plot=False
                )
                
                # Stats for report
                stats = {
                    "original_len": len(st.session_state.tvs_mapped_wo_outlier),
                    "train_len": len(train_df),
                    "val_wo_len": len(st.session_state.ds_result_val_wo),
                    "val_w_len": len(st.session_state.ds_result_val_w)
                }

                # Generate PDF
                generate_tvs_report(stats, plot_images, f_report)

                st.success(f"Files saved to `{folder_path}`")
                st.success(f"Report generated: `{f_report.name}`")

    st.button("Back", on_click=prev_step)