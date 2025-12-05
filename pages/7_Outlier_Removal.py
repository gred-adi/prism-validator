import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from playwright.sync_api import sync_playwright
import sys
import asyncio
from jinja2 import Environment
import re
from datetime import datetime

# Import Utils
from utils.model_dev_utils import cleaned_dataset_name_split, read_prism_csv
from utils.outlier_detection_utils import (
    detect_outliers_series, 
    detect_multivariate_outliers, 
    generate_outlier_plots,
    generate_correlation_analysis,
    generate_pairplot_visuals
)

st.set_page_config(page_title="Outlier Removal", page_icon="❇️", layout="wide")

st.title("❇️ Outlier Removal Wizard")
st.markdown("""
Refine your dataset by removing statistical outliers and anomalies. 
The wizard helps you analyze metric relationships to choose between **Pairwise Detection** (analyzing relationships against an operational state) or **Multivariate Detection** (analyzing the global structure).

**How to Use:**
1.  **Select Model:** Ensure your TDT files are processed in the Global Settings sidebar. Select the desired **TDT** and **Model**.
2.  **Upload Data:** Upload the `CLEANED_...WITH-OUTLIER` file.
3.  **Analyze (Optional):** Review correlation stats and visualizations to decide the best strategy.
4.  **Configure:** Apply the chosen algorithm (e.g., RANSAC for pairwise, Isolation Forest for multivariate).
5.  **Export:** Save the clean dataset along with a PDF report.
""")

# --- Constants ---
ALGO_DESCRIPTIONS = {
    "isoforest": """**Isolation Forest:** An unsupervised algorithm that isolates anomalies by randomly partitioning the data. Outliers are 'easier' to isolate (require fewer splits) than normal points. Best for detecting global anomalies in complex, non-linear clusters.
    
    * Contamination: The expected proportion of outliers in the dataset. Adjust this based on how clean you believe your data is (e.g., 0.01 = 1%).""",
    
    "residual": """**RANSAC (Residuals):** Fits a robust regression model (ignoring existing outliers) to find the trend between the Feature and Operational State. Outliers are flagged based on their distance (residual) from this trend line. Best for features that have a strong linear relationship with the Operational State.
    
    * Residual Threshold: The Z-Score cutoff for residuals. A higher value (e.g., 3.5) means only very extreme deviations are flagged. A lower value (e.g., 2.0) is stricter.""",
    
    "lof": """**Local Outlier Factor (LOF):** Measures the local density deviation of a data point compared to its neighbors. It flags points that have a significantly lower density than their neighbors. Best for finding local outliers in clusters.
    
    * Contamination: The expected proportion of outliers in the dataset.
    * n_neighbors: The number of neighbors to consider for density estimation (default: 20).""",
    
    "iqr": """**Interquartile Range (IQR):** A simple statistical method. Calculates the range between the 25th and 75th percentiles (IQR). Points outside `Q1 - 1.5*IQR` and `Q3 + 1.5*IQR` are flagged. Best for simple, univariate cutoff filtering.
    
    * IQR Factor: The multiplier for the IQR range (standard is 1.5). Increasing this makes the filter more lenient.""",
    
    "pca_recon": """**PCA Reconstruction Error:** Projects the data into a lower-dimensional space using Principal Component Analysis (PCA) and then reconstructs it. Points with high reconstruction error (large difference between original and reconstructed value) violate the correlation structure of the system. Best for finding multi-sensor consistency issues.
    
    * Contamination: The expected proportion of outliers (points with the highest reconstruction errors) to remove.""",
    
    "isoforest_global": """**Isolation Forest (Global):** Applies the Isolation Forest algorithm to the entire dataset (all features simultaneously) without a specific target variable. Identifies points that are anomalous in the high-dimensional space.
    
    * Contamination: The expected proportion of outliers in the dataset."""
}

# --- Initialize Session State ---
if 'or_step' not in st.session_state: st.session_state.or_step = 1

# Data States
if 'or_raw_df' not in st.session_state: st.session_state.or_raw_df = None
if 'or_header' not in st.session_state: st.session_state.or_header = None
if 'or_mapped_df' not in st.session_state: st.session_state.or_mapped_df = None
if 'or_mapping_table' not in st.session_state: st.session_state.or_mapping_table = None 
if 'or_mask' not in st.session_state: st.session_state.or_mask = None 
if 'or_summary_stats' not in st.session_state: st.session_state.or_summary_stats = None
if 'or_plot_images' not in st.session_state: st.session_state.or_plot_images = []
if 'or_config_summary' not in st.session_state: st.session_state.or_config_summary = {}
if 'or_selected_features' not in st.session_state: st.session_state.or_selected_features = [] # Store selected features for final PairPlot

# Recommendation State
if 'or_recommendation' not in st.session_state: st.session_state.or_recommendation = None
if 'or_analysis_imgs' not in st.session_state: st.session_state.or_analysis_imgs = {}

# Metadata
if 'site_name' not in st.session_state: st.session_state.site_name = ""
if 'system_name' not in st.session_state: st.session_state.system_name = ""
if 'model_name' not in st.session_state: st.session_state.model_name = ""
if 'sprint_name' not in st.session_state: st.session_state.sprint_name = ""
if 'inclusive_dates' not in st.session_state: st.session_state.inclusive_dates = ""

# --- Helper Functions ---
def next_step(): st.session_state.or_step += 1
def prev_step(): st.session_state.or_step -= 1

def load_and_map_data(uploaded_file, survey_df, model_name):
    try:
        # Read Raw
        df_raw = pd.read_csv(uploaded_file, index_col=False, low_memory=False)
        # Separate Header and Data
        df_clean, header = read_prism_csv(df_raw)
        
        # Get Mapping
        model_survey = survey_df[survey_df['Model'] == model_name]
        point_col = 'Canary Point Name' if 'Canary Point Name' in model_survey.columns else 'Point Name'
        
        # Create map: Canary Point Name -> Metric
        name_to_metric = pd.Series(
            model_survey['Metric'].values,
            index=model_survey[point_col]
        ).to_dict()

        # Create map: Canary Point Name -> Function
        name_to_function = pd.Series(
            model_survey['Function'].values,
            index=model_survey[point_col]
        ).to_dict()
        
        # Also map Metric -> Function (for when we map by metric directly)
        metric_to_function = pd.Series(
            model_survey['Function'].values,
            index=model_survey['Metric']
        ).to_dict()

        # Helper to strip prefixes
        def strip_prefix(s):
            # Removes "AP-TVI-", "AP-TSI-" etc
            return re.sub(r'^AP-[A-Z]{3}-', '', str(s))

        # Map Columns
        new_cols = []
        mapping_log = [] # Track mappings for review

        for col in df_clean.columns:
            if col == 'DATETIME':
                new_cols.append(col)
            else:
                mapped_metric = None
                mapped_function = "N/A"
                
                # Strategy 1: Direct match with Point Name
                if col in name_to_metric:
                    mapped_metric = name_to_metric[col]
                    mapped_function = name_to_function.get(col, "N/A")
                
                # Strategy 2: Strip prefix and check if it matches a Metric Name directly
                if mapped_metric is None:
                    stripped_col = strip_prefix(col)
                    if stripped_col in metric_to_function:
                        mapped_metric = stripped_col
                        mapped_function = metric_to_function[stripped_col]

                # Final Assignment
                if mapped_metric and str(mapped_metric) != 'nan':
                    new_cols.append(str(mapped_metric))
                    mapping_log.append({
                        'Original Column': col, 
                        'Mapped Metric': str(mapped_metric),
                        'Function': str(mapped_function)
                    })
                else:
                    new_cols.append(col)
                    mapping_log.append({
                        'Original Column': col, 
                        'Mapped Metric': 'Not Found (Kept Original)',
                        'Function': 'N/A'
                    })
        
        df_clean.columns = new_cols
        
        st.session_state.or_raw_df = df_clean
        st.session_state.or_header = header
        st.session_state.or_mapped_df = df_clean # Save mapped version
        st.session_state.or_mapping_table = pd.DataFrame(mapping_log) # Save mapping log
        return True
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return False

def generate_report_html(stats, plot_data, config, strategy):
    formatted_stats = {
        "original": f"{stats['original']:,}",
        "cleaned": f"{stats['cleaned']:,}",
        "removed": f"{stats['removed']:,}",
        "pct": stats['pct']
    }

    env = Environment()
    template_str = """
    <html>
    <head>
        <style>
            @page { size: A4 portrait; margin: 2cm; }
            @page landscape { size: A4 landscape; margin: 2cm; }
            
            body { font-family: "Helvetica", "Arial", sans-serif; color: #333; margin: 0; }
            
            h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
            h2 { color: #2980b9; margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            .section { margin-bottom: 20px; }
            
            /* Stats & Config Table */
            .info-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px; }
            .info-table th, .info-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .info-table th { background-color: #f2f2f2; font-weight: bold; width: 30%; }
            
            .description-box { background-color: #f9f9f9; padding: 15px; border-left: 5px solid #2980b9; margin-bottom: 20px; font-size: 13px; line-height: 1.5; }
            
            /* Page Break & Images */
            .page-break { page-break-before: always; }
            
            .landscape-section {
                page: landscape;
                width: 100%;
                page-break-before: always;
            }
            
            .img-container { text-align: center; margin-top: 10px; page-break-inside: avoid; }
            img { max-width: 95%; height: auto; border: 1px solid #ccc; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
            
            .plot-title { font-size: 16px; font-weight: bold; color: #333; margin-bottom: 5px; text-align: center; }
        </style>
    </head>
    <body>
        <h1>Outlier Removal Report</h1>
        <div style="font-size: 12px; color: #777; margin-bottom: 20px;">Generated on: {{ date }}</div>

        <!-- Processing Summary -->
        <div class="section">
            <h2>Processing Summary</h2>
            
            <table class="info-table">
                <tr><th>Strategy</th><td>{{ config.strategy }}</td></tr>
                <tr><th>Algorithm</th><td>{{ config.algorithm }}</td></tr>
                {% for key, val in config.params.items() %}
                <tr><th>{{ key }}</th><td>{{ val }}</td></tr>
                {% endfor %}
            </table>
            
            <h3>Impact Statistics</h3>
            <table class="info-table">
                <tr><th>Original Rows</th><td>{{ stats.original }}</td></tr>
                <tr><th>Cleaned Rows</th><td>{{ stats.cleaned }}</td></tr>
                <tr><th>Outliers Removed</th><td>{{ stats.removed }} ({{ stats.pct }}%)</td></tr>
            </table>

            <h3>Algorithm Description</h3>
            <div class="description-box">
                {{ config.description | safe }}
            </div>
        </div>
        
        <!-- PAIR PLOTS: BEFORE & AFTER -->
        <div class="page-break"></div>
        <h2>Data Distribution Analysis</h2>
        
        {% for item in plot_data %}
            {% if item.type == 'pairplot_before' %}
            <div class="section page-break">
                <div class="plot-title">Before Outlier Removal</div>
                <div class="img-container">
                    <img src="data:image/png;base64,{{ item.img }}" style="width: 100%;" />
                </div>
            </div>
            {% endif %}
        {% endfor %}
        
        {% for item in plot_data %}
            {% if item.type == 'pairplot_after' %}
            <div class="section page-break">
                <div class="plot-title">After Outlier Removal</div>
                <div class="img-container">
                    <img src="data:image/png;base64,{{ item.img }}" style="width: 100%;" />
                </div>
            </div>
            {% endif %}
        {% endfor %}
        
        <!-- OTHER VISUALIZATIONS -->
        <div class="page-break"></div>
        <h2>Detection Details</h2>
        
        {% for item in plot_data %}
            
            {% if item.type == 'pairwise' %}
            <div class="section page-break">
                <div class="plot-title">{{ item.title }}</div>
                <div class="img-container">
                    <p><strong>Time Series & Outliers</strong></p>
                    <img src="data:image/png;base64,{{ item.ts_img }}" style="width: 100%;" />
                </div>
                <div class="img-container">
                    <p><strong>Density Scatterplot</strong></p>
                    <img src="data:image/png;base64,{{ item.scatter_img }}" style="width: 80%;" />
                </div>
            </div>
            
            {% elif item.type == 'multivariate_summary' %}
            <div class="section">
                <div class="plot-title">{{ item.title }}</div>
                <div class="img-container">
                    <img src="data:image/png;base64,{{ item.img }}" />
                </div>
            </div>
            
            {% elif item.type == 'multivariate_ts' %}
            <div class="section landscape-section">
                <div class="plot-title">{{ item.title }}</div>
                <div class="img-container">
                    <img src="data:image/png;base64,{{ item.ts_img }}" style="width: 100%;" />
                </div>
            </div>
            {% endif %}
            
        {% endfor %}
    </body>
    </html>
    """
    template = env.from_string(template_str)
    return template.render(
        stats=formatted_stats, 
        plot_data=plot_data, 
        config=config, 
        strategy=strategy,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

# Progress Indicator
steps = ["Upload & Map", "Analyze & Recommend", "Configure & Detect", "Export"]
current = st.session_state.or_step
st.progress(current / len(steps), text=f"Step {current}: {steps[current-1]}")

# ==========================================
# STEP 1: UPLOAD
# ==========================================
if current == 1:
    st.header("Step 1: Upload Data")
    st.info("Please go to the **Global Settings** sidebar (on the left), upload your TDT Excel files, and click 'Generate & Load Files'. This is required to map metric names.")
   
    # TDT/Model Selection
    if st.session_state.survey_df is None:
        st.error("❌ TDT Data not found. Please go to the **Global Settings** sidebar and load your TDT files first.")
        st.stop()

    survey_df = st.session_state.survey_df
    
    c1, c2 = st.columns(2)
    with c1:
        tdt_list = sorted(survey_df['TDT'].unique())
        sel_tdt = st.selectbox("Select TDT", tdt_list)
    with c2:
        model_list = sorted(survey_df[survey_df['TDT'] == sel_tdt]['Model'].unique())
        sel_model = st.selectbox("Select Model", model_list)
        
    uploaded_file = st.file_uploader("Upload 'WITH-OUTLIER' Dataset", type=["csv"])
    
    if uploaded_file and sel_model:
        # Validate filename
        try:
            _, fname_model, _ = cleaned_dataset_name_split(uploaded_file.name)
            if fname_model != sel_model:
                st.warning(f"⚠️ Filename model (`{fname_model}`) doesn't match selected (`{sel_model}`). Proceed with caution.")
        except: pass
        
        if st.button("Load & Map Data", type="primary"):
            with st.spinner("Loading and mapping columns..."):
                if load_and_map_data(uploaded_file, survey_df, sel_model):
                    # Auto-fill metadata
                    try:
                        site, _, dates = cleaned_dataset_name_split(uploaded_file.name)
                        st.session_state.model_name = sel_model
                        if not st.session_state.site_name: st.session_state.site_name = site
                        if not st.session_state.inclusive_dates: st.session_state.inclusive_dates = dates
                    except: pass
                    # Do NOT auto-advance. Let user review mapping.
                    st.rerun()

    # --- Mapping Review Section ---
    if st.session_state.or_mapping_table is not None:
        st.divider()
        st.markdown("### 📋 Mapping Review")
        st.info("Please review the column mappings below. Columns marked 'Not Found' will retain their original names.")
        
        with st.expander("View Column Mapping Table", expanded=True):
            st.dataframe(st.session_state.or_mapping_table, use_container_width=True)
            
        if st.button("Confirm Mapping & Proceed ➡️", type="primary"):
            next_step()
            st.rerun()

# ==========================================
# STEP 2: ANALYZE & RECOMMEND (NEW)
# ==========================================
elif current == 2:
    st.header("Step 2: Analyze & Recommend")
    
    df = st.session_state.or_mapped_df
    if df is None:
        st.error("No data loaded.")
        st.button("Back", on_click=prev_step) # Ensure back button exists if data lost
        st.stop()
        
    numeric_cols = [c for c in df.columns if c != 'DATETIME']
    
    st.markdown("""
    This step helps identify relationships between variables to recommend an outlier detection strategy.
    
    * **Click 'Run Analysis'** to generate correlation heatmaps and receive a recommendation.
    * **Click 'Skip / Proceed'** to go directly to configuration if you already know your strategy.
    """)

    # --- Identify Focused Metrics ---
    focused_cols = []
    survey_df = st.session_state.survey_df
    model_name = st.session_state.model_name
    
    if survey_df is not None and model_name:
        m_survey = survey_df[survey_df['Model'] == model_name]
        if not m_survey.empty and 'Function' in m_survey.columns and 'Metric' in m_survey.columns:
            targets = m_survey[m_survey['Function'].isin(['Operational State', 'Fault Detection'])]['Metric'].tolist()
            focused_cols = [m for m in targets if m in numeric_cols]
            
    # --- Metrics Toggle ---
    col_anal1, col_anal2 = st.columns([3, 1])
    with col_anal1:
        if focused_cols:
            st.info(f"Defaulting to **{len(focused_cols)}** focused metrics (Operational State & Fault Detection).")
        else:
            st.info("No 'Operational State' or 'Fault Detection' tags found in TDT. Defaulting to first 6 numeric columns.")
            
    with col_anal2:
        use_all_metrics = st.toggle("Include all metrics", value=False, help="If on, analyzes all numeric columns. If off, focuses only on Operational State and Fault Detection.")

    # Logic: Only run if button clicked OR results already exist
    # Allow re-run if user changes toggle (by clicking Run again)
    if st.button("📊 Run Analysis", type="primary"):
        with st.spinner("Analyzing dataset correlations..."):
            
            # Determine columns based on toggle
            cols_to_analyze = numeric_cols if use_all_metrics else focused_cols
            if not cols_to_analyze: cols_to_analyze = numeric_cols[:6] # Fallback
            
            rec_text, rec_stats, rec_reason = generate_correlation_analysis(df, cols_to_analyze)
            imgs = generate_pairplot_visuals(df, cols_to_analyze)
            
            st.session_state.or_recommendation = {
                "strategy": rec_text,
                "stats": rec_stats,
                "reason": rec_reason
            }
            st.session_state.or_analysis_imgs = imgs
            st.rerun()
    
    # Display Recommendation & Visuals if available
    if st.session_state.or_recommendation:
        rec = st.session_state.or_recommendation
        
        if rec["strategy"] == "Pairwise":
            st.success(f"💡 **Recommendation: {rec['strategy']} Approach**")
        else:
            st.info(f"💡 **Recommendation: {rec['strategy']} Approach**")
            
        st.markdown(f"**Reason:** {rec['reason']}")
        
        # Display Visuals
        st.divider()
        st.subheader("Data Relationships")
        
        # We only have one image now (PairGrid), so full width is better
        if "pairplot" in st.session_state.or_analysis_imgs:
            st.image(f"data:image/png;base64,{st.session_state.or_analysis_imgs['pairplot']}", caption="Pairwise Relationships (Downsampled)", use_container_width=True)
            
    st.markdown("---")
    c_back, c_next = st.columns([1, 5])
    with c_back:
        st.button("⬅️ Back", on_click=prev_step)
    with c_next:
        # Button text changes based on whether analysis was run
        btn_text = "Proceed to Configuration ➡️" if st.session_state.or_recommendation else "Skip & Proceed to Configuration ⏩"
        st.button(btn_text, on_click=next_step, type="primary")

# ==========================================
# STEP 3: CONFIGURE & DETECT
# ==========================================
elif current == 3:
    st.header("Step 3: Configuration & Detection")
    
    df = st.session_state.or_mapped_df
    
    # --- Configuration UI ---
    st.subheader("Detection Strategy")
    
    # Pre-select based on recommendation if available
    default_strat_idx = 0
    if st.session_state.or_recommendation and st.session_state.or_recommendation["strategy"] == "Multivariate":
        default_strat_idx = 1
    
    strategy = st.radio(
        "Select Approach", 
        ["Pairwise (Op. State vs Features)", "Multivariate (Global Structure)"],
        index=default_strat_idx,
        horizontal=True
    )
    
    numeric_cols = [c for c in df.columns if c != 'DATETIME']
    
    # Determine Default Features based on TDT Function (Fault Detection / Op State)
    default_features = []
    default_op_state = None
    survey_df = st.session_state.survey_df
    model_name = st.session_state.model_name
    
    if survey_df is not None and model_name:
        # Filter for current model
        m_survey = survey_df[survey_df['Model'] == model_name]
        if not m_survey.empty and 'Function' in m_survey.columns and 'Metric' in m_survey.columns:
            # Get metrics with specific functions
            targets = m_survey[m_survey['Function'].isin(['Operational State', 'Fault Detection'])]['Metric'].tolist()
            # Intersect with available numeric columns in the uploaded file
            default_features = [m for m in targets if m in numeric_cols]
            
            # Identify Operational State
            op_state_rows = m_survey[m_survey['Function'] == 'Operational State']
            if not op_state_rows.empty:
                potential_op = op_state_rows['Metric'].iloc[0]
                if potential_op in numeric_cols:
                    default_op_state = potential_op
    
    # Fallback if no smart defaults found
    if not default_features:
        default_features = numeric_cols[:5]
    
    # Initialize variables for the algorithm call
    contamination = 0.01
    residual_thresh = 3.5
    iqr_factor = 1.5
    n_neighbors = 20
    percentile_cut = 0.01
    current_params = {}

    # 1. Pairwise Config
    if "Pairwise" in strategy:
        current_strategy = "pairwise"
        col1, col2 = st.columns(2)
        with col1:
            # Set default index for Op State
            op_state_idx = numeric_cols.index(default_op_state) if default_op_state else 0
            op_state = st.selectbox("Operational State (X-axis)", numeric_cols, index=op_state_idx)
            
            method = st.selectbox("Algorithm", ["isoforest", "residual", "lof", "iqr"], index=0)
        
        with col2:
            # --- Dynamic Hyperparameters ---
            if method == "isoforest":
                contamination = st.slider("Contamination (Expected Outlier %)", 0.001, 0.05, 0.005, 0.001, format="%.4f")
                current_params = {"Contamination": contamination}
            elif method == "residual":
                residual_thresh = st.slider("Residual Threshold (Z-Score)", 2.0, 5.0, 3.5, 0.1)
                current_params = {"Residual Threshold": residual_thresh}
            elif method == "lof":
                contamination = st.slider("Contamination (Expected Outlier %)", 0.001, 0.05, 0.005, 0.001, format="%.4f")
                n_neighbors = st.slider("N Neighbors", 5, 50, 20, 1)
                current_params = {"Contamination": contamination, "N Neighbors": n_neighbors}
            elif method == "iqr":
                iqr_factor = st.slider("IQR Factor", 1.0, 3.0, 1.5, 0.1)
                current_params = {"IQR Factor": iqr_factor}
        
        # Info Box for Algorithm
        st.info(ALGO_DESCRIPTIONS.get(method, ""))
        
        # Feature Selection
        # Ensure Op State isn't in the default feature list to check against itself
        smart_defaults = [f for f in default_features if f != op_state]
        
        target_features = st.multiselect(
            "Features to Check", 
            [c for c in numeric_cols if c != op_state], 
            default=smart_defaults
        )
        
        removal_logic = st.radio("Removal Logic", ["Strict (Remove row if outlier in ANY selected feature)", "Consensus (Remove row if outlier in 2+ features)"])

    # 2. Multivariate Config
    else:
        current_strategy = "multivariate"
        col1, col2 = st.columns(2)
        with col1:
            # Add Operational State selection for Visualization context
            # Set default index for Op State if found
            op_state_idx = numeric_cols.index(default_op_state) if default_op_state in numeric_cols else 0
            op_state = st.selectbox("Operational State (for Visualization context)", numeric_cols, index=op_state_idx)
            
            method = st.selectbox("Algorithm", ["pca_recon", "isoforest_global"]) 
        with col2:
            contamination = st.slider("Contamination (Expected Outlier %)", 0.001, 0.05, 0.01, 0.001)
            current_params = {"Contamination": contamination}
            
        # Info Box for Algorithm
        st.info(ALGO_DESCRIPTIONS.get(method, ""))
            
        target_features = st.multiselect(
            "Features included in Model", 
            numeric_cols, 
            default=default_features
        )
        # op_state is now used for visualization in multivariate too

    # --- Run Detection ---
    if st.button("🚀 Run Detection", type="primary"):
        if not target_features:
            st.error("Please select at least one feature.")
        else:
            # Store selected features for next step (PairGrid)
            st.session_state.or_selected_features = target_features
            
            with st.spinner("Running algorithms..."):
                final_mask = pd.Series(False, index=df.index)
                
                # Capture Config Summary for Report
                st.session_state.or_config_summary = {
                    "strategy": strategy,
                    "algorithm": method,
                    "params": current_params,
                    "description": ALGO_DESCRIPTIONS.get(method, "").replace("**", "<b>").replace("</b>:", "</b>").replace("\n", "<br>")
                }

                # Logic Execution
                if current_strategy == "pairwise":
                    # Run parallel flagging
                    temp_mask_df = pd.DataFrame(index=df.index)
                    progress_bar = st.progress(0)
                    
                    for i, feat in enumerate(target_features):
                        flag_series = detect_outliers_series(
                            df, op_state, feat, 
                            method=method, 
                            contamination=contamination, 
                            residual_threshold=residual_thresh,
                            n_neighbors=n_neighbors,
                            iqr_factor=iqr_factor
                        )
                        temp_mask_df[feat] = flag_series
                        progress_bar.progress((i+1)/len(target_features))
                    
                    progress_bar.empty()
                    
                    # Apply Logic
                    if "Strict" in removal_logic:
                        final_mask = temp_mask_df.any(axis=1)
                    else:
                        final_mask = (temp_mask_df.sum(axis=1) >= 2)
                        
                else: # Multivariate
                    final_mask = detect_multivariate_outliers(
                        df, target_features, method=method, contamination=contamination
                    )
                
                # Save State
                st.session_state.or_mask = final_mask
                
                # Stats
                n_total = len(df)
                n_removed = final_mask.sum()
                st.session_state.or_summary_stats = {
                    "original": n_total,
                    "removed": n_removed,
                    "cleaned": n_total - n_removed,
                    "pct": round((n_removed/n_total)*100, 2)
                }
                
                # Generate Plots
                # Limit features to avoid overloading the UI/Report
                features_to_plot = target_features[:10]
                
                st.session_state.or_plot_images = generate_outlier_plots(
                    df, 
                    final_mask, 
                    current_strategy, 
                    op_state, 
                    features_to_plot,
                    time_col='DATETIME' # Ensure we pass the time column
                )
                
                st.rerun()

    # --- Results View ---
    if st.session_state.or_mask is not None:
        st.divider()
        st.subheader("Detection Results")
        
        stats = st.session_state.or_summary_stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Original Rows", f"{stats['original']:,}")
        c2.metric("Outliers Found", f"{stats['removed']:,}")
        c3.metric("Cleaned Rows", f"{stats['cleaned']:,}", delta=f"-{stats['pct']:.2f}% removed", delta_color="inverse")
        
        st.subheader("Visualizations")
        # Display images stored in session state
        if st.session_state.or_plot_images:
            for item in st.session_state.or_plot_images:
                # Pairwise: Show Side-by-Side
                if item['type'] == 'pairwise':
                    st.markdown(f"#### {item['title']}")
                    vc1, vc2 = st.columns(2)
                    with vc1:
                        st.image(f"data:image/png;base64,{item['ts_img']}", caption="Time Series (Green=OpState, Blue=Feature, Red=Outlier)")
                    with vc2:
                        st.image(f"data:image/png;base64,{item['scatter_img']}", caption="Density Scatter (Red=Outlier)")
                    st.markdown("---")
                
                # Multivariate Summary (PCA)
                elif item['type'] == 'multivariate_summary':
                    st.markdown(f"#### {item['title']}")
                    st.image(f"data:image/png;base64,{item['img']}")
                    st.markdown("---")
                
                # Multivariate Time Series
                elif item['type'] == 'multivariate_ts':
                    # We can grid these to save space in UI
                    # But for now, simple list is fine
                    st.image(f"data:image/png;base64,{item['ts_img']}", caption=f"Time Series: {item['title']} (Red=Global Outlier)")

        
        if st.button("Proceed to Export ➡️", type="primary"):
            next_step()
            st.rerun()

    st.markdown("---")
    st.button("⬅️ Back", on_click=prev_step, key="back_step_2")

# ==========================================
# STEP 4: EXPORT
# ==========================================
elif current == 4:
    st.header("Step 4: Export")
    
    if st.session_state.or_mask is None:
        st.error("No results found.")
        st.stop()
        
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

    if st.button("💾 Save Files & Generate Report", type="primary"):
        if not all([st.session_state.site_name, st.session_state.system_name]):
            st.error("Please fill in metadata.")
        else:
            with st.spinner("Generating visuals and saving..."):
                # Paths
                base_path = Path(st.session_state.get('base_path', Path.cwd()))
                folder_path = base_path / st.session_state.site_name / st.session_state.system_name / st.session_state.sprint_name / st.session_state.model_name / "dataset"
                folder_path.mkdir(parents=True, exist_ok=True)
                
                # Apply Mask
                mask = st.session_state.or_mask
                df_raw = st.session_state.or_raw_df # Original mapped df
                df_clean = df_raw[~mask]
                df_outliers = df_raw[mask]
                header = st.session_state.or_header
                
                # --- NEW: Generate Before/After PairGrids for Report ---
                # Get the features used in detection (stored in Step 3)
                feats_to_plot = st.session_state.get('or_selected_features', [])
                if not feats_to_plot: # Fallback
                    feats_to_plot = [c for c in df_raw.columns if c != 'DATETIME'][:6]
                
                # Generate "Before" Plot
                imgs_before = generate_pairplot_visuals(df_raw, feats_to_plot, title_suffix="Before Removal")
                # Generate "After" Plot
                imgs_after = generate_pairplot_visuals(df_clean, feats_to_plot, title_suffix="After Removal")
                
                # Add these to the plot list for the report
                report_plots = st.session_state.or_plot_images.copy()
                if "pairplot" in imgs_before:
                    report_plots.insert(0, {'type': 'pairplot_before', 'img': imgs_before['pairplot']})
                if "pairplot" in imgs_after:
                    report_plots.insert(1, {'type': 'pairplot_after', 'img': imgs_after['pairplot']})
                
                # Filenames
                prefix = f"CLEANED-{st.session_state.model_name}-{st.session_state.inclusive_dates}"
                f_clean = folder_path / f"{prefix}-WITHOUT-OUTLIER.csv"
                f_outlier = folder_path / f"{prefix}-OUTLIERS-ONLY.csv"
                f_report = folder_path / f"{prefix}-OUTLIER-REPORT.pdf"
                
                # Save CSVs (Re-attach header)
                def save_w_header(df, path):
                    df_copy = df.copy()
                    df_copy.columns = header.columns 
                    final = pd.concat([header, df_copy], ignore_index=True)
                    final.to_csv(path, index=False)
                    
                save_w_header(df_clean, f_clean)
                save_w_header(df_outliers, f_outlier)
                
                # Generate PDF
                if sys.platform == "win32":
                    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                    
                html = generate_report_html(
                    st.session_state.or_summary_stats, 
                    report_plots, 
                    st.session_state.or_config_summary, # Pass full config
                    "Outlier Removal"
                )
                
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True)
                        page = browser.new_page()
                        page.set_content(html)
                        page.pdf(path=str(f_report), format="A4")
                        browser.close()
                    st.success(f"Report saved: `{f_report.name}`")
                except Exception as e:
                    st.error(f"PDF Error: {e}")
                    
                st.success(f"Files saved to `{folder_path}`")
                
                # Show the new plots in UI for confirmation
                st.markdown("### Generated Data Distribution Plots")
                c1, c2 = st.columns(2)
                with c1:
                    if "pairplot" in imgs_before:
                        st.image(f"data:image/png;base64,{imgs_before['pairplot']}", caption="Before Removal", use_container_width=True)
                with c2:
                    if "pairplot" in imgs_after:
                        st.image(f"data:image/png;base64,{imgs_after['pairplot']}", caption="After Removal", use_container_width=True)

    st.button("⬅️ Back", on_click=prev_step, key="back_step_4")