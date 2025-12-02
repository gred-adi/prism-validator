import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Import utils
from utils.model_dev_utils import cleaned_dataset_name_split, split_holdout, generate_split_holdout_report

st.set_page_config(page_title="Split Holdout Dataset", page_icon="2️⃣", layout="wide")

# --- Initialize Session State ---
if 'holdout_step' not in st.session_state: st.session_state.holdout_step = 1

# Data States
if 'holdout_raw_df' not in st.session_state: st.session_state.holdout_raw_df = None
if 'holdout_cleaned_df' not in st.session_state: st.session_state.holdout_cleaned_df = None
if 'holdout_raw_header' not in st.session_state: st.session_state.holdout_raw_header = None
if 'holdout_split_mark_date' not in st.session_state: st.session_state.holdout_split_mark_date = None
if 'holdout_split_pct' not in st.session_state: st.session_state.holdout_split_pct = 0.15

# Metadata States
if 'site_name' not in st.session_state: st.session_state.site_name = ""
if 'system_name' not in st.session_state: st.session_state.system_name = ""
if 'model_name' not in st.session_state: st.session_state.model_name = ""
if 'sprint_name' not in st.session_state: st.session_state.sprint_name = ""
if 'inclusive_dates' not in st.session_state: st.session_state.inclusive_dates = ""

# --- Helper Functions ---
def set_step(step):
    st.session_state.holdout_step = step

def next_step():
    st.session_state.holdout_step += 1

def prev_step():
    st.session_state.holdout_step -= 1

def load_data(raw_file, cleaned_file):
    """Reads files and populates session state using dynamic header detection."""
    try:
        # Parse filename for metadata defaults
        try:
            site, model, dates = cleaned_dataset_name_split(cleaned_file.name)
            if not st.session_state.site_name: st.session_state.site_name = site
            if not st.session_state.model_name: st.session_state.model_name = model
            if not st.session_state.inclusive_dates: st.session_state.inclusive_dates = dates
        except Exception:
            # Fallback if filename parsing fails (non-standard format)
            pass
        
        # Read Files with low_memory=False to avoid DtypeWarning on mixed columns
        # We read without header initially to inspect the structure
        raw_df_full = pd.read_csv(raw_file, header=None, low_memory=False)
        cleaned_df_full = pd.read_csv(cleaned_file, header=None, low_memory=False)
        
        # --- Dynamic Header Detection ---
        # Instead of hardcoding 4 rows, we find where the actual data starts.
        # We assume the first column contains timestamps.
        
        # Coerce the first column to datetime. Errors become NaT.
        # The first valid date index is the start of our data.
        first_col_dates = pd.to_datetime(raw_df_full.iloc[:, 0], errors='coerce')
        data_start_idx = first_col_dates.first_valid_index()
        
        if data_start_idx is None:
            # Fallback: If detection fails, try to force standard 4 rows
            st.warning("Could not auto-detect timestamp column. Assuming standard 4 header rows.")
            data_start_idx = 4
            
        # Split Header and Data based on detected index
        raw_header = raw_df_full.iloc[:data_start_idx].copy()
        raw_data = raw_df_full.iloc[data_start_idx:].copy()
        cleaned_data = cleaned_df_full.iloc[data_start_idx:].copy()
        
        # Assign Column Names
        # Standard PRISM files usually have column names in the 2nd row (index 1)
        # If detected data start is > 1, we try to grab row index 1 for names.
        # If data starts at 0 or 1, we might not have headers or they are at 0.
        name_row_idx = 1 if data_start_idx > 1 else 0
        col_names = raw_df_full.iloc[name_row_idx].astype(str).values
        
        raw_data.columns = col_names
        cleaned_data.columns = col_names
        
        # Rename first column to 'DATETIME' for internal consistency
        # (Using columns[0] ensures we target the right one regardless of name)
        raw_data.rename(columns={raw_data.columns[0]: 'DATETIME'}, inplace=True)
        cleaned_data.rename(columns={cleaned_data.columns[0]: 'DATETIME'}, inplace=True)
        
        # Convert to datetime using coerce to handle any lingering metadata rows
        raw_data['DATETIME'] = pd.to_datetime(raw_data['DATETIME'], errors='coerce')
        cleaned_data['DATETIME'] = pd.to_datetime(cleaned_data['DATETIME'], errors='coerce')
        
        # Drop rows where DATETIME became NaT (this cleans up any edge cases)
        raw_data = raw_data.dropna(subset=['DATETIME'])
        cleaned_data = cleaned_data.dropna(subset=['DATETIME'])
        
        # Sort by time
        raw_data.sort_values('DATETIME', inplace=True)
        cleaned_data.sort_values('DATETIME', inplace=True)
        
        st.session_state.holdout_raw_df = raw_data
        st.session_state.holdout_cleaned_df = cleaned_data
        st.session_state.holdout_raw_header = raw_header # Save exact header block to restore later
        
        return True
    except Exception as e:
        st.error(f"Error reading files: {e}")
        return False

# --- Page Layout ---
st.title("🔀 Holdout Splitting Wizard")
st.markdown("""
This tool splits your cleaned dataset into two parts: a **Training/Validation set** and a **Holdout set**.
The Holdout set is a portion of the most recent data that is kept separate and is used for the final, unbiased evaluation of the model.

**How to use:**
1.  **Upload Data (Step 1):** Upload both the `RAW` and `CLEANED` versions of your dataset.
2.  **Configure Split (Step 2):** Use the slider to define the percentage of recent data to allocate to the Holdout set. The timeline visualization helps you see the exact split point.
3.  **Export & Report (Step 3):** Confirm the output metadata, and the tool will save the `Train/Validation` and `Holdout` sets as separate CSV files. A PDF report summarizing the split will also be generated.
""")
steps = ["Upload Data", "Configure Split", "Export & Report"]
current_step = st.session_state.holdout_step
st.progress(current_step / len(steps), text=f"Step {current_step}: {steps[current_step-1]}")

# ==========================================
# STEP 1: UPLOAD DATA
# ==========================================
if current_step == 1:
    st.header("Step 1: Upload Data")
    st.write("Upload the **RAW** and **CLEANED** datasets. These files should be the output of the Data Cleansing step.")
    
    col1, col2 = st.columns(2)
    with col1:
        raw_file = st.file_uploader("Upload RAW dataset", type=["csv"], key="u_raw")
    with col2:
        cleaned_file = st.file_uploader("Upload CLEANED dataset", type=["csv"], key="u_clean")
        
    if raw_file and cleaned_file:
        if st.button("Load Files & Next", type="primary"):
            with st.spinner("Reading and processing files..."):
                success = load_data(raw_file, cleaned_file)
                if success:
                    next_step()
                    st.rerun()

# ==========================================
# STEP 2: CONFIGURE SPLIT
# ==========================================
elif current_step == 2:
    st.header("Step 2: Interactive Split Configuration")
    
    if st.session_state.holdout_cleaned_df is None:
        st.error("No data loaded.")
        st.button("Back", on_click=prev_step)
        st.stop()
        
    df = st.session_state.holdout_cleaned_df
    
    # --- Split Control (Slider) ---
    st.subheader("Define Holdout Size")
    st.write("Drag the slider to define how much recent data to set aside for the Holdout set.")
    
    # Slider returns percentage (0-100)
    split_pct_val = st.slider(
        "Holdout Percentage (%)", 
        min_value=5, 
        max_value=50, 
        value=int(st.session_state.holdout_split_pct * 100),
        format="%d%%",
        help="The percentage of data from the END of the timeline to reserve for Holdout."
    )
    
    # Convert to float (0.0 - 1.0) and index
    split_fraction = split_pct_val / 100.0
    st.session_state.holdout_split_pct = split_fraction
    
    total_rows = len(df)
    holdout_count = int(total_rows * split_fraction)
    train_count = total_rows - holdout_count
    
    # Determine the split DATE based on the count
    # We take the timestamp at the split index
    if total_rows > 0:
        split_date = df.iloc[train_count]['DATETIME']
    else:
        split_date = datetime.now()
        
    st.session_state.holdout_split_mark_date = split_date

    # --- Visualization: Density & Timeline ---
    st.subheader("Split Preview")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 1. Plot Histogram/Density of Data Availability
    # We use timestamps converted to numbers for histogramming
    dates = mdates.date2num(df['DATETIME'])
    
    # Plot histogram (density of points over time)
    n, bins, patches = ax.hist(dates, bins=100, color='#e0e0e0', edgecolor='white', label='Data Density')
    
    # 2. Add Vertical Line for Split
    split_num = mdates.date2num(split_date)
    ax.axvline(split_num, color='#FF4B4B', linestyle='--', linewidth=2, label=f'Split: {split_date.strftime("%Y-%m-%d")}')
    
    # 3. Highlight Regions
    # Get Time Limits
    start_num = mdates.date2num(df['DATETIME'].iloc[0])
    end_num = mdates.date2num(df['DATETIME'].iloc[-1])
    
    # Train Region (Green)
    ax.axvspan(start_num, split_num, color='#4CAF50', alpha=0.2, label=f'Train/Val ({100-split_pct_val}%)')
    # Holdout Region (Orange)
    ax.axvspan(split_num, end_num, color='#FFA726', alpha=0.3, label=f'Holdout ({split_pct_val}%)')
    
    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=15)
    ax.set_yticks([]) # Hide y-axis counts (density is relative)
    ax.set_title("Data Density Timeline & Split Point")
    ax.legend(loc='upper left')
    sns.despine(left=True)
    
    st.pyplot(fig)
    plt.close(fig)
    
    # --- Impact Stats Table ---
    st.markdown("### Segment Details")
    c1, c2 = st.columns(2)
    
    with c1:
        st.info(f"**Train / Validation Set**\n\n"
                f"📅 **End:** {split_date.strftime('%Y-%m-%d %H:%M')}\n\n"
                f"🔢 **Rows:** {train_count:,} ({(1-split_fraction)*100:.1f}%)")
        
    with c2:
        st.warning(f"**Holdout Set**\n\n"
                   f"📅 **Start:** {split_date.strftime('%Y-%m-%d %H:%M')}\n\n"
                   f"🔢 **Rows:** {holdout_count:,} ({split_fraction*100:.1f}%)")
        
    st.markdown("---")
    c_back, c_next = st.columns([1, 5])
    with c_back:
        st.button("⬅️ Back", on_click=prev_step)
    with c_next:
        st.button("Next: Review & Export ➡️", on_click=next_step, type="primary")

# ==========================================
# STEP 3: EXPORT
# ==========================================
elif current_step == 3:
    st.header("Step 3: Review & Export")
    
    st.write("Please confirm the metadata for the file naming convention.")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.site_name = st.text_input("Site Name", value=st.session_state.site_name)
            st.session_state.system_name = st.text_input("System Name", value=st.session_state.system_name)
            st.session_state.model_name = st.text_input("Model Name", value=st.session_state.model_name)
        with col2:
            st.session_state.sprint_name = st.text_input("Sprint Name", value=st.session_state.sprint_name)
            st.session_state.inclusive_dates = st.text_input("Inclusive Dates", value=st.session_state.inclusive_dates, help="Format: YYYYMMDD-YYYYMMDD")

    st.divider()
    
    # --- Calculate Stats for Preview ---
    split_mark = st.session_state.holdout_split_mark_date
    raw_df = st.session_state.holdout_raw_df
    cleaned_df = st.session_state.holdout_cleaned_df
    
    # Filter subsets
    train_val_df = cleaned_df[cleaned_df['DATETIME'] <= split_mark]
    holdout_df = cleaned_df[cleaned_df['DATETIME'] > split_mark]
    
    # Helper to calc stats row
    def get_stat_row(name, df):
        if df.empty:
            return {"Dataset": name, "Start Time": "N/A", "End Time": "N/A", "Rows": 0, "Duration (Days)": 0}
        start = df['DATETIME'].iloc[0]
        end = df['DATETIME'].iloc[-1]
        return {
            "Dataset": name,
            "Start Time": start.strftime("%Y-%m-%d %H:%M"),
            "End Time": end.strftime("%Y-%m-%d %H:%M"),
            "Rows": f"{len(df):,}",
            "Duration (Days)": (end - start).days + 1
        }

    stats_data = [
        get_stat_row("Raw", raw_df),
        get_stat_row("Cleaned", cleaned_df),
        get_stat_row("Train / Validation", train_val_df),
        get_stat_row("Holdout", holdout_df)
    ]
    stats_df_preview = pd.DataFrame(stats_data)

    # --- Visualization: Re-render Timeline ---
    st.subheader("Split Visualization")
    fig, ax = plt.subplots(figsize=(10, 3))
    dates = mdates.date2num(cleaned_df['DATETIME'])
    ax.hist(dates, bins=100, color='#e0e0e0', edgecolor='white', label='Data Density')
    split_num = mdates.date2num(split_mark)
    ax.axvline(split_num, color='#FF4B4B', linestyle='--', linewidth=2, label='Split Mark')
    
    # Highlight Regions
    start_num = mdates.date2num(cleaned_df['DATETIME'].iloc[0]) if not cleaned_df.empty else 0
    end_num = mdates.date2num(cleaned_df['DATETIME'].iloc[-1]) if not cleaned_df.empty else 0
    ax.axvspan(start_num, split_num, color='#4CAF50', alpha=0.2, label='Train/Val')
    ax.axvspan(split_num, end_num, color='#FFA726', alpha=0.3, label='Holdout')
    
    ax.set_title(f"Dataset Split (Split Date: {split_mark.strftime('%Y-%m-%d')})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=15)
    ax.legend(loc='upper left')
    st.pyplot(fig)
    
    # --- Statistics Table ---
    st.subheader("Split Statistics")
    st.dataframe(stats_df_preview, hide_index=True, use_container_width=True)
    
    st.divider()
    
    if st.button("🚀 Process Split & Save Files", type="primary"):
        meta_filled = all([
            st.session_state.site_name, st.session_state.system_name, 
            st.session_state.model_name, st.session_state.sprint_name, 
            st.session_state.inclusive_dates
        ])
        
        if not meta_filled:
            st.error("Please fill in all metadata fields above.")
        else:
            with st.spinner("Processing split and generating report..."):
                # Use the utility to perform the actual split and stats generation
                # We pass the calculated split date
                
                # Note: We pass remove_header_rows=0 because we already stripped headers
                # in the dynamic load_data function.
                train_val, holdout, used_mark, stats = split_holdout(
                    st.session_state.holdout_raw_df,
                    st.session_state.holdout_cleaned_df,
                    split_mark,
                    date_col="DATETIME", 
                    remove_header_rows=0 
                )
                
                # Restore Header: Prepend the original metadata rows
                header_df = st.session_state.holdout_raw_header
                
                # Align column names for concatenation
                train_val.columns = header_df.columns
                holdout.columns = header_df.columns
                
                final_train_val = pd.concat([header_df, train_val], ignore_index=True)
                final_holdout = pd.concat([header_df, holdout], ignore_index=True)
                
                # --- Save Files ---
                # USE GLOBAL BASE PATH
                base_path = Path(st.session_state.get('base_path', Path.cwd()))
                dataset_path = base_path / st.session_state.site_name / st.session_state.system_name / st.session_state.sprint_name / st.session_state.model_name / "dataset"
                dataset_path.mkdir(parents=True, exist_ok=True)

                train_val_out = dataset_path / f"CLEANED-{st.session_state.model_name}-{st.session_state.inclusive_dates}-WITH-OUTLIER.csv"
                holdout_out = dataset_path / f"{st.session_state.model_name}-{st.session_state.inclusive_dates}-HOLDOUT.csv"
                report_out = dataset_path / f"{st.session_state.model_name}-{st.session_state.inclusive_dates}-SPLIT-HOLDOUT-REPORT.pdf"

                final_train_val.to_csv(train_val_out, index=False, header=False) 
                final_holdout.to_csv(holdout_out, index=False, header=False)
                
                st.success(f"✅ Train/Validation set saved: `{train_val_out.name}`")
                st.success(f"✅ Holdout set saved: `{holdout_out.name}`")
                
                # --- Generate PDF Report ---
                # Pass the FIGURE generated above for the report
                report_success = generate_split_holdout_report(
                    stats,
                    str(split_mark),
                    report_out,
                    fig
                )
                plt.close(fig)
                
                if report_success:
                    st.success(f"📄 Report generated: `{report_out.name}`")
                else:
                    st.warning("Report generation failed.")
                    
    st.markdown("---")
    st.button("⬅️ Back", on_click=prev_step)
