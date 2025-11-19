import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
from datetime import datetime, timedelta, time
from dateutil import parser as dt_parser
import tempfile
import shutil
import os
import random
import time as py_time
import io
import re


st.set_page_config(page_title="Canary Historian Downloader", layout="wide")
st.title("Canary Historian Data Downloader")


# --- TDT Consolidation Functions ---
def _process_survey_sheets(file_path, survey_data_list):
    """
    Processes the 'Point Survey' and related sheets from a single TDT Excel file.
    """
    try:
        xls = pd.ExcelFile(file_path)
        if 'Point Survey' not in xls.sheet_names:
            st.warning(f"'Point Survey' sheet not found in {os.path.basename(file_path)}")
            return

        df_version = pd.read_excel(xls, 'Version', header=None, nrows=5)
        tdt_name = str(df_version.iloc[4, 3])
        df_point_survey = pd.read_excel(xls, 'Point Survey', header=None)

        df_attribute = pd.DataFrame()
        if 'Attribute' in xls.sheet_names:
            df_attribute = pd.read_excel(xls, 'Attribute', header=None).iloc[3:, [1, 4, 5, 6, 7]]
            df_attribute.columns = ['Metric', 'Function', 'Constraint', 'Filter Condition', 'Filter Value']

        df_calculation = pd.DataFrame()
        if 'Calculation' in xls.sheet_names:
            df_calculation = pd.read_excel(xls, 'Calculation', header=None).iloc[2:, [1, 2, 3, 5, 6, 7, 8]]
            df_calculation.columns = ['Metric', 'Calc Point Type', 'Calculation Description', 'Pseudo Code', 'Language', 'Input Point', 'PRiSM Code']

        start_col = 3
        while start_col < df_point_survey.shape[1]:
            end_col = start_col + 5
            model_name = str(df_point_survey.iloc[0, start_col])
            if pd.notna(model_name) and model_name.strip() != "":
                headers = list(df_point_survey.iloc[1, 1:3].values) + list(df_point_survey.iloc[1, start_col:end_col].values)
                sub_data = pd.concat([df_point_survey.iloc[2:, 1:3], df_point_survey.iloc[2:, start_col:end_col]], axis=1)
                sub_data = sub_data.dropna(subset=sub_data.columns[-5:], how='all')

                if not sub_data.empty:
                    sub_table_df = pd.DataFrame(sub_data.values, columns=headers)
                    sub_table_df['TDT'] = tdt_name
                    sub_table_df['Model'] = model_name

                    if not df_attribute.empty:
                        sub_table_df = sub_table_df.merge(df_attribute, how='inner', on='Metric')
                    if not df_calculation.empty:
                        sub_table_df = sub_table_df.merge(df_calculation, how='left', on='Metric')

                    survey_data_list.append(sub_table_df)
            start_col = end_col
    except Exception as e:
        st.error(f"Failed to process file {os.path.basename(file_path)}: {e}")

@st.cache_data
def generate_survey_df_from_folder(folder_path):
    """
    Scans a folder for TDT Excel files and consolidates them into a single DataFrame.
    """
    if not folder_path or not os.path.isdir(folder_path):
        st.error("Invalid folder path provided.")
        return None

    tdt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx') and not f.startswith('~')]
    if not tdt_files:
        raise FileNotFoundError(f"No Excel files (.xlsx) found in the folder: {folder_path}")

    all_survey_data = []
    progress_bar = st.progress(0)
    for i, file_path in enumerate(tdt_files):
        _process_survey_sheets(file_path, all_survey_data)
        progress_bar.progress((i + 1) / len(tdt_files))
    progress_bar.empty()

    if not all_survey_data:
        raise ValueError("No valid survey data could be consolidated from the provided files.")

    survey_df = pd.concat(all_survey_data, ignore_index=True)
    survey_cols = ['TDT', 'Model'] + [col for col in survey_df.columns if col not in ['TDT', 'Model']]
    survey_df = survey_df[survey_cols]
    return survey_df

def convert_df_to_excel_bytes(survey_df):
    """
    Converts the survey DataFrame into an in-memory Excel file.
    """
    output_survey = io.BytesIO()
    with pd.ExcelWriter(output_survey, engine='xlsxwriter') as writer:
        survey_df.to_excel(writer, sheet_name='Consolidated Point Survey', index=False)
    output_survey.seek(0)
    return output_survey

# --- Constants & Helpers ---
# Replace with your actual API details or use Streamlit secrets
API_URL = "https://cch.aboitizpower.com:55236/api/v2/getTagData"
API_TOKEN = st.secrets.get("api", {}).get("token", "")
AGGREGATE_NAMES = [
    "Average", "Count", "CountInStateNonZero", "CountInStateZero", "Delta", "DeltaBounds",
    "DeltaTotalCount", "DurationBad", "DurationGood", "DurationInStateNonZero", "DurationInStateZero",
    "End", "EndBound", "Instant", "Interpolative", "Maximum", "Maximum2", "MaximumActualTime",
    "MaximumActualTime2", "Minimum", "Minimum2", "MinimumActualTime", "MinimumActualTime2",
    "NumberOfTransitions", "PercentBad", "PercentGood", "Range", "Range2",
    "StandardDeviationPopulation", "StandardDeviationSample", "Start", "StartBound", "TimeAverage",
    "TimeAverage2", "Total", "Total2", "TotalPer24Hours", "TotalPerHour", "TotalPerMinute",
    "VariancePopulation", "VarianceSample", "WorstQuality", "WorstQuality2"
]
# A reasonable limit for the number of data points in a single API call
MAX_POINTS_PER_CHUNK = 10_000

def calculate_time_chunks(start_dt, end_dt, interval_str, num_tags):
    """
    Splits the date range into time chunks based on a max point limit.
    Returns a list of (start, end) datetime tuples for each chunk.
    """
    total_duration_sec = (end_dt - start_dt).total_seconds()

    parts = list(map(int, interval_str.split(':')))
    interval_sec = parts[0] * 3600 + parts[1] * 60 + parts[2]

    if interval_sec == 0:
        return [(start_dt, end_dt)]

    points_per_second = num_tags / interval_sec

    total_points = points_per_second * total_duration_sec

    if total_points <= MAX_POINTS_PER_CHUNK:
        return [(start_dt, end_dt)]

    num_chunks = int(total_points / MAX_POINTS_PER_CHUNK) + 1

    chunk_duration_sec = total_duration_sec / num_chunks

    chunks = []
    current_start = start_dt
    for _ in range(num_chunks):
        current_end = current_start + timedelta(seconds=chunk_duration_sec)
        chunks.append((current_start, current_end))
        current_start = current_end

    # Ensure the very last chunk ends exactly at the user-defined end date
    chunks[-1] = (chunks[-1][0], end_dt)

    return chunks

# --- Main Logic: Determine which dataframe to use ---
df_survey = None
data_source_selected = False

if 'survey_df' in st.session_state:
    st.info("Using the TDT data from the Home page.")
    df_survey = st.session_state.survey_df
    data_source_selected = True
else:
    st.warning("Please generate the TDT files on the Home page first.")
    st.stop()

# --- Step 2: Filter & Load Metrics ---
if data_source_selected and df_survey is not None:
    st.divider()
    st.header("Step 2 — Filter & Load Metrics")

    required_cols = ["TDT", "Model", "Metric", "Canary Point Name", "Canary Description", "DCS Description", "Unit"]
    missing = [c for c in required_cols if c not in df_survey.columns]
    if missing:
        st.error(f"Uploaded sheet is missing required columns: {missing}")
        st.stop()

    tdt_list = sorted(df_survey["TDT"].dropna().unique())
    selected_tdt = st.selectbox("Select TDT", ["-- Select TDT --"] + tdt_list)
    selected_model = None
    if selected_tdt and selected_tdt != "-- Select TDT --":
        model_list = sorted(df_survey[df_survey["TDT"] == selected_tdt]["Model"].dropna().unique())
        selected_model = st.selectbox("Select Model", ["-- Select Model --"] + model_list)

    if st.button("Load Metrics") and selected_model and selected_model != "-- Select Model --":
        filtered_df = df_survey[
            (df_survey["TDT"] == selected_tdt) &
            (df_survey["Model"] == selected_model)
        ].reset_index(drop=True)
        st.session_state.filtered_df = filtered_df
        st.session_state.selected_model = selected_model

        display_cols = ["Metric", "DCS Description", "Canary Point Name", "Canary Description", "Unit"]
        st.subheader("Filtered Metrics")
        st.dataframe(filtered_df[display_cols])

        st.info(f"{len(filtered_df)} total tags loaded for this model.")

        # Find tags where the point name is "PRiSM Calc"
        prism_calc_df = filtered_df[filtered_df["Canary Point Name"] == "PRiSM Calc"]

        # If any are found, display a warning with the count and list of metrics
        if not prism_calc_df.empty:
            count_prism_calc = len(prism_calc_df)
            metrics_list = prism_calc_df["Metric"].tolist()
            metrics_str = ", ".join(f"'{m}'" for m in metrics_list)

            st.info(
                f"⚠️ **{count_prism_calc} tag(s) are PRiSM Calc(s).** "
                f"These will be excluded from the download. \n\n**Metrics:** {metrics_str}"
            )

        # --- New Warning Logic ---
        # Find tags where the point name is "Not Found"
        not_found_df = filtered_df[filtered_df["Canary Point Name"] == "Not Found"]

        # If any are found, display a warning with the count and list of metrics
        if not not_found_df.empty:
            count_not_found = len(not_found_df)
            metrics_list = not_found_df["Metric"].tolist()
            metrics_str = ", ".join(f"'{m}'" for m in metrics_list)

            st.warning(
                f"⚠️ **{count_not_found} tag(s) have missing Canary Point Name(s).** "
                f"These will be excluded from the download. \n\n**Metrics:** {metrics_str}"
            )

# --- Step 3: Configure & Fetch ---
if "filtered_df" in st.session_state:
    st.divider()
    st.header("Step 3 — Configure API Query & Fetch Data")

    # --- MODIFIED: Filter out 'PRiSM Calc' and 'Not Found' metrics BEFORE creating the list ---
    # This ensures they do not appear in the multiselect and are not selected by default.
    valid_rows_mask = ~st.session_state.filtered_df["Canary Point Name"].isin(["PRiSM Calc", "Not Found"])
    valid_metrics = st.session_state.filtered_df.loc[valid_rows_mask, "Metric"].tolist()

    selected_metrics = st.multiselect("Select Metrics to Fetch", valid_metrics, default=valid_metrics)
    st.session_state.selected_metrics = selected_metrics

    col1, col2 = st.columns([1, 1])

    with col1:
        end_default = datetime.now()
        start_default = end_default - timedelta(days=7)
        start_date = st.date_input("Start Date", value=start_default)
        start_time = st.time_input("Start Time", time(0, 0))
        end_date = st.date_input("End Date", value=end_default)
        end_time = st.time_input("End Time", time(23, 59))
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)

    with col2:
        agg_name = st.selectbox("Aggregate Name", AGGREGATE_NAMES, index=AGGREGATE_NAMES.index("EndBound"))
        agg_interval = st.text_input("Aggregate Interval (HH:MM:SS)", value="0:01:00")
        include_quality = st.checkbox("Include quality in response (includeQuality)", value=False)
        token_input = st.text_input("API Token", value=API_TOKEN, type="password")

    if st.button("Fetch Data from Canary API"):
        start_fetch_time = py_time.perf_counter()

        if not st.session_state.selected_metrics:
            st.warning("No metrics selected. Please select at least one metric to fetch.")
            st.stop()

        # 1. Filter the DataFrame based on selected metrics in the multiselect
        # Since selected_metrics only contains valid metrics, we don't strictly need to filter again,
        # but filtering by selection is still required.
        active_df = st.session_state.filtered_df[
            st.session_state.filtered_df["Metric"].isin(st.session_state.selected_metrics)
        ]
        
        st.session_state.active_df = active_df
        
        # 2. Extract tags from the active_df
        tags = active_df["Canary Point Name"].astype(str).tolist()

        if not tags:
            st.error("No valid tags to query for the selected metrics.")
            st.stop()

        try:
            time_chunks = calculate_time_chunks(start_datetime, end_datetime, agg_interval, len(tags))

            st.info(f"Fetching data in {len(time_chunks)} chunks to prevent overload.")
            outer_pbar = st.progress(0, text="Fetching progress: 0%")

            # List to hold dataframes from each chunk
            data_dfs = []

            for i, (chunk_start, chunk_end) in enumerate(time_chunks):
                body = {
                    "apiToken": token_input,
                    "tags": tags,
                    "startTime": chunk_start.isoformat(sep=' '),
                    "endTime": chunk_end.isoformat(sep=' '),
                    "aggregateName": agg_name,
                    "aggregateInterval": agg_interval,
                    "includeQuality": include_quality,
                    "maxSize": 1_000_000,
                    "continuation": None
                }
                headers = {"Content-Type": "application/json"}

                accumulated_chunk = {tag: [] for tag in tags}

                inner_pbar_text = st.empty()

                page_num = 0
                while True:
                    page_num += 1
                    inner_pbar_text.text(f"Chunk {i+1}/{len(time_chunks)}: Fetching page {page_num}...")

                    resp = requests.post(API_URL, data=json.dumps(body), headers=headers, timeout=300)
                    resp.raise_for_status()
                    resp_json = resp.json()

                    page_data = resp_json.get("data") or resp_json.get("results") or {}

                    for tag, arr in page_data.items():
                        if tag in accumulated_chunk:
                            for point in arr:
                                t = point.get("t") or point.get("time") or point.get("timestamp")
                                v = point.get("v") or point.get("value")
                                if t is not None:
                                    accumulated_chunk[tag].append({"t": t, "v": v})

                    continuation_token = resp_json.get("continuation")
                    if not continuation_token:
                        break

                    body["continuation"] = continuation_token[0] if isinstance(continuation_token, list) else continuation_token
                    if not body["continuation"]:
                        break

                inner_pbar_text.empty()

                # --- Build DataFrame for this data chunk ---
                all_timestamps = set()
                for tag in tags:
                    for pt in accumulated_chunk.get(tag, []):
                        all_timestamps.add(pt["t"])

                sorted_ts = sorted(list(all_timestamps), key=lambda x: dt_parser.parse(x))

                rows = []
                lookup = {tag: {pt["t"]: pt["v"] for pt in accumulated_chunk.get(tag, [])} for tag in tags}

                for ts in sorted_ts:
                    row_data = {"Point Name": ts}
                    for tag in tags:
                        val = lookup.get(tag, {}).get(ts)
                        row_data[tag] = val if val is not None else None
                    rows.append(row_data)

                if rows:
                    chunk_df = pd.DataFrame(rows)
                    data_dfs.append(chunk_df)

                outer_pbar.progress((i + 1) / len(time_chunks), text=f"Fetching progress: {i+1}/{len(time_chunks)} chunks complete.")

            # --- Merge all data chunks and prepend metadata ---
            outer_pbar.progress(1.0, text="Merging data and finalizing...")

            if not data_dfs:
                st.warning("No data returned for the selected range and tags.")
                st.session_state.df_hist = None
                st.stop()

            final_data_df = pd.concat(data_dfs, ignore_index=True)

            # Create the metadata DataFrame just once
            meta_data = {
                "Point Name": ["Description", "Extended Name", "Extended Description", "Unit"]
            }
            for tag in tags:
                meta_data[tag] = [
                    st.session_state.active_df.loc[st.session_state.active_df["Canary Point Name"] == tag, "Canary Description"].iloc[0],
                    st.session_state.active_df.loc[st.session_state.active_df["Canary Point Name"] == tag, "Metric"].iloc[0],
                    st.session_state.active_df.loc[st.session_state.active_df["Canary Point Name"] == tag, "DCS Description"].iloc[0],
                    st.session_state.active_df.loc[st.session_state.active_df["Canary Point Name"] == tag, "Unit"].iloc[0],
                ]
            meta_df = pd.DataFrame(meta_data)

            # Combine metadata and data
            final_df = pd.concat([meta_df, final_data_df], ignore_index=True)

            st.session_state.df_hist = final_df
            st.session_state.start_datetime = start_datetime
            st.session_state.end_datetime = end_datetime

            end_fetch_time = py_time.perf_counter()

            total_seconds = end_fetch_time - start_fetch_time
            minutes, seconds = divmod(total_seconds, 60)
            hours, minutes = divmod(minutes, 60)

            st.success(f"Data fetching complete! Total time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            outer_pbar.empty()

        except requests.HTTPError as e:
            st.error(f"HTTP error while calling Canary API: {e} — Response text: {getattr(e.response, 'text', '')}")
            st.session_state.df_hist = None
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.session_state.df_hist = None

# --- Step 3: Preview & Download ---
if "df_hist" in st.session_state and st.session_state.df_hist is not None:
    st.divider()
    st.header("Step 3 — Preview & Download")

    # --- Data Preview Section ---
    st.subheader("Data Preview")

    # --- Data Statistics ---
    if st.button("Show Data Statistics"):
        st.info("Statistics are calculated on the raw data, after attempting to convert values to numbers.")

        # Create a mapping from Canary Point Name (tag) to Metric for user-friendly headers
        active_df = st.session_state.active_df
        tag_to_metric_map = pd.Series(
            active_df.Metric.values,
            index=active_df["Canary Point Name"]
        ).to_dict()

        # Isolate the data portion of the dataframe
        data_df = st.session_state.df_hist.iloc[4:].copy()
        data_cols = data_df.columns[1:]
        df_numeric_converted = data_df[data_cols].apply(pd.to_numeric, errors='coerce')

        if data_cols.empty:
            st.warning("No data columns available to describe.")
        else:
            stats = df_numeric_converted.describe(include='all')
            total_rows = len(df_numeric_converted)

            if total_rows > 0:
                null_counts = df_numeric_converted.isna().sum()
                percent_null = (null_counts / total_rows) * 100
                stats.loc['Percent Null'] = percent_null
            else:
                stats.loc['Percent Null'] = 0.0

            stats = stats.round(2)
            if 'Percent Null' in stats.index:
                stats.loc['Percent Null'] = stats.loc['Percent Null'].map('{:.2f}%'.format)

            # Rename the columns from tags to the friendly metric names
            stats.rename(columns=tag_to_metric_map, inplace=True)

            st.dataframe(stats)

# --- Plotting Section ---
    st.subheader("Data Plots")

    if st.button("Generate Data Plots"):
        # Mappings for tags, metrics, and units
        active_df = st.session_state.active_df
        tag_to_metric_map = pd.Series(
            active_df.Metric.values,
            index=active_df["Canary Point Name"]
        ).to_dict()
        metric_to_tag_map = {v: k for k, v in tag_to_metric_map.items()}
        # Create the new map for Metrics to Units
        metric_to_unit_map = pd.Series(
            active_df.Unit.values,
            index=active_df.Metric
        ).to_dict()

        # Get available tags from the data and map them to metric names
        available_tags = st.session_state.df_hist.columns[1:].tolist()
        metrics_to_plot = [tag_to_metric_map[tag] for tag in available_tags if tag in tag_to_metric_map]

        if metrics_to_plot:
            if len(metrics_to_plot) > 20:
                st.warning(f"⚠️ Plotting all {len(metrics_to_plot)} tags. This may take a moment and the plot will be very long.")

            # Data preparation
            tags_to_plot = [metric_to_tag_map[metric] for metric in metrics_to_plot]
            plot_df = st.session_state.df_hist.iloc[4:].copy()
            plot_df["Point Name"] = pd.to_datetime(plot_df["Point Name"])
            plot_df = plot_df.set_index("Point Name")
            plot_df_subset = plot_df[tags_to_plot].apply(pd.to_numeric, errors='coerce')
            plot_df_subset.columns = metrics_to_plot

            # Matplotlib code for individual subplots
            num_plots = len(metrics_to_plot)
            fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
            if num_plots == 1:
                axes = [axes]

            for i, metric_name in enumerate(metrics_to_plot):
                random_color = f"#{random.randint(0, 0xFFFFFF):06x}"
                axes[i].plot(plot_df_subset.index, plot_df_subset[metric_name], color=random_color)
                axes[i].set_title(metric_name)
                axes[i].grid(True)
                axes[i].set_xlabel("DATETIME")
                # Look up the unit for the current metric and set the Y-axis label
                unit = active_df.loc[active_df["Metric"] == metric_name, "Unit"].iloc[0]
                axes[i].set_ylabel(unit)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.warning("No data available to plot.")


    # --- Download CSV ---
    st.subheader("Download Data")
    model_name = st.session_state.selected_model
    start_str = st.session_state.start_datetime.strftime("%Y%m%d")
    end_str = st.session_state.end_datetime.strftime("%Y%m%d")
    filename = f"{model_name}-{start_str}-{end_str}-RAW.csv"
    csv_bytes = st.session_state.df_hist.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Historian CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv"
    )