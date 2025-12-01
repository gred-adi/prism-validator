import streamlit as st
import pandas as pd

from pathlib import Path
from utils.app_ui import get_model_info
from utils.model_val_utils import extract_numeric

st.set_page_config(page_title="Calculate Accuracy", page_icon="🅰️")

st.header("Calculate Accuracy")

data_file = st.file_uploader("Upload your model data file (model_name.dat)", type=["dat"])
fault_detection_file = st.file_uploader("Upload your fault detection file (fault_detection.csv)", type=["csv"])

if data_file is not None and fault_detection_file is not None:

    get_model_info("", data_file.name.removesuffix(".dat"), "")

    st.success(f"File '{data_file.name}' uploaded successfully.")

    if st.button("Calculate Accuracy", type="primary"):

        with st.spinner("Reading data..."):
            df_data = pd.read_csv(data_file, encoding="UTF-16", delimiter='\t')

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

        fault_detection_dataset = pd.read_csv(fault_detection_file)
        fault_detection_list = fault_detection_dataset.columns.tolist()
        fault_detection_list.remove('Name')
        fault_detection_list.remove('Minimum OMR')
        print(f"Number of unique metrics from fault diagnostics: {len(fault_detection_list)}")

        modeled_cols = []

        combined_metrics = list(set(fault_detection_list).union(modeled_cols))

        print(f"Number of unique metrics: {len(combined_metrics)}")

        df_combined = df_data[combined_metrics].apply(lambda col: col.apply(extract_numeric))

        df_combined.rename(columns=column_mapping, inplace=True)

        results = [
            {
                'Metrics': col,
                'Average - Relative Deviation (%)': round(abs(df_combined[col].mean()) * 100, 2),
                'Accuracy (%)': round((1 - abs(df_combined[col].mean())) * 100, 2),
            }
            for col in df_combined.columns
        ]

        df_scores = pd.DataFrame(results)
        st.dataframe(df_scores)

        base_path = Path.cwd()
        relative_deviation_path = base_path/ st.session_state.site_name / st.session_state.system_name / st.session_state.sprint_name / st.session_state.model_name / "relative_deviation"
        relative_deviation_path.mkdir(parents=True, exist_ok=True)
        accuracy_file_path = relative_deviation_path / f"{st.session_state.model_name}_Accuracy.csv"
        df_scores.to_csv(accuracy_file_path, index=False)
        st.success(f"File saved to {accuracy_file_path}")
