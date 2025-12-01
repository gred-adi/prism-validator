import streamlit as st
from pathlib import Path
from utils.model_val_utils import build_partial_config_table, merge_constraints, generate_report_from_table

st.set_page_config(page_title="Model FPR", page_icon="🅱️")

st.header("Model QA")
# Generate config meta table
st.write("Please enter the following information:")
site_name = st.text_input("Site Name (e.g., TVI/TSI)")
system_name = st.text_input("System Name (e.g., BOP)")

if st.button("Generate QA report", type="primary"):
    if not site_name or not system_name:
        st.error("Please enter both Site Name and System Name.")
        st.stop()

    # Can be moved to text input if necessary
    sub_ts_length = 60
    n_ts_above_thresh = 50
    time_interval = 1
    warning = 5.0
    alert = 10.0

    base_path = Path.cwd()
    dataset_path = base_path / site_name / system_name
    constraint_table = dataset_path / "constraint_table.csv"

    df_partial = build_partial_config_table(
        dataset_path,
        site_name,
        system_name,
        sub_ts_length,
        n_ts_above_thresh,
        time_interval,
        warning,
        alert
    )

    st.write("Partial Config Table")
    st.dataframe(df_partial)

    st.write("Meta Table")
    meta_table = merge_constraints(df_partial, constraint_table)
    st.dataframe(meta_table)

    meta_table_fpath =  dataset_path / "meta_table.csv"
    meta_table.to_csv(meta_table_fpath, index=False)

    with st.spinner("Generating Reports... (QA report generation may take a long time)"):
        try:
            generate_report_from_table(
                table_path = meta_table_fpath,
                local_path = dataset_path,
                regenerate = True
            )
            st.success(f"Meta Table saved to {meta_table_fpath}")
            st.success(f"Reports saved to {dataset_path} Summary Reports folder")
        except Exception as e:
            st.error(f"Report generation failed. {e}")
