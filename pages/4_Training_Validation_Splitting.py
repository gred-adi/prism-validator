import streamlit as st
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns

from utils.app_ui import render_sidebar, get_model_info
from utils.model_dev_utils import cleaned_dataset_name_split, read_prism_csv, corrfunc
from verstack.stratified_continuous_split import scsplit
from pathlib import Path
from fpdf import FPDF
from io import BytesIO

DEFAULT_OPERATIONAL_STATE = 'AP-TVI-GROSS LOAD' # Default value might have to change

st.set_page_config(page_title="Train Validation Split", page_icon="4️⃣")

st.title("🔀 Train-Validation Splitting Wizard")

st.write(
    "This page is for splitting the cleaned datasets with and without outliers into training and validation sets. To begin, please upload the datasets below."
)

if 'tvs_applied' not in st.session_state: st.session_state.tvs_applied = False
if 'cleaned_selected_dataset' not in st.session_state: st.session_state.cleaned_selected_dataset = None
if 'vis_data_splitting' not in st.session_state: st.session_state.vis_data_splitting = False
if 'ds_result_train' not in st.session_state: st.session_state.ds_result_train = None
if 'ds_result_validation_without_outlier' not in st.session_state: st.session_state.ds_result_validation_without_outlier = None
if 'df_result_validation_with_outlier_length' not in st.session_state: st.session_state.df_result_validation_with_outlier_length = None

outlier_file = st.file_uploader("Upload your CLEANED dataset WITH OUTLIERS file", type=["csv"], accept_multiple_files=False)
no_outlier_file = st.file_uploader("Upload your CLEANED dataset WITHOUT OUTLIERS file", type=["csv"], accept_multiple_files=False)
point_list_dataset_file = st.file_uploader("Upload your POINT LIST dataset file (project_points.csv)", type=["csv"], accept_multiple_files=False)
fault_detection_dataset_file = st.file_uploader("Upload your FAULT DETECTION POINTS dataset file (fault_detection.csv)", type=["csv"], accept_multiple_files=False)

if outlier_file is not None and no_outlier_file is not None and point_list_dataset_file is not None and fault_detection_dataset_file is not None:
    st.success(f"File '{outlier_file.name}' uploaded successfully.")
    st.success(f"File '{no_outlier_file.name}' uploaded successfully.")
    st.success(f"File '{point_list_dataset_file.name}' uploaded successfully.")
    st.success(f"File '{fault_detection_dataset_file.name}' uploaded successfully.")

    # Automatically extract site_name, model_name, inclusive_dates from filename if cleaned outlier file name format is correct
    auto_site_name, auto_model_name, auto_inclusive_dates = cleaned_dataset_name_split(outlier_file.name)

    # Get user inputs for site_name, system_name, model_name, sprint_name, inclusive_dates (Saves it to session_state now)
    get_model_info(auto_site_name, auto_model_name, auto_inclusive_dates)

    if (st.button("Split Dataset", type="primary")):

        site_name = st.session_state.site_name
        system_name = st.session_state.system_name
        model_name = st.session_state.model_name
        sprint_name = st.session_state.sprint_name
        inclusive_dates = st.session_state.inclusive_dates

        # Path to save datasets and visualizations after splitting
        base_path = Path.cwd()
        data_splitting_path = base_path / site_name / system_name / sprint_name / model_name / "data_splitting"
        dataset_path = base_path / site_name / system_name / sprint_name / model_name / "dataset"
        # Create folders if they don't exist
        data_splitting_path.mkdir(parents=True, exist_ok=True)
        # Should already exist but just in case
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Start main logic
        with st.spinner("Reading files..."):
            # Read datasets and files and put them in dataframes
            outlier_df = pd.read_csv(outlier_file, index_col=False)
            no_outlier_df = pd.read_csv(no_outlier_file, index_col=False)
            point_list_dataset = pd.read_csv(point_list_dataset_file)
            fault_detection_dataset = pd.read_csv(fault_detection_dataset_file)

        with st.spinner("Processing files..."):
            fault_detection_list = fault_detection_dataset.columns.tolist()
            fault_detection_list[0] = 'DATETIME'
            fault_detection_list.remove('Minimum OMR')
            modeled_cols = []
            combined = list(set(fault_detection_list).union(modeled_cols))
            fault_detection_list = ['DATETIME'] + [col for col in combined if col != 'DATETIME']

            cleaned_dataset_with_outlier, cleaned_dataset_with_outlier_header = read_prism_csv(outlier_df)
            cleaned_dataset_without_outlier, cleaned_dataset_without_outlier_header = read_prism_csv(no_outlier_df)
            del outlier_df, no_outlier_df # free memory
            gc.collect()

            # Data Point Mapping
            new_column = []
            for column in cleaned_dataset_without_outlier_header.columns:
                if column in point_list_dataset['Name'].tolist():
                    mapping_value = str(point_list_dataset[point_list_dataset['Name'] == column]['Metric'].values[0])
                    if (mapping_value == 'nan'):
                        new_column.append(column)
                    else:
                        new_column.append(mapping_value)
                else:
                    new_column.append(column)
                new_column[0] = 'DATETIME'
            cleaned_dataset_without_outlier.columns = new_column

            cleaned_selected_dataset = cleaned_dataset_without_outlier[fault_detection_list]
            st.session_state.cleaned_selected_dataset = cleaned_selected_dataset
            # st.dataframe(cleaned_selected_dataset.head())

            try:
                operational_state = point_list_dataset[point_list_dataset['Constrain']==True]['Metric'].values[0]
            except:
                operational_state = DEFAULT_OPERATIONAL_STATE

        with st.spinner("Processing dataset splitting..."):
            # Data Splitting
            ds_train, ds_validate = scsplit(
                cleaned_selected_dataset,
                stratify = cleaned_selected_dataset[operational_state],
                test_size = 0.2,
                train_size = 0.8,
                continuous = True,
                random_state = None
            )

            ds_result_train = cleaned_dataset_without_outlier[cleaned_dataset_without_outlier['DATETIME'].isin(ds_train['DATETIME'])]
            ds_result_validation_without_outlier = cleaned_dataset_without_outlier[cleaned_dataset_without_outlier['DATETIME'].isin(ds_validate['DATETIME'])]

            df_result_validation_with_outlier = cleaned_dataset_with_outlier[~cleaned_dataset_with_outlier['DATETIME'].isin(ds_train['DATETIME'])]
            # st.write(f"{ds_result_train.shape}")
            # st.write(f"{ds_result_validation_without_outlier.shape}")
            # st.write(f"{df_result_validation_with_outlier.shape}")

            # Export Datasets
            ds_result_train.columns = cleaned_dataset_without_outlier_header.columns
            export_train = pd.concat([cleaned_dataset_without_outlier_header, ds_result_train])

            ds_result_validation_without_outlier.columns = cleaned_dataset_without_outlier_header.columns
            export_validation_without_outlier = pd.concat([cleaned_dataset_without_outlier_header, ds_result_validation_without_outlier])

            df_result_validation_with_outlier.columns = cleaned_dataset_with_outlier_header.columns
            export_validation_with_outlier = pd.concat([cleaned_dataset_with_outlier_header, df_result_validation_with_outlier])

            st.session_state.ds_result_train = ds_result_train
            st.session_state.ds_result_validation_without_outlier = ds_result_validation_without_outlier
            st.session_state.df_result_validation_with_outlier_length = len(df_result_validation_with_outlier)

            # Save datasets
            train_out = data_splitting_path / f"TRAINING-{model_name}-{inclusive_dates}-WITHOUT-OUTLIER.csv"
            outlier_out = data_splitting_path / f"VALIDATION-{model_name}-{inclusive_dates}-WITH-OUTLIER.csv"
            no_outlier_out = data_splitting_path / f"VALIDATION-{model_name}-{inclusive_dates}-WITHOUT-OUTLIER.csv"

            # Visualization paths
            # vis_dataset = dataset_path / f"CLEANED-{model_name}-{inclusive_dates}-WITHOUT-OUTLIER.csv"
            vis_data_splitting = data_splitting_path / f"CLEANED-{model_name}-{inclusive_dates}-WITHOUT-OUTLIER.csv"
            st.session_state.vis_data_splitting = vis_data_splitting

            export_train.to_csv(train_out, index=False)
            export_validation_with_outlier.to_csv(outlier_out, index=False)
            export_validation_without_outlier.to_csv(no_outlier_out, index=False)

            st.success(f"Training dataset saved to: {train_out}")
            st.success(f"Validation dataset WITH OUTLIER saved to: {outlier_out}")
            st.success(f"Validation dataset WITHOUT OUTLIER saved to: {no_outlier_out}")

            st.session_state.tvs_applied = True

    if 'tvs_visualize' not in st.session_state: st.session_state.tvs_visualize = False
    if 'save_data' not in st.session_state: st.session_state.save_data = False
    if 'time_series' not in st.session_state: st.session_state.time_series = False
    if 'stats' not in st.session_state: st.session_state.stats = False
    if 'pair_plot' not in st.session_state: st.session_state.pair_plot = False
    if 'box_plot' not in st.session_state: st.session_state.box_plot = False
    cleaned_selected_dataset = st.session_state.cleaned_selected_dataset
    vis_data_splitting = st.session_state.vis_data_splitting
    ds_result_train = st.session_state.ds_result_train
    ds_result_validation_without_outlier = st.session_state.ds_result_validation_without_outlier
    df_outlier_length = st.session_state.df_result_validation_with_outlier_length

    if st.session_state.tvs_applied:
        st.header("Generate Visualizations & Report (Optional)")
        st.toggle("Visualize Data", value=st.session_state.tvs_visualize, key="tvs_visualize")
        st.toggle("Generate PDF", value=st.session_state.save_data, key="save_data")

        # Visualization options
        if st.session_state.tvs_visualize:
            st.write("Please select which graphs to generate.")
            st.toggle("Time Series for each parameter", value=st.session_state.time_series, key="time_series")
            st.toggle("Pair Plot (takes a long time)", value=st.session_state.pair_plot, key="pair_plot")
            st.toggle("Box Plot", value=st.session_state.box_plot, key="box_plot")

        # Generate Button
        # This button is now outside the "visualize" block and renamed.
        if st.button("Generate", type='primary'):

            # Check if any action is requested
            if not st.session_state.tvs_visualize and not st.session_state.save_data:
                st.warning("Please toggle on 'Visualize Data' or 'Generate PDF' to generate.")
                st.stop()

            with st.spinner("Generating..."):
                pdf = None
                report_path = None
                report_generation_failed = False

                # Initialize PDF
                if st.session_state.save_data:
                    try:
                        report_path = str(vis_data_splitting).replace('.csv', '-TRAIN-VALIDATION-REPORT.pdf')
                        pdf = FPDF(orientation="portrait")
                        pdf.add_page()
                        pdf.set_font("Arial", "B", 16)
                        pdf.cell(0, 10, "Data Visualization Report", 0, 1, "C")
                    except Exception as e:
                        st.error(f"Failed to initialize PDF: {e}")
                        report_generation_failed = True

                # Generate stats after split
                st.header("Statistics")
                try:
                    st.write(f"Original Cleaned Data: {len(cleaned_selected_dataset):,} rows")
                    st.write(f"Train Data: {len(ds_result_train):,} rows")
                    st.write(f"Validation With Outlier: {df_outlier_length:,} rows")
                    st.write(f"Validation Without Outlier: {len(ds_result_validation_without_outlier):,} rows")

                    def q25(x): return x.quantile(0.25)
                    def q50(x): return x.quantile(0.50)
                    def q75(x): return x.quantile(0.75)
                    agg_list = [q25, q50, q75, 'max', 'min', 'mean', 'std', 'size']

                    dataset_subset_stat = cleaned_selected_dataset.drop('DATETIME', axis=1).agg(agg_list)
                    st.dataframe(dataset_subset_stat)
                    dataset_subset_stat.to_csv(str(vis_data_splitting).replace('.csv', '_statistic.csv'))

                    # Add Stats to PDF (if PDF exists)
                    if st.session_state.save_data and pdf and not report_generation_failed:
                        pdf.add_page(orientation="portrait")
                        pdf.set_font("Arial", "B", 14)
                        pdf.cell(0, 10, "Statistics", 0, 1, "L")
                        pdf.ln(5)

                        # row counts
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(0, 8, "Row Counts", 0, 1)
                        pdf.set_font("Arial", "", 10)
                        pdf.cell(0, 6, f"Original Cleaned Data: {len(cleaned_selected_dataset):,} rows", 0, 1)
                        pdf.cell(0, 6, f"Train Data: {len(ds_result_train):,} rows", 0, 1)
                        pdf.cell(0, 6, f"Validation With Outlier: {df_outlier_length:,} rows", 0, 1)
                        pdf.cell(0, 6, f"Validation Without Outlier: {len(ds_result_validation_without_outlier):,} rows", 0, 1)
                        pdf.ln(10)

                        # figure 2
                        col_width_metric = 40
                        col_width_value = 60
                        for col in dataset_subset_stat.columns:
                            pdf.set_font("Arial", "B", 10)
                            pdf.multi_cell(0, 5, f"Metric: {col}", 0, "L")
                            pdf.ln(2)

                            pdf.set_font("Arial", "B", 8)
                            pdf.cell(col_width_metric, 8, "Statistic", 1, 0, "C")
                            pdf.cell(col_width_value, 8, "Value", 1, 0, "C")
                            pdf.ln()

                            pdf.set_font("Arial", "", 8)
                            for idx in dataset_subset_stat.index:
                                metric_name = str(idx)
                                metric_value = f"{dataset_subset_stat.loc[idx, col]:.4f}"
                                pdf.cell(col_width_metric, 8, metric_name, 1)
                                pdf.cell(col_width_value, 8, metric_value, 1)
                                pdf.ln()
                            pdf.ln(8)

                except Exception as e:
                    st.error(f"Failed to generate Statistics: {e}")

                # Generate Optional Visualizations
                if st.session_state.tvs_visualize:

                    # FIGURE 1
                    if st.session_state.time_series:
                        st.header("Time Series")
                        try:
                            fig, ax = plt.subplots(figsize=(22.5, (len(cleaned_selected_dataset.columns) - 1) * 2.5))
                            cleaned_selected_dataset[sorted(cleaned_selected_dataset.columns)].plot(x='DATETIME', subplots=True, grid=True, ax=ax)
                            for ax_i in plt.gcf().axes: ax_i.legend(loc=2)
                            st.pyplot(fig)
                            fig.savefig(str(vis_data_splitting).replace('.csv', '_timeseries.png'))

                            if st.session_state.save_data and pdf and not report_generation_failed:
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 14)
                                pdf.cell(0, 10, "Time Series for each Parameter", 0, 1, "L")
                                with BytesIO() as buffer:
                                    fig.savefig(buffer, format="png", bbox_inches='tight')
                                    buffer.seek(0)
                                    pdf.image(buffer, x=10, y=30, w=pdf.w - 20, type="PNG")
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Failed to generate Time Series plot: {e}")

                    # FIGURE 3
                    if st.session_state.pair_plot:
                        st.header("Pair Plot")
                        try:
                            g = sns.PairGrid(cleaned_selected_dataset, height=4, diag_sharey=False)
                            g.map_upper(corrfunc, cmap=plt.get_cmap('BrBG'), norm=plt.Normalize(vmin=-1, vmax=1))
                            g.map_lower(sns.scatterplot, s=50, color='#018571')
                            g.map_diag(sns.kdeplot, color='red', fill=True)
                            g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
                            st.pyplot(g)
                            g.savefig(str(vis_data_splitting).replace('.csv', '_pairplot.png'))

                            if st.session_state.save_data and pdf and not report_generation_failed:
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 14)
                                pdf.cell(0, 10, "Pair Plot", 0, 1, "L")
                                with BytesIO() as buffer:
                                    g.fig.savefig(buffer, format="png", bbox_inches='tight')
                                    buffer.seek(0)
                                    pdf.image(buffer, x=10, y=30, w=pdf.w - 20, type="PNG")
                            plt.close(g.fig)
                        except Exception as e:
                            st.error(f"Failed to generate Pair Plot: {e}")

                    # FIGURE 4
                    if st.session_state.box_plot:
                        st.header("Box Plot")
                        try:
                            df_plot = pd.concat({
                                'TRAIN': ds_result_train[sorted(ds_result_train.columns)].drop(columns=['Point Name']).melt(),
                                'VALIDATION': ds_result_validation_without_outlier[sorted(ds_result_validation_without_outlier.columns)].drop(columns=['Point Name']).melt()}, names=['data', 'old_index'])
                            df_plot = df_plot.reset_index(level=0).reset_index(drop=True)
                            g = sns.catplot(data=df_plot, kind='box', x='data', y='value', col='variable', col_wrap=6, height=3, aspect=1.25, sharey=False, palette={'TRAIN': 'blue', 'VALIDATION': 'orange'})
                            g.set(xlabel='', xticks=[])
                            g.set_titles('{col_name}', size=8.5)
                            st.pyplot(g)
                            g.savefig(str(vis_data_splitting).replace('.csv', '_boxplot.png'))

                            if st.session_state.save_data and pdf and not report_generation_failed:
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 14)
                                pdf.cell(0, 10, "Box Plot", 0, 1, "L")
                                with BytesIO() as buffer:
                                    g.fig.savefig(buffer, format="png", bbox_inches='tight')
                                    buffer.seek(0)
                                    pdf.image(buffer, x=10, y=30, w=pdf.w - 20, type="PNG")
                            plt.close(g.fig)
                        except Exception as e:
                            st.error(f"Failed to generate Box Plot: {e}")

                # Save pdf when visualize is on
                if st.session_state.save_data and pdf and report_path and not report_generation_failed:
                    try:
                        pdf.output(report_path)
                        st.success(f"PDF Report saved to {report_path}")
                    except Exception as e:
                        st.error(f"Failed to save PDF: {e}")

                # Clear session state if visualize was on or if the report was saved
                if st.session_state.tvs_visualize or st.session_state.save_data:
                    st.session_state.cleaned_selected_dataset = None
                    st.session_state.vis_data_splitting = None
                    st.session_state.ds_result_train = None
                    st.session_state.ds_result_validation_without_outlier = None