import streamlit as st
import pandas as pd

from pathlib import Path
from utils.model_dev_utils import cleaned_dataset_name_split, split_holdout, generate_split_holdout_report
from utils.app_ui import render_sidebar, get_model_info
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="Split Holdout Dataset", page_icon="2️⃣")

render_sidebar()

def get_and_cache_model_info_from_file():
    """
    Asks the user for the raw and cleaned datasets and asks for site_name,
    utility_name, model_name, sprint_name, and inclusive_dates. Also reads from the files for the
    raw and cleaned dataframes.

    Stores all the information and dataframes in the session state for later use.
    """
    raw_file = st.file_uploader("Upload your RAW dataset file", type=["csv"], accept_multiple_files=False)
    cleaned_file = st.file_uploader("Upload your CLEANED RAW dataset file", type=["csv"], accept_multiple_files=False)

    if raw_file is not None and cleaned_file is not None:
        st.success(f"File '{raw_file.name}' uploaded successfully.")
        st.success(f"File '{cleaned_file.name}' uploaded successfully.")

        # Automatically extract site_name, model_name, inclusive_dates from filename if cleaned file name format is correct
        auto_site_name, auto_model_name, auto_inclusive_dates = cleaned_dataset_name_split(cleaned_file.name)

        # Get user inputs for site_name, utility_name, model_name, sprint_name, inclusive_dates
        get_model_info(auto_site_name, auto_model_name, auto_inclusive_dates)

        with st.spinner("Reading Files..."):
            raw_df = pd.read_csv(raw_file)
            cleaned_df = pd.read_csv(cleaned_file)

        # Store in cache
        st.session_state.raw_df = raw_df
        st.session_state.cleaned_df = cleaned_df
        st.session_state.has_data = True
        # return raw_df, cleaned_df, site_name, utility_name, model_name, sprint_name, inclusive_dates

st.header("Split Holdout Dataset")

st.write(
    "This page is for splitting the holdout dataset from the cleaned dataset. To begin, please upload the raw and cleaned raw datasets below."
)

if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None

if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None

if 'has_data' not in st.session_state:
    st.session_state.has_data = False

if st.session_state.has_data == False:
    # raw_df and cleaned_df found in cache
    if st.session_state.raw_df is not None and st.session_state.cleaned_df is not None:
        st.info(f"Data for {st.session_state.model_name} found. Would you like to use it?")
        if st.button("Yes"):
            try:
                site_name = st.session_state.site_name
                utility_name = st.session_state.utility_name
                model_name = st.session_state.model_name
                sprint_name = st.session_state.sprint_name
                inclusive_dates = st.session_state.inclusive_dates
                raw_df = st.session_state.raw_df
                cleaned_df = st.session_state.cleaned_df

                # Might cause issues because there is no raw_df_header unless it came from data_cleaning, needs testing
                raw_df_header = st.session_state.raw_df_header

                # Add original headers to the dataframes to match current process
                raw_df.columns = raw_df_header.columns
                raw_df = pd.concat([raw_df_header, raw_df])

                cleaned_df.columns = raw_df_header.columns
                cleaned_df = pd.concat([raw_df_header, cleaned_df])

                st.session_state.has_data = True
            except KeyError or AttributeError:
                st.error("Cache was incomplete. Please upload the files.")
                get_and_cache_model_info_from_file()

        elif st.button("No"):
            # Add code to reset cache values
            get_and_cache_model_info_from_file()
    # raw_df and cleaned_df not found in cache
    else:
        get_and_cache_model_info_from_file()

if st.session_state.has_data == True:
    st.markdown(f"""
            Current Data:
            Site Name: {st.session_state.site_name}
            Utility Name: {st.session_state.utility_name}
            Model Name: {st.session_state.model_name}
            Sprint Name: {st.session_state.sprint_name}
            Inclusive Dates: {st.session_state.inclusive_dates}
            """)
    if st.button("Re-enter Data"):
        st.session_state.raw_df = None
        st.session_state.cleaned_df = None
        st.session_state.has_data = False

    site_name = st.session_state.site_name
    utility_name = st.session_state.utility_name
    model_name = st.session_state.model_name
    sprint_name = st.session_state.sprint_name
    inclusive_dates = st.session_state.inclusive_dates
    remove_header_rows = 4

    st.header("Split Mark")
    st.write("Specify the split mark to divide the cleaned dataset into training and holdout sets.")
    st.write("This can be a float (e.g., 0.1 for 10% holdout) or a date (YYYY-MM-DD) to split based on date. If 0.1 is selected, 10% of the data will be used as the holdout set. If a date is selected, all data after that date will be used as the holdout set.")
    split_mark = st.text_input("Split Mark (float for fraction or date in YYYY-MM-DD format)")
    # If float, convert to float, if not, keep it as string
    try:
        split_mark = float(split_mark)
    except ValueError:
        split_mark = split_mark

    if (st.button("Split Dataset", type="primary")):
        # if not site_name or not utility_name or not model_name or not sprint_name or not inclusive_dates:
        #     st.error("Please fill in all the missing fields: Site Name, Utility Name, Model Name, Sprint Name, and Inclusive Dates.")
        if not split_mark:
            st.error("Please specify a valid Split Mark (float or date).")
        else:
            with st.spinner("Processing dataset split..."):
                train_val, holdout, split_mark_used, stats = split_holdout(
                    st.session_state.raw_df,
                    st.session_state.cleaned_df,
                    split_mark,
                    date_col="Point Name",
                    remove_header_rows=remove_header_rows
                )

            with st.spinner("Saving datasets..."):
                # Path to save datasets after splitting
                base_path = Path.cwd()
                dataset_path = base_path / site_name / utility_name / sprint_name / model_name / "dataset"
                # Create folders if they don't exist
                dataset_path.mkdir(parents=True, exist_ok=True)

                train_val_out = dataset_path / f"CLEANED-{model_name}-{inclusive_dates}-WITH-OUTLIER.csv"
                holdout_out = dataset_path / f"{model_name}-{inclusive_dates}-HOLDOUT.csv"

                # If upload is successful, save the uploaded file to the dataset path and display the file path to the user
                train_val.to_csv(train_val_out, index=False)
                holdout.to_csv(holdout_out, index=False)
                st.success(f"Train/Validation dataset saved to: {train_val_out}")
                st.success(f"Holdout dataset saved to: {holdout_out}")

            # Display split statistics
            st.subheader("Split statistics")
            st.write(f"Split mark used: {split_mark_used}")

            # Prepare data for the plot
            plot_data = []
            # Define the keys we *want* to plot, normalized to lowercase
            sets_to_plot = ["cleaned", "train_val", "holdout"]
            # Define the desired order for the final plot
            dataset_order = ["Cleaned", "Train_Val", "Holdout"]

            # Iterate over the stats dictionary directly, like your table code
            for set_name, stat in stats.items():

                # Normalize the key from the dict to check against our list
                normalized_set_name = set_name.lower()

                # A more flexible check
                # This checks if "cleaned" is in "cleaned", or "trainval" is in "train_val"
                found = False
                for plot_key in sets_to_plot:
                    if plot_key in normalized_set_name:
                        found = True
                        break

                if found:
                    if stat.get("size", 0) > 0:
                        start = stat.get("start")
                        end = stat.get("end")

                        if start is not None and end is not None:
                            duration = (end - start)
                            plot_data.append({
                                "Dataset": set_name.title(),
                                "Start": start,
                                "Duration": duration
                            })

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                fig, ax = plt.subplots(figsize=(10, 3))

                ax.barh(
                    plot_df["Dataset"],
                    plot_df["Duration"],
                    left=plot_df["Start"]
                )

                ax.xaxis_date()
                ax.xaxis.set_major_formatter(
                    mdates.DateFormatter('%Y-%m-%d')
                )

                ax.set_xlabel("Date")
                ax.set_ylabel("Dataset")
                ax.set_title("Dataset Time Spans")
                ax.grid(axis='x', linestyle='--', alpha=0.7)

                ax.set_yticks(dataset_order)

                fig.autofmt_xdate()

                st.pyplot(fig)
            else:
                st.info("No data available to plot time spans.")

            for set_name, stat in stats.items():
                st.markdown(f"**{set_name.title()}**")
                if stat["size"] > 0:
                    # convert datetimes to readable strings
                    start = stat["start"].strftime("%Y-%m-%d %H:%M:%S") if stat["start"] is not None else None
                    end = stat["end"].strftime("%Y-%m-%d %H:%M:%S") if stat["end"] is not None else None
                    st.table({
                        "Metric": ["Start", "End", "Rows", "Number of days"],
                        "Value": [start, end, stat["size"], stat["num_days"]],
                    })
                else:
                    st.write("(Empty set)")

            try:
                report_file_path = dataset_path / f"CLEANED-{model_name}-{inclusive_dates}-SPLIT-HOLDOUT-REPORT.pdf"
                with st.spinner(f"Generating PDF report..."):
                    # Call the new utility function
                    success = generate_split_holdout_report(
                    stats,
                    split_mark_used,
                    report_file_path,
                    fig
                    )

                if success:
                    st.success(f"Report saved to {report_file_path}")
                else:
                    st.error("Failed to generate PDF report. ")

            except Exception as e:
                st.error(f"PDF generation failed: {e}")

            st.session_state.raw_df = None # free memory
            st.session_state.cleaned_df = None
            st.session_state.has_data = False
