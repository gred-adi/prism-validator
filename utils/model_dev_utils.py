import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st

from typing import Tuple, Union, Dict, Any
from pathlib import Path
from scipy.stats import pearsonr
from fpdf import FPDF
from io import BytesIO

def cleaned_dataset_name_split(
        filename:str
        ) -> Tuple [str, str, str]:
    """
    Given a filename, returns the site_name, model_name,
    and inclusive dates.

    filename format is expected to be:
        CLEANED-[model_name]-[inclusive_dates]-RAW.csv
        Example: CLEANED-AP-TVI-U1-BFP_A_MOTOR-20240601-20250801-RAW.csv

    Args:
        filename (str): The name of the cleaned dataset file.

    Returns:
        Tuple[str, str, str]: A tuple containing:
        site_name, model_name, inclusive_dates
    """
    # Remove the 'CLEANED-' prefix and suffix
    base_name = filename.removeprefix("CLEANED-")
    if base_name.endswith("-WITH-OUTLIER.csv"):
        base_name = base_name.removesuffix("-WITH-OUTLIER.csv")
    elif base_name.endswith("-WITHOUT-OUTLIER.csv"):
        base_name = base_name.removesuffix("-WITHOUT-OUTLIER.csv")
    elif base_name.endswith("-RAW.csv"):
        base_name = base_name.removesuffix("-RAW.csv")
    # Split the remaining string by the second to the last "-"
    parts = base_name.rsplit("-", 2)
    model_name = parts[0]
    inclusive_dates = parts[1] + "-" + parts[2]
    # Get the value between the first and second "-" in model_name (TVI in the example)
    site_name = model_name.split("-")[1]
    return site_name, model_name, inclusive_dates

def data_cleaning_read_prism_csv(df: pd.DataFrame, project_points: pd.DataFrame):
    # df = pd.read_csv(path, index_col=False)
    """
    Takes a PRISM DataFrame and returns a DataFrame with mapped metric names and the original PRISM header.
    Args:
        df (pd.DataFrame): PRISM DataFrame to be processed.
        project_points (pd.DataFrame): DataFrame containing project points mapping (most likely from project_points.csv).
    """
    df_header = df.iloc[:4,:]

    df.columns = df_header.columns
    df = df.iloc[4:,:]

    df.rename(columns={'Point Name':'DATETIME'}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.apply(pd.to_numeric)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    # Data Point Mapping
    new_column = []
    for column in df.columns:
        if column == 'DATETIME':
            new_column.append(column)
        elif column in project_points['Name'].tolist():
            mapping_value = str(project_points.loc[project_points['Name'] == column, 'Metric'].values[0])
            if (mapping_value == 'nan'):
                new_column.append(column)
            else:
                new_column.append(mapping_value)
        else:
            new_column.append(column)

    df.columns = new_column
    return df, df_header

def _generate_report_cover_page(pdf: FPDF,
                                raw_df: pd.DataFrame,
                                total_excluded: int,
                                stats_dict: dict = None,
                                numeric_filters: list = [],
                                datetime_filters: list = []):
    """Adds a summary page to the FPDF object."""
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Data Cleaning Report", 0, 1, "C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)

    start_time = raw_df['DATETIME'].min()
    end_time = raw_df['DATETIME'].max()
    total_rows = len(raw_df)

    pdf.cell(0, 8, f"Raw Data Timespan: {start_time} to {end_time}", 0, 1)
    pdf.cell(0, 8, f"Total Rows in Raw Data: {total_rows}", 0, 1)
    pdf.cell(0, 8, f"Total Rows Excluded: {total_excluded}", 0, 1)

    # This section now works for BOTH simple and full reports
    if stats_dict:
        pdf.ln(10) # Add a line break
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Filter Impact Statistics", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Excluded by Numeric: {stats_dict['pct_numeric']:.2f}% ({stats_dict['numeric_count']:,} points)", 0, 1)
        pdf.cell(0, 8, f"Excluded by Datetime: {stats_dict['pct_datetime']:.2f}% ({stats_dict['datetime_count']:,} points)", 0, 1)
        pdf.cell(0, 8, f"Total Excluded (Union): {stats_dict['pct_union']:.2f}% ({stats_dict['union_count']:,} points)", 0, 1)

    # --- NEW SECTION: Print the list of filters ---
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Active Filters Applied", 0, 1)
    pdf.set_font("Arial", "", 10) # Use a smaller font for the list

    if not numeric_filters and not datetime_filters:
        pdf.cell(0, 5, "No filters were applied.", 0, 1)

    for col, op, val in numeric_filters:
        pdf.cell(0, 5, f"- Numeric: {col} {op} {val}", 0, 1)

    for op, val in datetime_filters:
        if op == "between (includes edge values)":
            pdf.cell(0, 5, f"- Datetime: remove between {val[0]} and {val[1]}", 0, 1)
        else:
            pdf.cell(0, 5, f"- Datetime: {op} {val}", 0, 1)

def generate_simple_report(raw_df: pd.DataFrame, numeric_filters: list, datetime_filters: list, pdf_file_path: Path):
    """Generates a simple text-only PDF report."""

    total_points = len(raw_df)
    if total_points == 0:
        st.error("Cannot generate report from empty data.")
        return

    datetime_mask = pd.Series(False, index=raw_df.index)
    for op, val in datetime_filters:
        start_time = pd.to_datetime(val[0]) if op == "between (includes edge values)" else None
        end_time = pd.to_datetime(val[1]) if op == "between (includes edge values)" else None
        if op == "< (remove before)": datetime_mask = datetime_mask | (raw_df['DATETIME'] < pd.to_datetime(val))
        elif op == "> (remove after)": datetime_mask = datetime_mask | (raw_df['DATETIME'] > pd.to_datetime(val))
        elif op == "between (includes edge values)": datetime_mask = datetime_mask | ((raw_df['DATETIME'] >= start_time) & (raw_df['DATETIME'] <= end_time))

    all_numeric_mask = pd.Series(False, index=raw_df.index)
    for col, op, val in numeric_filters:
        if op == "<": all_numeric_mask = all_numeric_mask | (raw_df[col] < val)
        elif op == "<=": all_numeric_mask = all_numeric_mask | (raw_df[col] <= val)
        elif op == "==": all_numeric_mask = all_numeric_mask | (raw_df[col] == val)
        elif op == ">=": all_numeric_mask = all_numeric_mask | (raw_df[col] >= val)
        elif op == ">": all_numeric_mask = all_numeric_mask | (raw_df[col] > val)

    total_excluded = (all_numeric_mask | datetime_mask).sum()
    numeric_count = all_numeric_mask.sum()
    datetime_count = datetime_mask.sum()

    pct_numeric = (numeric_count / total_points) * 100
    pct_datetime = (datetime_count / total_points) * 100
    pct_union = (total_excluded / total_points) * 100

    stats_payload = {
        "numeric_count": numeric_count, "datetime_count": datetime_count, "union_count": total_excluded,
        "pct_numeric": pct_numeric, "pct_datetime": pct_datetime, "pct_union": pct_union
    }

    pdf = FPDF(orientation="portrait")

    _generate_report_cover_page(
        pdf,
        raw_df,
        total_excluded,
        stats_payload,
        numeric_filters,
        datetime_filters
    )

    try:
        pdf.output(pdf_file_path)
    except Exception as e:
        st.error(f"Failed to save PDF report: {e}")

def generate_data_cleaning_visualizations(raw_df: pd.DataFrame,
                                          cleaned_df: pd.DataFrame,
                                          numeric_filters: list,
                                          datetime_filters: list,
                                          selected_metrics: list,
                                          generate_report: bool,
                                          pdf_file_path: Path = None):
    """
    Generates and displays all data cleaning visualizations and consolidates them in a report
    if generate_report is True
    """
    pdf = None
    if generate_report:
        if not pdf_file_path:
            st.error("PDF file path is missing. Cannot generate report.")
            generate_report = False # Abort report generation
        else:
            pdf = FPDF(orientation="portrait") # Initialize PDF object

    # Create Masks for all filter types
    total_points = len(raw_df)
    if total_points == 0:
        st.info("No data to visualize.")
        return

    # Master mask for all datetime filters
    datetime_mask = pd.Series(False, index=raw_df.index)
    for op, val in datetime_filters:
        start_time = pd.to_datetime(val[0]) if op == "between (includes edge values)" else None
        end_time = pd.to_datetime(val[1]) if op == "between (includes edge values)" else None

        if op == "< (remove before)":
            datetime_mask = datetime_mask | (raw_df['DATETIME'] < pd.to_datetime(val))
        elif op == "> (remove after)":
            datetime_mask = datetime_mask | (raw_df['DATETIME'] > pd.to_datetime(val))
        elif op == "between (includes edge values)":
            datetime_mask = datetime_mask | ((raw_df['DATETIME'] >= start_time) & (raw_df['DATETIME'] <= end_time))

    # Master mask for all numeric filters
    all_numeric_mask = pd.Series(False, index=raw_df.index)
    for col, op, val in numeric_filters:
        if op == "<": all_numeric_mask = all_numeric_mask | (raw_df[col] < val)
        elif op == "<=": all_numeric_mask = all_numeric_mask | (raw_df[col] <= val)
        elif op == "==": all_numeric_mask = all_numeric_mask | (raw_df[col] == val)
        elif op == ">=": all_numeric_mask = all_numeric_mask | (raw_df[col] >= val)
        elif op == ">": all_numeric_mask = all_numeric_mask | (raw_df[col] > val)

    # Generate Filter Statistics
    numeric_count = all_numeric_mask.sum()
    datetime_count = datetime_mask.sum()
    union_count = (all_numeric_mask | datetime_mask).sum()
    pct_numeric = (numeric_count / total_points) * 100
    pct_datetime = (datetime_count / total_points) * 100
    pct_union = (union_count / total_points) * 100

    stats_payload = {
        "numeric_count": numeric_count, "datetime_count": datetime_count, "union_count": union_count,
        "pct_numeric": pct_numeric, "pct_datetime": pct_datetime, "pct_union": pct_union
    }

    st.markdown("### Filter Impact Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        ### Excluded by Numeric
        ## {pct_numeric:.2f}%
        {numeric_count:,} points
        """)

    with col2:
        st.markdown(f"""
        ### Excluded by Datetime
        ## {pct_datetime:.2f}%
        {datetime_count:,} points
        """)

    with col3:
        st.markdown(f"""
        ### Total Excluded (Union)
        ## {pct_union:.2f}%
        {union_count:,} points
        """)

    # Create first page for pdf
    if generate_report:
        _generate_report_cover_page(
            pdf,
            raw_df,
            stats_payload['union_count'],
            stats_payload,
            numeric_filters,
            datetime_filters
        )

    # Generate Graph 1 (Timeline)
    st.markdown("### Filter Timeline")

    numeric_points = raw_df.loc[all_numeric_mask, 'DATETIME']
    datetime_points = raw_df.loc[datetime_mask & ~all_numeric_mask, 'DATETIME']

    fig_timeline, ax_timeline = plt.subplots(figsize=(12, 3))
    ax_timeline.vlines(numeric_points, ymin=0, ymax=1, color='gray', alpha=0.7, linewidth=0.5, label='Removed (Numeric)')
    ax_timeline.vlines(datetime_points, ymin=0, ymax=1, color='purple', alpha=0.7, label='Removed (Datetime)')
    ax_timeline.get_yaxis().set_visible(False)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_xlabel('DATETIME')
    ax_timeline.legend()
    ax_timeline.grid(axis='x', linestyle='--', alpha=0.5)
    st.pyplot(fig_timeline)

    # Add timeline graph to pdf report
    if generate_report:
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Timeline Graph", 0, 1, "C")
        img_buffer = BytesIO()
        fig_timeline.savefig(img_buffer, format="png", bbox_inches='tight')
        img_buffer.seek(0)
        image_bytes = img_buffer.read()
        # Center image
        pdf.image(image_bytes, x=10, y=20, w=pdf.w - 20, type="PNG")

    plt.close(fig_timeline)

    # Generate Graphs 2 and 3 (Metric-Specific Plots)
    st.markdown("### Filtered Metric Plots")

    for metric in selected_metrics:
        st.subheader(f"Plot for: {metric}")

        # Create 3 new masks specific to this loop
        metric_specific_mask = pd.Series(False, index=raw_df.index)
        other_numeric_mask = pd.Series(False, index=raw_df.index)
        metric_filters_list = []

        for col, op, val in numeric_filters:
            if col not in raw_df.columns: continue

            if op == "<": mask = (raw_df[col] < val)
            elif op == "<=": mask = (raw_df[col] <= val)
            elif op == "==": mask = (raw_df[col] == val)
            elif op == ">=": mask = (raw_df[col] >= val)
            elif op == ">": mask = (raw_df[col] > val)
            else: continue

            if col == metric:
                metric_specific_mask = metric_specific_mask | mask
                metric_filters_list.append((op, val))
            else:
                other_numeric_mask = other_numeric_mask | mask

        # boolean conditions for coloring
        purple_condition = datetime_mask
        dark_gray_condition = metric_specific_mask & ~datetime_mask
        light_gray_condition = other_numeric_mask & ~datetime_mask & ~metric_specific_mask

        # Graph 2
        fig_metric, ax_metric = plt.subplots(figsize=(12, 5))

        sns.lineplot(
            data=raw_df,
            x='DATETIME',
            y=metric,
            ax=ax_metric,
            label=f"{metric} (Raw Data)",
            zorder=10,  # Make sure line is on top
            linewidth=0.5,
            color="green"
        )

        # Graph 3
        fig_metric_cleaned, ax_metric_cleaned = plt.subplots(figsize=(12, 5))

        sns.lineplot(
            data=cleaned_df,
            x='DATETIME',
            y=metric,
            ax=ax_metric_cleaned,
            label=f"{metric} (Cleaned Data)",
            zorder=10,  # Make sure line is on top
            linewidth=0.5,
            color='green',
            legend=None
        )



        # Get ymin and max from the raw dataset to be used as limits for both graphs
        xmin_raw, xmax_raw = ax_metric.get_xlim()
        ymin_raw, ymax_raw = ax_metric.get_ylim()

        # Apply global limits to BOTH plots
        ax_metric.set_ylim(ymin_raw, ymax_raw)
        ax_metric_cleaned.set_ylim(ymin_raw, ymax_raw)

        # Plot the highlights for Graph 2
        ax_metric.fill_between(
            raw_df['DATETIME'], ymin_raw, ymax_raw,
            where=purple_condition,
            color='purple', alpha=0.4, label='Datetime Filter', linewidth=0
        )

        ax_metric.fill_between(
            raw_df['DATETIME'], ymin_raw, ymax_raw,
            where=light_gray_condition,
            color='gray', alpha=0.6, label='Other Numeric Filter', linewidth=0
        )

        ax_metric.fill_between(
            raw_df['DATETIME'], ymin_raw, ymax_raw,
            where=dark_gray_condition,
            color='blue', alpha=0.4, label=f'{metric} Filter', linewidth=0
        )
        ax_metric.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
        ax_metric.grid(False)
        ax_metric.set_xlabel('DATETIME')
        ax_metric.set_ylabel(metric)
        ax_metric.set_title(f"Time Series for {metric} with Filter Highlights")

        #Finish Graph 3
        ax_metric_cleaned.set_xlim(xmin_raw, xmax_raw)
        ax_metric_cleaned.set_ylim(ymin_raw, ymax_raw) # Re-apply limits
        ax_metric_cleaned.grid(False)
        ax_metric_cleaned.set_xlabel('DATETIME')
        ax_metric_cleaned.set_ylabel(metric)
        ax_metric_cleaned.set_title(f"Time Series for {metric} for cleaned data")

        x_pos = cleaned_df['DATETIME'].min()

        for op, val in metric_filters_list:
            # Draw the horizontal dotted line
            ax_metric_cleaned.axhline(
                y=val,
                color='gray',
                linestyle='--',
                linewidth=1
            )
            # Draw the text label directly on the plot
            ax_metric_cleaned.text(
                x=x_pos,
                y=val,
                s=f" {op} {val}",
                color='gray',
                verticalalignment='top',
                horizontalalignment='left'
            )

        graph_col1, graph_col2 = st.columns(2)
        with graph_col1:
            st.pyplot(fig_metric)
        with graph_col2:
            st.pyplot(fig_metric_cleaned)

        if generate_report:
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, f"Plots for Metric: {metric}", 0, 1, "C")

            margin = 10
            gutter = 30

            # Width for both images (full page width minus margins)
            img_width = pdf.w - (margin * 2)

            # Calculate height based on your 12,5 fig_metric aspect ratio (height = width * 5/12)
            img_height = img_width * (5 / 12)

            # X position for both plots
            img_x = margin

            # Y position for Graph 2 (top)
            img_y1 = 30 # Position below the title

            # Y position for Graph 3 (bottom)
            img_y2 = img_y1 + img_height + gutter

            # Save Graph 2 (Raw) to buffer and add to top
            with BytesIO() as img_buffer_raw:
                fig_metric.savefig(img_buffer_raw, format="png", bbox_inches='tight')
                img_buffer_raw.seek(0)
                pdf.image(img_buffer_raw, x=img_x, y=img_y1, w=img_width, type="PNG")

            # Save Graph 3 (Cleaned) to buffer and add to bottom
            with BytesIO() as img_buffer_cleaned:
                fig_metric_cleaned.savefig(img_buffer_cleaned, format="png", bbox_inches='tight')
                img_buffer_cleaned.seek(0)
                pdf.image(img_buffer_cleaned, x=img_x, y=img_y2, w=img_width, type="PNG")

                plt.close(fig_metric)
                plt.close(fig_metric_cleaned)

    if generate_report:
        try:
            pdf.output(pdf_file_path)
        except Exception as e:
            st.error(f"Failed to save full PDF report: {e}")

def split_holdout(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    split_mark: Union[float, str, pd.Timestamp],
    date_col: str = "Point Name",
    verbose: bool = False, # Set to True to print stats for logging
    remove_header_rows: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, Dict[str, Dict[str, Any]]]:
    """
    Split cleaned_df into train/validation and holdout sets by a date or percentage split.
    Optionally removes header rows and prepends them to split outputs.
    Also returns split statistics for each set.
    """

    df_header = None
    if remove_header_rows > 0:
        df_header = cleaned_df.iloc[:remove_header_rows].reset_index(drop=True)
        raw_df = raw_df.iloc[remove_header_rows:, :].reset_index(drop=True)
        cleaned_df = cleaned_df.iloc[remove_header_rows:, :].reset_index(drop=True)

    raw_df[date_col] = pd.to_datetime(raw_df[date_col])
    cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col])

    raw_df_sorted = raw_df.sort_values(date_col).reset_index(drop=True)
    cleaned_df_sorted = cleaned_df.sort_values(date_col).reset_index(drop=True)

    if not isinstance(split_mark, float):
        if not isinstance(split_mark, pd.Timestamp):
            split_mark = pd.to_datetime(split_mark)
    else:
        n = len(raw_df_sorted)
        h_raw = int(round(n * split_mark))
        split_idx = n - h_raw - 1 if h_raw < n else n - 1
        split_mark = raw_df_sorted.iloc[split_idx][date_col]

    train_val_df = cleaned_df_sorted[cleaned_df_sorted[date_col] <= split_mark].reset_index(drop=True)
    holdout_df = cleaned_df_sorted[cleaned_df_sorted[date_col] > split_mark].reset_index(drop=True)

    def get_stats(df):
        if len(df) > 0:
            start = df[date_col].iloc[0]
            end = df[date_col].iloc[-1]
            num_days = (end - start).days + 1
            return {"start": start, "end": end, "size": len(df), "num_days": num_days}
        else:
            return {"start": None, "end": None, "size": 0, "num_days": 0}

    stats = {
        "raw": get_stats(raw_df_sorted),
        "cleaned": get_stats(cleaned_df_sorted),
        "train_val": get_stats(train_val_df),
        "holdout": get_stats(holdout_df),
    }

    if verbose:
        for key, stat in stats.items():
            print(f"\n{key.title()} set:")
            if stat['size'] > 0:
                print(f"  Start time: {stat['start']}")
                print(f"  End time:   {stat['end']}")
                print(f"  Size:       {stat['size']} rows")
                print(f"  Num days:   {stat['num_days']}")
            else:
                print("  (Empty set)")

    if df_header is not None:
        train_val_df = pd.concat([df_header, train_val_df], ignore_index=True)
        holdout_df = pd.concat([df_header, holdout_df], ignore_index=True)

    return train_val_df, holdout_df, split_mark, stats

def generate_split_holdout_report(stats: Dict[str, Any], split_mark_used: str, pdf_file_path: Path, fig: plt.figure):
    """
    Generates a PDF report with the data split figure and stats tables.
    """
    pdf = FPDF(orientation="landscape")
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Data Split Report", 0, 1, "C")
    pdf.ln(5)

    # Add figure to pdf
    if fig:
        pdf.set_font("Arial", "B", 14)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png", bbox_inches='tight')
            img_buffer.seek(0)
            image_bytes = img_buffer.read()
            pdf.image(image_bytes, x=10, y=30, w=pdf.w - 20, type="PNG")
        plt.close(fig)
        pdf.ln(120)
    else:
        pdf.cell(0, 10, "No time span chart to display.", 0, 1, "L")

    # Add stats table to pdf
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Split Statistics", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Split mark used: {split_mark_used}", 0, 1)
    pdf.ln(5)

    for set_name, stat in stats.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"{set_name.title()}", 0, 1)
        pdf.set_font("Arial", "", 10)

        if stat.get("size", 0) > 0:
            start_str = stat.get("start").strftime("%Y-%m-%d %H:%M:%S") if stat.get("start") else "N/A"
            end_str = stat.get("end").strftime("%Y-%m-%d %H:%M:%S") if stat.get("end") else "N/A"

            # Create a simple table with borders (1)
            pdf.cell(40, 6, "Metric", 1)
            pdf.cell(0, 6, "Value", 1)
            pdf.ln()

            pdf.cell(40, 6, "Start", 1)
            pdf.cell(0, 6, start_str, 1)
            pdf.ln()

            pdf.cell(40, 6, "End", 1)
            pdf.cell(0, 6, end_str, 1)
            pdf.ln()

            pdf.cell(40, 6, "Rows", 1)
            pdf.cell(0, 6, f"{stat.get('size'):,}", 1) # Added comma formatting
            pdf.ln()

            pdf.cell(40, 6, "Number of days", 1)
            pdf.cell(0, 6, str(stat.get("num_days", "N/A")), 1)
            pdf.ln()

            pdf.ln(5) # Space after table
        else:
            pdf.cell(0, 6, "(Empty set)", 0, 1)
            pdf.ln(5)

    try:
        pdf.output(pdf_file_path)
        return True
    except Exception as e:
        return False

def read_prism_csv(df: pd.DataFrame):
    # df = pd.read_csv(path, index_col=False)

    df_header = df.iloc[:4,:]

    df.columns = df.iloc[1].values
    df = df.iloc[4:,:]

    df.rename(columns={'Extended Name':'DATETIME'}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.apply(pd.to_numeric)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    return df, df_header

def corrfunc(x, y, **kwds):
    cmap = kwds['cmap']
    norm = kwds['norm']
    ax = plt.gca()
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
    r, _ = pearsonr(x, y)
    facecolor = cmap(norm(r))
    ax.set_facecolor(facecolor)
    lightness = (max(facecolor[:3]) + min(facecolor[:3]) ) / 2
    ax.annotate(f"r={r:.2f}", xy=(.5, .5), xycoords=ax.transAxes, color='white' if lightness < 0.7 else 'black', size=30, ha='center', va='center')