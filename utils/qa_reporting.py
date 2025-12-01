"""
qa_report.py

Functions for generating PDF reports.
"""

import os
from datetime import datetime
import pandas as pd
from fpdf import FPDF
import dataframe_image as dfi

from utils.qa_yaml_utils import safe_load_numpy_yaml
from PIL import Image



# def add_image_page(
#     pdf: FPDF, 
#     image_path: str, 
#     plot_title: str,
# ) -> None:
#     """
#     Add a new page to the PDF with an image and a centered title.
#     """
#     pdf.add_page()
#     pdf.set_font("Arial", style="B", size=18)
#     pdf.cell(0, 20, plot_title, ln=True, align='C')
#     pdf.image(image_path, x=10, y=30, w=280)
def add_image_page(pdf: FPDF, image_path: str, plot_title: str) -> None:
    """
    Add a new page to the PDF with an image (auto-fit) and a centered title.
    """
    # 1. Page settings (A4 landscape)
    PAGE_W = 297
    PAGE_H = 210
    X_MARGIN = 10
    TITLE_HEIGHT = 20
    TOP_MARGIN = 30
    BOTTOM_MARGIN = 10

    # 2. Area available for image
    max_w = PAGE_W - 2 * X_MARGIN
    max_h = PAGE_H - TOP_MARGIN - BOTTOM_MARGIN

    # 3. Get image size
    with Image.open(image_path) as img:
        img_w_px, img_h_px = img.size
        img_ratio = img_w_px / img_h_px
        box_ratio = max_w / max_h

        # 4. Decide scaling
        if img_ratio > box_ratio:
            # Image is wider than box, fit to width
            disp_w = max_w
            disp_h = max_w / img_ratio
        else:
            # Image is taller than box, fit to height
            disp_h = max_h
            disp_w = max_h * img_ratio

    # 5. Insert
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=18)
    pdf.cell(0, TITLE_HEIGHT, plot_title, ln=True, align='C')
    # Center the image horizontally
    x = (PAGE_W - disp_w) / 2
    y = TOP_MARGIN
    pdf.image(image_path, x=x, y=y, w=disp_w, h=disp_h)


def add_info_row(
    pdf: FPDF, 
    label: str, 
    value: str, 
    label2: str, 
    value2: str,
) -> None:
    """
    Add a row of dataset information to the PDF.
    """
    pdf.set_x(30)
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(40, 5, label, ln=0)
    pdf.set_font("Arial", size=12)
    pdf.cell(80, 5, f"{value}", ln=0)
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(40, 5, label2, ln=0)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 5, f"{value2}", ln=True)
    
def add_dataset_info(
    pdf: FPDF, 
    max_omr: float, 
    min_omr: float,
) -> None:
    """
    Add dataset information (specifically, OMR values) to the PDF below an image.
    """
    pdf.ln(150)
    add_info_row(pdf, "Minimum OMR: ", str(min_omr), "Maximum OMR: ", str(max_omr))

def add_images_from_folder(
    pdf: FPDF, 
    folder_path: str, 
    ks_df: pd.DataFrame, 
    p_value_threshold: float = 0.05,
) -> None:
    """
    Add images from a folder to the PDF along with KS statistic and P-value information.
    """
    for filename in os.listdir(folder_path):
        if filename.startswith("distribution"):
            pdf.add_page()
            plot_title = os.path.splitext(filename)[0]
            variable = plot_title.replace("distribution_comparison_", "")
            
            img_path = os.path.join(folder_path, filename)
            pdf.set_font("Arial", style="B", size=18)
            pdf.cell(0, 20, f"Distribution Comparison of {variable}", ln=True, align='C')
            pdf.image(img_path, x=10, y=30, w=280) 

            row = ks_df[ks_df["Variable"] == variable].iloc[0]
            ks_stat = row['KS Statistic']
            p_value = row['P-value']

            pdf.ln(145)
            pdf.set_x(20)
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(28, 5, "KS Statistic: ", ln=0)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 5, f"{ks_stat}", ln=True)

            pdf.set_x(20)
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(28, 5, "P-value: ", ln=0)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 5, f"{p_value}", ln=True)

            pdf.ln(5)
            pdf.set_x(20)
            confidence_level = int((1 - p_value_threshold) * 100)

            if p_value > p_value_threshold:
                conclusion = f"They are consistent at {confidence_level}% confidence level."
            else:
                conclusion = f"They are NOT consistent at {confidence_level}% confidence level."

            pdf.set_font("Arial", style="B", size=12)
            pdf.multi_cell(0, 5, conclusion)

def add_fpr_plot(
    pdf: FPDF, 
    image_path: str, 
    plot_title: str, 
    fpr_stats_cleaned_omr: dict, 
    fpr_stats_holdout_omr: dict, 
    fpr_stats_raw_omr: dict, 
    fprp_stats_holdout_omr: dict,
) -> None:
    """
    Add an FPR plot to the PDF along with a table of FPR and FPRP statistics.
    """
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=18)
    pdf.cell(0, 20, plot_title, ln=True, align='C')
    pdf.image(image_path, x=10, y=25, w=280)
    pdf.ln(140)
    
    indention = 55
    headers = ["Metric", "Warning (%)", "Alert (%)"]
    col_widths = [80, 50, 50]

    pdf.set_x(indention)
    pdf.set_font("Arial", 'B', size=12)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 5, header, border=1, align='C')
    pdf.ln()

    rows = [
        ("Inwindow Validation Data (without Outlier)", fpr_stats_cleaned_omr),
        ("Inwindow Validation Data (with Outlier)", fpr_stats_raw_omr),
        ("Holdout Data", fpr_stats_holdout_omr),
        ("Holdout Data (Persistence)", fprp_stats_holdout_omr),
    ]

    pdf.set_font("Arial", size=12)
    for label, stats in rows:
        pdf.set_x(indention)
        pdf.cell(col_widths[0], 5, label, border=1)
        pdf.cell(col_widths[1], 5, f"{stats['warning']:.2f}", border=1, align='C')
        pdf.cell(col_widths[2], 5, f"{stats['alert']:.2f}", border=1, align='C')
        pdf.ln()

def generate_dataset_description(dataset_info: dict) -> str:
    """
    Generate a description for a dataset based on its stats.
    """
    return (
        f"Start Time: {dataset_info['start_time']}\n"
        f"End Time: {dataset_info['end_time']}\n"
        f"Number of Records: {dataset_info['n_records']}\n"
        f"Window Size: {dataset_info['window_size']}\n"
    )


def generate_qa_report(
    model_name: str,
    fpr_file_path: str,
    ks_file_path: str,
    report_file_path: str,
    fpr_stats_cleaned_omr_fpath: str,
    fpr_stats_holdout_omr_fpath: str,
    fpr_stats_raw_omr_fpath: str,
    fprp_stats_holdout_omr_fpath: str,
    data_stats_fpath: str,
    ks_df_fpath: str,
    datasets_range_fpath: str,
) -> None:
    """
    Generate a QA report in PDF format.
    """
    fpr_stats_cleaned_omr = safe_load_numpy_yaml(fpr_stats_cleaned_omr_fpath)
    fpr_stats_holdout_omr = safe_load_numpy_yaml(fpr_stats_holdout_omr_fpath)
    fpr_stats_raw_omr = safe_load_numpy_yaml(fpr_stats_raw_omr_fpath)
    fprp_stats_holdout_omr = safe_load_numpy_yaml(fprp_stats_holdout_omr_fpath)
    data_stats = safe_load_numpy_yaml(data_stats_fpath)
    datasets_range = safe_load_numpy_yaml(datasets_range_fpath)
    ks_df = pd.read_csv(ks_df_fpath)

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # page 1: Cover page
    pdf.add_page()
    pdf.ln(20)
    pdf.set_font("Arial", style="B", size=30)
    pdf.cell(0, 20, model_name, ln=True, align='C')
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "Model QA Report", ln=True, align='C')

    indention = 35
    pdf.ln(20)
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Validation Dataset:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.set_x(indention)
    pdf.multi_cell(0, 10, generate_dataset_description(data_stats['validation']))

    pdf.ln(5)
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Holdout Dataset:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.set_x(indention)
    pdf.multi_cell(0, 10, generate_dataset_description(data_stats['holdout']))

    timestamp = datetime.now().strftime("%m-%d-%y %H:%M:%S")
    pdf.ln(10)
    pdf.set_font("Arial", style="I", size=12)
    pdf.cell(0, 0, f"Report generated on {timestamp}.")

    # page 2: Model performance analysis 
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Arial", style="B", size=30)
    pdf.cell(0, 20, "Model Performances Analysis", ln=True, align='C')

    # pages 3, 4, 5: OMR plots
    plot_titles = [
        "Validation Set (without outlier): OMR vs Time",
        "Validation Set (with outlier): OMR vs Time",
        "Holdout Set (with persistence): OMR vs Time",
        "False Positive Rates (with and without persistence)",
        "OMR Distributions",
    ]
    
    add_image_page(pdf, os.path.join(fpr_file_path, f"{model_name} - Cleaned Validation OMR.jpg"), plot_titles[0])
    add_dataset_info(
        pdf, 
        max_omr=datasets_range['cleaned_omr_range']['max'], 
        min_omr=datasets_range['cleaned_omr_range']['min']
    )

    add_image_page(pdf, os.path.join(fpr_file_path, f"{model_name} - Raw Validation OMR.jpg"), plot_titles[1])
    add_dataset_info(
        pdf, 
        max_omr=datasets_range['raw_omr_range']['max'], 
        min_omr=datasets_range['raw_omr_range']['min']
    )

    add_image_page(pdf, os.path.join(fpr_file_path, f"{model_name} - Holdout OMR.jpg"), plot_titles[2])
    add_dataset_info(
        pdf, 
        max_omr=datasets_range['holdout_omr_range']['max'], 
        min_omr=datasets_range['holdout_omr_range']['min']
    )

    # page 6: FPR plot
    add_fpr_plot(
        pdf=pdf,
        image_path=os.path.join(fpr_file_path, f"{model_name} - False Positive Rates (with and without persistence).jpg"),
        plot_title=plot_titles[3],
        fpr_stats_cleaned_omr=fpr_stats_cleaned_omr,
        fpr_stats_holdout_omr=fpr_stats_holdout_omr,
        fpr_stats_raw_omr=fpr_stats_raw_omr,
        fprp_stats_holdout_omr=fprp_stats_holdout_omr,
    )

    # page 7: OMR distributions
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=18)
    pdf.cell(0, 20, plot_titles[4], ln=True, align='C')
    pdf.image(os.path.join(fpr_file_path, f"{model_name} - OMR Distributions.jpg"), x=10, y=30, w=280)

    # page 8: KS caption
    pdf.add_page()
    pdf.ln(20)
    pdf.set_font("Arial", style="B", size=30)
    pdf.cell(0, 20, "Consistency between Validation and Holdout Datasets", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    caption = (
        """
        Comparing the consistency of validation and holdout datasets using the Kolmogorov-Smirnov (KS) statistic 
        is a critical step in model validation. It helps ensure that the model will perform reliably 
        on unseen data by verifying that the data used during development is representative of the 
        data it will encounter in practice. The KS statistic provides a measure of the difference 
        between the datasets, and the p-value helps determine whether that difference is significant 
        enough to warrant concern. This process helps prevent potential issues with model generalization, 
        leading to more robust and reliable models.
        """
    )
    pdf.multi_cell(0, 10, caption, align="C")

    # page 9 onwards: KS distribution comparison images and table
    add_images_from_folder(pdf, ks_file_path, ks_df)

    # last page: KS results summary image
    ks_plot_title = "Summary of the Inwindow Validation and Holdout Datasets Consistency"
    add_image_page(pdf, os.path.join(ks_file_path, "ks_results.jpg"), ks_plot_title)

    report_pdf_fpath = os.path.join(report_file_path, f"model_qa_report_{model_name}.pdf")
    pdf.output(report_pdf_fpath)


def generate_summary_fprp(sprint_path: str) -> pd.DataFrame:
    """
    Generates a summary DataFrame for False Positive Rate with Persistence (FPRP) statistics across models.

    This function iterates through each model folder within the specified `sprint_path`, retrieves the FPRP 
    statistics stored in a YAML file, and compiles them into a summary DataFrame. Each model's warning 
    and alert statistics are sorted in descending order of alert and warning values, and the summary is saved as a 
    .jpg image.
    """

    data = []

    for model_name in os.listdir(sprint_path):
        if model_name.startswith('.'):
            continue

        model_folder = os.path.join(sprint_path, model_name, 'performance_assessment_report', 'FPR')

        fprp_file = os.path.join(model_folder, 'fprp_stats_holdout_omr_df.yaml')
        
        if os.path.isfile(fprp_file):
            stats = safe_load_numpy_yaml(fprp_file)

            warning = stats.get('warning', None)
            alert = stats.get('alert', None)

            data.append([model_name, warning, alert])

    summary_fprp = pd.DataFrame(data, columns=['Model Name','FPR with persistence at 5.0% OMR (Warning)', 'FPR with persistence at 10.0% OMR (Alert)'])
    summary_fprp = summary_fprp.sort_values(by=[
        'FPR with persistence at 10.0% OMR (Alert)',
        'FPR with persistence at 5.0% OMR (Warning)',
    ], ascending=[False, False])
    summary_fprp_no_index = summary_fprp.reset_index(drop=True)
    dfi.export(summary_fprp_no_index, os.path.join(sprint_path, "fpr_summary.jpg"), table_conversion='matplotlib')

    return summary_fprp


def generate_sprint_summary_report(
    sprint_path: str,
    sprint_name: str,
) -> None:
    """
    Generates a summary PDF report for a sprint, displaying key FPR with Persistence (FPRP) metrics.

    This function creates a multi-page PDF report with:
    - A title page showing the sprint name.
    - A summary page with an image of FPR with Persistence scores for models in the sprint.
    - A comparison page with an image showing Warning and Alert FPRP scores across models.
    """
    
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # page 1: Title page 
    pdf.add_page()
    pdf.ln(80)
    pdf.set_font("Arial", style="B", size=30)
    pdf.cell(0, 20, sprint_name, ln=True, align='C')
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "Summary Results", ln=True, align='C')
    
    # page 2: fPR summary image
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=18)
    pdf.cell(0, 0, f"FPR with Persistance Scores of {sprint_name} Models", ln=True, align='C')
    pdf.image(os.path.join(sprint_path, "fpr_summary.jpg"), x=20, y=30, w=250)

    # page 3: FPR comparison plot
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=18)
    pdf.cell(0, 0, "Warning and Alert FPR with Persistence Comparison per Model", ln=True, align='C')
    pdf.image(os.path.join(sprint_path, "fprp_plots.jpg"), x=10, y=30, w=280)

    pdf.output(os.path.join(sprint_path, f"{sprint_name} Summary Results.pdf"))
