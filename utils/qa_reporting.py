import os
import base64
from datetime import datetime
import pandas as pd
from fpdf import FPDF
import dataframe_image as dfi
from playwright.sync_api import sync_playwright
from jinja2 import Environment
import sys
import asyncio

from utils.qa_yaml_utils import safe_load_numpy_yaml
from PIL import Image

# ==========================================
# HELPER FUNCTIONS (Shared / Legacy FPDF)
# ==========================================

def add_image_page(pdf: FPDF, image_path: str, plot_title: str) -> None:
    """Adds a new page to a PDF with an auto-fitting image and a title.

    Args:
        pdf (FPDF): The FPDF object.
        image_path (str): The path to the image file.
        plot_title (str): The title for the page.
    """
    PAGE_W = 297
    PAGE_H = 210
    X_MARGIN = 10
    TITLE_HEIGHT = 20
    TOP_MARGIN = 30
    BOTTOM_MARGIN = 10

    max_w = PAGE_W - 2 * X_MARGIN
    max_h = PAGE_H - TOP_MARGIN - BOTTOM_MARGIN

    try:
        with Image.open(image_path) as img:
            img_w_px, img_h_px = img.size
            img_ratio = img_w_px / img_h_px
            box_ratio = max_w / max_h

            if img_ratio > box_ratio:
                disp_w = max_w
                disp_h = max_w / img_ratio
            else:
                disp_h = max_h
                disp_w = max_h * img_ratio

        pdf.add_page()
        pdf.set_font("Arial", style="B", size=18)
        pdf.cell(0, TITLE_HEIGHT, plot_title, ln=True, align='C')
        x = (PAGE_W - disp_w) / 2
        y = TOP_MARGIN
        pdf.image(image_path, x=x, y=y, w=disp_w, h=disp_h)
    except Exception as e:
        print(f"Error adding image {image_path}: {e}")
        # Add a blank page with error message if image fails
        pdf.add_page()
        pdf.set_font("Arial", style="B", size=18)
        pdf.cell(0, 20, f"Error: Could not load image {plot_title}", ln=True, align='C')


def add_info_row(pdf: FPDF, label: str, value: str, label2: str, value2: str) -> None:
    """Adds a two-column informational row to the PDF.

    Args:
        pdf (FPDF): The FPDF object.
        label (str): The label for the first value.
        value (str): The first value.
        label2 (str): The label for the second value.
        value2 (str): The second value.
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
    
def add_dataset_info(pdf: FPDF, max_omr: float, min_omr: float) -> None:
    """Adds dataset information (min/max OMR) to the PDF.

    Args:
        pdf (FPDF): The FPDF object.
        max_omr (float): The maximum OMR value.
        min_omr (float): The minimum OMR value.
    """
    pdf.ln(150)
    add_info_row(pdf, "Minimum OMR: ", str(min_omr), "Maximum OMR: ", str(max_omr))

def add_images_from_folder(pdf: FPDF, folder_path: str, ks_df: pd.DataFrame, p_value_threshold: float = 0.05) -> None:
    """Adds a page for each distribution image in a folder to the PDF.

    Args:
        pdf (FPDF): The FPDF object.
        folder_path (str): The path to the folder containing the images.
        ks_df (pd.DataFrame): The DataFrame with KS test results.
        p_value_threshold (float, optional): The p-value threshold for consistency.
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

def add_fpr_plot(pdf: FPDF, image_path: str, plot_title: str, fpr_stats_cleaned_omr: dict, fpr_stats_holdout_omr: dict, fpr_stats_raw_omr: dict, fprp_stats_holdout_omr: dict) -> None:
    """Adds a page with an FPR plot and a summary table to the PDF.

    Args:
        pdf (FPDF): The FPDF object.
        image_path (str): The path to the FPR plot image.
        plot_title (str): The title for the page.
        fpr_stats_cleaned_omr (dict): FPR stats for cleaned OMR.
        fpr_stats_holdout_omr (dict): FPR stats for holdout OMR.
        fpr_stats_raw_omr (dict): FPR stats for raw OMR.
        fprp_stats_holdout_omr (dict): FPRP stats for holdout OMR.
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
    """Generates a formatted string describing a dataset.

    Args:
        dataset_info (dict): A dictionary with dataset statistics.

    Returns:
        str: A formatted string with the dataset description.
    """
    return (
        f"Start Time: {dataset_info['start_time']}<br>"
        f"End Time: {dataset_info['end_time']}<br>"
        f"Number of Records: {dataset_info['n_records']}<br>"
        f"Window Size: {dataset_info['window_size']}"
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
    """Generates a QA report PDF using FPDF.

    This function orchestrates the creation of a comprehensive QA report,
    including performance analysis and consistency checks, using the FPDF library.

    Args:
        model_name (str): The name of the model.
        fpr_file_path (str): The path to the FPR files.
        ks_file_path (str): The path to the KS test files.
        report_file_path (str): The path to save the report.
        fpr_stats_cleaned_omr_fpath (str): The path to the cleaned OMR FPR stats.
        fpr_stats_holdout_omr_fpath (str): The path to the holdout OMR FPR stats.
        fpr_stats_raw_omr_fpath (str): The path to the raw OMR FPR stats.
        fprp_stats_holdout_omr_fpath (str): The path to the holdout OMR FPRP stats.
        data_stats_fpath (str): The path to the data stats file.
        ks_df_fpath (str): The path to the KS test results CSV.
        datasets_range_fpath (str): The path to the datasets range file.
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
    desc_val = generate_dataset_description(data_stats['validation']).replace("<br>", "\n")
    pdf.multi_cell(0, 10, desc_val)

    pdf.ln(5)
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Holdout Dataset:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.set_x(indention)
    desc_hold = generate_dataset_description(data_stats['holdout']).replace("<br>", "\n")
    pdf.multi_cell(0, 10, desc_hold)

    timestamp = datetime.now().strftime("%m-%d-%y %H:%M:%S")
    pdf.ln(10)
    pdf.set_font("Arial", style="I", size=12)
    pdf.cell(0, 0, f"Report generated on {timestamp}.")

    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Arial", style="B", size=30)
    pdf.cell(0, 20, "Model Performances Analysis", ln=True, align='C')

    plot_titles = [
        "Validation Set (without outlier): OMR vs Time",
        "Validation Set (with outlier): OMR vs Time",
        "Holdout Set (with persistence): OMR vs Time",
        "False Positive Rates (with and without persistence)",
        "OMR Distributions",
    ]
    
    add_image_page(pdf, os.path.join(fpr_file_path, f"{model_name} - Cleaned Validation OMR.jpg"), plot_titles[0])
    add_dataset_info(pdf, max_omr=datasets_range['cleaned_omr_range']['max'], min_omr=datasets_range['cleaned_omr_range']['min'])

    add_image_page(pdf, os.path.join(fpr_file_path, f"{model_name} - Raw Validation OMR.jpg"), plot_titles[1])
    add_dataset_info(pdf, max_omr=datasets_range['raw_omr_range']['max'], min_omr=datasets_range['raw_omr_range']['min'])

    add_image_page(pdf, os.path.join(fpr_file_path, f"{model_name} - Holdout OMR.jpg"), plot_titles[2])
    add_dataset_info(pdf, max_omr=datasets_range['holdout_omr_range']['max'], min_omr=datasets_range['holdout_omr_range']['min'])

    add_fpr_plot(
        pdf=pdf,
        image_path=os.path.join(fpr_file_path, f"{model_name} - False Positive Rates (with and without persistence).jpg"),
        plot_title=plot_titles[3],
        fpr_stats_cleaned_omr=fpr_stats_cleaned_omr,
        fpr_stats_holdout_omr=fpr_stats_holdout_omr,
        fpr_stats_raw_omr=fpr_stats_raw_omr,
        fprp_stats_holdout_omr=fprp_stats_holdout_omr,
    )

    pdf.add_page()
    pdf.set_font("Arial", style="B", size=18)
    pdf.cell(0, 20, plot_titles[4], ln=True, align='C')
    pdf.image(os.path.join(fpr_file_path, f"{model_name} - OMR Distributions.jpg"), x=10, y=30, w=280)

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

    add_images_from_folder(pdf, ks_file_path, ks_df)

    ks_plot_title = "Summary of the Inwindow Validation and Holdout Datasets Consistency"
    add_image_page(pdf, os.path.join(ks_file_path, "ks_results.jpg"), ks_plot_title)

    report_pdf_fpath = os.path.join(report_file_path, f"model_qa_report_{model_name}.pdf")
    pdf.output(report_pdf_fpath)


def generate_summary_fprp(sprint_path: str) -> pd.DataFrame:
    """Generates a summary DataFrame of FPRP statistics for a sprint.

    This function scans a sprint directory, reads the FPRP statistics for each
    model, and compiles them into a summary DataFrame.

    Args:
        sprint_path (str): The path to the sprint directory.

    Returns:
        pd.DataFrame: A DataFrame with the FPRP summary.
    """
    data = []
    for model_name in os.listdir(sprint_path):
        if model_name.startswith('.'): continue
        model_folder = os.path.join(sprint_path, model_name, 'performance_assessment_report', 'FPR')
        fprp_file = os.path.join(model_folder, 'fprp_stats_holdout_omr_df.yaml')
        if os.path.isfile(fprp_file):
            stats = safe_load_numpy_yaml(fprp_file)
            data.append([model_name, stats.get('warning', None), stats.get('alert', None)])

    summary_fprp = pd.DataFrame(data, columns=['Model Name','FPR with persistence at 5.0% OMR (Warning)', 'FPR with persistence at 10.0% OMR (Alert)'])
    summary_fprp = summary_fprp.sort_values(by=['FPR with persistence at 10.0% OMR (Alert)', 'FPR with persistence at 5.0% OMR (Warning)'], ascending=[False, False])
    summary_fprp_no_index = summary_fprp.reset_index(drop=True)
    dfi.export(summary_fprp_no_index, os.path.join(sprint_path, "fpr_summary.jpg"), table_conversion='matplotlib')
    return summary_fprp


def generate_sprint_summary_report(sprint_path: str, sprint_name: str) -> None:
    """Generates a summary PDF report for a sprint.

    Args:
        sprint_path (str): The path to the sprint directory.
        sprint_name (str): The name of the sprint.
    """
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.ln(80)
    pdf.set_font("Arial", style="B", size=30)
    pdf.cell(0, 20, sprint_name, ln=True, align='C')
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "Summary Results", ln=True, align='C')
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=18)
    pdf.cell(0, 0, f"FPR with Persistance Scores of {sprint_name} Models", ln=True, align='C')
    pdf.image(os.path.join(sprint_path, "fpr_summary.jpg"), x=20, y=30, w=250)
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=18)
    pdf.cell(0, 0, "Warning and Alert FPR with Persistence Comparison per Model", ln=True, align='C')
    pdf.image(os.path.join(sprint_path, "fprp_plots.jpg"), x=10, y=30, w=280)
    pdf.output(os.path.join(sprint_path, f"{sprint_name} Summary Results.pdf"))

# ==========================================
# NEW PLAYWRIGHT-BASED REPORT GENERATOR
# ==========================================

def get_image_base64(path):
    """Loads an image and returns it as a base64 encoded string.

    Args:
        path (str): The path to the image file.

    Returns:
        str: The base64 encoded image string, or None if the file doesn't exist.
    """
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    return None

def generate_qa_report_playwright(
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
) -> bool:
    """Generates a QA report PDF using Playwright.

    This function orchestrates the creation of a comprehensive QA report by
    rendering an HTML template with the provided data and converting it to a
    PDF using Playwright.

    Args:
        model_name (str): The name of the model.
        fpr_file_path (str): The path to the FPR files.
        ks_file_path (str): The path to the KS test files.
        report_file_path (str): The path to save the report.
        fpr_stats_cleaned_omr_fpath (str): The path to the cleaned OMR FPR stats.
        fpr_stats_holdout_omr_fpath (str): The path to the holdout OMR FPR stats.
        fpr_stats_raw_omr_fpath (str): The path to the raw OMR FPR stats.
        fprp_stats_holdout_omr_fpath (str): The path to the holdout OMR FPRP stats.
        data_stats_fpath (str): The path to the data stats file.
        ks_df_fpath (str): The path to the KS test results CSV.
        datasets_range_fpath (str): The path to the datasets range file.

    Returns:
        bool: True if the report was generated successfully, False otherwise.
    """
    
    # 1. Load Data
    try:
        fpr_stats_cleaned_omr = safe_load_numpy_yaml(fpr_stats_cleaned_omr_fpath)
        fpr_stats_holdout_omr = safe_load_numpy_yaml(fpr_stats_holdout_omr_fpath)
        fpr_stats_raw_omr = safe_load_numpy_yaml(fpr_stats_raw_omr_fpath)
        fprp_stats_holdout_omr = safe_load_numpy_yaml(fprp_stats_holdout_omr_fpath)
        data_stats = safe_load_numpy_yaml(data_stats_fpath)
        datasets_range = safe_load_numpy_yaml(datasets_range_fpath)
        ks_df = pd.read_csv(ks_df_fpath)
    except Exception as e:
        print(f"Error loading stats files: {e}")
        return False

    # 2. Prepare Images (Convert to Base64)
    images = {}
    
    # OMR Plots (Filenames match those generated by generate_report_plots in qa_plotting.py)
    img_map = {
        "cleaned_omr": f"{model_name} - Cleaned Validation OMR.jpg",
        "raw_omr": f"{model_name} - Raw Validation OMR.jpg",
        "holdout_omr": f"{model_name} - Holdout OMR.jpg",
        "fpr_rates": f"{model_name} - False Positive Rates (with and without persistence).jpg",
        "omr_dist": f"{model_name} - OMR Distributions.jpg",
    }
    
    for key, filename in img_map.items():
        images[key] = get_image_base64(os.path.join(fpr_file_path, filename))

    # KS Plots
    ks_images = []
    for filename in os.listdir(ks_file_path):
        if filename.startswith("distribution"):
            variable = filename.replace("distribution_comparison_", "").replace(".jpg", "").replace(".png", "")
            b64_img = get_image_base64(os.path.join(ks_file_path, filename))
            
            # Get stats for this var
            row = ks_df[ks_df["Variable"] == variable]
            if not row.empty:
                ks_stat = row.iloc[0]['KS Statistic']
                p_value = row.iloc[0]['P-value']
                consistent = p_value > 0.05
            else:
                ks_stat, p_value, consistent = "N/A", "N/A", False
                
            ks_images.append({
                "title": variable,
                "img": b64_img,
                "ks_stat": ks_stat,
                "p_value": p_value,
                "consistent": consistent
            })
    
    # KS Summary
    images["ks_summary"] = get_image_base64(os.path.join(ks_file_path, "ks_results.jpg"))

    # 3. HTML Template
    template_str = """
    <html>
    <head>
        <style>
            @page { size: A4 landscape; margin: 1cm; }
            body { font-family: "Helvetica", "Arial", sans-serif; color: #333; margin: 0; padding: 0; }
            
            /* Page Breaks */
            .page-break { page-break-after: always; }
            .title-page { display: flex; flex-direction: column; justify-content: center; align-items: center; height: 95vh; text-align: center; }
            
            /* Headings */
            h1 { color: #2c3e50; font-size: 32px; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; width: 100%; text-align: center; }
            h2 { color: #2980b9; font-size: 24px; margin-top: 20px; border-bottom: 1px solid #eee; }
            h3 { color: #555; font-size: 18px; margin-bottom: 10px; text-align: center; }
            
            /* Content Layout */
            .stats-container { width: 80%; margin: 20px auto; border: 1px solid #ddd; padding: 20px; border-radius: 8px; background-color: #f9f9f9; }
            .stat-row { margin-bottom: 10px; font-size: 14px; }
            .label { font-weight: bold; color: #555; width: 150px; display: inline-block; }
            
            /* Images */
            .img-container { text-align: center; margin: 20px 0; }
            img { max-width: 95%; max-height: 15cm; border: 1px solid #ccc; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
            
            /* Tables */
            table { width: 80%; margin: 20px auto; border-collapse: collapse; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; font-size: 14px; }
            th { background-color: #f2f2f2; font-weight: bold; }
            
            .meta-info { font-size: 12px; color: #888; text-align: center; margin-top: 50px; }
            
            .ks-info { font-size: 14px; margin: 10px auto; width: 80%; text-align: left; background: #fff; padding: 10px; border-left: 4px solid #2980b9; }
        </style>
    </head>
    <body>
        <!-- PAGE 1: COVER -->
        <div class="title-page page-break">
            <h1 style="border:none; font-size: 48px;">{{ model_name }}</h1>
            <h2 style="border:none; color: #7f8c8d;">Model QA Report</h2>
            
            <div class="stats-container" style="text-align: left;">
                <h3>Validation Dataset</h3>
                <div class="stat-row">{{ data_stats['validation'] | safe }}</div>
                
                <h3 style="margin-top: 30px;">Holdout Dataset</h3>
                <div class="stat-row">{{ data_stats['holdout'] | safe }}</div>
            </div>
            
            <div class="meta-info">Generated on: {{ generation_date }}</div>
        </div>

        <!-- PAGE 2: Performance Title -->
        <div class="title-page page-break">
            <h1 style="border:none; margin-top: 20%;">Model Performance Analysis</h1>
        </div>

        <!-- PAGE 3: Cleaned OMR -->
        <div class="page-break">
            <h3>Validation Set (without outlier): OMR vs Time</h3>
            <div class="img-container">
                {% if images['cleaned_omr'] %}<img src="data:image/jpeg;base64,{{ images['cleaned_omr'] }}">{% else %} <p>Image not found</p> {% endif %}
            </div>
            <div class="stats-container" style="text-align: center;">
                <strong>Min OMR:</strong> {{ ranges['cleaned_omr_range']['min'] }} &nbsp;&nbsp;|&nbsp;&nbsp; 
                <strong>Max OMR:</strong> {{ ranges['cleaned_omr_range']['max'] }}
            </div>
        </div>

        <!-- PAGE 4: Raw OMR -->
        <div class="page-break">
            <h3>Validation Set (with outlier): OMR vs Time</h3>
            <div class="img-container">
                {% if images['raw_omr'] %}<img src="data:image/jpeg;base64,{{ images['raw_omr'] }}">{% endif %}
            </div>
            <div class="stats-container" style="text-align: center;">
                <strong>Min OMR:</strong> {{ ranges['raw_omr_range']['min'] }} &nbsp;&nbsp;|&nbsp;&nbsp; 
                <strong>Max OMR:</strong> {{ ranges['raw_omr_range']['max'] }}
            </div>
        </div>

        <!-- PAGE 5: Holdout OMR -->
        <div class="page-break">
            <h3>Holdout Set (with persistence): OMR vs Time</h3>
            <div class="img-container">
                {% if images['holdout_omr'] %}<img src="data:image/jpeg;base64,{{ images['holdout_omr'] }}">{% endif %}
            </div>
            <div class="stats-container" style="text-align: center;">
                <strong>Min OMR:</strong> {{ ranges['holdout_omr_range']['min'] }} &nbsp;&nbsp;|&nbsp;&nbsp; 
                <strong>Max OMR:</strong> {{ ranges['holdout_omr_range']['max'] }}
            </div>
        </div>

        <!-- PAGE 6: FPR Rates -->
        <div class="page-break">
            <h3>False Positive Rates (with and without persistence)</h3>
            <div class="img-container">
                {% if images['fpr_rates'] %}<img src="data:image/jpeg;base64,{{ images['fpr_rates'] }}">{% endif %}
            </div>
            
            <table>
                <thead>
                    <tr><th>Metric</th><th>Warning (%)</th><th>Alert (%)</th></tr>
                </thead>
                <tbody>
                    <tr><td>Inwindow Validation (No Outlier)</td><td>{{ fpr_cleaned['warning']|round(2) }}</td><td>{{ fpr_cleaned['alert']|round(2) }}</td></tr>
                    <tr><td>Inwindow Validation (With Outlier)</td><td>{{ fpr_raw['warning']|round(2) }}</td><td>{{ fpr_raw['alert']|round(2) }}</td></tr>
                    <tr><td>Holdout Data</td><td>{{ fpr_holdout['warning']|round(2) }}</td><td>{{ fpr_holdout['alert']|round(2) }}</td></tr>
                    <tr><td>Holdout Data (Persistence)</td><td>{{ fprp_holdout['warning']|round(2) }}</td><td>{{ fprp_holdout['alert']|round(2) }}</td></tr>
                </tbody>
            </table>
        </div>

        <!-- PAGE 7: OMR Dist -->
        <div class="page-break">
            <h3>OMR Distributions</h3>
            <div class="img-container">
                {% if images['omr_dist'] %}<img src="data:image/jpeg;base64,{{ images['omr_dist'] }}">{% endif %}
            </div>
        </div>

        <!-- PAGE 8: KS Intro -->
        <div class="title-page page-break">
            <h1 style="border:none;">Consistency between Validation and Holdout Datasets</h1>
            <p style="width: 70%; font-size: 16px; line-height: 1.6; color: #555;">
                Comparing the consistency of validation and holdout datasets using the Kolmogorov-Smirnov (KS) statistic 
                is a critical step in model validation. It helps ensure that the model will perform reliably 
                on unseen data by verifying that the data used during development is representative of the 
                data it will encounter in practice.
            </p>
        </div>

        <!-- PAGE 9+: KS Plots -->
        {% for ks in ks_images %}
        <div class="page-break">
            <h3>Distribution Comparison of {{ ks.title }}</h3>
            <div class="img-container">
                <img src="data:image/jpeg;base64,{{ ks.img }}">
            </div>
            <div class="ks-info">
                <strong>KS Statistic:</strong> {{ ks.ks_stat }}<br>
                <strong>P-value:</strong> {{ ks.p_value }}<br><br>
                <em>Conclusion: They are {% if ks.consistent %}consistent{% else %}NOT consistent{% endif %} at 95% confidence level.</em>
            </div>
        </div>
        {% endfor %}

        <!-- PAGE LAST: KS Summary -->
        <div class="page-break">
            <h3>Summary of Dataset Consistency</h3>
            <div class="img-container">
                {% if images['ks_summary'] %}<img src="data:image/jpeg;base64,{{ images['ks_summary'] }}">{% endif %}
            </div>
        </div>

    </body>
    </html>
    """
    
    # 4. Render
    env = Environment()
    template = env.from_string(template_str)
    
    html_content = template.render(
        model_name=model_name,
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data_stats={k: generate_dataset_description(v) for k, v in data_stats.items()},
        ranges=datasets_range,
        images=images,
        fpr_cleaned=fpr_stats_cleaned_omr,
        fpr_raw=fpr_stats_raw_omr,
        fpr_holdout=fpr_stats_holdout_omr,
        fprp_holdout=fprp_stats_holdout_omr,
        ks_images=ks_images
    )
    
    # 5. Save PDF
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html_content)
            
            # Save final PDF
            final_pdf_path = os.path.join(report_file_path, f"model_qa_report_{model_name}.pdf")
            page.pdf(path=final_pdf_path, format="A4", landscape=True, margin={'top': '1cm', 'bottom': '1cm', 'left': '1cm', 'right': '1cm'})
            
            browser.close()
            return True
    except Exception as e:
        print(f"Playwright PDF Error: {e}")
        return False