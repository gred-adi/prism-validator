import pandas as pd
from playwright.sync_api import sync_playwright
from jinja2 import Environment
import base64
from datetime import datetime
import io
import zipfile
import streamlit as st
import asyncio
import sys

# --- Configuration: Column Mappings for TDT Reports ---
# This ensures the PDF report matches the specific columns shown in the Web App.
TDT_VALIDATION_COLUMNS = {
    "tdt_point_survey": {
        "details": [
            'TDT', 'Model', 'Metric', 'Point Type', 'KKS Point Name', 
            'DCS Description', 'Canary Point Name', 'Canary Description', 'Unit'
        ]
    },
    "tdt_calculation": {
        "details": [
            'TDT', 'Metric', 'Point Type', 'Calc Point Type', 'Calculation Description', 
            'Pseudo Code', 'Language', 'Input Point', 'PRiSM Code'
        ]
    },
    "tdt_attribute": {
        "function_validation": {
            "details": [
                'TDT', 'Metric', 'Function', 'Constraint', 'Diag_Count', 
                'Filter Condition', 'Filter Value'
            ]
        },
        "filter_audit": {
            "details": [
                'TDT', 'Metric', 'Function', 'Filter Condition', 'Filter Value'
            ]
        }
    },
    "tdt_diagnostics": {
        "details": [
            'TDT', 'Failure Mode', 'Metric', 'Direction', 'Weighting'
        ]
    }
}

def generate_pdf_report(browser, report_data, tdt_model_name, selected_submodules, report_type, page_size="A3", orientation="landscape"):
    """Generates a PDF report from HTML content using Playwright.

    This function takes report data as HTML, injects it into a Jinja2
    template, and uses a running Playwright browser instance to render the
    HTML and convert it into a PDF byte stream. It includes a title page,
    a table of contents, and content pages for each submodule.

    Args:
        browser (playwright.sync_api.Browser):
            An active Playwright browser instance.
        report_data (dict):
            A dictionary where keys are submodule names and values are the
            HTML content blocks to be included in the report.
        tdt_model_name (str):
            The name of the TDT or Model for which the report is generated.
        selected_submodules (list):
            A list of submodule names to be included in the report and the
            table of contents.
        report_type (str):
            The type of the report (e.g., "TDT", "PRISM").
        page_size (str, optional):
            The page size for the PDF. Defaults to "A3".
        orientation (str, optional):
            The page orientation ('landscape' or 'portrait'). Defaults to "landscape".

    Returns:
        bytes: The generated PDF report as a byte stream, or None if an
               error occurred.
    """
    try:
        # Initialize Jinja2 environment without a specific loader to avoid path issues
        env = Environment()
        
        # HTML Template
        template = env.from_string("""
        <html>
        <head>
            <style>
                /* --- General Styling --- */
                body { font-family: "Helvetica", "Arial", sans-serif; font-size: 9pt; color: #333; margin: 0; }
                h1 { color: #2c3e50; font-size: 20pt; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-bottom: 20px; }
                h2 { color: #2980b9; font-size: 16pt; margin-top: 25px; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; break-after: avoid; page-break-after: avoid; }
                h3 { color: #16a085; font-size: 13pt; margin-top: 15px; margin-bottom: 8px; break-after: avoid; page-break-after: avoid; }
                h4 { color: #7f8c8d; font-size: 11pt; margin-top: 10px; margin-bottom: 5px; font-weight: bold; break-after: avoid; page-break-after: avoid; }
                p { margin-bottom: 10px; line-height: 1.4; }

                /* --- Page Breaks --- */
                .title-page { text-align: center; page-break-after: always; break-after: page; display: flex; flex-direction: column; justify-content: center; height: 80vh; }
                .toc-page { page-break-after: always; break-after: page; }
                .content-page { page-break-before: always; break-before: page; }

                /* --- Table of Contents --- */
                .toc ul { list-style-type: none; padding-left: 0; }
                .toc li { margin-bottom: 6px; border-bottom: 1px dotted #ccc; }
                .toc a { text-decoration: none; color: #333; display: block; padding: 5px 0; width: 100%; }

                /* --- Table Styling --- */
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 15px;
                    font-size: 8pt;
                    page-break-inside: auto;
                    table-layout: fixed; /* Ensures column widths are respected */
                }
                tr { page-break-inside: avoid; break-inside: avoid; }
                th, td {
                    border: 1px solid #bdc3c7;
                    padding: 4px 6px;
                    text-align: left;
                    vertical-align: top;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    word-break: break-word; 
                    white-space: normal;
                }
                thead { display: table-header-group; }
                th { background-color: #ecf0f1; color: #2c3e50; font-weight: bold; }
                td { background-clip: padding-box; }

                /* --- Compact Index Column Styling --- */
                /* Targets the first header and first cell of every row */
                table tr th:first-child,
                table tr td:first-child {
                    width: 35px;          /* Fixed compact width for index */
                    min-width: 35px;
                    max-width: 35px;
                    text-align: center;   /* Center the index number */
                    color: #666;          /* Slightly muted color */
                    background-color: #f9f9f9; /* Light background for index */
                    font-size: 7.5pt;
                }
            </style>
        </head>
        <body>
            <!-- 1. Title Page -->
            <div class="title-page">
                <h1 style="border: none; font-size: 32pt; margin-bottom: 10px;">{{ report_type }} Validation Report</h1>
                <h2 style="border: none; font-size: 24pt; color: #555;">{{ tdt_model_name }}</h2>
                <p style="margin-top: 50px; font-size: 12pt; color: #777;">Generated on: {{ generation_date }}</p>
            </div>

            <!-- 2. Table of Contents -->
            <div class="toc-page">
                <h1>Table of Contents</h1>
                <div class="toc">
                    <ul>
                        {% for submodule in selected_submodules %}
                            <li><a href="#{{ submodule | replace(' ', '-') }}">{{ submodule }}</a></li>
                        {% endfor %}
                    </ul>
                </div>
            </div>

            <!-- 3. Content Pages -->
            {% for submodule in selected_submodules %}
                <div class="content-page" id="{{ submodule | replace(' ', '-') }}">
                    {% if report_data[submodule] %}
                        {{ report_data[submodule] | safe }}
                    {% else %}
                        <h2>{{ submodule }}</h2>
                        <p>No data found for this section.</p>
                    {% endif %}
                </div>
            {% endfor %}
        </body>
        </html>
        """)

        generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_out = template.render(
            report_type=report_type,
            tdt_model_name=tdt_model_name,
            generation_date=generation_date,
            selected_submodules=selected_submodules,
            report_data=report_data
        )

        # Open a new page in the EXISTING browser instance
        page = browser.new_page()
        page.set_content(html_out)
        
        # Define Footer
        footer_html = """
            <div style="font-size: 10px; width: 100%; text-align: right; padding-right: 0.5in; padding-bottom: 0.2in; color: #555; font-family: sans-serif;">
                Page <span class="pageNumber"></span> of <span class="totalPages"></span>
            </div>
        """

        # Generate PDF
        pdf_bytes = page.pdf(
            format=page_size,
            landscape=(orientation.lower() == "landscape"),
            margin={'top': '0.5in', 'bottom': '0.5in', 'left': '0.5in', 'right': '0.5in'},
            print_background=True,
            display_header_footer=True,
            header_template='<div></div>', 
            footer_template=footer_html
        )
        
        page.close() # Close just this page, keep browser open
        return pdf_bytes

    except Exception as e:
        st.error(f"❌ Error generating PDF for {tdt_model_name}: {str(e)}")
        if "Executable doesn't exist" in str(e):
             st.warning("⚠️ **Action Required:** It looks like the browser engine is not installed. Please stop the app and run `playwright install chromium` in your terminal.")
        return None

def display_report_generation_tab(st, session_state, report_type, validation_filter_cols, submodule_options, highlight_function, axis=None):
    """Renders the report generation tab UI in the Streamlit app.

    This function creates the user interface for selecting TDTs, report
    sections, and PDF settings. It handles the logic for filtering data based
    on user selections, orchestrating the PDF generation process, and
    providing a download link for the final report(s) as a single PDF or a
    ZIP archive.

    Args:
        st (streamlit): The Streamlit module instance.
        session_state (streamlit.runtime.state.SessionState):
            The Streamlit session state object, used to access loaded data and
            validation results.
        report_type (str):
            The type of report being generated (e.g., "TDT", "PRISM").
        validation_filter_cols (dict):
            A dictionary mapping submodule keys to the column name used for
            filtering (e.g., 'TDT' or 'Model').
        submodule_options (dict):
            A dictionary of available report sections, where keys are the
            display names and values are the corresponding keys in the session
            state.
        highlight_function (function):
            The function used to apply conditional styling to the DataFrames
            before rendering them as HTML.
        axis (int or None, optional):
            The axis along which the highlight function is applied. Defaults to None.
    """
    st.header("Report Generation")
    st.markdown(f"""
    Generate comprehensive PDF reports for selected {report_type} configurations.
    """)

    overview_df = session_state.get('overview_df')
    if overview_df is None or overview_df.empty:
        st.warning("No TDT data loaded. Please load TDT files on the Home page.")
        return

    available_tdts = sorted(overview_df['TDT'].unique())

    if not available_tdts:
        st.warning("No TDTs found.")
        return

    # --- Selection UI ---
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col1:
        st.subheader("Select TDTs")
        select_all_tdts = st.checkbox("Select All TDTs", key=f"{report_type}_select_all")
        selected_tdts = []
        if select_all_tdts:
            selected_tdts = available_tdts
            st.info(f"Selected {len(available_tdts)} TDTs")
        else:
            with st.container(height=300):
                for tdt in available_tdts:
                    if st.checkbox(tdt, key=f"cb_{report_type}_{tdt}"):
                        selected_tdts.append(tdt)

    with col2:
        st.subheader("Select Sections")
        selected_submodules = {}
        with st.container(height=300):
            for name, key in submodule_options.items():
                results = session_state.validation_states[key].get("results")
                has_results = results is not None and (
                    (isinstance(results, dict) and any(v is not None for v in results.values())) or 
                    (isinstance(results, pd.DataFrame) and not results.empty)
                )
                
                is_enabled = st.toggle(
                    name,
                    value=has_results,
                    disabled=not has_results,
                    key=f"toggle_{report_type}_{key}"
                )
                if is_enabled:
                    selected_submodules[name] = key

    with col3:
        st.subheader("PDF Settings")
        page_size = st.selectbox(
            "Page Size", 
            ["A3", "A4", "Letter", "Legal"], 
            index=0,
            key=f"{report_type}_page_size"
        )
        orientation = st.selectbox(
            "Orientation",
            ["Landscape", "Portrait"],
            index=0,
            key=f"{report_type}_orientation"
        )
        
    st.markdown("---")

    if st.button(f"Generate {report_type} Reports", disabled=not selected_tdts or not selected_submodules):
        progress_bar = st.progress(0, text="Initializing...")
        pdfs_to_zip = []
        total_tdts = len(selected_tdts)
        
        # --- FIX: Enforce ProactorEventLoop for Windows ---
        # This fixes 'NotImplementedError' in asyncio when launching subprocesses (Playwright)
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        # --- PLAYWRIGHT LIFECYCLE START ---
        try:
            with sync_playwright() as p:
                # Launch the browser ONCE for the entire batch
                browser = p.chromium.launch(headless=True)
                
                for i, tdt_name in enumerate(selected_tdts):
                    progress_text = f"Processing {tdt_name}... ({i+1}/{total_tdts})"
                    progress_bar.progress((i + 1) / total_tdts, text=progress_text)

                    # Filter Data Logic
                    models_in_tdt = overview_df[overview_df['TDT'] == tdt_name]['Model'].unique()
                    rendered_sections = {}

                    # --- Loop through Submodules ---
                    for submodule_name, submodule_key in selected_submodules.items():
                        results = session_state.validation_states[submodule_key].get("results", {})
                        filter_col = validation_filter_cols.get(submodule_key)
                        
                        if not results or not filter_col:
                            continue

                        items_to_filter = [tdt_name] if filter_col == 'TDT' else models_in_tdt
                        filtered_results = {}
                        
                        # Data Filtering Logic
                        for res_key, data in results.items():
                            if isinstance(data, pd.DataFrame):
                                if filter_col in data.columns:
                                    filtered_df = data[data[filter_col].isin(items_to_filter)].copy()
                                    if not filtered_df.empty:
                                        filtered_results[res_key] = filtered_df
                                else:
                                    filtered_results[res_key] = data
                            elif isinstance(data, dict):
                                nested_filtered = {}
                                for sub_key, sub_df in data.items():
                                    if isinstance(sub_df, pd.DataFrame) and filter_col in sub_df.columns:
                                        f_sub_df = sub_df[sub_df[filter_col].isin(items_to_filter)].copy()
                                        if not f_sub_df.empty:
                                            nested_filtered[sub_key] = f_sub_df
                                if nested_filtered:
                                    filtered_results[res_key] = nested_filtered

                        # HTML Generation
                        html_block = f"<h2>{submodule_name}</h2>"
                        if not filtered_results:
                            html_block += "<p>No data found for this TDT/Model configuration.</p>"
                        else:
                            priority_keys = ['summary', 'matches', 'mismatches', 'all_entries']
                            keys_ordered = [k for k in priority_keys if k in filtered_results] + \
                                        [k for k in filtered_results if k not in priority_keys]

                            for table_name in keys_ordered:
                                table_data = filtered_results[table_name]
                                title = table_name.replace('_', ' ').title()
                                
                                if isinstance(table_data, pd.DataFrame):
                                    # --- COLUMNS FILTERING & STYLING LOGIC ---
                                    final_df = table_data
                                    hide_issue = False
                                    
                                    # 1. Check if there is a specific column mapping for this table
                                    cols_to_keep = None
                                    if submodule_key in TDT_VALIDATION_COLUMNS:
                                        mapping = TDT_VALIDATION_COLUMNS[submodule_key]
                                        if isinstance(mapping, dict) and table_name in mapping:
                                            cols_to_keep = mapping[table_name]
                                    
                                    # 2. Filter Columns if mapping exists
                                    if cols_to_keep:
                                        # Determine actual columns to keep (intersection with existing)
                                        cols_actual = [c for c in cols_to_keep if c in table_data.columns]
                                        
                                        # Ensure 'Issue' is kept for styling, but flag it for hiding
                                        if 'Issue' in table_data.columns:
                                            if 'Issue' not in cols_actual:
                                                cols_actual.append('Issue')
                                                hide_issue = True
                                        
                                        final_df = table_data[cols_actual]
                                    
                                    # 3. Apply Style
                                    styler = final_df.style.apply(highlight_function, axis=axis)
                                    
                                    # 4. Hide 'Issue' column if it was strictly for logic
                                    if hide_issue:
                                        styler.hide(['Issue'], axis="columns")

                                    html_block += f"<h3>{title}</h3>"
                                    html_block += styler.to_html()

                                elif isinstance(table_data, dict):
                                    html_block += f"<h3>{title}</h3>"
                                    for sub_title, sub_df in table_data.items():
                                        # --- COLUMNS FILTERING & STYLING LOGIC (Nested) ---
                                        final_df = sub_df
                                        hide_issue = False
                                        
                                        # 1. Check mapping for nested dict
                                        cols_to_keep = None
                                        if submodule_key in TDT_VALIDATION_COLUMNS:
                                            mapping = TDT_VALIDATION_COLUMNS[submodule_key]
                                            # Navigate: submodule -> table_name (e.g. function_validation) -> sub_title (e.g. details)
                                            if isinstance(mapping, dict) and table_name in mapping:
                                                nested_mapping = mapping[table_name]
                                                if isinstance(nested_mapping, dict) and sub_title in nested_mapping:
                                                    cols_to_keep = nested_mapping[sub_title]
                                        
                                        # 2. Filter Columns
                                        if cols_to_keep:
                                            cols_actual = [c for c in cols_to_keep if c in sub_df.columns]
                                            if 'Issue' in sub_df.columns:
                                                if 'Issue' not in cols_actual:
                                                    cols_actual.append('Issue')
                                                    hide_issue = True
                                            final_df = sub_df[cols_actual]

                                        # 3. Apply Style & Hide
                                        styler = final_df.style.apply(highlight_function, axis=axis)
                                        if hide_issue:
                                            styler.hide(['Issue'], axis="columns")

                                        sub_title_clean = sub_title.replace('_', ' ').title()
                                        html_block += f"<h4>{sub_title_clean}</h4>"
                                        html_block += styler.to_html()

                        rendered_sections[submodule_name] = html_block

                    # --- GENERATE PDF ---
                    # Pass the EXISTING browser instance
                    pdf_bytes = generate_pdf_report(
                        browser=browser,
                        report_data=rendered_sections,
                        tdt_model_name=tdt_name,
                        selected_submodules=list(selected_submodules.keys()),
                        report_type=report_type,
                        page_size=page_size,
                        orientation=orientation.lower()
                    )
                    
                    if pdf_bytes:
                        pdfs_to_zip.append({
                            "name": f"{tdt_name}_{report_type}_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            "data": pdf_bytes
                        })
                
                # Browser automatically closes when exiting the 'with' block
                browser.close() 

        except Exception as e:
            st.error(f"Critical Playwright Error: {e}")
            if "Executable doesn't exist" in str(e):
                st.warning("⚠️ **Browser not found.** Please run `playwright install` in your terminal.")

        progress_bar.empty()
        # --- PLAYWRIGHT LIFECYCLE END ---

        # Download Logic
        if not pdfs_to_zip:
            if not st.session_state.get('error_shown', False):
                st.warning("No reports generated.")
        elif len(pdfs_to_zip) == 1:
            st.download_button(
                label="📥 Download PDF Report",
                data=pdfs_to_zip[0]["data"],
                file_name=pdfs_to_zip[0]["name"],
                mime="application/pdf"
            )
        else:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for pdf in pdfs_to_zip:
                    zf.writestr(pdf["name"], pdf["data"])
            
            st.download_button(
                label=f"📥 Download All Reports (.zip)",
                data=zip_buffer.getvalue(),
                file_name=f"{report_type}_Reports_{datetime.now().strftime('%Y%m%d')}.zip",
                mime="application/zip"
            )