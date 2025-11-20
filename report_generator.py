"""
This module provides functionality to generate PDF reports from styled Pandas DataFrames.
"""

import pandas as pd
from weasyprint import HTML
from jinja2 import Environment, FileSystemLoader
import base64
from datetime import datetime

def generate_pdf_report(report_data, tdt_model_name, selected_submodules, report_type):
    """
    Generates a PDF report from a dictionary of pre-rendered HTML tables.

    Args:
        report_data (dict): A dictionary where keys are submodule names and values are HTML strings of tables.
        tdt_model_name (str): The name of the TDT or Model for the report title.
        selected_submodules (list): A list of submodule names to include in the report.
        report_type (str): The type of report ('TDT' or 'PRISM').

    Returns:
        bytes: The generated PDF as a byte string.
    """
    env = Environment(loader=FileSystemLoader('.'))
    template = env.from_string("""
    <html>
    <head>
        <style>
            /* --- Page Layout and Numbering --- */
            @page {
                size: letter landscape;
                margin: 0.75in;
                @bottom-right {
                    content: "Page " counter(page) " of " counter(pages);
                }
            }

            /* --- General Styling --- */
            body { font-family: sans-serif; }
            h1, h2, h3, h4, h5 { color: #333; }
            a { text-decoration: none; color: inherit; }

            /* --- Page Structure --- */
            .title-page { text-align: center; page-break-after: always; }
            .toc-page { page-break-after: always; }
            .content-page { page-break-before: always; }

            /* --- Table of Contents --- */
            .toc ul { list-style-type: none; padding-left: 0; }
            .toc li { margin-bottom: 10px; }
            .toc a { display: block; }
            .toc .leader {
                display: inline-block;
                width: 100%;
                border-bottom: 1px dotted #ccc;
            }

            /* --- Table Styling --- */
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
                page-break-inside: auto;
            }
            tr { page-break-inside: avoid; page-break-after: auto; }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                word-wrap: break-word; /* Handle long text */
            }
            thead { display: table-header-group; } /* Repeat headers on new pages */
            th { background-color: #f2f2f2; }

        </style>
    </head>
    <body>
        <!-- 1. Title Page -->
        <div class="title-page">
            <h1>{{ report_type }} Validation Report</h1>
            <h2>{{ tdt_model_name }}</h2>
            <p>Generated on: {{ generation_date }}</p>
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
                    <p>No data available for this section.</p>
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

    return HTML(string=html_out, base_url='.').write_pdf()

def display_report_generation_tab(st, session_state, report_type, validation_filter_cols, submodule_options, highlight_function, axis=None):
    """
    Renders the report generation tab UI in a Streamlit app.

    Args:
        st: The Streamlit module.
        session_state: The Streamlit session state object.
        report_type (str): The type of report ('TDT' or 'PRISM').
        validation_filter_cols (dict): Mapping of validation keys to filter column names.
        submodule_options (dict): Mapping of human-readable submodule names to validation keys.
        highlight_function (function): The function to use for styling DataFrames.
        axis (int or None): The axis to apply the highlight function on.
    """
    st.header("Report Generation")
    st.markdown("""
    Select the TDTs and validation sections you wish to include in the PDF reports.
    - A separate PDF will be generated for each selected TDT.
    - If multiple reports are generated, they will be downloaded as a single `.zip` file.
    """)

    # --- NEW: TDT-based item selection ---
    overview_df = session_state.get('overview_df')
    if overview_df is None or overview_df.empty:
        st.warning("No TDT data is loaded. Please go to the Home page and load TDT files.")
        return

    available_tdts = sorted(overview_df['TDT'].unique())

    if not available_tdts:
        st.warning("No TDTs found in the loaded data.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Select TDTs")
        select_all_tdts = st.checkbox("Select All TDTs", key=f"{report_type}_select_all_tdts")
        selected_tdts = []
        if select_all_tdts:
            selected_tdts = available_tdts
            for tdt in available_tdts:
                st.checkbox(tdt, value=True, key=f"cb_{report_type}_{tdt}")
        else:
            for tdt in available_tdts:
                if st.checkbox(tdt, key=f"cb_{report_type}_{tdt}"):
                    selected_tdts.append(tdt)

    with col2:
        st.subheader("Select Report Sections")
        selected_submodules = {}
        for name, key in submodule_options.items():
            has_results = session_state.validation_states[key].get("results") is not None
            is_enabled = st.toggle(
                name,
                value=has_results,
                disabled=not has_results,
                key=f"toggle_{report_type}_{key}",
                help="Run the validation on its tab to enable this section." if not has_results else ""
            )
            if is_enabled:
                selected_submodules[name] = key

    st.markdown("---")

    if st.button(f"Generate & Download {report_type} Reports", disabled=not selected_tdts or not selected_submodules):
        progress_bar = st.progress(0, text=f"Initializing {report_type} report generation...")
        pdfs_to_zip = []

        total_tdts = len(selected_tdts)
        for i, tdt_name in enumerate(selected_tdts):
            progress_text = f"Generating report for {tdt_name}... ({i+1}/{total_tdts})"
            progress_bar.progress((i + 1) / total_tdts, text=progress_text)

            # --- NEW: Get models for the current TDT ---
            models_in_tdt = overview_df[overview_df['TDT'] == tdt_name]['Model'].unique()

            rendered_sections = {}
            for submodule_name, submodule_key in selected_submodules.items():
                results = session_state.validation_states[submodule_key].get("results", {})
                filter_col = validation_filter_cols.get(submodule_key)
                if not results or not filter_col: continue

                # --- NEW: Determine items to filter by ---
                # If the filter column is 'TDT', we just use the tdt_name.
                # If it's 'MODEL', we use the list of models for this TDT.
                items_to_filter = [tdt_name] if filter_col == 'TDT' else models_in_tdt

                filtered_results = {}
                for res_key, data in results.items():
                    # Case 1: The item is a DataFrame
                    if isinstance(data, pd.DataFrame) and filter_col in data.columns:
                        # Use .isin() to filter by one or more items
                        filtered_df = data[data[filter_col].isin(items_to_filter)].copy()
                        if not filtered_df.empty:
                            filtered_results[res_key] = filtered_df
                    # Case 2: The item is a dictionary of DataFrames (like 'mismatches')
                    elif isinstance(data, dict):
                        nested_filtered_dfs = {}
                        for sub_key, sub_df in data.items():
                            if isinstance(sub_df, pd.DataFrame) and filter_col in sub_df.columns:
                                # Use .isin() here as well
                                filtered_sub_df = sub_df[sub_df[filter_col].isin(items_to_filter)].copy()
                                if not filtered_sub_df.empty:
                                    nested_filtered_dfs[sub_key] = filtered_sub_df
                        if nested_filtered_dfs:
                            filtered_results[res_key] = nested_filtered_dfs

                # --- NEW: Dynamic Table Generation ---
                html_block = f"<h2>{submodule_name}</h2>"
                if not filtered_results:
                    html_block += "<p>No data found for this section.</p>"
                else:
                    # Iterate through all tables in the filtered results
                    for table_name, table_data in filtered_results.items():
                        title = table_name.replace('_', ' ').title()

                        # Handle nested dictionaries of DataFrames (like mismatches)
                        if isinstance(table_data, dict):
                            html_block += f"<h3>{title}</h3>"
                            has_content = False
                            for sub_title, sub_df in table_data.items():
                                if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                                    has_content = True
                                    sub_title_clean = sub_title.replace('_', ' ').title()
                                    html_block += f"<h4>{sub_title_clean}</h4>"
                                    html_block += sub_df.style.apply(highlight_function, axis=axis).to_html()
                            if not has_content:
                                 html_block = html_block.rstrip(f"<h3>{title}</h3>")

                        # Handle regular DataFrames
                        elif isinstance(table_data, pd.DataFrame) and not table_data.empty:
                            html_block += f"<h3>{title}</h3>"
                            html_block += table_data.style.apply(highlight_function, axis=axis).to_html()

                rendered_sections[submodule_name] = html_block

            pdf_bytes = generate_pdf_report(
                report_data=rendered_sections,
                tdt_model_name=tdt_name, # Pass TDT name to the report generator
                selected_submodules=list(selected_submodules.keys()),
                report_type=report_type
            )
            pdfs_to_zip.append({
                "name": f"{tdt_name}_{report_type}_Validation_Report_{datetime.now().strftime('%Y-%m-%d')}.pdf",
                "data": pdf_bytes
            })

        progress_bar.empty()

        if not pdfs_to_zip:
            st.warning("No data was found for the selected TDTs and sections. No reports were generated.")
        elif len(pdfs_to_zip) == 1:
            st.download_button(
                label=f"✅ Download {report_type} PDF Report",
                data=pdfs_to_zip[0]["data"],
                file_name=pdfs_to_zip[0]["name"],
                mime="application/pdf",
                key=f"{report_type}_report_download_single"
            )
        elif len(pdfs_to_zip) > 1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for pdf in pdfs_to_zip:
                    zf.writestr(pdf["name"], pdf["data"])

            st.download_button(
                label=f"✅ Download {report_type} Reports as .zip ({len(pdfs_to_zip)} files)",
                data=zip_buffer.getvalue(),
                file_name=f"{report_type}_Validation_Reports_{datetime.now().strftime('%Y-%m-%d')}.zip",
                mime="application/zip",
                key=f"{report_type}_report_download_zip"
            )
