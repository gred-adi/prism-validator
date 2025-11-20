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
                @page { size: letter landscape; margin: 1in; }
                body { font-family: sans-serif; }
                h1, h2, h3, h4, h5 { color: #333; }
                .title-page { text-align: center; }
                .page-break { page-break-before: always; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="title-page">
                <h1>{{ report_type }} Validation Report</h1>
                <h2>{{ tdt_model_name }}</h2>
                <p>Generated on: {{ generation_date }}</p>
            </div>

            {% for submodule in selected_submodules %}
                <div class="page-break"></div>
                <h2>{{ submodule }}</h2>
                {% if report_data[submodule] %}
                    {{ report_data[submodule] | safe }}
                {% else %}
                    <p>No data available for this section.</p>
                {% endif %}
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
    Select the items and validation sections you wish to include in the PDF reports.
    - A separate PDF will be generated for each selected item.
    - If multiple reports are generated, they will be downloaded as a single `.zip` file.
    """)

    available_items = set()
    for key, filter_col in validation_filter_cols.items():
        results = session_state.validation_states[key].get("results")
        if results and not results.get('summary', pd.DataFrame()).empty:
            summary_df = results['summary']
            if filter_col in summary_df.columns:
                available_items.update(summary_df[filter_col].unique())

    sorted_items = sorted(list(available_items))

    if not sorted_items:
        st.warning("No validation results are available. Please run at least one validation to generate a report.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(f"Select {report_type}s")
        select_all = st.checkbox(f"Select All {report_type}s", key=f"{report_type}_select_all")
        selected_items = []
        if select_all:
            selected_items = sorted_items
            for item in sorted_items:
                st.checkbox(item, value=True, key=f"cb_{report_type}_{item}")
        else:
            for item in sorted_items:
                if st.checkbox(item, key=f"cb_{report_type}_{item}"):
                    selected_items.append(item)

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

    if st.button(f"Generate & Download {report_type} Reports", disabled=not selected_items or not selected_submodules):
        progress_bar = st.progress(0, text=f"Initializing {report_type} report generation...")
        pdfs_to_zip = []

        total_items = len(selected_items)
        for i, item_name in enumerate(selected_items):
            progress_text = f"Generating report for {item_name}... ({i+1}/{total_items})"
            progress_bar.progress((i + 1) / total_items, text=progress_text)

            rendered_sections = {}
            for submodule_name, submodule_key in selected_submodules.items():
                results = session_state.validation_states[submodule_key].get("results", {})
                filter_col = validation_filter_cols.get(submodule_key)
                if not results or not filter_col: continue

                filtered_results = {}
                for res_key, df in results.items():
                    if isinstance(df, pd.DataFrame) and filter_col in df.columns:
                        filtered_results[res_key] = df[df[filter_col] == item_name].copy()

                html_block = f"<h3>{submodule_name}</h3>"
                summary_df = filtered_results.get('summary')

                if summary_df is None or summary_df.empty:
                    html_block += "<p>No data found for this section.</p>"
                else:
                    html_block += "<h4>Summary</h4>" + summary_df.style.apply(highlight_function, axis=axis).to_html()

                    mismatches = filtered_results.get('mismatches', {})
                    if mismatches:
                        html_block += "<h4>Mismatches</h4>"
                        if isinstance(mismatches, dict):
                            for m_type, m_df in mismatches.items():
                                if not m_df.empty:
                                    html_block += f"<h5>{m_type.replace('_', ' ').title()}</h5>" + m_df.style.apply(highlight_function, axis=axis).to_html()
                        elif isinstance(mismatches, pd.DataFrame) and not mismatches.empty:
                            html_block += mismatches.style.apply(highlight_function, axis=axis).to_html()

                    matches_df = filtered_results.get('matches')
                    if matches_df is not None and not matches_df.empty:
                        html_block += "<h4>Matches</h4>" + matches_df.to_html(index=False)

                    all_entries_df = filtered_results.get('all_entries')
                    if all_entries_df is not None and not all_entries_df.empty:
                         html_block += "<h4>All Entries</h4>" + all_entries_df.style.apply(highlight_function, axis=axis).to_html()

                rendered_sections[submodule_name] = html_block

            pdf_bytes = generate_pdf_report(
                report_data=rendered_sections,
                tdt_model_name=item_name,
                selected_submodules=list(selected_submodules.keys()),
                report_type=report_type
            )
            pdfs_to_zip.append({
                "name": f"{item_name}_{report_type}_Validation_Report_{datetime.now().strftime('%Y-%m-%d')}.pdf",
                "data": pdf_bytes
            })

        progress_bar.empty()

        if len(pdfs_to_zip) == 1:
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
