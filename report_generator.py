import pandas as pd
from weasyprint import HTML, CSS
from jinja2 import Environment, FileSystemLoader
import base64
from datetime import datetime
import io
import zipfile

def generate_pdf_report(report_data, tdt_model_name, selected_submodules, report_type):
    """
    Generates a PDF report from a dictionary of pre-rendered HTML tables.
    Includes Table of Contents and Page Numbers.
    """
    env = Environment(loader=FileSystemLoader('.'))
    
    # HTML Template with enhanced CSS for PDF structure
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
                    font-size: 10pt;
                    font-family: sans-serif;
                    color: #555;
                }
            }

            /* --- General Styling --- */
            body { font-family: "Helvetica", "Arial", sans-serif; font-size: 10pt; color: #333; }
            h1 { color: #2c3e50; font-size: 24pt; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-bottom: 20px; }
            h2 { color: #2980b9; font-size: 18pt; margin-top: 30px; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            h3 { color: #16a085; font-size: 14pt; margin-top: 20px; margin-bottom: 10px; }
            h4 { color: #7f8c8d; font-size: 12pt; margin-top: 15px; margin-bottom: 5px; font-weight: bold; }
            p { margin-bottom: 10px; line-height: 1.5; }

            /* --- Page Breaks --- */
            .title-page { text-align: center; page-break-after: always; display: flex; flex-direction: column; justify-content: center; height: 80vh; }
            .toc-page { page-break-after: always; }
            .content-page { page-break-before: always; }

            /* --- Table of Contents --- */
            .toc ul { list-style-type: none; padding-left: 0; }
            .toc li { margin-bottom: 8px; border-bottom: 1px dotted #ccc; }
            .toc a { text-decoration: none; color: #333; display: block; padding: 5px 0; width: 100%; }
            .toc a::after { content: target-counter(attr(href), page); float: right; }

            /* --- Table Styling --- */
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
                font-size: 9pt;
                page-break-inside: auto;
            }
            tr { page-break-inside: avoid; page-break-after: auto; }
            th, td {
                border: 1px solid #bdc3c7;
                padding: 6px 8px;
                text-align: left;
                vertical-align: top;
                word-wrap: break-word;
            }
            thead { display: table-header-group; }
            th { background-color: #ecf0f1; color: #2c3e50; font-weight: bold; }
            
            /* Highlight styling handled by Pandas Styler, but ensuring compatibility */
            td { background-clip: padding-box; }

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

    return HTML(string=html_out, base_url='.').write_pdf()

def display_report_generation_tab(st, session_state, report_type, validation_filter_cols, submodule_options, highlight_function, axis=None):
    """
    Renders the report generation tab UI.
    
    Implements:
    1. TDT-Level Consolidation: Loops through selected TDTs and aggregates model data.
    2. Comprehensive Content: Iterates through all results tables (Summary, Matches, Mismatches).
    """
    st.header("Report Generation")
    st.markdown(f"""
    Generate comprehensive PDF reports for selected {report_type} configurations.
    
    **Features:**
    - **Per-TDT Reports:** Generates one PDF file per TDT, consolidating data for all associated models.
    - **Full Details:** Includes Summary, Matches, Mismatches, and All Entries tables.
    - **Navigable PDF:** Includes a clickable Table of Contents and page numbers.
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
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select TDTs")
        select_all_tdts = st.checkbox("Select All TDTs", key=f"{report_type}_select_all")
        selected_tdts = []
        if select_all_tdts:
            selected_tdts = available_tdts
            st.info(f"Selected {len(available_tdts)} TDTs")
        else:
            for tdt in available_tdts:
                if st.checkbox(tdt, key=f"cb_{report_type}_{tdt}"):
                    selected_tdts.append(tdt)

    with col2:
        st.subheader("Select Sections")
        selected_submodules = {}
        for name, key in submodule_options.items():
            results = session_state.validation_states[key].get("results")
            # Check if results exist and are not empty/None
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

    st.markdown("---")

    if st.button(f"Generate {report_type} Reports", disabled=not selected_tdts or not selected_submodules):
        progress_bar = st.progress(0, text="Initializing...")
        pdfs_to_zip = []

        total_tdts = len(selected_tdts)
        
        # --- Loop through TDTs (TDT-Level Consolidation) ---
        for i, tdt_name in enumerate(selected_tdts):
            progress_text = f"Processing {tdt_name}... ({i+1}/{total_tdts})"
            progress_bar.progress((i + 1) / total_tdts, text=progress_text)

            # Get all models associated with this TDT for filtering
            models_in_tdt = overview_df[overview_df['TDT'] == tdt_name]['Model'].unique()
            
            rendered_sections = {}

            # --- Loop through Submodules (Comprehensive Content) ---
            for submodule_name, submodule_key in selected_submodules.items():
                results = session_state.validation_states[submodule_key].get("results", {})
                filter_col = validation_filter_cols.get(submodule_key)
                
                if not results or not filter_col:
                    continue

                # Determine items to filter by (Consolidation Logic)
                # If the validator groups by TDT, we filter by TDT name.
                # If it groups by Model (e.g. Metric Mapping), we filter by ALL models in this TDT.
                items_to_filter = [tdt_name] if filter_col == 'TDT' else models_in_tdt

                filtered_results = {}
                
                # Iterate through ALL tables in results (Summary, Mismatches, etc.)
                for res_key, data in results.items():
                    # Handle DataFrame (e.g., 'summary', 'all_entries')
                    if isinstance(data, pd.DataFrame):
                        if filter_col in data.columns:
                            filtered_df = data[data[filter_col].isin(items_to_filter)].copy()
                            if not filtered_df.empty:
                                filtered_results[res_key] = filtered_df
                        else:
                            # Fallback: If filter col is missing (rare), include if it's generic
                            filtered_results[res_key] = data

                    # Handle Dictionary of DataFrames (e.g., 'mismatches')
                    elif isinstance(data, dict):
                        nested_filtered = {}
                        for sub_key, sub_df in data.items():
                            if isinstance(sub_df, pd.DataFrame) and filter_col in sub_df.columns:
                                f_sub_df = sub_df[sub_df[filter_col].isin(items_to_filter)].copy()
                                if not f_sub_df.empty:
                                    nested_filtered[sub_key] = f_sub_df
                        if nested_filtered:
                            filtered_results[res_key] = nested_filtered

                # Generate HTML for this section
                html_block = f"<h2>{submodule_name}</h2>"
                
                if not filtered_results:
                    html_block += "<p>No data found for this TDT/Model configuration.</p>"
                else:
                    # Order specific keys first if present for better readability
                    priority_keys = ['summary', 'matches', 'mismatches', 'all_entries']
                    keys_ordered = [k for k in priority_keys if k in filtered_results] + \
                                   [k for k in filtered_results if k not in priority_keys]

                    for table_name in keys_ordered:
                        table_data = filtered_results[table_name]
                        title = table_name.replace('_', ' ').title()
                        
                        if isinstance(table_data, pd.DataFrame):
                            html_block += f"<h3>{title}</h3>"
                            html_block += table_data.style.apply(highlight_function, axis=axis).to_html()
                        
                        elif isinstance(table_data, dict):
                            html_block += f"<h3>{title}</h3>"
                            for sub_title, sub_df in table_data.items():
                                sub_title_clean = sub_title.replace('_', ' ').title()
                                html_block += f"<h4>{sub_title_clean}</h4>"
                                html_block += sub_df.style.apply(highlight_function, axis=axis).to_html()

                rendered_sections[submodule_name] = html_block

            # Generate PDF for this TDT
            pdf_bytes = generate_pdf_report(
                report_data=rendered_sections,
                tdt_model_name=tdt_name,
                selected_submodules=list(selected_submodules.keys()),
                report_type=report_type
            )
            
            pdfs_to_zip.append({
                "name": f"{tdt_name}_{report_type}_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                "data": pdf_bytes
            })

        progress_bar.empty()

        # Download Logic
        if not pdfs_to_zip:
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