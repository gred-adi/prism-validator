# PRISM Configuration Validator
## 📖 Overview
The **PRISM Configuration Validator** is a comprehensive Streamlit web application designed to automate and streamline the process of validating Microsoft SQL Server-based PRISM configurations. It compares live database data against reference technical design templates (TDTs), providing detailed summaries and mismatch reports across various configuration sections.

The application features a modular architecture, allowing for easy extension and maintenance of different validation types.<br>

## ✨ Key Features
*   **Modular Validation Tabs:** Each validation type is organized into its own dedicated tab for a clean user experience.
Consolidation Overview
    *   Metric Validation (Template)
    *   Metric Mapping Validation (Project)
    *   Filter Validation (Project)
    *   Failure Diagnostics Validation (Template)
    *   Absolute Deviation Validation
*   **Dynamic File Generation:** Automatically consolidates multiple TDT Excel files from a selected local folder into the required reference files, eliminating manual preparation.
*   **Secure Database Connection:** Uses Streamlit's secrets management to handle database credentials securely.
*   **Dynamic SQL Queries:** Fetches model deployment configurations based on user-provided asset lists.
*   **Interactive Reports:** Presents validation results in clear, filterable tables, including summaries, matches, and categorized mismatches.
*   **Optional File Downloads:** Allows users to download the consolidated reference files for offline analysis.<br>

## 🚀 Getting Started
Follow these instructions to set up and run the PRISM Validator on your local machine.

### 1. Prerequisites
Before you begin, ensure you have the following installed:
  * **Python:** Version 3.9 or higher.
  * **uv:** The fast Python package installer. If you don't have it, install it with:
    ```
    pip install uv
    ```
  * **Git:** For cloning the repository.
  * **Microsoft ODBC Driver for SQL Server:** This is a critical system-level requirement. The application cannot connect to the database without it. Download and install it from the official Microsoft website.

### 2. Setup Instructions

**Step 1: Clone the Repository**
  
  Open your terminal or command prompt and clone the project to your local machine.
  
  ```
  git clone [https://github.com/your-username/prism-validator.git](https://github.com/your-username/prism-validator.git)
  cd prism-validator
  ```
      
**Step 2: Create the Secrets File**

The application requires a `secrets.toml` file to store your database credentials.
1. Create a new folder named `.streamlit` inside the `prism-validator` directory.
2. Inside the .streamlit folder, create a new file named secrets.toml.
3. Add your database credentials to this file using the following format:

    ```
    [db]
    host = "YOUR_DATABASE_HOST"
    database = "YOUR_DATABASE_NAME"
    user = "YOUR_USERNAME"
    password = "YOUR_PASSWORD"
    ```
        
**Step 3: Set Up Virtual Environment and Install Dependencies with** `uv`

Using uv is the recommended way to manage the project environment.
  1. Create a virtual environment:
      ```
      uv venv
      ```
  2. Activate the environment:
    
      *  On Windows (Command Prompt):
          ```
          .venv\Scripts\activate
          ```
      *  On macOS and Linux (Bash):
          ```
          source .venv/bin/activate
          ```
  3. Install all the project dependencies from the pyproject.toml file:
      ```
      uv sync
      ```

### 3. Running the Application
Once the setup is complete, you can run the Streamlit application with a single command:
```
streamlit run app.py
```
Your web browser will automatically open a new tab with the running application.<br><br>


# 💻 How to Use the App

1.   **Connect to Database:** Fill in your database credentials in the sidebar (they will be pre-filled from your secrets file) and click **"Connect to Database"**.
2.   **Generate Reference Files:**
      * Click the **"Select TDT Folder"** button to open your computer's file browser and choose the folder containing your TDT Excel files.
      * Once the folder is selected, click **"Generate & Load Files"**. The app will process the files and show a success message.
3.   **Upload Statistics File:** Click "Browse files" to upload the `Consolidated Statistics file`.
4.   **Navigate to a Validation Tab:** Go to any of the validation tabs (e.g., "Metric Mapping Validation"). The "Run Validation" button will be enabled.
5.   **Run Validation:** Click the **"Run..."** button to perform the comparison. The results, including a summary and detailed tables, will appear on the screen.
6.   **Filter Results:** Use the dropdown menus to filter the detailed match and mismatch tables by TDT or Model.<br><br>


# 📁 Project Structure
```
prism-validator/
│
├── .streamlit/
│   └── secrets.toml         # Stores database credentials (you must create this)
│
├── app.py                   # Main Streamlit application UI and orchestration
├── db_utils.py              # Database connection and query execution
├── file_generator.py        # Logic to consolidate TDT files
│
├── validations/             # Package containing all validation-specific modules
│   ├── __init__.py          # Makes 'validations' a Python package, allowing imports
│   │
│   ├── metric_validation/
│   │   ├── __init__.py      # Makes this sub-directory a package
│   │   ├── query.py         # Contains the specific SQL query for this validation
│   │   ├── parser.py        # Handles parsing the required Excel sheets for this validation
│   │   └── validator.py     # Contains the core data comparison and result generation logic
│   │
│   ├── metric_mapping_validation/
│   │   ├── __init__.py
│   │   ├── query.py
│   │   ├── parser.py
│   │   └── validator.py
│   │
│   ├── filter_validation/
│   │   ├── __init__.py
│   │   ├── query.py
│   │   ├── parser.py
│   │   └── validator.py
│   │
│   ├── failure_diagnostics_validation/
│   │   ├── __init__.py
│   │   ├── query.py
│   │   ├── parser.py
│   │   └── validator.py
│   │
│   ├── absolute_deviation_validation/
│   │   ├── __init__.py
│   │   ├── query.py
│   │   ├── parser.py
│   │   └── validator.py
│   │
│   └── model_deployment_config/
│       ├── __init__.py
│       └── query.py         # Fetches deployment info; no validation needed
│
├── pyproject.toml           # Project metadata and dependencies for uv/pip
└── README.md                # This file
```
