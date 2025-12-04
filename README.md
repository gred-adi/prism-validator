# PRISM Dev/QA Web Toolkit

## 📖 Overview
The **PRISM Dev/QA Web Toolkit** is a comprehensive Streamlit application designed to centralize and streamline the workflows for validating Technical Design Templates (TDTs), auditing live PRISM configurations, and preparing data for model development.

The application features a modular, multi-page architecture that guides users through validation, data extraction, and data preparation tasks.

## ✨ Key Features
*   **TDT and PRISM Validation:** Perform offline integrity checks on TDT files and validate them against live PRISM database configurations.
*   **Data Extraction:** Fetch historical time-series data from Canary Historian based on TDT tag definitions.
*   **Data Preparation Wizard:** A step-by-step wizard for data cleansing, holdout splitting, outlier removal, and train-validation splitting.
*   **Secure Database Connection:** Uses Streamlit's secrets management to handle database credentials securely.
*   **Dynamic Reporting:** Generate detailed PDF reports for validation results and data preparation steps.
*   **Interactive UI:** An intuitive and interactive user interface for configuring tasks and reviewing results.

## 🚀 Getting Started
Follow these instructions to set up and run the PRISM Dev/QA Web Toolkit on your local machine.

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
```
git clone https://github.com/gred-adi/prism-validator.git
cd prism-validator
```

**Step 2: Initialize the Development Environment**
  
  Prerequisites: You'll need uv installed before you can initialize the development environment.
  1. Open a terminal in the `prism-validator` directory.
  2. Run the following command to initialize the development environment:
     ```
     uv init
     ```
     This command will set up the necessary files like `pyproject.toml` and `python-environment`.

**Step 3: Configure** `pyproject.toml`
  
  The pyproject.toml file defines the dependencies and configurations for your Python project. You can edit it to specify the versions and libraries you need.
  1. Open the `pyproject.toml` file created during the `uv init` process.
  2. Replace its entire content with the following:
      ```
      [project]
      name = "prism-validator"
      version = "1.0.0"
      description = "A comprehensive Streamlit application designed to centralize and streamline the workflows for validating Technical Design Templates (TDTs), auditing live PRISM configurations, and preparing data for model development."
      readme = "README.md"
      requires-python = ">=3.9"
      dependencies = [
          "click",
          "dataframe-image",
          "fpdf2",
          "ipykernel",
          "jinja2",
          "lxml",
          "matplotlib",
          "numpy",
          "openpyxl",
          "pandas",
          "playwright",
          "plotly",
          "pyodbc",
          "requests",
          "scikit-learn",
          "scipy",
          "seaborn",
          "streamlit",
          "verstack",
          "xlsxwriter"
      ]
      ```
      
**Step 4: Set Up Virtual Environment and Install Dependencies**
  1. Install all the project dependencies from the `pyproject.toml` file by running the following command:
      ```
      uv sync
      ```
      This will install all the dependencies listed in the `pyproject.toml file` and set up a virtual environment (`.venv`), ensuring your project’s dependencies are isolated from your system Python.
      
**Step 5: Create the Secrets File**

The application requires a `secrets.toml` file to store your database credentials.
1. Create a new folder named `.streamlit` inside the `prism-validator` directory.
2. Inside the `.streamlit` folder, create a new file named `secrets.toml`.
3. Add your database credentials to this file using the following format:

    ```
    [db]
    host = "PRISM_DATABASE_HOST"
    database = "PRISM_DATABASE_NAME"
    user = "PRISM_USERNAME"
    password = "PRISM_PASSWORD"

    [api]
    token = "CANARY_API_TOKEN"
    ```

### 3. Running the Application
Instead of running terminal commands manually every time, you can configure the provided `run_app.bat` script for one-click launching.
  1. **Locate the Script:** Find the `run_app.bat` file in the root directory of the project.
  2. **Edit the File:** Right-click `run_app.bat` and select Edit (or open it with a text editor like Notepad).
  3. **Configure Your Path:** Look for the line:
     ```
     cd /d "[Your App Location]"
     ```
     Replace `[Your App Location]` with the full path to your `prism-validator` folder. _Example:_ `cd /d "C:\Users\YourName\Documents\prism-validator"`
  4. **Save and Close.**

     **To run the app:** Simply double-click the `run_app.bat file`. This will automatically activate the virtual environment and launch the Streamlit server in your default browser.

## 💻 How to Use the App

1.  **Global Settings:** In the sidebar on the left, upload your TDT Excel files and click **"Generate & Load Files"**. This step is required for most modules.
2.  **Navigate to a Module:** Select a module from the navigation on the left to begin a task.
3.  **Follow Instructions:** Each module provides a "How to use" section with step-by-step instructions.
4.  **Connect to Database:** If a module requires a database connection, use the sidebar to enter your credentials and connect.

# 📁 Project Structure
The repository is organized into a modular structure to separate concerns and improve maintainability.

```
prism-validator/
│
├── .streamlit/
│   └── secrets.toml         # Stores database credentials (you must create this)
│
├── app.py                   # Main Streamlit application UI and orchestration
├── db_utils.py              # Database connection and query execution
├── file_generator.py        # Logic to consolidate TDT files
├── report_generator.py      # Logic for generating PDF reports
├── style_utils.py           # DataFrame styling utilities
│
├── pages/                   # Each file represents a page in the Streamlit app
│   ├── 1_PRISM_Config_Validator.py
│   ├── 2_Canary_Historian_Downloader.py
│   └── ... (other pages)
│
├── utils/                   # Utility functions for data processing, plotting, etc.
│   ├── model_dev_utils.py
│   └── ... (other utils)
│
├── validations/             # Package containing all validation-specific modules
│   ├── __init__.py          # Makes 'validations' a Python package
│   │
│   ├── prism_validations/     # Validations that compare TDTs to the PRISM database
│   │   ├── metric_validation/
│   │   │   ├── query.py         # SQL query for this validation
│   │   │   ├── parser.py        # Excel parsing logic
│   │   │   └── validator.py     # Data comparison and result generation logic
│   │   └── ... (other validation modules)
│   │
│   └── tdt_validations/       # Validations that check the internal consistency of TDT files
│       ├── attribute_validation/
│       │   └── validator.py     # Logic to validate attributes within a TDT
│       └── ... (other validation modules)
│
├── pyproject.toml           # Project metadata and dependencies for uv/pip
└── README.md                # This file
```

## 🤝 Contributing
Contributions are welcome! If you have a feature request, bug report, or want to contribute to the code, please follow these steps:

1.  **Open an Issue:** For any significant changes, please open an issue first to discuss what you would like to change.
2.  **Fork the Repository:** Create a fork of the repository to your own GitHub account.
3.  **Create a Branch:** Create a new branch for your changes (`git checkout -b feature/your-feature-name`).
4.  **Make Your Changes:** Implement your changes and ensure the code is well-documented.
5.  **Submit a Pull Request:** Open a pull request from your branch to the `main` branch of the original repository.

We appreciate your contributions to make this tool better for everyone.
