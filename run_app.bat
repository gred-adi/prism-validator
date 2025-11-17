@echo off
echo Changing directory to the app location...
:: The /d switch is important to change drives if necessary (e.g., from C: to G:)
cd /d "C:\Users\Gredoble\Documents\prism_validator\prism-validator"

echo Activating virtual environment...
:: Use 'call' to run the activate script and then return to this script
call ".venv\Scripts\activate.bat"

echo Starting Streamlit app (app.py)...
:: Now that the venv is active, run streamlit
streamlit run app.py

echo Streamlit server has stopped.
:: Pause the window so you can see any error messages
pause