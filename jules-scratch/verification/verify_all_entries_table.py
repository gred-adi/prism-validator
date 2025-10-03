from playwright.sync_api import sync_playwright, expect
import time

def run_verification(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    try:
        page.goto("http://localhost:8501")
        time.sleep(5) # Add a hard wait for the app to initialize

        # Wait for the main title to ensure the page is loaded
        expect(page.get_by_text("PRISM Configuration Validator")).to_be_visible(timeout=10000)

        # 1. Connect to Database
        # Use user-facing labels for locators
        page.get_by_label("Host").fill("localhost")
        page.get_by_label("Database").fill("prism")
        page.get_by_label("User").fill("user")
        page.get_by_label("Password").fill("password")
        page.get_by_role("button", name="Connect to Database").click()
        expect(page.get_by_text("✅ Connection successful!")).to_be_visible()

        # 2. Generate Files from TDT Folder - We will mock this by uploading files directly
        # For this test, we will focus on Absolute Deviation, which needs a stats file.

        # 3. Upload Statistics File
        stats_file_path = "validations/absolute_deviation_validation/test_data/Consolidated_Statistics_for_Abs_Dev.xlsx"
        page.locator('input[type="file"]').nth(0).set_input_files(stats_file_path)
        time.sleep(2) # Allow time for upload

        # 4. Navigate to Absolute Deviation Validation tab
        page.get_by_role("tab", name="Absolute Deviation Validation").click()

        # 5. Run Validation
        page.get_by_role("button", name="Run Absolute Deviation Validation").click()

        # Wait for summary to appear to ensure validation is complete
        expect(page.get_by_text("Validation Summary")).to_be_visible(timeout=10000)

        # 6. Filter by Model
        page.get_by_role("combobox").first.select_option("MODEL_A")

        # 7. Take a screenshot of the "All Entries" table
        # Wait for the table to be visible
        expect(page.get_by_text("All Entries (Filtered)")).to_be_visible(timeout=10000)

        # Take screenshot
        page.screenshot(path="jules-scratch/verification/all_entries_table_verification.png")

    finally:
        browser.close()

with sync_playwright() as playwright:
    run_verification(playwright)