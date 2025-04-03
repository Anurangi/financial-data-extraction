import re
import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def setup_driver(headless=True):
    """Set up and return a Chrome WebDriver with optional headless mode."""
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def sanitize_filename(filename):
    """Sanitize the filename by replacing invalid characters and spaces."""
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)  # Remove invalid characters
    filename = re.sub(r'\s+', '_', filename)  # Replace spaces with underscores
    return filename

def get_pdf_links(symbol):
    """Get PDF links from the quarterly reports page of the company profile."""
    url = f"https://www.cse.lk/pages/company-profile/company-profile.component.html?symbol={symbol}"
    driver = setup_driver()
    driver.get(url)
    
    try:
        financials_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Financials')]"))
        )
        financials_tab.click()
        WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'Quarterly Reports')]"))
        )
        
        quarterly_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Quarterly Reports')]"))
        )
        quarterly_tab.click()
        WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "tr"))
        )
    except Exception as e:
        logger.error(f"Error accessing Quarterly Reports tab: {e}")
        driver.quit()
        return []
    
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    
    pdf_links = []
    rows = soup.find_all('tr')
    
    for row in rows:
        if len(pdf_links) >= 20:
            break  # Stop collecting after 20 quarters
        
        try:
            date_cell = row.find('td')
            if not date_cell:
                continue
            title_cell = date_cell.find_next('td').find('div', class_='col-lg-10 col-md-10')
            
            if title_cell:
                title = title_cell.get_text(strip=True)
                if "Interim Financial Statement" in title or "Quarterly" in title:
                    pdf_link = row.find('a', href=True)
                    if pdf_link and pdf_link['href'].endswith(".pdf"):
                        pdf_url = pdf_link['href']
                        pdf_links.append((date_cell.get_text(strip=True), title, pdf_url if pdf_url.startswith("http") else f"https://www.cse.lk{pdf_url}"))
        except Exception as e:
            logger.error(f"Error processing row: {e}")
    
    return pdf_links

def download_pdfs(pdf_links, company_name, save_dir="reports"):
    """Download the PDFs and save them to a local directory."""
    save_path = os.path.join(save_dir, sanitize_filename(company_name))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    session = requests.Session()  # Use a session for multiple requests
    for uploaded_date, title, pdf_url in pdf_links:
        sanitized_title = sanitize_filename(title)
        sanitized_date = sanitize_filename(uploaded_date)
        pdf_name = f"{sanitized_date} - {sanitized_title}.pdf"
        pdf_path = os.path.join(save_path, pdf_name)
        
        try:
            response = session.get(pdf_url)
            response.raise_for_status()
            
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded: {pdf_name}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {pdf_url}: {e}")
        except Exception as e:
            logger.error(f"Error saving file {pdf_name}: {e}")

def main():
    """Main function to drive the PDF extraction and downloading process."""
    symbols = {
        "REXP.N0000": "REXP",  # Map symbols to company names
        "DIPD.N0000": "DIPD"
    }
    
    for symbol, company_name in symbols.items():
        logger.info(f"Fetching quarterly reports for {company_name}...")
        pdf_links = get_pdf_links(symbol)
        if pdf_links:
            logger.info(f"Found {len(pdf_links)} PDF reports for {company_name}.")
            download_pdfs(pdf_links, company_name)
        else:
            logger.info(f"No PDF reports found for {company_name}.")

if __name__ == "__main__":
    main()
