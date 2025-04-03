# financial_data_pipeline.py
"""
Scrapes the web, get the necessary pdfs and then extract text in the relevent pages.
"""
import re
import os
import time
import argparse
import logging
import requests
import fitz  # PyMuPDF
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

class FinancialDataPipeline:
    """
    A pipeline for extracting financial data from the Colombo Stock Exchange (CSE).
    This pipeline:
    1. Scrapes quarterly report PDFs from the CSE website
    2. Downloads the PDFs to a local directory
    3. Extracts income statement text from the PDFs
    4. Validates and saves the extracted text
    """
    
    def __init__(self, output_dir="reports", headless=True, log_file="pipeline.log"):
        """
        Initialize the pipeline with configurable parameters.
        
        Args:
            output_dir (str): Directory to save reports
            headless (bool): Whether to run browser in headless mode
            log_file (str): Path to the log file
        """
        self.output_dir = output_dir
        self.headless = headless
        
        # Set up logging
        self.setup_logging(log_file)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory: {output_dir}")
    
    def setup_logging(self, log_file):
        """Set up logging configuration."""
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            
        # Update log file path to be inside logs directory
        log_file_path = os.path.join(logs_dir, log_file)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()
            ]
        )
    
    def setup_driver(self):
        """Set up and return a Chrome WebDriver."""
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    
    def sanitize_filename(self, filename):
        """Sanitize the filename by replacing invalid characters and spaces."""
        filename = re.sub(r'[\\/*?:"<>|]', '_', filename)  # Remove invalid characters
        filename = re.sub(r'\s+', '_', filename)  # Replace spaces with underscores
        return filename
    
    def get_pdf_links(self, symbol):
        """
        Get PDF links from the quarterly reports page of the company profile.
        
        Args:
            symbol (str): The company symbol on CSE
            
        Returns:
            list: A list of tuples (date, title, url) for each PDF
        """
        url = f"https://www.cse.lk/pages/company-profile/company-profile.component.html?symbol={symbol}"
        driver = self.setup_driver()
        driver.get(url)
        
        try:
            # Navigate to the Financials tab
            financials_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Financials')]"))
            )
            financials_tab.click()
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'Quarterly Reports')]"))
            )
            
            # Navigate to the Quarterly Reports tab
            quarterly_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Quarterly Reports')]"))
            )
            quarterly_tab.click()
            WebDriverWait(driver, 5).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "tr"))
            )
        except Exception as e:
            self.logger.error(f"Error accessing Quarterly Reports tab: {e}")
            driver.quit()
            return []
        
        # Parse the page content
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
                            pdf_links.append((
                                date_cell.get_text(strip=True), 
                                title, 
                                pdf_url if pdf_url.startswith("http") else f"https://www.cse.lk{pdf_url}"
                            ))
            except Exception as e:
                self.logger.error(f"Error processing row: {e}")
        
        return pdf_links
    
    def download_pdfs(self, pdf_links, company_name):
        """
        Download the PDFs and save them to a local directory.
        
        Args:
            pdf_links (list): List of tuples (date, title, url) for each PDF
            company_name (str): Name of the company
            
        Returns:
            list: Paths to the downloaded PDF files
        """
        save_path = os.path.join(self.output_dir, self.sanitize_filename(company_name))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        downloaded_files = []
        session = requests.Session()  # Use a session for multiple requests
        
        for uploaded_date, title, pdf_url in pdf_links:
            sanitized_title = self.sanitize_filename(title)
            sanitized_date = self.sanitize_filename(uploaded_date)
            pdf_name = f"{sanitized_date}_{sanitized_title}.pdf"
            pdf_path = os.path.join(save_path, pdf_name)
            
            try:
                # Check if file already exists to avoid re-downloading
                if os.path.exists(pdf_path):
                    self.logger.info(f"File already exists: {pdf_name}")
                    downloaded_files.append(pdf_path)
                    continue
                
                response = session.get(pdf_url)
                response.raise_for_status()
                
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
                self.logger.info(f"Downloaded: {pdf_name}")
                downloaded_files.append(pdf_path)
                
                # Add a small delay to avoid overwhelming the server
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to download {pdf_url}: {e}")
            except Exception as e:
                self.logger.error(f"Error saving file {pdf_name}: {e}")
        
        return downloaded_files
    
    def extract_pdf_text(self, pdf_path):
        """
        Extract text from a PDF file focusing on income statement pages.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            tuple: (extracted_text, pages_found)
        """
        try:
            # Open the PDF file
            pdf_document = fitz.open(pdf_path)
            
            # Keywords to look for in the page
            target_keywords = [
                "Consolidated Income Statement", 
                "STATEMENT OF PROFIT OR LOSS",
            ]
            
            full_text = ""
            pages_found = []  # Track pages where keywords are found
            group_page_found = False  # Flag to track if we have already found the group data

            # Loop through each page in the PDF
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)  # Load the page
                page_text = page.get_text("text")  # Extract text from the page
                
                # Check if any of the target keywords are present on the page
                if any(keyword in page_text for keyword in target_keywords):
                    if "GROUP" in page_text and not group_page_found:
                        # If the page mentions "GROUP" and it's the first one, extract it
                        self.logger.info(f"Group data found on page {page_num + 1} of {pdf_path}")
                        full_text += page_text  # Add the text from the relevant page
                        pages_found.append(page_num + 1)  # Store the page number
                        group_page_found = True  # Mark that we've found the group data
                        break  # Stop once the group data page is found and processed
                    elif not group_page_found:
                        # If no group data is found yet, process the first occurrence of any page with the target keywords
                        self.logger.info(f"First occurrence of target keywords found on page {page_num + 1} of {pdf_path}")
                        full_text += page_text
                        pages_found.append(page_num + 1)
                        break  # Stop once the first valid page is processed
            
            if full_text:
                self.logger.info(f"Text successfully extracted from relevant pages in {pdf_path}")
                self.logger.info(f"Pages extracted: {pages_found}")
            else:
                self.logger.warning(f"No relevant pages found in {pdf_path}.")
            
            return full_text, pages_found
        
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return "", []
    
    def validate_extracted_data(self, extracted_text):
        """
        Validate if the extracted text contains relevant financial data.
        
        Args:
            extracted_text (str): Extracted text from the PDF
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        validation_keywords = [
            r"Revenue", r"Cost of Goods Sold", r"Gross Profit", r"Net Profit",
            r"Operating Expenses", r"Other Operating Income", r"Distribution Costs"
        ]
        
        # Check if the text matches expected patterns for an income statement
        for keyword in validation_keywords:
            if not re.search(keyword, extracted_text, re.IGNORECASE):
                self.logger.warning(f"Validation failed: {keyword} not found in the extracted text.")
                return False
        
        self.logger.info("Validation passed: All required keywords found in the extracted text.")
        return True
    
    def write_extracted_text_to_file(self, pdf_path, extracted_text):
        """
        Write the extracted text to a file.
        
        Args:
            pdf_path (str): Path to the PDF file
            extracted_text (str): Text to write to file
            
        Returns:
            str: Path to the output file
        """
        try:
            output_file = os.path.splitext(pdf_path)[0] + ".txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            self.logger.info(f"Extracted text written to {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Error writing extracted text to file for {pdf_path}: {e}")
            return None
    
    def process_pdfs_in_directory(self, directory_path):
        """
        Process all PDF files in the given directory to extract relevant data.
        
        Args:
            directory_path (str): Path to the directory containing PDFs
            
        Returns:
            list: Paths to the output text files
        """
        output_files = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory_path, filename)
                self.logger.info(f"Processing {pdf_path}")
                
                # Extract text from the PDF
                extracted_text, pages_found = self.extract_pdf_text(pdf_path)
                
                # Write the extracted text to a file regardless of validation
                if extracted_text:
                    output_file = self.write_extracted_text_to_file(pdf_path, extracted_text)
                    if output_file:
                        output_files.append(output_file)
                else:
                    self.logger.error(f"No relevant text extracted from {pdf_path}.")
        
        return output_files
    
    def run_pipeline(self, symbols):
        """
        Run the complete pipeline for the given symbols.
        
        Args:
            symbols (dict): Dictionary mapping symbols to company names
            
        Returns:
            dict: Results of the pipeline
        """
        results = {}
        
        for symbol, company_name in symbols.items():
            company_results = {
                "downloaded_pdfs": [],
                "extracted_text_files": []
            }
            
            # Step 1: Get PDF links
            self.logger.info(f"Fetching quarterly reports for {company_name} ({symbol})...")
            pdf_links = self.get_pdf_links(symbol)
            
            if not pdf_links:
                self.logger.warning(f"No PDF reports found for {company_name}.")
                results[company_name] = company_results
                continue
            
            self.logger.info(f"Found {len(pdf_links)} PDF reports for {company_name}.")
            
            # Step 2: Download PDFs
            downloaded_pdfs = self.download_pdfs(pdf_links, company_name)
            company_results["downloaded_pdfs"] = downloaded_pdfs
            
            # Step 3: Process downloaded PDFs
            company_dir = os.path.join(self.output_dir, self.sanitize_filename(company_name))
            if os.path.exists(company_dir) and os.listdir(company_dir):
                self.logger.info(f"Processing PDFs in {company_dir}")
                extracted_files = self.process_pdfs_in_directory(company_dir)
                company_results["extracted_text_files"] = extracted_files
            else:
                self.logger.warning(f"No PDFs found in {company_dir}")
            
            results[company_name] = company_results
        
        return results

def main():
    """Main entry point for the financial data extraction pipeline."""
    parser = argparse.ArgumentParser(description="Financial Data Extraction Pipeline")
    parser.add_argument("--symbols", nargs="+", default=["REXP.N0000", "DIPD.N0000"],
                        help="List of company symbols to process")
    parser.add_argument("--output_dir", default="reports",
                        help="Directory to save reports")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run browser in headless mode")
    parser.add_argument("--log_file", default="pipeline.log",
                        help="Path to the log file")
    args = parser.parse_args()
    
    # Define company mapping
    symbols = {symbol: symbol.split('.')[0] for symbol in args.symbols}
    
    # Initialize and run the pipeline
    pipeline = FinancialDataPipeline(
        output_dir=args.output_dir,
        headless=args.headless,
        log_file=args.log_file
    )
    
    results = pipeline.run_pipeline(symbols)
    
    # Print summary
    print("\n=== Pipeline Execution Summary ===")
    for company, result in results.items():
        print(f"\nCompany: {company}")
        print(f"  - PDFs downloaded: {len(result['downloaded_pdfs'])}")
        print(f"  - Text files extracted: {len(result['extracted_text_files'])}")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()