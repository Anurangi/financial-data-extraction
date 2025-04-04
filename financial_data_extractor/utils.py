"""
Utility functions for the financial data extraction pipeline.
"""
import os
import re
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def setup_logging(log_file, root_dir=None):
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Name of the log file
        root_dir (str, optional): Root directory path. If None, use current directory.
        
    Returns:
        logger: Configured logger
    """
    # Determine logs directory
    if root_dir:
        logs_dir = os.path.join(root_dir, "logs")
    else:
        # Use the parent directory of the current script as default
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(script_dir))
        logs_dir = os.path.join(root_dir, "logs")
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Update log file path to be inside logs directory
    log_file_path = os.path.join(logs_dir, log_file)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def setup_driver(headless=True):
    """
    Set up and return a Chrome WebDriver.
    
    Args:
        headless (bool): Whether to run browser in headless mode
        
    Returns:
        WebDriver: Configured Chrome WebDriver
    """
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def sanitize_filename(filename):
    """
    Sanitize the filename by replacing invalid characters and spaces.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)  # Remove invalid characters
    filename = re.sub(r'\s+', '_', filename)  # Replace spaces with underscores
    return filename


def create_directory(directory_path):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path (str): Path to the directory to create
        
    Returns:
        bool: True if directory was created or already exists, False otherwise
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            return True
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False


def validate_extracted_data(extracted_text, logger):
    """
    Validate if the extracted text contains relevant financial data.
    
    Args:
        extracted_text (str): Extracted text from the PDF
        logger: Logger instance
        
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
            logger.warning(f"Validation failed: {keyword} not found in the extracted text.")
            return False
    
    logger.info("Validation passed: All required keywords found in the extracted text.")
    return True