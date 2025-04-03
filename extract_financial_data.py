import fitz  # PyMuPDF
import logging
import re
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="extraction_log.txt")
logger = logging.getLogger(__name__)

def extract_pdf_text(pdf_path):
    """
    Extract text from a PDF file using PyMuPDF, focusing on pages with
    "Consolidated Income Statement" or "STATEMENT OF PROFIT OR LOSS".
    Prioritize the 'group' version of the 'STATEMENT OF PROFIT OR LOSS'.
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
                    logger.info(f"Group data found on page {page_num + 1} of {pdf_path}")
                    full_text += page_text  # Add the text from the relevant page
                    pages_found.append(page_num + 1)  # Store the page number
                    group_page_found = True  # Mark that we've found the group data
                    break  # Stop once the group data page is found and processed
                elif not group_page_found:
                    # If no group data is found yet, process the first occurrence of any page with the target keywords
                    logger.info(f"First occurrence of STATEMENT OF PROFIT OR LOSS found on page {page_num + 1} of {pdf_path}")
                    full_text += page_text
                    pages_found.append(page_num + 1)
                    break  # Stop once the first valid page is processed
        
        if full_text:
            logger.info(f"Text successfully extracted from relevant pages in {pdf_path}")
            logger.info(f"Pages extracted: {pages_found}")
        else:
            logger.warning(f"No relevant pages found in {pdf_path}.")
        
        return full_text, pages_found
    
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return "", []

def validate_extracted_data(extracted_text):
    """
    Validate if the extracted text contains relevant financial data such as
    'Revenue', 'COGS', 'Net Profit', etc., in the context of an income statement.
    """
    validation_keywords = [
        r"Revenue", r"Cost of Goods Sold", r"Gross Profit", r"Net Profit",
        r"Operating Expenses", r"Other Operating Income", r"Distribution Costs"
    ]
    
    # Check if the text matches expected patterns for an income statement
    for keyword in validation_keywords:
        if not re.search(keyword, extracted_text, re.IGNORECASE):  # Added re.IGNORECASE
            logger.warning(f"Validation failed: {keyword} not found in the extracted text.")
            return False
    
    logger.info("Validation passed: All required keywords found in the extracted text.")
    return True

def write_extracted_text_to_file(pdf_path, extracted_text):
    """
    Write the extracted text to a file in the same directory as the PDF.
    The text will be saved with the same name as the PDF, but with a '.txt' extension.
    """
    try:
        output_file = os.path.splitext(pdf_path)[0] + ".txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        logger.info(f"Extracted text written to {output_file}")
    except Exception as e:
        logger.error(f"Error writing extracted text to file for {pdf_path}: {e}")

def process_pdfs_in_directory(directory_path):
    """
    Process all PDF files in the given directory to extract relevant data.
    """
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            logger.info(f"Found PDF: {pdf_path}")  # Debug line
            logger.info(f"Processing {pdf_path}")
            
            # Extract text from the PDF
            extracted_text, pages_found = extract_pdf_text(pdf_path)
            
            # Write the extracted text to a file regardless of validation
            if extracted_text:
                write_extracted_text_to_file(pdf_path, extracted_text)
            else:
                logger.error(f"No relevant text extracted from {pdf_path}.")

def main():
    # Process PDFs in the 'DIPD' and 'REXP' directories
    dipd_directory = r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\reports\DIPD"
    rexp_directory = r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\reports\REXP"
    
    logger.info("Starting extraction for DIPD PDFs...")
    process_pdfs_in_directory(dipd_directory)
    
    logger.info("Starting extraction for REXP PDFs...")
    process_pdfs_in_directory(rexp_directory)

# Only run the script when it's executed directly (not imported)
if __name__ == "__main__":
    main()
