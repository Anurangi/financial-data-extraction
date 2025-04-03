import os
import glob
import pandas as pd
import json
import openai  # Using the legacy version

def extract_with_openai(file_path, api_key, company_type):
    """
    Extract financial data from a text file using OpenAI's GPT-3.5 Turbo.
    Compatible with older openai library versions.
    
    Args:
        file_path: Path to the financial statement text file
        api_key: OpenAI API key
        company_type: Type of company format ("DIPD" or "REXP")
    """
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract filename
    filename = os.path.basename(file_path)
    
    # Set OpenAI API key
    openai.api_key = api_key
    
    # Create the prompt for GPT - adjusted based on company type
    if company_type == "DIPD":
        company_name = "DIPPED PRODUCTS PLC"
        prompt = f"""
        Extract the financial data for the LATEST QUARTER (3 months period) from the following financial statement for Dipped Products PLC.
        
        Focus specifically on extracting these key metrics (provide exactly these metrics in your output):
        1. Revenue from contracts with customers
        2. Cost of sales
        3. Gross profit
        4. Other income and gains
        5. Distribution costs
        6. Administrative expenses
        7. Other expenses
        8. Finance costs
        9. Finance income
        10. Profit before tax
        11. Tax expense
        12. Profit for the period
        13. Quarter end date (in format DD/MM/YYYY)
        
        Return ONLY a JSON object with the following structure:
        {{
            "Revenue from contracts with customers": "value",
            "Cost of sales": "value",
            "Gross profit": "value",
            "Other income and gains": "value", 
            "Distribution costs": "value",
            "Administrative expenses": "value",
            "Other expenses": "value",
            "Finance costs": "value",
            "Finance income": "value",
            "Profit before tax": "value",
            "Tax expense": "value",
            "Profit for the period": "value",
            "Quarter end date": "value"
        }}
        """
    else:  # REXP format
        company_name = "RICHARD PIERIS EXPORTS PLC"
        prompt = f"""
        Extract the financial data for the LATEST QUARTER (3 months period) from the following financial statement for Richard Pieris Exports PLC.
        
        Focus specifically on extracting these key metrics (provide exactly these metrics in your output):
        1. Revenue
        2. Cost of Sales
        3. Gross Profit
        4. Other Operating Income
        5. Distribution Costs
        6. Administrative Expenses
        7. Other Operating Expense
        8. Profit from Operations
        9. Finance Income
        10. Finance Cost
        11. Other Financial Items
        12. Share of Profit of Associate
        13. Profit Before Tax
        14. Taxation
        15. Profit for the Period
        16. Quarter end date (in format DD/MM/YYYY)
        
        Return ONLY a JSON object with the following structure:
        {{
            "Revenue": "value",
            "Cost of Sales": "value",
            "Gross Profit": "value",
            "Other Operating Income": "value", 
            "Distribution Costs": "value",
            "Administrative Expenses": "value",
            "Other Operating Expense": "value",
            "Profit from Operations": "value",
            "Finance Income": "value",
            "Finance Cost": "value",
            "Other Financial Items": "value",
            "Share of Profit of Associate": "value",
            "Profit Before Tax": "value",
            "Taxation": "value",
            "Profit for the Period": "value",
            "Quarter end date": "value"
        }}
        """
    
    # Add common instructions and financial statement
    prompt += """
    If any value is not found, use null.
    Format all numerical values as strings, preserving the original format (with commas, decimal points, etc.).
    If values are in parentheses (indicating negative numbers), convert them to use a negative sign instead.
    Make sure to extract the most recent 3-month period data, not the 6-month data.
    ONLY return the JSON object, no additional text.
    
    Here is the financial statement:
    """ + content[:15000]  # Limit content length to avoid token limits
    
    # Call OpenAI API using legacy approach
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial data extraction assistant. Extract the requested financial metrics from quarterly reports and return them exactly in the JSON format specified."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Use 0 for more deterministic results
        )
        
        # Extract the JSON string from the response
        json_str = response.choices[0].message.content
        
        # Clean the JSON if there are markdown code blocks
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON
        try:
            data = json.loads(json_str)
            
            # Add file information
            data['filename'] = filename
            data['company_name'] = company_name
            
            return data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from OpenAI response for {file_path}: {e}")
            print(f"Response content: {json_str}")
            return {"filename": filename, "error": "JSON parsing error", "company_name": company_name}
            
    except Exception as e:
        print(f"Error calling OpenAI API for {file_path}: {e}")
        return {"filename": filename, "error": str(e), "company_name": company_name}

def process_company_files(company_type, api_key):
    """
    Process all text files for a specific company and extract financial data using OpenAI.
    
    Args:
        company_type: Type of company format ("DIPD" or "REXP")
        api_key: OpenAI API key
    """
    # Base directory path
    base_dir = r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\reports"
    
    # Company-specific directory
    directory = os.path.join(base_dir, company_type)
    
    # Output path for CSV
    output_csv = os.path.join(directory, f"{company_type.lower()}_financial_data_openai.csv")
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
    
    # Get all text files in the directory
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    
    if not file_paths:
        print(f"No text files found in {directory}")
        return
    
    print(f"Found {len(file_paths)} text files to process for {company_type}")
    
    # Extract data from each file
    all_data = []
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        data = extract_with_openai(file_path, api_key, company_type)
        all_data.append(data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved extracted data to {output_csv}")

def main():
    # Replace with your OpenAI API key
    api_key = "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"
    
    # Process both company types
    print("Processing DIPD files...")
    process_company_files("DIPD", api_key)
    
    print("\nProcessing REXP files...")
    process_company_files("REXP", api_key)
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()