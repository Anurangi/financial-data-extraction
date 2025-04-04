import os
import pandas as pd

def create_unified_dataset():
    """
    Create a unified dataset by merging DIPD and REXP financial data
    with standardized column names.
    """
    # Base directory path
    base_dir = r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\dataset_creator\data"
    
    # Input CSV paths
    dipd_csv = os.path.join(base_dir, "dipd_financial_data_openai.csv")
    rexp_csv = os.path.join(base_dir, "rexp_financial_data_openai.csv")
    
    # Output path for unified CSV
    output_csv = os.path.join(base_dir, "unified_financial_data.csv")
    
    # Check if input files exist
    if not os.path.exists(dipd_csv):
        print(f"DIPD CSV file not found: {dipd_csv}")
        return
    
    if not os.path.exists(rexp_csv):
        print(f"REXP CSV file not found: {rexp_csv}")
        return
    
    # Read the CSV files
    print("Reading input CSV files...")
    dipd_df = pd.read_csv(dipd_csv)
    rexp_df = pd.read_csv(rexp_csv)
    
    # Define column mapping for standardization
    # Format: {'standardized_column_name': {'DIPD': 'original_dipd_column', 'REXP': 'original_rexp_column'}}
    column_mapping = {
        'Company': {'DIPD': 'company_name', 'REXP': 'company_name'},
        'Filename': {'DIPD': 'filename', 'REXP': 'filename'},
        'Quarter End Date': {'DIPD': 'Quarter end date', 'REXP': 'Quarter end date'},
        'Revenue': {'DIPD': 'Revenue from contracts with customers', 'REXP': 'Revenue'},
        'Cost of Sales': {'DIPD': 'Cost of sales', 'REXP': 'Cost of Sales'},
        'Gross Profit': {'DIPD': 'Gross profit', 'REXP': 'Gross Profit'},
        'Other Income': {'DIPD': 'Other income and gains', 'REXP': 'Other Operating Income'},
        'Distribution Costs': {'DIPD': 'Distribution costs', 'REXP': 'Distribution Costs'},
        'Administrative Expenses': {'DIPD': 'Administrative expenses', 'REXP': 'Administrative Expenses'},
        'Other Expenses': {'DIPD': 'Other expenses', 'REXP': 'Other Operating Expense'},
        'Finance Income': {'DIPD': 'Finance income', 'REXP': 'Finance Income'},
        'Finance Costs': {'DIPD': 'Finance costs', 'REXP': 'Finance Cost'},
        'Profit Before Tax': {'DIPD': 'Profit before tax', 'REXP': 'Profit Before Tax'},
        'Tax Expense': {'DIPD': 'Tax expense', 'REXP': 'Taxation'},
        'Profit for Period': {'DIPD': 'Profit for the period', 'REXP': 'Profit for the Period'},
    }
    
    # REXP specific columns that don't have direct DIPD equivalents
    rexp_specific_columns = {
        'Other Financial Items': 'Other Financial Items',
        'Share of Profit of Associate': 'Share of Profit of Associate',
        'Profit from Operations': 'Profit from Operations'
    }
    
    # Create new dataframes with standardized columns
    print("Standardizing DIPD data...")
    dipd_standardized = pd.DataFrame()
    
    # Map standard columns for DIPD
    for std_col, sources in column_mapping.items():
        dipd_col = sources['DIPD']
        if dipd_col in dipd_df.columns:
            dipd_standardized[std_col] = dipd_df[dipd_col]
        else:
            dipd_standardized[std_col] = None
    
    # Add REXP-specific columns with None values for DIPD
    for std_col in rexp_specific_columns.keys():
        dipd_standardized[std_col] = None
    
    print("Standardizing REXP data...")
    rexp_standardized = pd.DataFrame()
    
    # Map standard columns for REXP
    for std_col, sources in column_mapping.items():
        rexp_col = sources['REXP']
        if rexp_col in rexp_df.columns:
            rexp_standardized[std_col] = rexp_df[rexp_col]
        else:
            rexp_standardized[std_col] = None
    
    # Add REXP-specific columns
    for std_col, rexp_col in rexp_specific_columns.items():
        if rexp_col in rexp_df.columns:
            rexp_standardized[std_col] = rexp_df[rexp_col]
        else:
            rexp_standardized[std_col] = None
    
    # Combine the standardized dataframes
    print("Combining datasets...")
    combined_df = pd.concat([dipd_standardized, rexp_standardized], ignore_index=True)
    
    # Clean and format numeric values
    print("Cleaning and formatting data...")
    numeric_columns = [
        'Revenue', 'Cost of Sales', 'Gross Profit', 'Other Income',
        'Distribution Costs', 'Administrative Expenses', 'Other Expenses',
        'Finance Income', 'Finance Costs', 'Profit Before Tax', 'Tax Expense',
        'Profit for Period', 'Other Financial Items', 'Share of Profit of Associate',
        'Profit from Operations'
    ]
    
    for col in numeric_columns:
        if col in combined_df.columns:
            # Convert string values to numeric, preserving negative values
            combined_df[col] = combined_df[col].apply(
                lambda x: pd.to_numeric(
                    str(x).replace(',', '').replace('(', '-').replace(')', ''),
                    errors='coerce'
                ) if pd.notna(x) else None
            )
    
    # Parse dates if needed
    if 'Quarter End Date' in combined_df.columns:
        try:
            combined_df['Quarter End Date'] = pd.to_datetime(
                combined_df['Quarter End Date'],
                format='%d/%m/%Y', 
                errors='coerce'
            )
        except Exception as e:
            print(f"Warning: Could not convert all dates: {e}")
    
    # Save to CSV
    print(f"Saving unified dataset to: {output_csv}")
    combined_df.to_csv(output_csv, index=False)
    print("Unified dataset created successfully!")
    
    # Return the dataframe for further use if needed
    return combined_df

if __name__ == "__main__":
    create_unified_dataset()