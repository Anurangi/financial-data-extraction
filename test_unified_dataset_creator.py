import unittest
import os
import pandas as pd
import tempfile
from unittest.mock import patch, mock_open
from io import StringIO

# Import the module to test
import sys
sys.path.append('.')  # Add current directory to path
# Import from the correct module name
from unified_dataset_creator import create_unified_dataset

class TestFinancialDataStandardization(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Sample DIPD data
        self.dipd_data = pd.DataFrame({
            'company_name': ['DIPD Corp', 'DIPD Corp'],
            'filename': ['report1.pdf', 'report2.pdf'],
            'Quarter end date': ['31/03/2023', '30/06/2023'],
            'Revenue from contracts with customers': [1000000, 1100000],
            'Cost of sales': [600000, 650000],
            'Gross profit': [400000, 450000],
            'Other income and gains': [50000, 55000],
            'Distribution costs': [70000, 75000],
            'Administrative expenses': [100000, 110000],
            'Other expenses': [30000, 35000],
            'Finance income': [5000, 5500],
            'Finance costs': [10000, 11000],
            'Profit before tax': [245000, 279500],
            'Tax expense': [45000, 50000],
            'Profit for the period': [200000, 229500]
        })
        
        # Sample REXP data
        self.rexp_data = pd.DataFrame({
            'company_name': ['REXP Inc', 'REXP Inc'],
            'filename': ['report3.pdf', 'report4.pdf'],
            'Quarter end date': ['31/03/2023', '30/06/2023'],
            'Revenue': [2000000, 2200000],
            'Cost of Sales': [1200000, 1300000],
            'Gross Profit': [800000, 900000],
            'Other Operating Income': [100000, 110000],
            'Distribution Costs': [150000, 165000],
            'Administrative Expenses': [200000, 220000],
            'Other Operating Expense': [50000, 55000],
            'Finance Income': [10000, 11000],
            'Finance Cost': [20000, 22000],
            'Profit from Operations': [500000, 570000],
            'Other Financial Items': [15000, 16500],
            'Share of Profit of Associate': [25000, 27500],
            'Profit Before Tax': [530000, 603000],
            'Taxation': [100000, 110000],
            'Profit for the Period': [430000, 493000]
        })
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Set up paths
        self.base_dir = self.temp_dir.name
        self.dipd_dir = os.path.join(self.base_dir, "DIPD")
        self.rexp_dir = os.path.join(self.base_dir, "REXP")
        os.makedirs(self.dipd_dir, exist_ok=True)
        os.makedirs(self.rexp_dir, exist_ok=True)
        
        # Set up file paths
        self.dipd_csv = os.path.join(self.dipd_dir, "dipd_financial_data_openai.csv")
        self.rexp_csv = os.path.join(self.rexp_dir, "rexp_financial_data_openai.csv")
        self.output_csv = os.path.join(self.base_dir, "unified_financial_data.csv")
        
        # Save test data to CSV files
        self.dipd_data.to_csv(self.dipd_csv, index=False)
        self.rexp_data.to_csv(self.rexp_csv, index=False)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.temp_dir.cleanup()
        
    @patch('unified_dataset_creator.os.path.join')
    def test_file_paths(self, mock_join):
        """Test that the correct file paths are constructed."""
        # Set up the mock to return predefined paths
        mock_join.side_effect = lambda *args: os.path.join(*args)
        
        # Call the function with mock
        with patch('unified_dataset_creator.os.path.exists', return_value=True), \
             patch('pandas.read_csv', return_value=pd.DataFrame()), \
             patch('pandas.DataFrame.to_csv'):
            create_unified_dataset()
        
        # Check that join was called with the correct arguments
        expected_calls = [
            ((r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\reports", "DIPD", "dipd_financial_data_openai.csv"),),
            ((r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\reports", "REXP", "rexp_financial_data_openai.csv"),),
            ((r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\reports", "unified_financial_data.csv"),)
        ]
        
        # Convert the mock_join.call_args_list to a comparable format
        actual_calls = [call[0] for call in mock_join.call_args_list]
        
        # Assert each expected call was made
        for expected_call in expected_calls:
            self.assertIn(expected_call, actual_calls)

    def test_file_existence_checks(self):
        """Test that the function checks if input files exist."""
        # Set up a mock path that doesn't exist
        with patch('financial_data_standardization.os.path.join', return_value="/nonexistent/path"), \
             patch('financial_data_standardization.os.path.exists', return_value=False), \
             patch('builtins.print') as mock_print:
            result = create_unified_dataset()
            
            # Check that the function returns None when files don't exist
            self.assertIsNone(result)
            
            # Check that an appropriate message was printed
            mock_print.assert_called_with("DIPD CSV file not found: /nonexistent/path")

    @patch('unified_dataset_creator.os.path')
    def test_integration(self, mock_path):
        """Test the full integration of the function using our test files."""
        # Mock os.path functions to use our temporary directory
        def mock_join(*args):
            if args[0] == r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\reports":
                return os.path.join(self.base_dir, *args[1:])
            return os.path.join(*args)
            
        mock_path.join.side_effect = mock_join
        mock_path.exists.return_value = True
        
        # Run the function with our test directory structure
        with patch('unified_dataset_creator.os.path.join', side_effect=mock_join), \
             patch('unified_dataset_creator.os.path.exists', return_value=True), \
             patch('pandas.read_csv') as mock_read_csv, \
             patch('pandas.DataFrame.to_csv') as mock_to_csv:
            
            # Make the mock read_csv return our test data
            mock_read_csv.side_effect = lambda path: self.dipd_data if "DIPD" in path else self.rexp_data
            
            result = create_unified_dataset()
            
            # Verify the result DataFrame has the expected structure
            self.assertIsNotNone(result)
            self.assertIn('Company Type', result.columns)
            self.assertIn('Revenue', result.columns)
            self.assertIn('Profit Before Tax', result.columns)
            
            # Verify to_csv was called with the expected path
            mock_to_csv.assert_called_once()

    def test_column_standardization(self):
        """Test that columns are correctly standardized."""
        # Override paths to use our test files
        with patch('unified_dataset_creator.os.path.join') as mock_join, \
             patch('unified_dataset_creator.os.path.exists', return_value=True):
             
            mock_join.side_effect = lambda *args: (
                self.dipd_csv if "DIPD" in args else 
                self.rexp_csv if "REXP" in args else 
                self.output_csv
            )
            
            # Call the function
            result = create_unified_dataset()
            
            # Verify that DIPD records have DIPD as Company Type
            dipd_records = result[result['Company Type'] == 'DIPD']
            self.assertEqual(len(dipd_records), 2)
            
            # Verify that REXP records have REXP as Company Type
            rexp_records = result[result['Company Type'] == 'REXP']
            self.assertEqual(len(rexp_records), 2)
            
            # Check that standard column names are used
            for col in ['Revenue', 'Cost of Sales', 'Gross Profit', 'Profit Before Tax']:
                self.assertIn(col, result.columns)
                
            # Check that REXP-specific columns are included
            for col in ['Other Financial Items', 'Share of Profit of Associate', 'Profit from Operations']:
                self.assertIn(col, result.columns)
                
            # Check that DIPD records have None for REXP-specific columns
            for col in ['Other Financial Items', 'Share of Profit of Associate', 'Profit from Operations']:
                self.assertTrue(pd.isna(dipd_records[col]).all())

    def test_numeric_formatting(self):
        """Test that numeric values are correctly formatted."""
        # Create data with various formatting issues
        test_dipd = pd.DataFrame({
            'company_name': ['DIPD Corp'],
            'filename': ['report1.pdf'],
            'Quarter end date': ['31/03/2023'],
            'Revenue from contracts with customers': ['1,000,000'],
            'Cost of sales': ['(600,000)'],  # Negative value in parentheses
            'Gross profit': [400000],
            'Profit before tax': ['400,000.50'],  # With decimal
            'Tax expense': ['N/A'],  # Non-numeric value
            'Profit for the period': [None]  # Missing value
        })
        
        # Override paths and read_csv to use our test data
        with patch('unified_dataset_creator.os.path.join') as mock_join, \
             patch('unified_dataset_creator.os.path.exists', return_value=True), \
             patch('pandas.read_csv') as mock_read_csv, \
             patch('pandas.DataFrame.to_csv'):
             
            mock_join.side_effect = lambda *args: "/mock/path"
            mock_read_csv.side_effect = lambda path: test_dipd if "DIPD" in path else pd.DataFrame()
            
            # Call the function
            result = create_unified_dataset()
            
            # Check numeric conversion
            self.assertEqual(result.loc[0, 'Revenue'], 1000000.0)
            self.assertEqual(result.loc[0, 'Cost of Sales'], -600000.0)  # Should convert to negative
            self.assertEqual(result.loc[0, 'Gross Profit'], 400000.0)
            self.assertEqual(result.loc[0, 'Profit Before Tax'], 400000.5)  # Should preserve decimal
            self.assertTrue(pd.isna(result.loc[0, 'Tax Expense']))  # 'N/A' should become NaN
            self.assertTrue(pd.isna(result.loc[0, 'Profit for Period']))  # None should remain None

    def test_date_parsing(self):
        """Test that dates are correctly parsed."""
        # Create data with various date formats
        test_data = pd.DataFrame({
            'company_name': ['Company A', 'Company B', 'Company C'],
            'filename': ['file1.pdf', 'file2.pdf', 'file3.pdf'],
            'Quarter end date': ['31/03/2023', '30/06/2023', 'Invalid Date']
        })
        
        # Override paths and read_csv to use our test data
        with patch('financial_data_standardization.os.path.join') as mock_join, \
             patch('financial_data_standardization.os.path.exists', return_value=True), \
             patch('pandas.read_csv') as mock_read_csv, \
             patch('pandas.DataFrame.to_csv'):
             
            mock_join.side_effect = lambda *args: "/mock/path"
            mock_read_csv.side_effect = lambda path: test_data
            
            # Call the function
            result = create_unified_dataset()
            
            # Check date parsing
            self.assertEqual(result.loc[0, 'Quarter End Date'].strftime('%Y-%m-%d'), '2023-03-31')
            self.assertEqual(result.loc[1, 'Quarter End Date'].strftime('%Y-%m-%d'), '2023-06-30')
            self.assertTrue(pd.isna(result.loc[2, 'Quarter End Date']))  # Invalid date should be NaN

if __name__ == '__main__':
    unittest.main()