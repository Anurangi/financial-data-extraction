import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
from datetime import datetime
import mlflow
import mlflow.statsmodels
import io
from PIL import Image

def simple_forecasting_pipeline(input_file, output_folder='./output', experiment_name="gross_profit_forecasting"):
    """
    A simple forecasting pipeline for gross profit with MLflow tracking
    
    Parameters:
    -----------
    input_file : str
        Path to the CSV file with financial data
    output_folder : str
        Folder to save outputs
    experiment_name : str
        Name of the MLflow experiment to use
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    # 1. Load the data
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file)
    
    # 2. Basic preprocessing
    data['Quarter End Date'] = pd.to_datetime(data['Quarter End Date'])
    data['Year'] = data['Quarter End Date'].dt.year
    data['Quarter'] = data['Quarter End Date'].dt.quarter
    
    # 3. Get unique companies
    companies = data['Company'].unique()
    print(f"Found {len(companies)} companies in the data")
    
    # 4. Simple forecasting for each company
    results = []
    
    for company in companies:
        print(f"\nProcessing {company}...")
        
        # Start MLflow run for this company
        with mlflow.start_run(run_name=f"{company}_forecast"):
            # Log company name as a parameter
            mlflow.log_param("company", company)
            
            # Filter data for this company
            company_data = data[data['Company'] == company].copy()
            company_data = company_data.sort_values('Quarter End Date')
            
            # Check for missing quarters
            date_diffs = []
            for i in range(1, len(company_data)):
                prev_date = company_data.iloc[i-1]['Quarter End Date']
                curr_date = company_data.iloc[i]['Quarter End Date']
                diff_months = (curr_date.year - prev_date.year) * 12 + (curr_date.month - prev_date.month)
                date_diffs.append(diff_months)
            
            # Log data quality information
            has_gaps = any(diff > 3 for diff in date_diffs)
            mlflow.log_param("has_data_gaps", has_gaps)
            mlflow.log_param("data_points", len(company_data))
            
            if has_gaps:
                print(f"  Note: Found gaps in quarterly data for {company}")
                missing_quarters = [i+1 for i, diff in enumerate(date_diffs) if diff > 3]
                print(f"  Missing quarters after positions: {missing_quarters}")
                print(f"  This may affect forecast accuracy.")
                mlflow.log_param("missing_quarters", str(missing_quarters))
            
            # Create time series for gross profit
            ts_data = company_data.set_index('Quarter End Date')['Gross Profit']
            
            # EDA: Create basic plots
            plt.figure(figsize=(10, 6))
            plt.plot(ts_data.index, ts_data.values, marker='o')
            plt.title(f'Gross Profit - {company}')
            plt.xlabel('Date')
            plt.ylabel('Gross Profit')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot to file and log to MLflow
            time_series_path = f"{output_folder}/{company.replace(' ', '_')}_time_series.png"
            plt.savefig(time_series_path)
            mlflow.log_artifact(time_series_path, "plots")
            
            # Log time series data as CSV
            time_series_csv = f"{output_folder}/{company.replace(' ', '_')}_ts_data.csv"
            ts_data.to_csv(time_series_csv)
            mlflow.log_artifact(time_series_csv, "data")
            
            # Simple statistics
            stats = {
                'Company': company,
                'Min': ts_data.min(),
                'Max': ts_data.max(),
                'Mean': ts_data.mean(),
                'Latest Value': ts_data.iloc[-1],
                'Data Points': len(ts_data)
            }
            
            # Log statistics as metrics
            mlflow.log_metric("min_gross_profit", stats['Min'])
            mlflow.log_metric("max_gross_profit", stats['Max'])
            mlflow.log_metric("mean_gross_profit", stats['Mean'])
            mlflow.log_metric("latest_gross_profit", stats['Latest Value'])
            
            print(f"Statistics for {company}:")
            for key, value in stats.items():
                if key != 'Company':
                    print(f"  {key}: {value:,.2f}")
            
            # Simple forecast with Exponential Smoothing
            # Use the last 4 periods as a test set
            train_size = max(len(ts_data) - 4, int(len(ts_data) * 0.7))
            train_data = ts_data.iloc[:train_size]
            test_data = ts_data.iloc[train_size:]
            
            # Log train/test split info
            mlflow.log_param("train_size", train_size)
            mlflow.log_param("test_size", len(test_data))
            mlflow.log_param("train_test_ratio", train_size / len(ts_data) if len(ts_data) > 0 else 0)
            
            # Train the model
            try:
                # Set model parameters
                model_params = {
                    "trend": "add",
                    "seasonal": "add",
                    "seasonal_periods": 4  # Quarterly data
                }
                
                # Log model parameters
                for param_name, param_value in model_params.items():
                    mlflow.log_param(param_name, param_value)
                
                model = ExponentialSmoothing(
                    train_data,
                    **model_params
                ).fit()
                
                # Log the model to MLflow
                mlflow.statsmodels.log_model(model, "model")
                
                # Forecast for the test period plus 4 additional quarters
                forecast_periods = len(test_data) + 4
                mlflow.log_param("forecast_periods", forecast_periods)
                forecast = model.forecast(forecast_periods)
                
                # Convert forecast index to pandas DatetimeIndex if it's not already
                if not isinstance(forecast.index, pd.DatetimeIndex):
                    # Create proper date index
                    last_train_date = train_data.index[-1]
                    forecast_index = pd.date_range(
                        start=last_train_date + pd.Timedelta(days=1),
                        periods=forecast_periods,
                        freq='Q'
                    )
                    forecast = pd.Series(forecast.values, index=forecast_index)
                
                # Calculate accuracy on test data
                if len(test_data) > 0:
                    try:
                        # Align test data and forecast by date
                        test_forecast = forecast.loc[test_data.index] if any(idx in forecast.index for idx in test_data.index) else forecast[:len(test_data)]
                        test_vals = test_data.values
                        forecast_vals = test_forecast.values
                        
                        # Calculate metrics
                        mape = np.mean(np.abs((test_vals - forecast_vals) / test_vals)) * 100
                        rmse = np.sqrt(np.mean((test_vals - forecast_vals) ** 2))
                        mae = np.mean(np.abs(test_vals - forecast_vals))
                        
                        # Log metrics to MLflow
                        mlflow.log_metric("test_mape", mape)
                        mlflow.log_metric("test_rmse", rmse)
                        mlflow.log_metric("test_mae", mae)
                        
                        print(f"  Test MAPE: {mape:.2f}%")
                        print(f"  Test RMSE: {rmse:.2f}")
                        print(f"  Test MAE: {mae:.2f}")
                        
                        stats['Test MAPE'] = mape
                    except Exception as e:
                        print(f"  Error calculating metrics: {str(e)}")
                        mlflow.log_param("metrics_error", str(e))
                        print(f"  Test data shape: {test_data.shape}, Forecast shape: {test_forecast.shape}")
                        
                # Plot the forecast
                plt.figure(figsize=(12, 6))
                
                # Format dates consistently for plotting
                if isinstance(ts_data.index, pd.DatetimeIndex):
                    historical_dates = [d.strftime('%Y-%m-%d') for d in ts_data.index]
                else:
                    historical_dates = [str(d) for d in ts_data.index]
                    
                if isinstance(forecast.index, pd.DatetimeIndex):
                    forecast_dates = [d.strftime('%Y-%m-%d') for d in forecast.index]
                else:
                    forecast_dates = [str(d) for d in forecast.index]
                
                plt.plot(historical_dates, ts_data.values, marker='o', label='Historical')
                plt.plot(forecast_dates, forecast.values, 'r--o', label='Forecast')
                
                # Add a vertical line to separate historical and future forecast
                if len(test_data) > 0:
                    plt.axvline(x=test_data.index[0].strftime('%Y-%m-%d'), color='gray', linestyle='--')
                
                plt.title(f'Gross Profit Forecast - {company}')
                plt.xlabel('Date')
                plt.ylabel('Gross Profit')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save forecast plot and log to MLflow
                forecast_path = f"{output_folder}/{company.replace(' ', '_')}_forecast.png"
                plt.savefig(forecast_path)
                mlflow.log_artifact(forecast_path, "plots")
                
                # Log forecast data as CSV
                forecast_df = pd.DataFrame({
                    'Date': forecast.index,
                    'Forecast': forecast.values
                })
                forecast_csv = f"{output_folder}/{company.replace(' ', '_')}_forecast.csv"
                forecast_df.to_csv(forecast_csv)
                mlflow.log_artifact(forecast_csv, "forecasts")
                
                # Save forecast to results
                future_forecast = forecast[len(test_data):]
                for date, value in zip(future_forecast.index, future_forecast.values):
                    # Ensure date is properly formatted as string
                    date_str = None
                    if isinstance(date, (pd.Timestamp, datetime)):
                        date_str = date.strftime('%Y-%m-%d')
                    else:
                        # Handle the case where date might not be a timestamp object
                        try:
                            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
                        except:
                            idx = list(future_forecast.index).index(date)
                            date_str = f"{datetime.now().year + 1}-{(idx+1)*3:02d}-30"  # Fallback format
                            print(f"  Warning: Using fallback date format for {date}: {date_str}")
                    
                    results.append({
                        'Company': company,
                        'Forecast Date': date_str,
                        'Gross Profit Forecast': value
                    })
                
            except Exception as e:
                print(f"  Error forecasting for {company}: {str(e)}")
                mlflow.log_param("forecasting_error", str(e))
                
                # Try a simpler approach if this fails
                if len(ts_data) >= 4:
                    print("  Using moving average as fallback...")
                    mlflow.log_param("fallback_method", "moving_average")
                    
                    # Simple moving average of last 4 quarters
                    avg_value = ts_data.iloc[-4:].mean()
                    mlflow.log_metric("moving_avg_forecast", avg_value)
                    
                    # Generate proper quarterly future dates (Mar 31, Jun 30, Sep 30, Dec 31)
                    last_date = ts_data.index[-1]
                    future_dates = []
                    
                    # Check if last_date is a timestamp or a different type
                    if isinstance(last_date, (pd.Timestamp, datetime)):
                        # Start from the next quarter after the last date
                        next_date = pd.Timestamp(last_date) + pd.DateOffset(days=1)
                        
                        # Normalize to start of next quarter
                        next_quarter_start = pd.Timestamp(year=next_date.year, month=(next_date.month - 1) // 3 * 3 + 1, day=1)
                        next_quarter_start += pd.DateOffset(months=3)
                        
                        # Generate 4 quarter-end dates
                        for i in range(4):
                            # Calculate quarter end date (last day of Mar, Jun, Sep, or Dec)
                            quarter_month = (next_quarter_start.month - 1) // 3 * 3 + 3  # Convert to quarter end month (3, 6, 9, 12)
                            quarter_end = pd.Timestamp(year=next_quarter_start.year, month=quarter_month, day=1) + pd.DateOffset(months=1, days=-1)
                            future_dates.append(quarter_end)
                            next_quarter_start += pd.DateOffset(months=3)
                    else:
                        # If the index is not a timestamp (could be int, string, etc.), use hardcoded next 4 quarters
                        print(f"  Using hardcoded quarters due to non-timestamp index type: {type(last_date)}")
                        mlflow.log_param("non_timestamp_index", str(type(last_date)))
                        
                        # Get current date and generate next 4 quarters
                        current_date = datetime.now()
                        next_quarter_start = pd.Timestamp(year=current_date.year, month=(current_date.month - 1) // 3 * 3 + 1, day=1)
                        next_quarter_start += pd.DateOffset(months=3)
                        
                        for i in range(4):
                            quarter_month = (next_quarter_start.month - 1) // 3 * 3 + 3
                            quarter_end = pd.Timestamp(year=next_quarter_start.year, month=quarter_month, day=1) + pd.DateOffset(months=1, days=-1)
                            future_dates.append(quarter_end)
                            next_quarter_start += pd.DateOffset(months=3)
                    
                    # Log fallback forecast dates
                    mlflow.log_param("fallback_forecast_dates", str([d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in future_dates]))
                    
                    # Add to results
                    for date in future_dates:
                        # Ensure date is properly formatted as string
                        date_str = None
                        if isinstance(date, (pd.Timestamp, datetime)):
                            date_str = date.strftime('%Y-%m-%d')
                        else:
                            # Handle the case where date might not be a timestamp object
                            try:
                                date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
                            except:
                                date_str = f"{datetime.now().year + 1}-{(i+1)*3:02d}-30"  # Fallback format
                                print(f"  Warning: Using fallback date format for {date}: {date_str}")
                        
                        results.append({
                            'Company': company,
                            'Forecast Date': date_str,
                            'Gross Profit Forecast': avg_value
                        })
    
    # 5. Save forecast results to CSV
    if results:
        # Validate all dates are proper formatted strings
        for result in results:
            # Ensure date is a string in YYYY-MM-DD format
            if not isinstance(result['Forecast Date'], str):
                try:
                    if hasattr(result['Forecast Date'], 'strftime'):
                        result['Forecast Date'] = result['Forecast Date'].strftime('%Y-%m-%d')
                    else:
                        result['Forecast Date'] = str(result['Forecast Date'])
                        print(f"Warning: Unexpected date format for {result['Company']}: {result['Forecast Date']}")
                except Exception as e:
                    print(f"Error formatting date for {result['Company']}: {str(e)}")
                    result['Forecast Date'] = str(result['Forecast Date'])
        
        forecast_df = pd.DataFrame(results)
        
        # Add one more validation to check if any date is not in proper format
        for i, row in forecast_df.iterrows():
            date_str = row['Forecast Date']
            if not (isinstance(date_str, str) and len(date_str) >= 10 and date_str[4] == '-' and date_str[7] == '-'):
                print(f"Warning: Potentially invalid date format at row {i}: {date_str} for {row['Company']}")
        
        # Create a timestamp for the output file
        timestamp = datetime.now().strftime('%Y%m%d')
        output_file = f"{output_folder}/gross_profit_forecast_{timestamp}.csv"
        forecast_df.to_csv(output_file, index=False)
        print(f"\nForecast saved to {output_file}")
        
        # Log the final forecast results to MLflow
        with mlflow.start_run(run_name=f"combined_forecasts_{timestamp}"):
            mlflow.log_param("companies_count", len(companies))
            mlflow.log_param("total_forecasts", len(forecast_df))
            mlflow.log_artifact(output_file, "final_forecasts")
            
            # Create and log a summary of companies and their forecast values
            summary_df = forecast_df.groupby('Company')['Gross Profit Forecast'].agg(['mean', 'min', 'max']).reset_index()
            summary_file = f"{output_folder}/forecast_summary_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            mlflow.log_artifact(summary_file, "summary")
    
    print("\nForecasting pipeline completed!")
    return forecast_df

if __name__ == "__main__":
    # Replace with your actual file path
    input_file = "C:/Users/Akshila.Anurangi/financial-data-extraction-pipeline/dataset_creator/data/unified_financial_data.csv"
    
    # Set MLflow tracking URI (uncomment and modify if using a remote tracking server)
    # mlflow.set_tracking_uri("http://localhost:5000")
    
    # Run the pipeline
    simple_forecasting_pipeline(input_file)