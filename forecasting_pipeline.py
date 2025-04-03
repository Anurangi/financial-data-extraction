import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess the data
def load_data(file_path):
    # Read the data
    data = pd.read_csv(file_path)
    
    # Convert date strings to datetime objects
    data['Quarter End Date'] = pd.to_datetime(data['Quarter End Date'])
    
    # Set date as index for time series analysis
    data = data.sort_values(by=['Company', 'Quarter End Date'])
    
    return data

# Exploratory Data Analysis
def perform_eda(data):
    # Basic info
    print("Data shape:", data.shape)
    print("\nCompanies in the dataset:", data['Company'].unique())
    
    # Time range
    print("\nTime range:", data['Quarter End Date'].min(), "to", data['Quarter End Date'].max())
    
    # Check for missing values
    print("\nMissing values:\n", data.isnull().sum())
    
    # Financial metrics summary by company
    print("\nFinancial metrics summary by company:")
    financial_cols = ['Revenue', 'Cost of Sales', 'Gross Profit', 'Profit Before Tax', 'Profit for Period']
    company_summary = data.groupby('Company')[financial_cols].agg(['mean', 'std', 'min', 'max'])
    print(company_summary)
    
    # Create visualizations
    plt.figure(figsize=(14, 8))
    
    # Revenue over time by company
    for company in data['Company'].unique():
        company_data = data[data['Company'] == company]
        plt.plot(company_data['Quarter End Date'], company_data['Revenue'], marker='o', linestyle='-', label=company)
    
    plt.title('Revenue Over Time by Company')
    plt.xlabel('Quarter End Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('revenue_over_time.png')
    
    # Profit margin over time
    plt.figure(figsize=(14, 8))
    for company in data['Company'].unique():
        company_data = data[data['Company'] == company]
        company_data['Profit Margin'] = company_data['Profit for Period'] / company_data['Revenue'] * 100
        plt.plot(company_data['Quarter End Date'], company_data['Profit Margin'], marker='o', linestyle='-', label=company)
    
    plt.title('Profit Margin Over Time by Company')
    plt.xlabel('Quarter End Date')
    plt.ylabel('Profit Margin (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('profit_margin_over_time.png')
    
    return None

# Time Series Analysis
def analyze_time_series(data, company, target_variable):
    # Filter data for the selected company
    company_data = data[data['Company'] == company].copy()
    
    # Set date as index
    company_data.set_index('Quarter End Date', inplace=True)
    
    # Ensure the data is sorted by date
    company_data = company_data.sort_index()
    
    # Get the time series data for the target variable
    ts_data = company_data[target_variable]
    
    # Plot the time series
    plt.figure(figsize=(14, 7))
    ts_data.plot()
    plt.title(f'{target_variable} Time Series for {company}')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{company}_{target_variable}_time_series.png')
    
    # Seasonal Decomposition
    try:
        # Check if we have enough data points
        if len(ts_data) > 8:  # Need at least 2 periods for seasonal decomposition
            # Try to decompose with appropriate periodicity
            # For quarterly data, period=4
            decomposition = seasonal_decompose(ts_data, model='multiplicative', period=4)
            
            # Plot decomposition
            plt.figure(figsize=(14, 12))
            plt.subplot(411)
            plt.plot(decomposition.observed)
            plt.title('Observed')
            plt.subplot(412)
            plt.plot(decomposition.trend)
            plt.title('Trend')
            plt.subplot(413)
            plt.plot(decomposition.seasonal)
            plt.title('Seasonality')
            plt.subplot(414)
            plt.plot(decomposition.resid)
            plt.title('Residuals')
            plt.tight_layout()
            plt.savefig(f'{company}_{target_variable}_decomposition.png')
    except:
        print(f"Could not perform seasonal decomposition for {company} {target_variable} - possibly insufficient data")
    
    return ts_data

# Model building using SARIMA (manually specifying parameters instead of auto_arima)
def build_sarima_model(ts_data, company, target_variable, forecast_periods=4):
    # Split data into train and test sets
    train_size = int(len(ts_data) * 0.8)
    train, test = ts_data[:train_size], ts_data[train_size:]
    
    # For quarterly data, we typically use these parameters:
    # p=1, d=1, q=1 (non-seasonal components)
    # P=1, D=1, Q=1, s=4 (seasonal components)
    # You can adjust these based on your data
    p, d, q = 1, 1, 1
    P, D, Q, s = 1, 1, 1, 4
    
    # Build SARIMA model
    model = SARIMAX(train, 
                    order=(p, d, q), 
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    results = model.fit(disp=False)
    print(results.summary())
    
    # Make predictions on the test set
    predictions = results.get_forecast(steps=len(test))
    predictions_ci = predictions.conf_int()
    predictions_mean = predictions.predicted_mean
    
    # Calculate error metrics
    mse = mean_squared_error(test, predictions_mean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions_mean)
    
    print(f"\nTest Set Metrics for {company} {target_variable}:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Test Data')
    plt.plot(test.index, predictions_mean, label='Predictions', color='red')
    plt.fill_between(predictions_ci.index,
                    predictions_ci.iloc[:, 0],
                    predictions_ci.iloc[:, 1], color='pink', alpha=0.3)
    
    plt.title(f'SARIMA Model: Actual vs Predicted {target_variable} for {company}')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{company}_{target_variable}_predictions.png')
    
    # Generate future forecasts
    future_forecast = results.get_forecast(steps=forecast_periods)
    future_forecast_mean = future_forecast.predicted_mean
    future_forecast_ci = future_forecast.conf_int()
    
    # Create future date index for the forecast
    last_date = ts_data.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=forecast_periods, freq='3M')
    
    # Plot the historical data and forecasts
    plt.figure(figsize=(14, 7))
    plt.plot(ts_data.index, ts_data, label='Historical Data')
    plt.plot(forecast_index, future_forecast_mean, label='Forecast', color='red')
    plt.fill_between(forecast_index,
                    future_forecast_ci.iloc[:, 0],
                    future_forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
    
    plt.title(f'SARIMA Forecast: {target_variable} for {company}')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{company}_{target_variable}_forecast.png')
    
    return results, future_forecast_mean, forecast_index

# Model building using Exponential Smoothing
def build_ets_model(ts_data, company, target_variable, forecast_periods=4):
    # Split data into train and test sets
    train_size = int(len(ts_data) * 0.8)
    train, test = ts_data[:train_size], ts_data[train_size:]
    
    # Build ETS model (Holt-Winters Exponential Smoothing)
    model = ExponentialSmoothing(train, 
                               seasonal_periods=4, 
                               trend='add', 
                               seasonal='add',
                               use_boxcox=True)
    
    results = model.fit()
    
    # Make predictions on the test set
    predictions = results.forecast(len(test))
    
    # Calculate error metrics
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    
    print(f"\nTest Set Metrics for {company} {target_variable} (ETS model):")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Test Data')
    plt.plot(test.index, predictions, label='Predictions', color='red')
    
    plt.title(f'ETS Model: Actual vs Predicted {target_variable} for {company}')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{company}_{target_variable}_ets_predictions.png')
    
    # Generate future forecasts
    future_forecast = results.forecast(forecast_periods)
    
    # Create future date index for the forecast
    last_date = ts_data.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=forecast_periods, freq='3M')
    
    # Plot the historical data and forecasts
    plt.figure(figsize=(14, 7))
    plt.plot(ts_data.index, ts_data, label='Historical Data')
    plt.plot(forecast_index, future_forecast, label='Forecast', color='red')
    
    plt.title(f'ETS Forecast: {target_variable} for {company}')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{company}_{target_variable}_ets_forecast.png')
    
    return results, future_forecast, forecast_index

# Compare multiple forecasting models
def compare_models(ts_data, company, target_variable):
    # Split data into train and test sets
    train_size = int(len(ts_data) * 0.8)
    train, test = ts_data[:train_size], ts_data[train_size:]
    
    # List to store model results
    model_predictions = {}
    model_mse = {}
    
    # 1. SARIMA model
    try:
        p, d, q = 1, 1, 1
        P, D, Q, s = 1, 1, 1, 4
        
        sarima_model = SARIMAX(train, 
                        order=(p, d, q), 
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        sarima_results = sarima_model.fit(disp=False)
        sarima_pred = sarima_results.get_forecast(steps=len(test)).predicted_mean
        sarima_mse = mean_squared_error(test, sarima_pred)
        
        model_predictions['SARIMA'] = sarima_pred
        model_mse['SARIMA'] = sarima_mse
        
        print(f"SARIMA MSE: {sarima_mse:.2f}")
    except:
        print("Could not fit SARIMA model")
    
    # 2. ETS model
    try:
        ets_model = ExponentialSmoothing(train, 
                                   seasonal_periods=4, 
                                   trend='add', 
                                   seasonal='add',
                                   use_boxcox=True)
        
        ets_results = ets_model.fit()
        ets_pred = ets_results.forecast(len(test))
        ets_mse = mean_squared_error(test, ets_pred)
        
        model_predictions['ETS'] = ets_pred
        model_mse['ETS'] = ets_mse
        
        print(f"ETS MSE: {ets_mse:.2f}")
    except:
        print("Could not fit ETS model")
    
    # 3. Simple ARIMA model
    try:
        arima_model = sm.tsa.ARIMA(train, order=(2, 1, 2))
        arima_results = arima_model.fit()
        arima_pred = arima_results.forecast(steps=len(test))
        arima_mse = mean_squared_error(test, arima_pred)
        
        model_predictions['ARIMA'] = arima_pred
        model_mse['ARIMA'] = arima_mse
        
        print(f"ARIMA MSE: {arima_mse:.2f}")
    except:
        print("Could not fit ARIMA model")
    
    # Plot and compare results
    if model_predictions:
        plt.figure(figsize=(14, 7))
        plt.plot(train.index, train, label='Training Data')
        plt.plot(test.index, test, label='Actual Test Data')
        
        for name, pred in model_predictions.items():
            plt.plot(test.index, pred, label=f'{name} (MSE: {model_mse[name]:.2f})')
        
        plt.title(f'Model Comparison for {target_variable} - {company}')
        plt.xlabel('Date')
        plt.ylabel(target_variable)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{company}_{target_variable}_model_comparison.png')
        
        # Determine the best model
        if model_mse:
            best_model = min(model_mse.items(), key=lambda x: x[1])
            print(f"\nBest model for {company} {target_variable}: {best_model[0]} with MSE {best_model[1]:.2f}")
            return best_model[0]
    
    return None

# Main execution function
def main():
    # Path to the data file
    file_path = r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\reports\unified_financial_data.csv"
    
    # Load data
    data = load_data(file_path)
    
    # Perform EDA
    perform_eda(data)
    
    # Select companies and variables to forecast
    companies = data['Company'].unique()
    target_variables = ['Revenue', 'Gross Profit', 'Profit for Period']
    
    # Dictionary to store forecast results
    forecasts = {}
    
    # Analyze and forecast for each company and target variable
    for company in companies:
        company_forecasts = {}
        
        for variable in target_variables:
            print(f"\n{'='*50}")
            print(f"Analyzing {variable} for {company}")
            print(f"{'='*50}")
            
            # Time series analysis
            ts_data = analyze_time_series(data, company, variable)
            
            if len(ts_data) >= 8:  # Need minimum data points for meaningful forecast
                # Compare models to find the best one
                best_model_name = compare_models(ts_data, company, variable)
                
                # Build forecast with the best model or default to SARIMA if comparison failed
                if best_model_name == 'ETS':
                    model, future_forecast, forecast_index = build_ets_model(ts_data, company, variable)
                else:  # Default to SARIMA
                    model, future_forecast, forecast_index = build_sarima_model(ts_data, company, variable)
                
                # Store forecasts
                company_forecasts[variable] = {
                    'forecast_values': future_forecast,
                    'forecast_dates': forecast_index
                }
            else:
                print(f"Insufficient data points for {company} {variable} to create a reliable forecast")
        
        forecasts[company] = company_forecasts
    
    print("\nForecasting process completed!")
    return forecasts

# Call main function
if __name__ == "__main__":
    forecasts = main()