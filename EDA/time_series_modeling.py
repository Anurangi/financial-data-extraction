import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("Set2")

# File path and output directory
file_path = r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\dataset_creator\data\unified_financial_data.csv"
output_dir = r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\EDA\Forecasting"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to save figures
def save_fig(fig, filename):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Figure saved to {filepath}")

# Function to save model results
def save_results(text, filename):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        f.write(text)
    print(f"Results saved to {filepath}")

# Load the data
print("Loading financial data...")
df = pd.read_csv(file_path)

# Convert 'Quarter End Date' to datetime
df['Quarter End Date'] = pd.to_datetime(df['Quarter End Date'])

# Ensure 'Gross Profit' is numeric
df['Gross Profit'] = pd.to_numeric(df['Gross Profit'], errors='coerce')

# Create 'Quarter' column with YYYY-Q format for easier reference
df['Quarter'] = df['Quarter End Date'].dt.to_period('Q')

# Filter out rows with missing Gross Profit values
df = df.dropna(subset=['Gross Profit'])

# ---------------------- Time Series Preprocessing ----------------------
print("\n===== Time Series Preprocessing =====")

# Create separate time series for each company
companies = df['Company'].unique()
time_series_data = {}

for company in companies:
    # Filter data for this company
    company_data = df[df['Company'] == company].copy()
    
    # Sort by date
    company_data = company_data.sort_values('Quarter End Date')
    
    # Create a time series with quarterly frequency
    ts = company_data.set_index('Quarter End Date')['Gross Profit']
    
    # Ensure the index is of DatetimeIndex type with quarterly frequency
    if len(ts) > 1:
        ts.index = pd.DatetimeIndex(ts.index.values, freq=pd.infer_freq(ts.index))
    
    # Store in the dictionary
    time_series_data[company] = ts

# ---------------------- Exploratory Analysis ----------------------
print("\n===== Exploratory Time Series Analysis =====")

# Visualize the time series
fig, ax = plt.subplots(figsize=(15, 8))
for company, ts in time_series_data.items():
    ts.plot(ax=ax, marker='o', label=company)

ax.set_title('Gross Profit Over Time by Company')
ax.set_xlabel('Date')
ax.set_ylabel('Gross Profit')
ax.grid(True, alpha=0.3)
ax.legend()
save_fig(fig, "01_gross_profit_time_series.png")

# ---------------------- Seasonal Decomposition ----------------------
print("\n===== Seasonal Decomposition =====")

for company, ts in time_series_data.items():
    # Check if we have enough data points for seasonal decomposition
    if len(ts) >= 4:  # Need at least 4 points for quarterly data
        try:
            # Decompose the time series
            decomposition = seasonal_decompose(ts, model='additive', period=4)
            
            # Plot the decomposition
            fig = plt.figure(figsize=(14, 10))
            
            # Trend
            ax1 = plt.subplot(411)
            ax1.plot(decomposition.trend)
            ax1.set_title('Trend')
            ax1.grid(True, alpha=0.3)
            
            # Seasonal
            ax2 = plt.subplot(412)
            ax2.plot(decomposition.seasonal)
            ax2.set_title('Seasonality')
            ax2.grid(True, alpha=0.3)
            
            # Residual
            ax3 = plt.subplot(413)
            ax3.plot(decomposition.resid)
            ax3.set_title('Residuals')
            ax3.grid(True, alpha=0.3)
            
            # Original
            ax4 = plt.subplot(414)
            ax4.plot(ts)
            ax4.set_title('Original Time Series')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_fig(fig, f"02_{company.replace(' ', '_')}_seasonal_decomposition.png")
            
            # ACF and PACF plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            plot_acf(ts, lags=min(20, len(ts)-1), ax=ax1)
            plot_pacf(ts, lags=min(20, len(ts)-1), ax=ax2)
            plt.tight_layout()
            save_fig(fig, f"03_{company.replace(' ', '_')}_acf_pacf.png")
            
        except Exception as e:
            print(f"Could not perform seasonal decomposition for {company}: {e}")
    else:
        print(f"Not enough data points for seasonal decomposition for {company}")

# ---------------------- ARIMA Model (instead of Auto ARIMA) ----------------------
print("\n===== ARIMA Model =====")

# Results dictionary
model_results = {}

# Common ARIMA orders to try based on financial data
arima_orders = [(1,1,1), (2,1,2), (1,1,2), (2,1,1)]

for company, ts in time_series_data.items():
    if len(ts) >= 8:  # Need sufficient data for training and testing
        print(f"\nFitting ARIMA models for {company}...")
        
        # Split into train and test sets
        train_size = int(len(ts) * 0.8)
        train, test = ts[:train_size], ts[train_size:]
        
        best_rmse = float('inf')
        best_order = None
        best_model = None
        
        # Try different ARIMA orders and select the best one
        for order in arima_orders:
            try:
                # Fit the model
                model = ARIMA(train, order=order)
                fitted_model = model.fit()
                
                # Make forecasts for test period
                forecasts = fitted_model.forecast(steps=len(test))
                
                # Calculate error
                rmse = np.sqrt(mean_squared_error(test, forecasts))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = order
                    best_model = fitted_model
                    
                print(f"  ARIMA{order} RMSE: {rmse:.2f}")
                
            except Exception as e:
                print(f"  Error fitting ARIMA{order}: {e}")
        
        if best_model is not None:
            # Fit the best model on the full dataset
            full_model = ARIMA(ts, order=best_order).fit()
            
            # Make forecasts
            forecast_periods = 4  # Forecast 4 quarters ahead
            forecasts = full_model.forecast(steps=forecast_periods)
            forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_periods+1, freq='Q')[1:]
            
            # Calculate error metrics on test set
            test_forecasts = best_model.forecast(steps=len(test))
            mse = mean_squared_error(test, test_forecasts)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test, test_forecasts)
            
            # Store results
            model_results[company] = {
                'ARIMA': {
                    'model': f"ARIMA{best_order}",
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'AIC': best_model.aic,
                    'BIC': best_model.bic
                }
            }
            
            # Plot the results
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot actual data
            ts.plot(ax=ax, label='Actual', marker='o')
            
            # Plot train/test split
            ax.axvline(train.index[-1], color='k', linestyle='--', alpha=0.5)
            
            # Plot in-sample predictions for training data
            in_sample_pred = best_model.fittedvalues
            ax.plot(train.index, in_sample_pred, color='green', label='In-sample Predictions')
            
            # Plot test forecasts
            ax.plot(test.index, test_forecasts, color='red', label='Test Forecasts', marker='x')
            
            # Plot future forecasts
            ax.plot(forecast_index, forecasts, color='purple', label='Future Forecasts', marker='x', linestyle='--')
            
            ax.set_title(f'Gross Profit ARIMA Forecast for {company}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Gross Profit')
            ax.grid(True, alpha=0.3)
            ax.legend()
            save_fig(fig, f"04_{company.replace(' ', '_')}_arima_forecast.png")
            
        else:
            print(f"Could not find a suitable ARIMA model for {company}")
    else:
        print(f"Not enough data points for ARIMA modeling for {company}")

# ---------------------- SARIMA Model ----------------------
print("\n===== SARIMA Model =====")

for company, ts in time_series_data.items():
    if len(ts) >= 8:  # Need sufficient data for training and testing
        try:
            print(f"\nFitting SARIMA model for {company}...")
            
            # Split into train and test sets
            train_size = int(len(ts) * 0.8)
            train, test = ts[:train_size], ts[train_size:]
            
            # Define SARIMA model parameters
            # These are common parameters for quarterly data
            order = (1, 1, 1)  # (p, d, q)
            seasonal_order = (1, 1, 1, 4)  # (P, D, Q, s)
            
            # Fit the model
            sarima = SARIMAX(train, 
                            order=order, 
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            sarima_result = sarima.fit(disp=False)
            
            # Print model summary
            print(f"SARIMA model summary for {company}:")
            print(sarima_result.summary())
            
            # Make forecasts
            forecast_periods = min(8, len(test) + 4)  # Forecast test period + 4 more quarters
            sarima_forecast = sarima_result.get_forecast(steps=forecast_periods)
            mean_forecast = sarima_forecast.predicted_mean
            conf_int = sarima_forecast.conf_int()
            
            # Create proper forecast index
            forecast_index = pd.date_range(start=train.index[-1], periods=forecast_periods+1, freq='Q')[1:]
            mean_forecast.index = forecast_index
            conf_int.index = forecast_index
            
            # Calculate error metrics on test set
            if len(test) > 0:
                test_forecast = mean_forecast[:len(test)]
                mse = mean_squared_error(test, test_forecast)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test, test_forecast)
                
                # Store results
                if company not in model_results:
                    model_results[company] = {}
                
                model_results[company]['SARIMA'] = {
                    'model': f"SARIMA{order}x{seasonal_order}",
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'AIC': sarima_result.aic,
                    'BIC': sarima_result.bic
                }
                
                # Plot the results
                fig, ax = plt.subplots(figsize=(15, 8))
                
                # Plot actual data
                ts.plot(ax=ax, label='Actual', marker='o')
                
                # Plot train/test split
                ax.axvline(train.index[-1], color='k', linestyle='--', alpha=0.5)
                
                # Plot forecasts
                mean_forecast.plot(ax=ax, label='SARIMA Forecast', style='--', marker='x')
                
                # Add confidence intervals
                ax.fill_between(conf_int.index, 
                               conf_int.iloc[:, 0], 
                               conf_int.iloc[:, 1], 
                               color='k', alpha=0.1)
                
                ax.set_title(f'Gross Profit SARIMA Forecast for {company}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Gross Profit')
                ax.grid(True, alpha=0.3)
                ax.legend()
                save_fig(fig, f"05_{company.replace(' ', '_')}_sarima_forecast.png")
            else:
                print(f"Test set is empty for {company}, skipping error calculation")
        except Exception as e:
            print(f"Error fitting SARIMA model for {company}: {e}")
    else:
        print(f"Not enough data points for SARIMA modeling for {company}")

# ---------------------- Simple Exponential Smoothing ----------------------
print("\n===== Exponential Smoothing =====")

from statsmodels.tsa.holtwinters import ExponentialSmoothing

for company, ts in time_series_data.items():
    if len(ts) >= 8:  # Need sufficient data for training and testing
        try:
            print(f"\nFitting Exponential Smoothing model for {company}...")
            
            # Split into train and test sets
            train_size = int(len(ts) * 0.8)
            train, test = ts[:train_size], ts[train_size:]
            
            # Fit Holt-Winters Exponential Smoothing model
            hw_model = ExponentialSmoothing(
                train,
                seasonal_periods=4,  # Quarterly data
                trend='add',
                seasonal='add',
                use_boxcox=True,
                initialization_method='estimated'
            )
            
            hw_fit = hw_model.fit(optimized=True, remove_bias=True)
            
            # Make forecasts
            forecast_periods = min(8, len(test) + 4)  # Forecast test period + 4 more quarters
            hw_forecast = hw_fit.forecast(forecast_periods)
            
            # Calculate error metrics on test set
            if len(test) > 0:
                test_forecast = hw_forecast[:len(test)]
                mse = mean_squared_error(test, test_forecast)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test, test_forecast)
                
                # Store results
                if company not in model_results:
                    model_results[company] = {}
                
                model_results[company]['ExponentialSmoothing'] = {
                    'model': "Holt-Winters Exponential Smoothing",
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'AIC': hw_fit.aic,
                    'BIC': hw_fit.bic
                }
                
                # Plot the results
                fig, ax = plt.subplots(figsize=(15, 8))
                
                # Plot actual data
                ts.plot(ax=ax, label='Actual', marker='o')
                
                # Plot train/test split
                ax.axvline(train.index[-1], color='k', linestyle='--', alpha=0.5)
                
                # Plot forecasts
                hw_forecast.plot(ax=ax, label='Exponential Smoothing Forecast', style='--', marker='x')
                
                ax.set_title(f'Gross Profit Exponential Smoothing Forecast for {company}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Gross Profit')
                ax.grid(True, alpha=0.3)
                ax.legend()
                save_fig(fig, f"06_{company.replace(' ', '_')}_exponential_smoothing_forecast.png")
            else:
                print(f"Test set is empty for {company}, skipping error calculation")
        except Exception as e:
            print(f"Error fitting Exponential Smoothing model for {company}: {e}")
    else:
        print(f"Not enough data points for Exponential Smoothing for {company}")

# ---------------------- Model Comparison ----------------------
print("\n===== Model Comparison =====")

# Prepare model comparison table
comparison_text = "# Gross Profit Forecasting Model Comparison\n\n"

for company, models in model_results.items():
    comparison_text += f"## {company}\n\n"
    comparison_text += "| Model | MSE | RMSE | MAE | AIC | BIC |\n"
    comparison_text += "|-------|-----|------|-----|-----|-----|\n"
    
    for model_name, metrics in models.items():
        if isinstance(metrics, dict):  # Handle nested dictionaries for SARIMA
            comparison_text += f"| {metrics['model']} | {metrics['MSE']:.2f} | {metrics['RMSE']:.2f} | {metrics['MAE']:.2f} | {metrics.get('AIC', 'N/A')} | {metrics.get('BIC', 'N/A')} |\n"
        else:
            comparison_text += f"| {model_name} | {metrics:.2f} | N/A | N/A | N/A | N/A |\n"
    
    comparison_text += "\n"

save_results(comparison_text, "07_model_comparison.md")

# ---------------------- Forecast Summary ----------------------
print("\n===== Forecast Summary =====")

# Create a summary of the forecasts
summary_text = "# Gross Profit Forecast Summary\n\n"

for company, models in model_results.items():
    best_model = min(models.items(), key=lambda x: x[1]['RMSE'] if isinstance(x[1], dict) and 'RMSE' in x[1] else float('inf'))
    best_model_name = best_model[0]
    
    if isinstance(best_model[1], dict):
        metrics = best_model[1]
        summary_text += f"## {company}\n\n"
        summary_text += f"- Best model: **{metrics['model']}**\n"
        summary_text += f"- RMSE: {metrics['RMSE']:.2f}\n"
        summary_text += f"- MAE: {metrics['MAE']:.2f}\n\n"
        summary_text += "### Key Findings\n\n"
        summary_text += f"- The {metrics['model']} model performed best for {company} based on error metrics.\n"
        
        # Add insights based on the data trends
        ts = time_series_data[company]
        avg_growth = ts.pct_change().mean() * 100
        
        if avg_growth > 0:
            summary_text += f"- The company shows an average growth of {avg_growth:.2f}% quarter-over-quarter in Gross Profit.\n"
        else:
            summary_text += f"- The company shows an average decline of {abs(avg_growth):.2f}% quarter-over-quarter in Gross Profit.\n"
        
        # Check for seasonality
        if len(ts) >= 8:
            try:
                # Calculate quarterly averages
                ts.index = pd.PeriodIndex(ts.index, freq='Q')
                quarterly_avg = ts.groupby(ts.index.quarter).mean()
                max_quarter = quarterly_avg.idxmax()
                min_quarter = quarterly_avg.idxmin()
                
                summary_text += f"- Q{max_quarter} typically shows the highest Gross Profit, while Q{min_quarter} shows the lowest.\n"
                summary_text += f"- The difference between the highest and lowest quarterly averages is {(quarterly_avg.max() - quarterly_avg.min()):.2f}.\n"
            except Exception as e:
                print(f"Error calculating quarterly averages for {company}: {e}")
        
        summary_text += "\n### Recommendations\n\n"
        summary_text += "- Continue monitoring the accuracy of forecasts against actual results.\n"
        summary_text += "- Consider updating the models as new quarterly data becomes available.\n"
        summary_text += "- Use these forecasts for budgeting and financial planning purposes.\n\n"

save_results(summary_text, "08_forecast_summary.md")

print("\nForecasting analysis completed. All results have been saved to:", output_dir)