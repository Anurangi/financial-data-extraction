# Deployment Architecture for Financial Forecasting System

# Step 1: Package the model for deployment
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

class FinancialForecastModel:
    """
    Model wrapper class to standardize forecasting functionality
    """
    def __init__(self, model_type='sarima', period=4, company=None, target_variable=None):
        self.model_type = model_type
        self.period = period
        self.company = company
        self.target_variable = target_variable
        self.model = None
        self.last_date = None
        
    def fit(self, time_series_data):
        """Train the forecasting model"""
        # Store the last date for future forecasting
        self.last_date = time_series_data.index[-1]
        
        if self.model_type == 'sarima':
            # SARIMA model
            p, d, q = 1, 1, 1
            P, D, Q, s = 1, 1, 1, 4
            
            model = SARIMAX(time_series_data, 
                           order=(p, d, q), 
                           seasonal_order=(P, D, Q, s),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            
            self.model = model.fit(disp=False)
            
        elif self.model_type == 'ets':
            # Exponential Smoothing model
            model = ExponentialSmoothing(time_series_data, 
                                       seasonal_periods=4, 
                                       trend='add', 
                                       seasonal='add',
                                       use_boxcox=True)
            
            self.model = model.fit()
        
        return self
    
    def predict(self, periods=None):
        """Generate forecasts"""
        if periods is None:
            periods = self.period
            
        if self.model_type == 'sarima':
            forecast = self.model.get_forecast(steps=periods)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()
            
            # Create future date index
            import pandas as pd
            forecast_index = pd.date_range(
                start=self.last_date + pd.DateOffset(months=3), 
                periods=periods, 
                freq='3M'
            )
            
            # Create result DataFrame
            result = pd.DataFrame({
                'forecast': forecast_mean.values,
                'lower_ci': forecast_ci.iloc[:, 0].values,
                'upper_ci': forecast_ci.iloc[:, 1].values
            }, index=forecast_index)
            
        elif self.model_type == 'ets':
            forecast_values = self.model.forecast(periods)
            
            # Create future date index
            import pandas as pd
            forecast_index = pd.date_range(
                start=self.last_date + pd.DateOffset(months=3), 
                periods=periods, 
                freq='3M'
            )
            
            # Create result DataFrame (no confidence intervals for ETS)
            result = pd.DataFrame({
                'forecast': forecast_values.values
            }, index=forecast_index)
        
        return result
    
    def save(self, filename):
        """Save the model to disk"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'period': self.period,
            'company': self.company,
            'target_variable': self.target_variable,
            'last_date': self.last_date
        }
        joblib.dump(model_data, filename)
        
    @classmethod
    def load(cls, filename):
        """Load a model from disk"""
        model_data = joblib.load(filename)
        
        instance = cls(
            model_type=model_data['model_type'],
            period=model_data['period'],
            company=model_data['company'],
            target_variable=model_data['target_variable']
        )
        
        instance.model = model_data['model']
        instance.last_date = model_data['last_date']
        
        return instance


# Step 2: Model Training and Saving Script
def train_and_save_models(data_path, output_dir):
    """Train and save forecasting models for all companies and variables"""
    import os
    import pandas as pd
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = pd.read_csv(data_path)
    data['Quarter End Date'] = pd.to_datetime(data['Quarter End Date'])
    
    # Get unique companies and target variables
    companies = data['Company'].unique()
    target_variables = ['Revenue', 'Gross Profit', 'Profit for Period']
    
    # Dictionary to store model paths
    model_paths = {}
    
    for company in companies:
        company_data = data[data['Company'] == company].copy()
        company_data.set_index('Quarter End Date', inplace=True)
        company_data = company_data.sort_index()
        
        model_paths[company] = {}
        
        for variable in target_variables:
            # Get time series data
            ts_data = company_data[variable]
            
            if len(ts_data) >= 8:  # Ensure we have enough data points
                # Train SARIMA model
                sarima_model = FinancialForecastModel(
                    model_type='sarima',
                    period=4,
                    company=company,
                    target_variable=variable
                )
                sarima_model.fit(ts_data)
                
                # Train ETS model
                ets_model = FinancialForecastModel(
                    model_type='ets',
                    period=4,
                    company=company,
                    target_variable=variable
                )
                ets_model.fit(ts_data)
                
                # Save models
                sarima_path = os.path.join(output_dir, f"{company.replace(' ', '_')}_{variable}_sarima.joblib")
                ets_path = os.path.join(output_dir, f"{company.replace(' ', '_')}_{variable}_ets.joblib")
                
                sarima_model.save(sarima_path)
                ets_model.save(ets_path)
                
                model_paths[company][variable] = {
                    'sarima': sarima_path,
                    'ets': ets_path
                }
                
                print(f"Saved models for {company} - {variable}")
    
    return model_paths


# Step 3: Flask API for model serving
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load model mappings
MODEL_DIR = "models"
model_mappings = {}  # To be populated on startup

@app.route('/api/forecast', methods=['POST'])
def forecast():
    data = request.json
    company = data.get('company')
    variable = data.get('variable')
    model_type = data.get('model_type', 'sarima')
    periods = data.get('periods', 4)
    
    # Validate inputs
    if not company or not variable:
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Check if model exists
    if company not in model_mappings or variable not in model_mappings[company]:
        return jsonify({"error": f"No model found for {company} - {variable}"}), 404
    
    if model_type not in model_mappings[company][variable]:
        return jsonify({"error": f"Model type {model_type} not available"}), 404
    
    # Load model
    model_path = model_mappings[company][variable][model_type]
    model = FinancialForecastModel.load(model_path)
    
    # Generate forecast
    forecast_result = model.predict(periods)
    
    # Convert to JSON-serializable format
    forecast_data = {
        'dates': forecast_result.index.strftime('%Y-%m-%d').tolist(),
        'forecast': forecast_result['forecast'].tolist(),
        'company': company,
        'variable': variable,
        'model_type': model_type
    }
    
    # Add confidence intervals if available
    if 'lower_ci' in forecast_result.columns and 'upper_ci' in forecast_result.columns:
        forecast_data['lower_ci'] = forecast_result['lower_ci'].tolist()
        forecast_data['upper_ci'] = forecast_result['upper_ci'].tolist()
    
    return jsonify(forecast_data)

@app.route('/api/companies', methods=['GET'])
def get_companies():
    return jsonify(list(model_mappings.keys()))

@app.route('/api/variables', methods=['GET'])
def get_variables():
    company = request.args.get('company')
    if not company or company not in model_mappings:
        return jsonify([])
    return jsonify(list(model_mappings[company].keys()))

@app.route('/api/model_types', methods=['GET'])
def get_model_types():
    company = request.args.get('company')
    variable = request.args.get('variable')
    
    if not company or not variable:
        return jsonify([])
    
    if company in model_mappings and variable in model_mappings[company]:
        return jsonify(list(model_mappings[company][variable].keys()))
    
    return jsonify([])

def load_model_mappings():
    """Load available models on startup"""
    import os
    import glob
    
    global model_mappings
    model_mappings = {}
    
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.joblib"))
    
    for model_file in model_files:
        filename = os.path.basename(model_file)
        parts = filename.split('_')
        
        if len(parts) >= 3:
            company_parts = parts[:-2]  # Company name might have underscores
            company = ' '.join(company_parts)
            variable = parts[-2]
            model_type = parts[-1].replace('.joblib', '')
            
            if company not in model_mappings:
                model_mappings[company] = {}
            
            if variable not in model_mappings[company]:
                model_mappings[company][variable] = {}
            
            model_mappings[company][variable][model_type] = model_file

if __name__ == '__main__':
    # Load model mappings on startup
    load_model_mappings()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)


# Step 4: Dashboard Frontend (Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from datetime import datetime

def load_data():
    """Load the original data for historical context"""
    data = pd.read_csv("unified_financial_data.csv")
    data['Quarter End Date'] = pd.to_datetime(data['Quarter End Date'])
    return data

def get_companies():
    """Get list of companies from the API"""
    response = requests.get("http://localhost:5000/api/companies")
    return response.json()

def get_variables(company):
    """Get list of variables for a company from the API"""
    response = requests.get(f"http://localhost:5000/api/variables?company={company}")
    return response.json()

def get_model_types(company, variable):
    """Get list of model types available for a company/variable from the API"""
    response = requests.get(f"http://localhost:5000/api/model_types?company={company}&variable={variable}")
    return response.json()

def get_forecast(company, variable, model_type, periods):
    """Get forecast from the API"""
    payload = {
        "company": company,
        "variable": variable,
        "model_type": model_type,
        "periods": periods
    }
    response = requests.post("http://localhost:5000/api/forecast", json=payload)
    return response.json()

def main():
    st.title("Financial Forecasting Dashboard")
    
    # Load the historical data
    data = load_data()
    
    # Sidebar for user inputs
    st.sidebar.header("Forecast Parameters")
    
    # Get list of companies from API
    companies = get_companies()
    company = st.sidebar.selectbox("Select Company", companies)
    
    if company:
        # Get variables for selected company
        variables = get_variables(company)
        variable = st.sidebar.selectbox("Select Financial Metric", variables)
        
        if variable:
            # Get model types for selected company/variable
            model_types = get_model_types(company, variable)
            model_type = st.sidebar.selectbox("Select Model Type", model_types)
            
            # Other parameters
            periods = st.sidebar.slider("Forecast Periods (Quarters)", 1, 8, 4)
            
            # Filter historical data for the selected company
            company_data = data[data['Company'] == company].copy()
            company_data = company_data.sort_values('Quarter End Date')
            
            # Display historical data
            st.header(f"Historical {variable} for {company}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(company_data['Quarter End Date'], company_data[variable], marker='o')
            ax.set_xlabel('Date')
            ax.set_ylabel(variable)
            ax.grid(True)
            st.pyplot(fig)
            
            # Display forecast if all parameters are selected
            if model_type:
                st.header(f"{variable} Forecast")
                
                with st.spinner("Generating forecast..."):
                    forecast_data = get_forecast(company, variable, model_type, periods)
                
                if 'error' in forecast_data:
                    st.error(f"Error: {forecast_data['error']}")
                else:
                    # Create forecast dataframe
                    forecast_df = pd.DataFrame({
                        'Date': pd.to_datetime(forecast_data['dates']),
                        'Forecast': forecast_data['forecast']
                    })
                    
                    # Check if we have confidence intervals
                    has_ci = 'lower_ci' in forecast_data and 'upper_ci' in forecast_data
                    
                    if has_ci:
                        forecast_df['Lower CI'] = forecast_data['lower_ci']
                        forecast_df['Upper CI'] = forecast_data['upper_ci']
                    
                    # Display forecast table
                    st.subheader("Forecast Values")
                    st.dataframe(forecast_df)
                    
                    # Plot forecast
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot historical data
                    ax.plot(company_data['Quarter End Date'], company_data[variable], 
                            marker='o', label='Historical')
                    
                    # Plot forecast
                    ax.plot(forecast_df['Date'], forecast_df['Forecast'], 
                            marker='o', linestyle='--', color='red', label='Forecast')
                    
                    # Add confidence intervals if available
                    if has_ci:
                        ax.fill_between(forecast_df['Date'], 
                                        forecast_df['Lower CI'], 
                                        forecast_df['Upper CI'],
                                        color='red', alpha=0.2, label='95% Confidence Interval')
                    
                    ax.set_xlabel('Date')
                    ax.set_ylabel(variable)
                    ax.legend()
                    ax.grid(True)
                    
                    st.pyplot(fig)

if __name__ == "__main__":
    main()


# Step 5: Docker Deployment
# Dockerfile for Flask API
"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY api.py .

EXPOSE 5000

CMD ["python", "api.py"]
"""

# Dockerfile for Streamlit Dashboard
"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard.py .
COPY unified_financial_data.csv .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py"]
"""

# Docker Compose file
"""
version: '3'

services:
  model-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    restart: always

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8501:8501"
    depends_on:
      - model-api
    environment:
      - API_URL=http://model-api:5000
    restart: always
"""

# Step 6: Cloud Deployment (AWS)
# AWS Elastic Beanstalk or AWS ECS could be used for deployment
"""
aws ecr create-repository --repository-name financial-forecasting-api
aws ecr create-repository --repository-name financial-forecasting-dashboard

# Build and push Docker images
docker build -f Dockerfile.api -t financial-forecasting-api .
docker build -f Dockerfile.dashboard -t financial-forecasting-dashboard .

docker tag financial-forecasting-api:latest [AWS_ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/financial-forecasting-api:latest
docker tag financial-forecasting-dashboard:latest [AWS_ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/financial-forecasting-dashboard:latest

aws ecr get-login-password | docker login --username AWS --password-stdin [AWS_ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com

docker push [AWS_ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/financial-forecasting-api:latest
docker push [AWS_ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/financial-forecasting-dashboard:latest

# Create ECS cluster, tasks, and services using AWS CLI or CloudFormation
"""

# Step 7: CI/CD Pipeline (GitHub Actions)
"""
name: Deploy Financial Forecasting System

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push API image to Amazon ECR
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
      run: |
        docker build -f Dockerfile.api -t $ECR_REGISTRY/financial-forecasting-api:${{ github.sha }} .
        docker push $ECR_REGISTRY/financial-forecasting-api:${{ github.sha }}
        docker tag $ECR_REGISTRY/financial-forecasting-api:${{ github.sha }} $ECR_REGISTRY/financial-forecasting-api:latest
        docker push $ECR_REGISTRY/financial-forecasting-api:latest
    
    - name: Build and push Dashboard image to Amazon ECR
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
      run: |
        docker build -f Dockerfile.dashboard -t $ECR_REGISTRY/financial-forecasting-dashboard:${{ github.sha }} .
        docker push $ECR_REGISTRY/financial-forecasting-dashboard:${{ github.sha }}
        docker tag $ECR_REGISTRY/financial-forecasting-dashboard:${{ github.sha }} $ECR_REGISTRY/financial-forecasting-dashboard:latest
        docker push $ECR_REGISTRY/financial-forecasting-dashboard:latest
    
    - name: Update ECS services
      run: |
        aws ecs update-service --cluster financial-forecasting --service api-service --force-new-deployment
        aws ecs update-service --cluster financial-forecasting --service dashboard-service --force-new-deployment
"""