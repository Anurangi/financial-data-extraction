import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc
import numpy as np
from datetime import datetime
import os
import logging
import sys

# Set up logging
logs_dir = r"C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

log_file_path = os.path.join(logs_dir, "dashboard.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.info("Dashboard application starting...")

try:
    # Load data
    data_path = r'C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\dataset_creator\data\unified_financial_data.csv'
    logger.info(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    data = pd.read_csv(data_path)
    logger.info(f"Data loaded successfully. Shape: {data.shape}")

    # Data preprocessing
    logger.info("Processing data...")
    data['Quarter End Date'] = pd.to_datetime(data['Quarter End Date'])
    data['Year'] = data['Quarter End Date'].dt.year
    data['Quarter'] = data['Quarter End Date'].dt.quarter
    data['Year-Quarter'] = data['Year'].astype(str) + '-Q' + data['Quarter'].astype(str)

    # Calculate additional metrics
    data['Gross Margin %'] = (data['Gross Profit'] / data['Revenue']) * 100
    data['Net Margin %'] = (data['Profit for Period'] / data['Revenue']) * 100
    data['Operating Margin %'] = ((data['Revenue'] + data['Other Income'] + data['Distribution Costs'] + 
                                 data['Administrative Expenses']) / data['Revenue']) * 100

    # Handle missing values
    data = data.fillna(0)
    logger.info("Data processing completed")

except Exception as e:
    logger.error(f"Error during data loading/processing: {str(e)}")
    # Continue with initialization but will show error in the app

# Initialize the Dash app
logger.info("Initializing Dash application...")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define available metrics for dropdown
metric_options = [
    {'label': 'Revenue', 'value': 'Revenue'},
    {'label': 'Gross Profit', 'value': 'Gross Profit'},
    {'label': 'Profit for Period', 'value': 'Profit for Period'},
    {'label': 'Gross Margin %', 'value': 'Gross Margin %'},
    {'label': 'Net Margin %', 'value': 'Net Margin %'},
    {'label': 'Operating Margin %', 'value': 'Operating Margin %'}
]

# Define layout
try:
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Financial Performance Dashboard", 
                          style={'textAlign': 'center', 'color': '#2C3E50', 'marginTop': 20}), 
                  width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Filters", style={'marginTop': 10}),
                    html.Label("Select Company:"),
                    dcc.Dropdown(
                        id='company-filter',
                        options=[{'label': company, 'value': company} for company in data['Company'].unique()],
                        value=data['Company'].unique().tolist(),
                        multi=True
                    ),
                    html.Label("Select Year Range:"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=data['Year'].min(),
                        max=data['Year'].max(),
                        step=1,
                        marks={str(year): str(year) for year in range(data['Year'].min(), data['Year'].max()+1)},
                        value=[data['Year'].min(), data['Year'].max()]
                    ),
                    html.Label("Select Metric:"),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=metric_options,
                        value='Revenue',
                        clearable=False
                    ),
                    html.Label("Comparison Mode:"),
                    dcc.RadioItems(
                        id='comparison-mode',
                        options=[
                            {'label': 'Quarter over Quarter', 'value': 'qoq'},
                            {'label': 'Year over Year', 'value': 'yoy'}
                        ],
                        value='qoq',
                        labelStyle={'display': 'block'}
                    )
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Key Performance Indicators", className="card-title")),
                    dbc.CardBody(id='kpi-cards')
                ], style={'marginBottom': '20px'}),
                
                dbc.Card([
                    dbc.CardHeader(html.H4("Quarterly Trend Analysis", className="card-title")),
                    dbc.CardBody([
                        dcc.Graph(id='quarterly-trend')
                    ])
                ], style={'marginBottom': '20px'})
            ], width=9)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Company Comparison", className="card-title")),
                    dbc.CardBody([
                        dcc.Graph(id='company-comparison')
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Profitability Analysis", className="card-title")),
                    dbc.CardBody([
                        dcc.Graph(id='profitability-analysis')
                    ])
                ])
            ], width=6)
        ], style={'marginTop': '20px'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Financial Data Table", className="card-title")),
                    dbc.CardBody([
                        html.Div(id='data-table-container')
                    ])
                ])
            ], width=12)
        ], style={'marginTop': '20px', 'marginBottom': '40px'})
        
    ], fluid=True)
    
    logger.info("Dashboard layout created successfully")
    
except Exception as e:
    logger.error(f"Error creating dashboard layout: {str(e)}")
    # Create a minimal layout to show the error
    app.layout = dbc.Container([
        html.H1("Error Initializing Dashboard"),
        html.P(f"An error occurred: {str(e)}"),
        html.P("Please check the logs for details.")
    ])

# Define callbacks
@app.callback(
    [Output('kpi-cards', 'children'),
     Output('quarterly-trend', 'figure'),
     Output('company-comparison', 'figure'),
     Output('profitability-analysis', 'figure'),
     Output('data-table-container', 'children')],
    [Input('company-filter', 'value'),
     Input('year-slider', 'value'),
     Input('metric-dropdown', 'value'),
     Input('comparison-mode', 'value')]
)
def update_dashboard(companies, year_range, selected_metric, comparison_mode):
    logger.info(f"Updating dashboard with: companies={companies}, year_range={year_range}, metric={selected_metric}, mode={comparison_mode}")
    
    try:
        if not isinstance(companies, list):
            companies = [companies]
            
        # Filter data based on selections
        filtered_data = data[
            (data['Company'].isin(companies)) & 
            (data['Year'] >= year_range[0]) & 
            (data['Year'] <= year_range[1])
        ]
        
        logger.debug(f"Filtered data shape: {filtered_data.shape}")
        
        # Generate KPI cards
        latest_year = filtered_data['Year'].max() if not filtered_data.empty else 0
        latest_quarter = filtered_data[filtered_data['Year'] == latest_year]['Quarter'].max() if latest_year > 0 else 0
        
        latest_data = filtered_data[
            (filtered_data['Year'] == latest_year) & 
            (filtered_data['Quarter'] == latest_quarter)
        ] if latest_year > 0 and latest_quarter > 0 else pd.DataFrame()
        
        kpi_cards = []
        for company in companies:
            company_latest = latest_data[latest_data['Company'] == company]
            if not company_latest.empty:
                # Get previous year data for comparison
                prev_year_data = data[
                    (data['Company'] == company) & 
                    (data['Year'] == latest_year - 1) & 
                    (data['Quarter'] == latest_quarter)
                ]
                
                revenue_current = company_latest['Revenue'].values[0]
                gross_profit_current = company_latest['Gross Profit'].values[0]
                net_profit_current = company_latest['Profit for Period'].values[0]
                
                revenue_previous = prev_year_data['Revenue'].values[0] if not prev_year_data.empty else 0
                revenue_change = ((revenue_current - revenue_previous) / revenue_previous * 100) if revenue_previous != 0 else 0
                
                card = dbc.Card([
                    dbc.CardBody([
                        html.H5(company, className="card-title"),
                        html.Div([
                            html.Div([
                                html.P("Revenue", className="card-text"),
                                html.H3(f"{revenue_current:,.0f}", className="card-text"),
                                html.P(f"{revenue_change:.1f}% YoY", 
                                     style={'color': 'green' if revenue_change > 0 else 'red'})
                            ], style={'width': '33%', 'display': 'inline-block'}),
                            html.Div([
                                html.P("Gross Profit", className="card-text"),
                                html.H3(f"{gross_profit_current:,.0f}", className="card-text"),
                            ], style={'width': '33%', 'display': 'inline-block'}),
                            html.Div([
                                html.P("Net Profit", className="card-text"),
                                html.H3(f"{net_profit_current:,.0f}", className="card-text"),
                            ], style={'width': '33%', 'display': 'inline-block'})
                        ])
                    ])
                ], style={'marginBottom': '10px'})
                
                kpi_cards.append(card)
        
        # Quarterly trend analysis
        if comparison_mode == 'qoq':
            quarterly_df = filtered_data.sort_values(['Company', 'Year', 'Quarter'])
            fig_trend = px.line(quarterly_df, x='Year-Quarter', y=selected_metric, color='Company',
                             title=f'Quarterly {selected_metric} Trend',
                             labels={selected_metric: selected_metric, 'Year-Quarter': 'Year-Quarter'},
                             markers=True)
            
        else:  # year over year comparison
            # Pivot data to compare quarters across years
            pivot_df = filtered_data.pivot_table(
                index=['Company', 'Quarter'], 
                columns='Year', 
                values=selected_metric,
                aggfunc='sum'
            ).reset_index()
            
            fig_trend = make_subplots(rows=1, cols=1, shared_xaxes=True)
            
            for company in companies:
                company_data = pivot_df[pivot_df['Company'] == company]
                for year in range(year_range[0], year_range[1] + 1):
                    if year in company_data.columns:
                        fig_trend.add_trace(
                            go.Scatter(
                                x=company_data['Quarter'],
                                y=company_data[year],
                                mode='lines+markers',
                                name=f"{company} - {year}"
                            )
                        )
            
            fig_trend.update_layout(
                title=f'Year over Year {selected_metric} Comparison by Quarter',
                xaxis_title='Quarter',
                yaxis_title=selected_metric,
                legend_title='Company - Year'
            )
        
        # Company comparison bar chart
        comparison_df = filtered_data.groupby(['Company', 'Year'])[selected_metric].sum().reset_index()
        fig_comparison = px.bar(comparison_df, x='Year', y=selected_metric, color='Company', barmode='group',
                             title=f'Annual {selected_metric} Comparison',
                             labels={selected_metric: selected_metric, 'Year': 'Year'})
        
        # Profitability analysis
        profitability_df = filtered_data.copy()
        if 'Margin' not in selected_metric:  # Only create this plot for non-margin metrics
            profitability_metrics = ['Gross Margin %', 'Net Margin %', 'Operating Margin %']
            profitability_df = profitability_df.melt(
                id_vars=['Company', 'Year', 'Quarter', 'Year-Quarter'],
                value_vars=profitability_metrics,
                var_name='Margin Type',
                value_name='Margin Percentage'
            )
            fig_profitability = px.line(profitability_df, x='Year-Quarter', y='Margin Percentage', 
                                     color='Company', line_dash='Margin Type',
                                     title='Profitability Margin Trends',
                                     labels={'Margin Percentage': 'Percentage (%)', 'Year-Quarter': 'Year-Quarter'})
        else:
            # If a margin metric is selected in the main trend, show expense breakdown instead
            expense_df = filtered_data.copy()
            expense_cols = ['Cost of Sales', 'Distribution Costs', 'Administrative Expenses', 'Other Expenses']
            # Convert to absolute values for better visualization
            for col in expense_cols:
                expense_df[col] = expense_df[col].abs()
                
            expense_df = expense_df.melt(
                id_vars=['Company', 'Year', 'Quarter', 'Year-Quarter'],
                value_vars=expense_cols,
                var_name='Expense Type',
                value_name='Amount'
            )
            fig_profitability = px.area(expense_df, x='Year-Quarter', y='Amount', color='Expense Type',
                                     facet_row='Company', 
                                     title='Expense Breakdown Analysis',
                                     labels={'Amount': 'Amount', 'Year-Quarter': 'Year-Quarter'})
        
        # Data table
        table_data = filtered_data[['Company', 'Year', 'Quarter', 'Revenue', 'Gross Profit', 
                                  'Profit for Period', 'Gross Margin %', 'Net Margin %']]
        table = DataTable(
            columns=[{"name": i, "id": i} for i in table_data.columns],
            data=table_data.to_dict('records'),
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '10px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
        
        logger.info("Dashboard update completed successfully")
        return kpi_cards, fig_trend, fig_comparison, fig_profitability, table
        
    except Exception as e:
        logger.error(f"Error updating dashboard: {str(e)}", exc_info=True)
        # Return empty placeholders to avoid breaking the app
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error loading data",
            annotations=[{
                "text": f"An error occurred: {str(e)}",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return [], empty_fig, empty_fig, empty_fig, html.Div("Error loading data table")

if __name__ == '__main__':
    logger.info("Starting Dash server...")
    app.run(debug=True)
    logger.info("Dash server stopped")