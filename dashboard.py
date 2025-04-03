import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Load CSV file
df = pd.read_csv(r'C:\Users\Akshila.Anurangi\financial-data-extraction-pipeline\reports\unified_financial_data.csv')

# Convert 'Quarter End Date' to datetime format
df['Quarter End Date'] = pd.to_datetime(df['Quarter End Date'])

# Ensure data is sorted properly
df = df.sort_values('Quarter End Date')

# Calculate KPIs
total_revenue = df['Revenue'].sum()
average_profit = df['Profit for Period'].mean()
gross_profit_margin = (df['Gross Profit'].sum() / total_revenue) * 100 if total_revenue else 0

# Initialize Dash app
app = dash.Dash(__name__)

# Revenue Over Time (Improved)
revenue_fig = px.line(
    df, x='Quarter End Date', y='Revenue', color='Company',
    title='Revenue Over Time',
    template='plotly_dark', markers=True, line_shape='spline'
)

# Profit for Period Over Time (Now a Line Chart)
profit_fig = px.line(
    df, x='Quarter End Date', y='Profit for Period', color='Company',
    title='Profit for Period Over Time',
    template='plotly_dark', markers=True, line_shape='spline'
)

# Gross Profit Per Quarter (Bar Chart)
gross_profit_fig = px.bar(
    df, x='Quarter End Date', y='Gross Profit', color='Company',
    title='Gross Profit Per Quarter',
    template='plotly_dark'
)

# Layout with KPI Cards
app.layout = html.Div([
    html.H1("📊 Financial Dashboard", style={'text-align': 'center'}),

    # KPI Cards
    html.Div([
        html.Div([
            html.H3("💰 Total Revenue"),
            html.P(f"${total_revenue:,.2f}", style={'font-size': '20px', 'font-weight': 'bold'})
        ], style={'padding': '20px', 'border': '1px solid #ddd', 'border-radius': '10px', 'text-align': 'center', 'background-color': '#222', 'color': 'white'}),

        html.Div([
            html.H3("📈 Avg. Profit for Period"),
            html.P(f"${average_profit:,.2f}", style={'font-size': '20px', 'font-weight': 'bold'})
        ], style={'padding': '20px', 'border': '1px solid #ddd', 'border-radius': '10px', 'text-align': 'center', 'background-color': '#222', 'color': 'white'}),

        html.Div([
            html.H3("📊 Gross Profit Margin"),
            html.P(f"{gross_profit_margin:.2f}%", style={'font-size': '20px', 'font-weight': 'bold'})
        ], style={'padding': '20px', 'border': '1px solid #ddd', 'border-radius': '10px', 'text-align': 'center', 'background-color': '#222', 'color': 'white'}),
    ], style={'display': 'flex', 'justify-content': 'space-around', 'margin-bottom': '20px'}),

    # Graphs
    dcc.Graph(figure=revenue_fig),
    dcc.Graph(figure=profit_fig),
    dcc.Graph(figure=gross_profit_fig),
])

# Run app
if __name__ == '__main__':
    app.run(debug=True)
