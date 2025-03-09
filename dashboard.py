# IMPORTS
import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update
import pandas as pd
import numpy as np
import time
import base64
import io
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go

# Custom Modules (Ensure these files exist)
from data_processing import generate_blade_data
from charts import x_bar_r_chart, cp_cpk_chart, pareto_chart, variability_boxplot, imr_chart, p_chart
from taguchi_optimization import generate_taguchi_doe

# Initialize the App
app = Dash(__name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
           suppress_callback_exceptions=True)

# Layout
app.layout = html.Div([
    # HEADER
    html.Div([
        html.H1("AEROSPACE SQC DASHBOARD", style={"color": "white", "textAlign": "center", "marginBottom": "0"}),
        html.P("Advanced Process Monitoring and Optimization", style={"color": "#D3D3D3", "textAlign": "center", "marginTop": "0"})
    ], style={"backgroundColor": "#007BFF", "padding": "20px"}),

    # NAVIGATION BAR
    dcc.Tabs(id="tabs", value='tab-data-summary', children=[
        dcc.Tab(label='Data & Upload', value='tab-data-summary'),
        dcc.Tab(label='Control Charts', value='tab-control-charts'),
        dcc.Tab(label='Process Capability', value='tab-capability'),
        dcc.Tab(label='DOE Optimization', value='tab-doe'),
        dcc.Tab(label='Defect Analysis', value='tab-defects'),
        dcc.Tab(label='Variability Analysis', value='tab-variability'),
        dcc.Tab(label='Summary Dashboard', value='tab-summary'),
        dcc.Tab(label='Process Improvement Suggestions', value='tab-suggestions'),
    ], style={"backgroundColor": "#F8F9FA", "color": "#007BFF", "fontSize": "16px"}),

    # DYNAMIC UPLOAD/EXPORT SECTION
    html.Div(id="file-upload-export-section"),

    # ALERTS AND METRICS
    html.Div(id="alert-notification", style={"color": "red", "textAlign": "center", "marginBottom": "20px"}),
    html.Div(id="performance-metrics", style={"color": "#00FF00", "textAlign": "center", "marginBottom": "20px"}),

    # TAB CONTENT
    html.Div(id='tabs-content', style={"padding": "20px"}),

    # INTERVAL COMPONENT FOR REAL-TIME UPDATES
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0),

    # DATA STORAGE
    dcc.Store(id='stored-data', storage_type='memory')
])

# CALLBACK TO SHOW FILE UPLOAD/EXPORT BUTTONS ONLY IN DATA TAB
@app.callback(
    Output("file-upload-export-section", "children"),
    Input("tabs", "value")
)
def show_file_upload_export(tab):
    if tab == 'tab-data-summary':
        return html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag & Drop or ', html.A('Select CSV', style={"color": "#007BFF"})]),
                style={
                    'width': '95%', 'height': '60px', 'borderWidth': '2px',
                    'borderStyle': 'dashed', 'borderColor': '#007BFF', 'margin': '20px auto'
                }
            ),
            html.Div([
                html.Button("Export Data", id="export-data-button", style={
                    "backgroundColor": "#007BFF", "color": "white", "padding": "10px", "width": "150px",
                    "border": "none", "cursor": "pointer", "marginTop": "10px"
                }),
                html.Button("Simulate Data", id="simulate-data-button", style={
                    "backgroundColor": "#FF4136", "color": "white", "padding": "10px", "width": "150px",
                    "border": "none", "cursor": "pointer", "marginTop": "10px"
                }),
                dcc.Download(id="download-data")
            ], style={"textAlign": "center"}),
        ])
    return None  # Hide for other tabs

# LOAD DATA CALLBACK (HANDLES CSV UPLOAD AND SIMULATED DATA)
@app.callback(
    Output('stored-data', 'data'),
    Output('alert-notification', 'children'),
    Input('upload-data', 'contents'),
    Input('simulate-data-button', 'n_clicks'),
    State('upload-data', 'filename'),
    State('stored-data', 'data')
)
def load_data(contents, n_clicks, filename, stored_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return stored_data, ""

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'upload-data' and contents:
        try:
            content_type, content = contents.split(',')  # Split base64 content
            decoded = base64.b64decode(content)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            # Validate required columns
            required_columns = {"Thickness (mm)", "Defects"}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            return df.to_json(), f"Uploaded: {filename}"
        except Exception as e:
            return stored_data, f"Error: Invalid CSV format ({str(e)})"

    elif trigger_id == 'simulate-data-button' and n_clicks:
        try:
            df = generate_blade_data()
            return df.to_json(), f"Simulated Data Loaded at {time.strftime('%H:%M:%S')}"
        except Exception as e:
            return stored_data, f"Error generating simulated data ({str(e)})"

    return stored_data, ""

# RENDER TAB CONTENT CALLBACK
@app.callback(
    Output('tabs-content', 'children'),
    Output('performance-metrics', 'children'),
    Input('tabs', 'value'),
    Input('stored-data', 'data')
)
def render_tab_content(tab, data_json):
    if data_json is None:
        return html.Div("No data available."), ""

    try:
        df = pd.read_json(data_json)
        
        # Validate required columns
        required_columns = {"Thickness (mm)", "Defects"}
        if not required_columns.issubset(df.columns):
            return html.Div(f"Error: Missing required columns {required_columns}"), ""

    except Exception as e:
        return html.Div(f"Error loading data: {str(e)}"), ""

    start_time = time.time()

    # Calculate Key Metrics
    mean_thickness = df["Thickness (mm)"].mean()
    std_thickness = df["Thickness (mm)"].std()
    defect_rate = df["Defects"].mean() * 100

    # Create a styled Key Metrics section
    key_metrics_section = html.Div([
        html.H3("Key Metrics", style={"color": "#007BFF", "textAlign": "center"}),
        html.Div([
            html.Div([
                html.H5("Mean Thickness", style={"textAlign": "center", "color": "#555"}),
                html.P(f"{mean_thickness:.3f} mm", style={"textAlign": "center", "fontSize": "20px", "fontWeight": "bold", "color": "#007BFF"})
            ], style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "5px", "margin": "10px", "boxShadow": "2px 2px 5px rgba(0,0,0,0.1)"}),
            
            html.Div([
                html.H5("Std Dev", style={"textAlign": "center", "color": "#555"}),
                html.P(f"{std_thickness:.3f} mm", style={"textAlign": "center", "fontSize": "20px", "fontWeight": "bold", "color": "#FFA500"})
            ], style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "5px", "margin": "10px", "boxShadow": "2px 2px 5px rgba(0,0,0,0.1)"}),
            
            html.Div([
                html.H5("Defect Rate", style={"textAlign": "center", "color": "#555"}),
                html.P(f"{defect_rate:.1f}%", style={"textAlign": "center", "fontSize": "20px", "fontWeight": "bold", "color": "#FF4136"})
            ], style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "5px", "margin": "10px", "boxShadow": "2px 2px 5px rgba(0,0,0,0.1)"}),
            
            html.Div([
                html.H5("Data Points", style={"textAlign": "center", "color": "#555"}),
                html.P(f"{len(df)}", style={"textAlign": "center", "fontSize": "20px", "fontWeight": "bold", "color": "#28A745"})
            ], style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "5px", "margin": "10px", "boxShadow": "2px 2px 5px rgba(0,0,0,0.1)"}),
        ], style={"display": "flex", "justifyContent": "space-around", "alignItems": "center", "flexWrap": "wrap"}),
    ])

    if tab == 'tab-data-summary':
        return (
            html.Div([
                key_metrics_section,
                html.H3("Raw Data & Metrics", style={"color": "#007BFF", "marginTop": "20px"}),
                dash_table.DataTable(
                    df.to_dict('records'),
                    [{"name": i, "id": i} for i in df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px', 'fontFamily': 'monospace'},
                    style_header={'backgroundColor': '#007BFF', 'color': 'white', 'fontWeight': 'bold'}
                ),
                html.Div([
                    html.H4("Additional Insights", style={"color": "#FFA500"}),
                    dcc.Markdown(f"""
                        - **Range**: `{df['Thickness (mm)'].max() - df['Thickness (mm)'].min():.2f} mm`
                        - **Variance**: `{df['Thickness (mm)'].var():.2f}`
                        - **Skewness**: `{df['Thickness (mm)'].skew():.2f}`
                    """),
                ]),
            ]),
            f"Loaded in {time.time() - start_time:.2f} seconds"
        )

    elif tab == 'tab-control-charts':
        x_bar_r_fig = x_bar_r_chart(df, "Thickness (mm)")
        imr_fig = imr_chart(df, "Thickness (mm)")
        p_chart_fig = p_chart(df, "Defects")

        mean = df["Thickness (mm)"].mean()
        std = df["Thickness (mm)"].std()
        ucl = mean + 3 * std
        lcl = mean - 3 * std

        for fig in [x_bar_r_fig, imr_fig]:
            fig.add_hline(y=ucl, line_color="red", annotation_text="UCL", line_dash="dot")
            fig.add_hline(y=lcl, line_color="red", annotation_text="LCL", line_dash="dot")

        return (
            html.Div([
                html.H3("Control Charts (Annotated)", style={"color": "#007BFF"}),
                dcc.Markdown("""
                    **Interpretation**:
                    - Control charts monitor process stability over time.
                    - Points outside UCL/LCL indicate out-of-control conditions.
                    - Formulas:
                      - UCL = Mean + 3σ
                      - LCL = Mean - 3σ
                    - Critical Values: ±3σ corresponds to 99.73% confidence level.
                    - Significance: Helps detect shifts, trends, or anomalies in the process.
                """),
                html.Div([
                    dcc.Graph(figure=x_bar_r_fig),
                    dcc.Graph(figure=imr_fig),
                    dcc.Graph(figure=p_chart_fig),
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "20px"}),
            ]),
            f"Computed in {time.time() - start_time:.2f} seconds"
        )

    elif tab == 'tab-capability':
        usl, lsl = 5.2, 4.8
        cp, cpk = calculate_cp_cpk(df, "Thickness (mm)", usl, lsl)
        cp_cpk_fig = cp_cpk_chart(df, "Thickness (mm)", usl, lsl)

        return (
            html.Div([
                html.H3("Process Capability (Cp/Cpk)", style={"color": "#007BFF"}),
                dcc.Markdown(f"""
                    **Interpretation**:
                    - Cp measures the process's potential capability.
                    - Cpk measures the actual capability, accounting for centering.
                    - Formulas:
                      - Cp = (USL - LSL) / (6σ)
                      - Cpk = min[(USL - μ)/(3σ), (μ - LSL)/(3σ)]
                    - Critical Values:
                      - Cp > 1.33: Process is capable.
                      - Cpk > 1.33: Process is centered and capable.
                    - Significance: Ensures the process meets specification limits consistently.
                """),
                dcc.Markdown(f"""
                    - **Cp**: `{cp:.2f}`
                    - **Cpk**: `{cpk:.2f}`
                    - **USL**: `{usl} mm`
                    - **LSL**: `{lsl} mm`
                """),
                dcc.Graph(figure=cp_cpk_fig),
            ]),
            f"Computed in {time.time() - start_time:.2f} seconds"
        )

    elif tab == 'tab-doe':
        doe_df = generate_taguchi_doe()
        return (
            html.Div([
                html.H3("DOE Optimization", style={"color": "#007BFF"}),
                dcc.Markdown("""
                    **Interpretation**:
                    - Design of Experiments (DOE) identifies optimal factor settings.
                    - Taguchi methods minimize variability and maximize robustness.
                    - Formulas:
                      - Signal-to-Noise Ratio (SNR): SNR = -10log₁₀(Σ(y²)/n)
                      - Optimal Factor Levels: Factors with highest SNR are selected.
                    - Critical Values: Higher SNR indicates better performance.
                    - Significance: Reduces experimentation costs and improves quality.
                """),
                dash_table.DataTable(
                    doe_df.to_dict('records'),
                    [{"name": i, "id": i} for i in doe_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': '#007BFF', 'color': 'white', 'fontWeight': 'bold'}
                ),
            ]),
            f"Computed in {time.time() - start_time:.2f} seconds"
        )

    elif tab == 'tab-defects':
        defect_counts = df["Defects"].value_counts()
        pareto_fig = pareto_chart(defect_counts)

        return (
            html.Div([
                html.H3("Defect Analysis", style={"color": "#007BFF"}),
                dcc.Markdown("""
                    **Interpretation**:
                    - Pareto chart identifies the most frequent defects.
                    - Follows the Pareto Principle (80/20 Rule): Focus on top 20% of defects causing 80% of problems.
                    - Formulas:
                      - Cumulative Frequency = Σ(Frequency)
                      - Cumulative Percentage = (Cumulative Frequency / Total Frequency) × 100
                    - Critical Values: Top 20% defects are prioritized for corrective actions.
                    - Significance: Guides root cause analysis and defect reduction efforts.
                """),
                dcc.Graph(figure=pareto_fig),
                html.Div([
                    html.H4("Top Defects"),
                    dash_table.DataTable(
                        defect_counts.reset_index().to_dict('records'),
                        [{"name": "Defect Type", "id": "index"}, {"name": "Count", "id": "Defects"}],
                        style_header={'backgroundColor': '#FF4136', 'color': 'white'}
                    )
                ]),
            ]),
            f"Computed in {time.time() - start_time:.2f} seconds"
        )

    elif tab == 'tab-variability':
        boxplot_fig = variability_boxplot(df, "Thickness (mm)")
        scatter_fig = px.scatter(df, x="Thickness (mm)", y="Defects", trendline="ols")

        return (
            html.Div([
                html.H3("Variability Analysis", style={"color": "#007BFF"}),
                dcc.Markdown("""
                    **Interpretation**:
                    - Boxplot visualizes the spread and outliers in thickness data.
                    - Scatter plot examines the relationship between thickness and defects.
                    - Formulas:
                      - Outliers: Data points beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
                      - Correlation Coefficient (r): Measures linear relationship strength.
                    - Critical Values:
                      - |r| > 0.7: Strong correlation.
                      - |r| < 0.3: Weak correlation.
                    - Significance: Identifies variability sources and defect correlations.
                """),
                dcc.Graph(figure=boxplot_fig),
                dcc.Graph(figure=scatter_fig),
            ]),
            f"Computed in {time.time() - start_time:.2f} seconds"
        )

    elif tab == 'tab-summary':
        usl, lsl = 5.2, 4.8
        cp, cpk = calculate_cp_cpk(df, "Thickness (mm)", usl, lsl)
        anomalies = detect_anomalies(df, "Thickness (mm)")
        defect_counts = df["Defects"].value_counts()

        decomposition = seasonal_decompose(df["Thickness (mm)"], model="additive", period=10)
        trend_fig = go.Figure(data=go.Scatter(x=df.index, y=decomposition.trend, name="Trend"))
        seasonal_fig = go.Figure(data=go.Scatter(x=df.index, y=decomposition.seasonal, name="Seasonality"))
        residual_fig = go.Figure(data=go.Scatter(x=df.index, y=decomposition.resid, name="Residual"))

        return (
            html.Div([
                html.H3("Summary Dashboard", style={"color": "#007BFF"}),
                dcc.Markdown("""
                    **Interpretation**:
                    - Summary consolidates key analyses: capability, defects, anomalies, and time-series trends.
                    - Time-series decomposition separates data into trend, seasonality, and residuals.
                    - Formulas:
                      - Trend: Long-term movement in data.
                      - Seasonality: Repeating short-term cycles.
                      - Residual: Random fluctuations after removing trend and seasonality.
                    - Critical Values: Significant residuals indicate unexplained variability.
                    - Significance: Provides a holistic view of process performance and areas for improvement.
                """),
                html.Div([
                    html.H4("Process Capability"),
                    dcc.Markdown(f"""
                        - **Cp**: `{cp:.2f}`
                        - **Cpk**: `{cpk:.2f}`
                        - **USL**: `{usl} mm`
                        - **LSL**: `{lsl} mm`
                    """),
                ]),
                html.Div([
                    html.H4("Top Defects"),
                    dcc.Graph(figure=pareto_chart(defect_counts)),
                ]),
                html.Div([
                    html.H4("Anomalies Detected"),
                    dash_table.DataTable(
                        data=anomalies.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in anomalies.columns],
                        style_header={'backgroundColor': '#FF4136', 'color': 'white'}
                    ),
                ]),
                html.Div([
                    html.H4("Time-Series Decomposition"),
                    dcc.Graph(figure=trend_fig),
                    dcc.Graph(figure=seasonal_fig),
                    dcc.Graph(figure=residual_fig),
                ]),
            ]),
            f"Computed in {time.time() - start_time:.2f} seconds"
        )

    elif tab == 'tab-suggestions':
        usl, lsl = 5.2, 4.8
        cp, cpk = calculate_cp_cpk(df, "Thickness (mm)", usl, lsl)
        anomalies = detect_anomalies(df, "Thickness (mm)")
        defect_counts = df["Defects"].value_counts()

        suggestions = []
        if cp < 1.33 or cpk < 1.33:
            suggestions.append("- Improve process centering to increase Cpk.")
        if anomalies.shape[0] > 0:
            suggestions.append("- Investigate anomalies detected in the thickness data.")
        if defect_counts.max() / defect_counts.sum() > 0.8:
            suggestions.append("- Focus on reducing the most frequent defect type (Pareto Principle).")

        return (
            html.Div([
                html.H3("Actionable Recommendations", style={"color": "#007BFF"}),
                dcc.Markdown("""
                    **Interpretation**:
                    - Recommendations are based on identified issues in process capability, defects, and anomalies.
                    - Formulas:
                      - Process Centering: Adjust mean closer to target value.
                      - Anomaly Investigation: Use root cause analysis tools (e.g., Fishbone Diagram).
                      - Defect Reduction: Implement corrective actions for top defects.
                    - Critical Values: Address high-priority issues first.
                    - Significance: Guides continuous improvement efforts.
                """),
                html.Ul([html.Li(s, style={"fontSize": "14px", "padding": "5px"}) for s in suggestions]),
                dcc.Markdown("""
                    **Root Cause Tips**:
                    - Use DOE results to identify optimal parameter settings.
                    - Check material batches for defect clusters.
                    - Review maintenance logs for anomaly timestamps.
                """),
            ]),
            f"Computed in {time.time() - start_time:.2f} seconds"
        )

# EXPORT DATA CALLBACK
@app.callback(
    Output("download-data", "data"),
    Input("export-data-button", "n_clicks"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def export_data(n_clicks, data_json):
    df = pd.read_json(data_json)
    return dcc.send_data_frame(df.to_csv, "SQC_data.csv", index=False)

# ANOMALY DETECTION
def detect_anomalies(data, column):
    model = IsolationForest(contamination=0.05)
    data = data.copy()
    data['Anomaly'] = model.fit_predict(data[[column]])
    return data[data['Anomaly'] == -1]

# PROCESS CAPABILITY CALCULATION
def calculate_cp_cpk(data, column, usl, lsl):
    std = data[column].std()
    mean = data[column].mean()
    cp = (usl - lsl) / (6 * std)
    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
    return cp, cpk

# RUN THE APP
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)