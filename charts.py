import plotly.express as px

# X-Bar and R Chart
def x_bar_r_chart(df, column):
    fig = px.line(df, y=column, title="X-Bar and R Chart")
    return fig

# I-MR Chart (Individual and Moving Range)
def imr_chart(df, column):
    fig = px.line(df, y=column, title="I-MR Chart")
    return fig

# P-Chart (Proportion of Defects)
def p_chart(df, column):
    fig = px.line(df, y=column, title="P-Chart")
    return fig

# Cp/Cpk Chart
def cp_cpk_chart(df, column, usl, lsl):
    fig = px.histogram(df, x=column, title="Cp/Cpk Chart")
    fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
    fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
    return fig

# Pareto Chart
def pareto_chart(defect_counts):
    fig = px.bar(defect_counts, x=defect_counts.index, y=defect_counts.values, title="Pareto Chart")
    return fig

# Variability Boxplot
def variability_boxplot(df, column):
    fig = px.box(df, y=column, title="Variability Boxplot")
    return fig