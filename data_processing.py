import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Simulated Data Generation (Aerospace-Specific)
# data_processing.py
import pandas as pd
import numpy as np

def generate_blade_data():
    np.random.seed(42)
    data = {
        "Thickness (mm)": np.random.normal(5.0, 0.1, 100),
        "Defects": np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    }
    return pd.DataFrame(data)

    data = {
        "Thickness (mm)": thickness,
        "Defects": defects,
        "Material": materials
    }
    return pd.DataFrame(data)

# Anomaly Detection
def detect_anomalies(data, column):
    model = IsolationForest(contamination=0.05)
    data['Anomaly'] = model.fit_predict(data[[column]])
    anomalies = data[data['Anomaly'] == -1]
    return anomalies

# Process Capability Calculation
def calculate_cp_cpk(data, column, usl, lsl):
    std = data[column].std()
    mean = data[column].mean()
    cp = (usl - lsl) / (6 * std)
    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
    return cp, cpk