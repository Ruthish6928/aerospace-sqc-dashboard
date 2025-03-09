# Aerospace-sqc-dashboard
Here’s a **clean, minimal README.md** without license details and focused only on used components:

```markdown
# Aerospace Blade SQC Dashboard 🚀

## Overview
Real-time quality control dashboard for aerospace blade manufacturing, analyzing thickness (mm) and defects using statistical tools and machine learning.

## Features ✨
- **CSV Upload/Simulation**: Load real or synthetic manufacturing data
- **Anomaly Detection**: Flags thickness outliers using Isolation Forest
- **Process Capability**: Calculates Cp/Cpk against 5.2/4.8 mm specs
- **Control Charts**: X-bar/R charts and Individual-Moving Range charts
- **Defect Analysis**: Pareto charts and defect distribution pies
- **DOE Optimization**: Taguchi method for parameter tuning
- **Real-Time Alerts**: Triggers warnings for defect spikes (>10 defects)

## Installation 🔧
```bash
git clone https://github.com/your-username/aerospace-sqc-dashboard.git
cd aerospace-sqc-dashboard
pip install dash pandas numpy scikit-learn statsmodels plotly
```

## Usage 📊
1. **Run the app**:
   ```bash
   python dashboard.py
   ```
2. **Upload CSV** with these columns:
   ```csv
   Thickness (mm),Defects
   5.02,0
   4.98,1
   ```
3. **Simulate data** with the red button if no CSV available

## Folder Structure 📂
```
aerospace-sqc-dashboard/
├── dashboard.py          # Main app logic
├── data_processing.py    # Synthetic data generation
├── charts.py             # Chart rendering functions
└── taguchi_optimization.py # DOE implementation
```

## Troubleshooting ❗
- **"Undefined variable" errors**:  
  Ensure all dependencies are installed (`pip install -r requirements.txt`)
- **CSV upload errors**:  
  Verify columns match `Thickness (mm)` and `Defects` (binary 0/1)
- **Dashboard layout issues**:  
  Check for unclosed parentheses/brackets in `dcc.Upload` or `html.Div` components

For support, contact your.email@example.com
```

This version:  
✅ Removes license section  
✅ Focuses only on components present in your code  
✅ Includes critical setup/usage details  
✅ Highlights troubleshooting steps for errors you encountered  

Let me know if you need further refinements!
