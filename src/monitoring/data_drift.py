# In src/monitoring/data_drift.py
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

def check_data_drift(reference_data_path, current_data_path, output_dir):
    """
    Check for data drift between reference and current datasets
    """
    # Load data
    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)
    
    # Initialize results
    drift_results = {}
    
    # Check for drift in numerical features
    num_cols = reference_data.select_dtypes(include=['float64', 'int64']).columns
    
    for col in num_cols:
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(
            reference_data[col].dropna(), 
            current_data[col].dropna()
        )
        
        # Check if drift detected (p < 0.05)
        is_drift = p_value < 0.05
        
        drift_results[col] = {
            'ks_statistic': float(ks_stat),
            'p_value': float(p_value),
            'drift_detected': bool(is_drift)
        }
        
        # Visualize distribution comparison
        plt.figure(figsize=(10, 6))
        sns.kdeplot(reference_data[col], label='Reference')
        sns.kdeplot(current_data[col], label='Current')
        plt.title(f'Distribution Comparison for {col}')
        plt.legend()
        plt.savefig(f'{output_dir}/drift_{col}.png')
        plt.close()
    
    # Save drift results
    with open(f'{output_dir}/drift_results.json', 'w') as f:
        json.dump(drift_results, f, indent=4)
    
    # Generate summary report
    drift_detected = any([result['drift_detected'] for result in drift_results.values()])
    
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'features_checked': len(drift_results),
        'features_with_drift': sum([1 for r in drift_results.values() if r['drift_detected']]),
        'drift_detected': drift_detected
    }
    
    with open(f'{output_dir}/drift_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    return drift_detected, drift_results

if __name__ == "__main__":
    # Example usage
    reference_path = r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\processed\X_train.csv"
    current_path = r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\processed\X_test.csv"
    output_dir = r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\src\monitoring"
    
    os.makedirs(output_dir, exist_ok=True)
    
    drift_detected, results = check_data_drift(reference_path, current_path, output_dir)
    
    if drift_detected:
        print("Data drift detected! Consider retraining the model.")
    else:
        print("No significant data drift detected.")