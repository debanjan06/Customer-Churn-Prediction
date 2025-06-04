# In src/evaluation/explain_model.py
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

def explain_model():
    # Load data and model
    X_test = pd.read_csv(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\processed\X_test.csv")
    with open(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\models\best_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values (use a sample if the dataset is large)
    sample_size = min(100, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test.iloc[sample_indices]
    
    # Generate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Create summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\dashboard\visualization\model_shap\shap_summary.png")
    
    # Create force plot for a single prediction
    plt.figure(figsize=(20, 3))
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        X_sample.iloc[0], 
        matplotlib=True, 
        show=False
    )
    plt.tight_layout()
    plt.savefig(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\models\shap_force_plot.png")
    
    return shap_values

if __name__ == "__main__":
    explain_model()  