# In src/modeling/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import xgboost as xgb
import pickle
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import os

def train_models():
    # Set MLflow tracking URI to your local mlflow directory
    mlflow_tracking_uri = r"file:///C:/Users/DEBANJAN SHIL/Documents/Churn_Prediction/mlflow"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Set experiment name
    experiment_name = "churn_prediction_experiment"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        # Experiment already exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name="churn_prediction_model_v1"):
        print("Started MLflow run")
        
        # Load data
        df = pd.read_csv(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\processed\engineered_features.csv")
        
        # Separate features and target
        X = df.drop('Churn_Yes', axis=1)
        y = df['Churn_Yes']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Log dataset info
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features_count", len(X.columns))
        mlflow.log_param("target_distribution", y.value_counts().to_dict())
        
        print("Logged dataset parameters")
        
        # Save test datasets for later use in explanation and evaluation
        X_test.to_csv(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\processed\X_test.csv", index=False)
        y_test.to_csv(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\processed\y_test.csv", index=False)
        
        # Train Random Forest
        rf_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        }
        
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        
        # Evaluate Random Forest
        rf_preds = rf_model.predict(X_test)
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_probs)
        
        # Log Random Forest metrics and params
        for param_name, param_value in rf_params.items():
            mlflow.log_param(f"rf_{param_name}", param_value)
        
        mlflow.log_metric("rf_auc", rf_auc)
        mlflow.log_metric("rf_accuracy", (rf_preds == y_test).mean())
        
        print(f"Random Forest AUC: {rf_auc}")
        
        # Train XGBoost
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "tree_method": "gpu_hist",  # Use GPU acceleration
            "random_state": 42
        }
        
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        
        # Evaluate XGBoost
        xgb_preds = xgb_model.predict(X_test)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_probs)
        
        # Log XGBoost metrics and params
        for param_name, param_value in xgb_params.items():
            mlflow.log_param(f"xgb_{param_name}", param_value)
        
        mlflow.log_metric("xgb_auc", xgb_auc)
        mlflow.log_metric("xgb_accuracy", (xgb_preds == y_test).mean())
        
        print(f"XGBoost AUC: {xgb_auc}")
        
        # Generate classification reports
        print("\nRandom Forest Classification Report:")
        rf_report = classification_report(y_test, rf_preds, output_dict=True)
        print(classification_report(y_test, rf_preds))
        
        print("\nXGBoost Classification Report:")
        xgb_report = classification_report(y_test, xgb_preds, output_dict=True)
        print(classification_report(y_test, xgb_preds))
        
        # Log detailed classification metrics
        mlflow.log_metric("rf_precision", rf_report['weighted avg']['precision'])
        mlflow.log_metric("rf_recall", rf_report['weighted avg']['recall'])
        mlflow.log_metric("rf_f1", rf_report['weighted avg']['f1-score'])
        
        mlflow.log_metric("xgb_precision", xgb_report['weighted avg']['precision'])
        mlflow.log_metric("xgb_recall", xgb_report['weighted avg']['recall'])
        mlflow.log_metric("xgb_f1", xgb_report['weighted avg']['f1-score'])
        
        # Save best model
        if xgb_auc > rf_auc:
            best_model = xgb_model
            best_model_name = "XGBoost"
            best_preds = xgb_preds
            best_probs = xgb_probs
            best_auc = xgb_auc
            print("XGBoost is the better model")
        else:
            best_model = rf_model
            best_model_name = "Random Forest"
            best_preds = rf_preds
            best_probs = rf_probs
            best_auc = rf_auc
            print("Random Forest is the better model")
        
        # Log best model info
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_auc", best_auc)
        
        # Create visualization directory for model evaluation results
        viz_dir = r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\visualizations\model_eval"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot ROC curves
        try:
            plt.figure(figsize=(10, 8))
            
            # Random Forest ROC
            fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
            plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})')
            
            # XGBoost ROC
            fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
            plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_auc:.3f})')
            
            # Plot diagonal
            plt.plot([0, 1], [0, 1], 'k--')
            
            # Formatting
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves for Churn Prediction Models')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            roc_curve_path = os.path.join(viz_dir, 'roc_curves.png')
            plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log the plot to MLflow
            mlflow.log_artifact(roc_curve_path, "plots")
            
            print(f"ROC curve saved and logged to MLflow")
        except Exception as e:
            print(f"Error creating ROC curve plot: {e}")
        
        # Create feature importance plot
        try:
            if hasattr(best_model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                
                # Get top 20 features
                feature_names = X.columns
                importances = best_model.feature_importances_
                indices = np.argsort(importances)[::-1][:20]
                
                plt.title(f'Top 20 Feature Importances - {best_model_name}')
                plt.bar(range(20), importances[indices])
                plt.xticks(range(20), [feature_names[i] for i in indices], rotation=45, ha='right')
                plt.tight_layout()
                
                feature_importance_path = os.path.join(viz_dir, 'feature_importance.png')
                plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Log the plot to MLflow
                mlflow.log_artifact(feature_importance_path, "plots")
                
                print("Feature importance plot saved and logged to MLflow")
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
        
        # Log model to MLflow with input example and signature
        try:
            mlflow.sklearn.log_model(
                best_model, 
                "model",
                input_example=X_test.iloc[0:5],
                registered_model_name="churn_prediction_model"
            )
            print("Model logged to MLflow successfully")
        except Exception as e:
            print(f"Error logging model to MLflow: {e}")
        
        # Save model locally
        models_dir = r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\models"
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, "best_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Model saved locally to: {model_path}")
        
        # Also save the model name for reference
        model_name_path = os.path.join(models_dir, "best_model_name.txt")
        with open(model_name_path, 'w') as f:
            f.write(best_model_name)
        
        # Log additional artifacts
        try:
            mlflow.log_artifact(model_path, "local_model")
            mlflow.log_artifact(model_name_path, "local_model")
            print("Local model files logged to MLflow")
        except Exception as e:
            print(f"Error logging local model files: {e}")
        
        # Log feature names
        try:
            feature_names_path = "feature_names.txt"
            with open(feature_names_path, "w") as f:
                for feature in X.columns:
                    f.write(f"{feature}\n")
            mlflow.log_artifact(feature_names_path, "model_info")
            os.remove(feature_names_path)  # Clean up local file
            print("Feature names logged to MLflow")
        except Exception as e:
            print(f"Error logging feature names: {e}")
        
        print("MLflow run completed successfully!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return best_model

if __name__ == "__main__":
    train_models()