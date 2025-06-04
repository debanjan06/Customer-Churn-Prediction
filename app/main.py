#!/usr/bin/env python3
"""
Fixed FastAPI Main - Returns Actual Model Predictions
"""

import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
import os
import logging

app = FastAPI(title="Churn Prediction API")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
model_path = "models/best_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

logger.info(f"Loading model from: {model_path}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

logger.info("Model loaded successfully!")

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: Optional[float] = None
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: Optional[str] = None
    InternetService: str
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    top_factors: List[dict]

def engineer_features(input_df):
    """Apply the same feature engineering as training"""
    try:
        df = input_df.copy()
        
        # Handle TotalCharges
        if df['TotalCharges'].isnull().any() or df['TotalCharges'].iloc[0] == 0:
            df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']
        
        # Create engineered features
        df['tenure_years'] = df['tenure'] / 12
        
        # Handle division by zero for charges_per_month
        df['charges_per_month'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']
        )
        
        # Service count feature
        service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        def count_services(row):
            count = 0
            for col in service_columns:
                if col in row:
                    val = str(row[col]).lower()
                    if val == 'yes' or 'fiber' in val or 'dsl' in val:
                        count += 1
            return count
        
        df['num_services'] = df.apply(count_services, axis=1)
        
        # Charges to tenure ratio
        df['charges_to_tenure_ratio'] = np.where(
            df['tenure'] > 0,
            df['MonthlyCharges'] / df['tenure'],
            df['MonthlyCharges']
        )
        
        # One-hot encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        return df_encoded
        
    except Exception as e:
        logger.error(f"Feature engineering error: {str(e)}")
        raise e

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer_data: CustomerData):
    try:
        logger.info("Received prediction request")
        
        # Convert input data to DataFrame
        input_dict = customer_data.dict()
        input_df = pd.DataFrame([input_dict])
        
        logger.info(f"Input data: {input_dict}")
        
        # Handle TotalCharges if None
        if input_dict['TotalCharges'] is None:
            input_df['TotalCharges'] = input_df['tenure'] * input_df['MonthlyCharges']
        
        # Apply feature engineering
        processed_input = engineer_features(input_df)
        
        logger.info(f"Processed features: {processed_input.shape}")
        logger.info(f"Feature columns: {list(processed_input.columns)}")
        
        # Get expected feature names from model (if available)
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            logger.info(f"Model expects {len(expected_features)} features")
            
            # Align features with training data
            missing_features = set(expected_features) - set(processed_input.columns)
            extra_features = set(processed_input.columns) - set(expected_features)
            
            # Add missing features as zeros
            for feature in missing_features:
                processed_input[feature] = 0
            
            # Remove extra features
            processed_input = processed_input[expected_features]
            
            logger.info(f"Aligned features: {processed_input.shape}")
        
        # Make prediction
        churn_prob = model.predict_proba(processed_input)[0, 1]
        churn_pred = churn_prob >= 0.5
        
        logger.info(f"Prediction: {churn_prob:.4f}, Decision: {churn_pred}")
        
        # Get feature importances (simplified approach)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = processed_input.columns
            
            # Create feature importance dictionary
            feature_imp = dict(zip(feature_names, importances))
            
            # Get top 3 factors
            top_factors = [
                {"factor": k.replace('_', ' ').title(), "importance": float(v)}
                for k, v in sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:3]
            ]
        else:
            # Fallback based on common churn factors
            top_factors = [
                {"factor": "Contract Type", "importance": 0.25},
                {"factor": "Tenure", "importance": 0.18},
                {"factor": "Monthly Charges", "importance": 0.15}
            ]
        
        # Create response
        response = {
            "customer_id": f"customer_{hash(str(input_dict)) % 10000:04d}",
            "churn_probability": float(churn_prob),
            "churn_prediction": bool(churn_pred),
            "top_factors": top_factors
        }
        
        logger.info(f"Response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Churn Prediction API", 
        "status": "Model loaded successfully",
        "model_type": str(type(model)),
        "features_expected": getattr(model, 'n_features_in_', "Unknown")
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_type": str(type(model))
    }

@app.get("/model-info")
def model_info():
    """Get information about the loaded model"""
    try:
        info = {
            "model_type": str(type(model)),
            "feature_count": getattr(model, 'n_features_in_', "Unknown")
        }
        
        if hasattr(model, 'feature_names_in_'):
            info["feature_names"] = model.feature_names_in_.tolist()
        else:
            info["feature_names"] = "Not available"
            
        return info
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug-prediction")
def debug_prediction():
    """Debug endpoint to test feature engineering"""
    try:
        # Sample data for testing
        sample_data = {
            "tenure": 12,
            "MonthlyCharges": 70.5,
            "TotalCharges": 846,
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check"
        }
        
        df = pd.DataFrame([sample_data])
        processed = engineer_features(df)
        
        return {
            "original_features": len(df.columns),
            "processed_features": len(processed.columns),
            "sample_prediction": "Would make prediction here",
            "processed_columns": processed.columns.tolist()[:10]  # First 10 for brevity
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)