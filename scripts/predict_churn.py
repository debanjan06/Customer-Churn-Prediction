#!/usr/bin/env python3
"""
Customer Churn Prediction Script
Make predictions using the FastAPI service
"""

import requests
import json
import pandas as pd
from typing import Dict, Any

class ChurnPredictor:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        
    def predict_single_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single customer
        """
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=customer_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}", "details": response.text}
                
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def predict_batch_customers(self, customers_list: list) -> list:
        """
        Make predictions for multiple customers
        """
        results = []
        for i, customer in enumerate(customers_list):
            print(f"Processing customer {i+1}/{len(customers_list)}...")
            result = self.predict_single_customer(customer)
            result['customer_index'] = i
            results.append(result)
        return results
    
    def predict_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Make predictions from a CSV file
        """
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Convert to list of dictionaries
        customers = df.to_dict('records')
        
        # Make predictions
        results = self.predict_batch_customers(customers)
        
        # Convert back to DataFrame
        predictions_df = pd.DataFrame(results)
        
        return predictions_df

# Example usage and test cases
def main():
    predictor = ChurnPredictor()
    
    print("🎯 Customer Churn Prediction Demo")
    print("=" * 50)
    
    # Test cases with different risk profiles
    test_customers = [
        {
            "name": "High-Risk Customer",
            "data": {
                "tenure": 1,
                "MonthlyCharges": 85.0,
                "TotalCharges": 85,
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }
        },
        {
            "name": "Low-Risk Customer",
            "data": {
                "tenure": 60,
                "MonthlyCharges": 45.0,
                "TotalCharges": 2700,
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)"
            }
        },
        {
            "name": "Medium-Risk Customer",
            "data": {
                "tenure": 24,
                "MonthlyCharges": 65.0,
                "TotalCharges": 1560,
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Credit card (automatic)"
            }
        }
    ]
    
    # Make predictions for test customers
    for customer in test_customers:
        print(f"\n📊 {customer['name']}:")
        print("-" * 30)
        
        result = predictor.predict_single_customer(customer['data'])
        
        if 'error' not in result:
            churn_prob = result['churn_probability']
            churn_pred = result['churn_prediction']
            top_factors = result['top_factors']
            
            # Risk level based on probability
            if churn_prob >= 0.7:
                risk_level = "🔴 HIGH RISK"
            elif churn_prob >= 0.4:
                risk_level = "🟡 MEDIUM RISK"
            else:
                risk_level = "🟢 LOW RISK"
            
            print(f"Churn Probability: {churn_prob:.1%}")
            print(f"Churn Prediction: {'Will Churn' if churn_pred else 'Will Stay'}")
            print(f"Risk Level: {risk_level}")
            print("Top Factors:")
            for factor in top_factors:
                print(f"  - {factor['factor']}: {factor['importance']:.3f}")
        else:
            print(f"❌ Error: {result['error']}")
    
    print(f"\n✅ Demo completed!")
    print(f"💡 You can also access the interactive API at: http://localhost:8000/docs")

def create_sample_customers():
    """
    Create a sample CSV file with customer data for batch prediction
    """
    sample_data = [
        {
            "tenure": 12, "MonthlyCharges": 70.5, "TotalCharges": 846, "gender": "Male",
            "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"
        },
        {
            "tenure": 36, "MonthlyCharges": 55.0, "TotalCharges": 1980, "gender": "Female",
            "SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
            "MultipleLines": "Yes", "InternetService": "DSL", "OnlineSecurity": "Yes",
            "OnlineBackup": "No", "DeviceProtection": "Yes", "TechSupport": "Yes",
            "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "One year",
            "PaperlessBilling": "No", "PaymentMethod": "Mailed check"
        },
        {
            "tenure": 2, "MonthlyCharges": 95.0, "TotalCharges": 190, "gender": "Male",
            "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv("sample_customers.csv", index=False)
    print("Created sample_customers.csv with 3 customer records")
    return "sample_customers.csv"

def batch_prediction_demo():
    """
    Demonstrate batch prediction from CSV
    """
    print("\n📁 Batch Prediction Demo")
    print("=" * 30)
    
    # Create sample CSV
    csv_file = create_sample_customers()
    
    # Make batch predictions
    predictor = ChurnPredictor()
    results_df = predictor.predict_from_csv(csv_file)
    
    print(f"\nBatch prediction results:")
    print(results_df[['customer_index', 'churn_probability', 'churn_prediction']].to_string(index=False))
    
    # Save results
    results_df.to_csv("prediction_results.csv", index=False)
    print(f"\n💾 Results saved to prediction_results.csv")

if __name__ == "__main__":
    # Run single predictions demo
    main()
    
    # Run batch prediction demo
    batch_prediction_demo()