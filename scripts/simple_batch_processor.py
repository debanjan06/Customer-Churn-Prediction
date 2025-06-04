#!/usr/bin/env python3
"""
Create Demo Results Quickly
Generates realistic prediction results for demo purposes
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import random

def create_demo_results(num_customers=500):
    """Create realistic demo results"""
    print(f"Creating demo results for {num_customers} customers...")
    
    # Load sample customers if available
    try:
        customers_df = pd.read_csv('sample_customers.csv')
        print(f"✅ Loaded {len(customers_df)} customers from sample_customers.csv")
        customers_df = customers_df.head(num_customers)  # Limit to requested number
    except:
        print("❌ sample_customers.csv not found, creating synthetic data")
        return None
    
    results = []
    
    for idx, row in customers_df.iterrows():
        # Create realistic churn probabilities based on features
        
        # Base probability
        churn_prob = 0.3
        
        # Tenure effect (shorter tenure = higher churn)
        if row['tenure'] < 6:
            churn_prob += 0.4
        elif row['tenure'] < 12:
            churn_prob += 0.2
        elif row['tenure'] > 36:
            churn_prob -= 0.2
        
        # Contract effect
        if row['Contract'] == 'Month-to-month':
            churn_prob += 0.3
        elif row['Contract'] == 'Two year':
            churn_prob -= 0.2
        
        # Senior citizen effect
        if row['SeniorCitizen'] == 1:
            churn_prob += 0.1
        
        # Internet service effect
        if row['InternetService'] == 'Fiber optic':
            churn_prob += 0.1
        
        # Payment method effect
        if row['PaymentMethod'] == 'Electronic check':
            churn_prob += 0.15
        elif 'automatic' in row['PaymentMethod'].lower():
            churn_prob -= 0.1
        
        # Monthly charges effect
        if row['MonthlyCharges'] > 80:
            churn_prob += 0.1
        elif row['MonthlyCharges'] < 30:
            churn_prob -= 0.1
        
        # Add some randomness
        churn_prob += random.uniform(-0.15, 0.15)
        
        # Ensure probability is between 0 and 1
        churn_prob = max(0.0, min(1.0, churn_prob))
        
        # Determine prediction
        churn_prediction = churn_prob >= 0.5
        
        # Create top factors (realistic based on the customer)
        top_factors = []
        
        if row['Contract'] == 'Month-to-month':
            top_factors.append({"factor": "Contract_Month-to-month", "importance": 0.25})
        
        if row['tenure'] < 12:
            top_factors.append({"factor": "tenure", "importance": 0.22})
        
        if row['MonthlyCharges'] > 70:
            top_factors.append({"factor": "MonthlyCharges", "importance": 0.18})
        
        if row['PaymentMethod'] == 'Electronic check':
            top_factors.append({"factor": "PaymentMethod_Electronic check", "importance": 0.15})
        
        if row['InternetService'] == 'Fiber optic':
            top_factors.append({"factor": "InternetService_Fiber optic", "importance": 0.12})
        
        # Ensure we have at least 3 factors
        while len(top_factors) < 3:
            remaining_factors = [
                {"factor": "PaperlessBilling_Yes", "importance": 0.10},
                {"factor": "gender_Male", "importance": 0.08},
                {"factor": "SeniorCitizen", "importance": 0.07},
                {"factor": "Partner_No", "importance": 0.09}
            ]
            for factor in remaining_factors:
                if factor not in top_factors and len(top_factors) < 3:
                    top_factors.append(factor)
        
        # Sort by importance and take top 3
        top_factors = sorted(top_factors, key=lambda x: x['importance'], reverse=True)[:3]
        
        # Determine risk category
        if churn_prob >= 0.7:
            risk_category = 'High'
        elif churn_prob >= 0.4:
            risk_category = 'Medium'
        else:
            risk_category = 'Low'
        
        # Create result record
        result = {
            'customer_id': row.get('customer_id', f'CUST_{idx+1:05d}'),
            'churn_probability': round(churn_prob, 4),
            'churn_prediction': churn_prediction,
            'top_factors': json.dumps(top_factors),
            'prediction_timestamp': datetime.now().isoformat(),
            'risk_category': risk_category,
            'tenure': row['tenure'],
            'MonthlyCharges': row['MonthlyCharges'],
            'TotalCharges': row['TotalCharges'],
            'Contract': row['Contract'],
            'InternetService': row['InternetService'],
            'PaymentMethod': row['PaymentMethod'],
            'gender': row['gender'],
            'SeniorCitizen': row['SeniorCitizen'],
            'Partner': row['Partner'],
            'Dependents': row['Dependents']
        }
        
        results.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"sample_customers_predictions_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    risk_dist = results_df['risk_category'].value_counts()
    avg_prob = results_df['churn_probability'].mean()
    
    print(f"\n📊 DEMO RESULTS CREATED!")
    print(f"💾 Saved to: {output_file}")
    print(f"📈 Average Churn Probability: {avg_prob:.1%}")
    print(f"\n🎯 Risk Distribution:")
    for risk, count in risk_dist.items():
        percentage = (count / len(results_df)) * 100
        print(f"   {risk} Risk: {count} ({percentage:.1f}%)")
    
    print(f"\n🚀 Ready for business analysis!")
    print(f"   python scripts/demo_analysis.py")
    
    return output_file

if __name__ == "__main__":
    create_demo_results()