#!/usr/bin/env python3
"""
Quick Sample Data Generator for Demo
Creates realistic customer data for churn prediction
"""

import pandas as pd
import random
import sys

def create_sample_customers(num_customers=500, filename="sample_customers.csv"):
    """Create sample customer data"""
    print(f"Creating {num_customers} sample customers...")
    
    # Data generators
    genders = ['Male', 'Female']
    yes_no = ['Yes', 'No']
    internet_services = ['DSL', 'Fiber optic', 'No']
    contracts = ['Month-to-month', 'One year', 'Two year']
    payment_methods = [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)',
        'Credit card (automatic)'
    ]
    
    customers = []
    for i in range(num_customers):
        # Generate realistic data
        tenure = random.randint(0, 72)
        is_senior = random.choice([0, 1])
        has_partner = random.choice(yes_no)
        has_dependents = random.choice(yes_no) if has_partner == 'Yes' else 'No'
        
        # Contract influences tenure
        if tenure < 12:
            contract = 'Month-to-month'
        elif tenure < 24:
            contract = random.choice(['Month-to-month', 'One year'])
        else:
            contract = random.choice(contracts)
        
        # Calculate charges
        base_charge = random.uniform(18, 40)
        internet = random.choice(internet_services)
        
        if internet == 'Fiber optic':
            base_charge += random.uniform(20, 50)
        elif internet == 'DSL':
            base_charge += random.uniform(10, 30)
        
        monthly_charges = round(base_charge, 2)
        total_charges = round(monthly_charges * tenure, 2) if tenure > 0 else monthly_charges
        
        customer = {
            'customer_id': f'CUST_{i+1:05d}',
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'gender': random.choice(genders),
            'SeniorCitizen': is_senior,
            'Partner': has_partner,
            'Dependents': has_dependents,
            'PhoneService': random.choice(yes_no),
            'MultipleLines': random.choice(yes_no + ['No phone service']),
            'InternetService': internet,
            'OnlineSecurity': random.choice(yes_no + ['No internet service']) if internet != 'No' else 'No internet service',
            'OnlineBackup': random.choice(yes_no + ['No internet service']) if internet != 'No' else 'No internet service',
            'DeviceProtection': random.choice(yes_no + ['No internet service']) if internet != 'No' else 'No internet service',
            'TechSupport': random.choice(yes_no + ['No internet service']) if internet != 'No' else 'No internet service',
            'StreamingTV': random.choice(yes_no + ['No internet service']) if internet != 'No' else 'No internet service',
            'StreamingMovies': random.choice(yes_no + ['No internet service']) if internet != 'No' else 'No internet service',
            'Contract': contract,
            'PaperlessBilling': random.choice(yes_no),
            'PaymentMethod': random.choice(payment_methods)
        }
        customers.append(customer)
    
    # Create DataFrame and save
    df = pd.DataFrame(customers)
    df.to_csv(filename, index=False)
    
    print(f"✅ Created {filename} with {len(customers)} customers")
    print(f"Sample data preview:")
    print(df.head())
    
    return filename

if __name__ == "__main__":
    # Get number of customers from command line or use default
    num_customers = 1000
    if len(sys.argv) > 1:
        try:
            num_customers = int(sys.argv[1])
        except:
            print("Usage: python create_sample.py [number_of_customers]")
            sys.exit(1)
    
    create_sample_customers(num_customers)