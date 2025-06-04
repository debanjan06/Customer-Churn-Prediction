# In src/feature_engineering/create_features.py
import pandas as pd
import numpy as np

def engineer_features(df):
    # Create new features
    
    # 1. Calculate tenure in years
    df['tenure_years'] = df['tenure'] / 12
    
    # 2. Create total charges per month feature
    df['charges_per_month'] = df['TotalCharges'] / df['tenure']
    df['charges_per_month'].fillna(df['MonthlyCharges'], inplace=True)
    
    # 3. Create feature for total number of services
    service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['num_services'] = df[service_columns].apply(
        lambda row: sum(row.values != 'No'), axis=1
    )
    
    # 4. Create average monthly charges to tenure ratio
    df['charges_to_tenure_ratio'] = df['MonthlyCharges'] / df['tenure']
    df['charges_to_tenure_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['charges_to_tenure_ratio'].fillna(df['MonthlyCharges'], inplace=True)
    
    # 5. Handle categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    return df

# Apply feature engineering
df = pd.read_csv(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\processed\cleaned_data.csv")
df_engineered = engineer_features(df)  
df_engineered.to_csv(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\processed\engineered_features.csv", index=False) 