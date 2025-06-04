# In a Jupyter notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic exploration
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check for empty strings or whitespace
print("Checking for empty strings or whitespace:")
for col in df.columns:
    if df[col].dtype == 'object':
        empty_count = (df[col] == '') | (df[col].str.isspace())
        if empty_count.any():
            print(f"{col}: {empty_count.sum()} empty/whitespace values")

# Fix TotalCharges column - convert to numeric
# This column appears to be a string that should be numeric
print("\nConverting TotalCharges to numeric:")
print(f"Before conversion - dtype: {df['TotalCharges'].dtype}")
print(f"Sample values: {df['TotalCharges'].head()}")

# Replace empty strings with NaN and convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"After conversion - dtype: {df['TotalCharges'].dtype}")
print(f"NaN values in TotalCharges: {df['TotalCharges'].isna().sum()}")

# Fill NaN values with 0 (for new customers with tenure = 0)
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Visualize churn distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=df)
plt.title('Customer Churn Distribution')
plt.savefig(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\dashboard\visualization\eda\churn_distribution.png")

# Create a correlation matrix only for numeric columns
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.savefig(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\dashboard\visualization\eda\correlation_matrix.png")

# Explore categorical variables
plt.figure(figsize=(15, 10))
categorical_columns = df.select_dtypes(include=['object']).columns
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)
    df_counts = df.groupby([column, 'Churn']).size().unstack()
    df_counts.plot(kind='bar', stacked=False, ax=plt.gca())
    plt.title(f'Churn by {column}')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    if i >= 9:  # Only show first 9 to avoid overcrowding
        break
plt.tight_layout()
plt.savefig(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\dashboard\visualization\eda\categorical_features.png")

# Save cleaned data
df.to_csv(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\processed\cleaned_data.csv", index=False)

# Additional insights - monthly charges vs tenure colored by churn
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df, alpha=0.7)
plt.title('Monthly Charges vs Tenure by Churn Status')
plt.savefig(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\dashboard\visualization\eda\charges_tenure_scatter.png")

# Create feature for total services subscribed
service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']

# Function to count services
def count_services(row):
    count = 0
    for col in service_columns:
        if row[col] == 'Yes' or row[col] == 'Fiber optic' or row[col] == 'DSL':
            count += 1
    return count

# Apply the function
df['TotalServices'] = df.apply(count_services, axis=1)

# Visualize number of services vs churn
plt.figure(figsize=(10, 6))
sns.countplot(x='TotalServices', hue='Churn', data=df)
plt.title('Number of Services vs Churn')
plt.savefig(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\dashboard\visualization\eda\services_churn.png")

# Print key statistics about churn
print("\nChurn Rate Analysis:")
churn_rate = df['Churn'].value_counts(normalize=True)['Yes'] * 100
print(f"Overall Churn Rate: {churn_rate:.2f}%")

print("\nChurn Rate by Contract Type:")
print(df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()['Yes'] * 100)

print("\nChurn Rate by Tenure Group:")
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                           labels=['0-12 mo', '13-24 mo', '25-36 mo', '37-48 mo', '49-60 mo', '61-72 mo'])
print(df.groupby('TenureGroup')['Churn'].value_counts(normalize=True).unstack()['Yes'] * 100)

# Save the enhanced data with new features
df.to_csv(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\processed\enhanced_data.csv", index=False)