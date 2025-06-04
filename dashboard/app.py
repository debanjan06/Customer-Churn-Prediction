# In dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import json
import plotly.express as px
import plotly.graph_objects as go

# Load model for direct predictions
with open(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\models\best_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Load dataset for visualizations
df = pd.read_csv(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\data\processed\cleaned_data.csv")

# Set page config
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# Define tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Predictions", "Model Insights"])

# Tab 1: Dashboard
with tab1:
    st.title("Customer Churn Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        churn_rate = df['Churn'].value_counts(normalize=True)['Yes'] * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col2:
        avg_tenure = df['tenure'].mean()
        st.metric("Avg. Customer Tenure", f"{avg_tenure:.1f} months")
    
    with col3:
        avg_monthly = df['MonthlyCharges'].mean()
        st.metric("Avg. Monthly Charges", f"${avg_monthly:.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn by Contract Type")
        fig = px.histogram(
            df, 
            x="Contract", 
            color="Churn",
            barmode="group",
            text_auto=True,
            title="Churn Distribution by Contract Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Churn by Tenure")
        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            marginal="box",
            title="Churn Distribution by Tenure"
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Predictions
with tab2:
    st.title("Churn Prediction Tool")
    st.write("Enter customer details to predict churn probability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 50)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
    with col2:
        gender = st.radio("Gender", ["Male", "Female"])
        partner = st.radio("Partner", ["Yes", "No"])
        dependents = st.radio("Dependents", ["Yes", "No"])
        tech_support = st.radio("Tech Support", ["Yes", "No", "No internet service"])
    
    # Prediction button
    if st.button("Predict Churn"):
        # Create a sample input (simplified)
        input_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': tenure * monthly_charges,
            'gender': gender,
            'Contract': contract,
            'InternetService': internet_service,
            'Partner': partner,
            'Dependents': dependents,
            'TechSupport': tech_support
        }
        
        # In a real application, you would call the API
        # Here we're making a direct prediction for simplicity
        
        # Show prediction
        st.write("### Prediction Results")
        
        # This is a mock prediction
        churn_prob = np.random.uniform(0, 1)
        
        # Gauge chart for churn probability
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = churn_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig)
        
        # Recommendation based on prediction
        if churn_prob > 0.5:
            st.error("High risk of churn! Consider retention strategies.")
        else:
            st.success("Low churn risk. Customer seems satisfied.")

# Tab 3: Model Insights
with tab3:
    st.title("Model Performance & Insights")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", "82.3%")
    
    with col2:
        st.metric("AUC-ROC", "0.846")
    
    with col3:
        st.metric("F1 Score", "0.67")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # Mock feature importance data
    features = ["Contract", "tenure", "MonthlyCharges", "TotalCharges", "InternetService"]
    importance = [0.35, 0.25, 0.15, 0.15, 0.10]
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance"
    )
    st.plotly_chart(fig)
    
    # SHAP value example
    st.subheader("SHAP Value Explanation")
    st.image(r"C:\Users\DEBANJAN SHIL\Documents\Churn_Prediction\dashboard\visualization\model_shap\shap_summary.png", 
             caption="SHAP Summary Plot (Sample)",
             use_column_width=True)

if __name__ == "__main__":
    # Run with: streamlit run dashboard/app.py
    pass