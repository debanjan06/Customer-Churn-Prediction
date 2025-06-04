# Customer Churn Prediction Project

An end-to-end machine learning project to predict customer churn with MLOps best practices.

## Project Structure

- `data/`: Raw and processed datasets
- `src/`: Source code for the project
- `app/`: FastAPI application for model serving
- `dashboard/`: Streamlit dashboard for visualizations
- `models/`: Saved model artifacts
- `mlflow/`: MLflow tracking and model registry

## Setup Instructions

```bash
conda create -n churn_project python=3.10
conda activate churn_project
pip install -r requirements.txt