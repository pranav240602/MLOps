import pandas as pd

def load_data():
    """Load the housing dataset"""
    print("Loading housing data...")
    df = pd.read_csv('/opt/airflow/dags/data/housing.csv')
    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())
    return "Data loaded successfully!"
