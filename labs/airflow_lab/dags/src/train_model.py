import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def train_model():
    """Train Random Forest model for house price prediction"""
    print("Starting model training...")
    
    # Load processed data
    df = pd.read_csv('/opt/airflow/dags/data/processed_housing.csv')
    
    # Split features and target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('/opt/airflow/dags/model', exist_ok=True)
    with open('/opt/airflow/dags/model/rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved successfully!")
    
    return "Model trained successfully!"
