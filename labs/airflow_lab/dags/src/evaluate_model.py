import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

def evaluate_model():
    """Load model and evaluate on test data"""
    print("Starting model evaluation...")
    
    # Load processed data
    df = pd.read_csv('/opt/airflow/dags/data/processed_housing.csv')
    
    # Split features and target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    
    # Split train/test (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Load saved model
    with open('/opt/airflow/dags/model/rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print("="*50 + "\n")
    
    # Save predictions
    results_df = pd.DataFrame({
        'actual_price': y_test.values,
        'predicted_price': y_pred
    })
    results_df.to_csv('/opt/airflow/dags/data/predictions.csv', index=False)
    print("Predictions saved to predictions.csv")
    
    print("Evaluation complete!")
    return None
