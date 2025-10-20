import pandas as pd

def feature_engineering():
    """Perform feature engineering on the dataset"""
    print("Starting feature engineering...")
    df = pd.read_csv('/opt/airflow/dags/data/housing.csv')
    
    # Handle missing values
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    
    # Create new features
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    
    print("New features created: rooms_per_household, bedrooms_per_room, population_per_household")
    
    # Convert ocean_proximity to numeric (one-hot encoding)
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
    
    print(f"Feature engineering completed! New shape: {df.shape}")
    
    # Save processed data
    df.to_csv('/opt/airflow/dags/data/processed_housing.csv', index=False)
    print("Processed data saved!")
    
    return "Feature engineering complete!"
