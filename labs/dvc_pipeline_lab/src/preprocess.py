
import os
import yaml
import pandas as pd
import numpy as np


def load_params(params_path='params.yaml'):
    """Load parameters from YAML file."""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params['preprocess']


def load_raw_data(data_path='data/raw/combined_stocks.csv'):
    """Load raw stock data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    return df


def clean_data(df, params):
    """
    Clean the stock data.
    
    Args:
        df: Input DataFrame
        params: Preprocessing parameters
    
    Returns:
        Cleaned DataFrame
    """
    print("\nCleaning data...")
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by Ticker and Date
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Check for missing values
    print(f"\nMissing values before cleaning:")
    print(df.isnull().sum())
    
    # Handle missing values
    if params['remove_nulls']:
        initial_len = len(df)
        df = df.dropna()
        print(f"Removed {initial_len - len(df)} rows with missing values")
    else:
        # Forward fill missing values within each ticker
        fill_method = params['fill_method']
        df = df.groupby('Ticker').apply(lambda x: x.fillna(method=fill_method))
        df = df.reset_index(drop=True)
        print(f"Filled missing values using {fill_method} method")
    
    # Remove stocks with insufficient data
    min_points = params['min_data_points']
    ticker_counts = df['Ticker'].value_counts()
    valid_tickers = ticker_counts[ticker_counts >= min_points].index
    
    initial_tickers = df['Ticker'].nunique()
    df = df[df['Ticker'].isin(valid_tickers)]
    final_tickers = df['Ticker'].nunique()
    
    if initial_tickers != final_tickers:
        print(f"Removed {initial_tickers - final_tickers} tickers with less than {min_points} data points")
    
    # Remove duplicate entries
    initial_len = len(df)
    df = df.drop_duplicates(subset=['Ticker', 'Date'], keep='first')
    if initial_len != len(df):
        print(f"Removed {initial_len - len(df)} duplicate entries")
    
    # Basic data validation
    df = df[df['Volume'] >= 0]  # Remove negative volumes
    df = df[df['Close'] > 0]    # Remove zero or negative prices
    
    print(f"\nData after cleaning:")
    print(f"Total records: {len(df)}")
    print(f"Unique tickers: {df['Ticker'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


def add_basic_features(df):
    """
    Add basic calculated features.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with additional features
    """
    print("\nAdding basic features...")
    
    # Daily returns
    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
    
    # Price change
    df['Price_Change'] = df.groupby('Ticker')['Close'].diff()
    
    # High-Low spread
    df['HL_Spread'] = df['High'] - df['Low']
    
    # Close-Open spread
    df['CO_Spread'] = df['Close'] - df['Open']
    
    print("Added features: Daily_Return, Price_Change, HL_Spread, CO_Spread")
    
    return df


def save_processed_data(df, output_path='data/processed/stock_data_cleaned.csv'):
    """Save processed data."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")


def main():
    """Main preprocessing function."""
    
    print("=" * 60)
    print("Stock Data Preprocessing")
    print("=" * 60)
    
    # Load parameters
    params = load_params()
    
    # Load raw data
    df = load_raw_data()
    
    # Clean data
    df = clean_data(df, params)
    
    # Add basic features
    df = add_basic_features(df)
    
    # Save processed data
    save_processed_data(df)
    
    print("\nPreprocessing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
