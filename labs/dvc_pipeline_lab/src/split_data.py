"""
Split stock market data into training and testing sets.
Maintains time series order (no shuffling).
"""

import os
import yaml
import pandas as pd
import numpy as np


def load_params(params_path='params.yaml'):
    """Load parameters from YAML file."""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params['split']


def load_features(data_path='data/features/stock_data_features.csv'):
    """Load feature data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df)} records")
    return df


def split_data(df, params):
    """
    Split data into train and test sets.
    For time series, we don't shuffle and use temporal split.
    
    Args:
        df: Input DataFrame
        params: Split parameters
    
    Returns:
        train_df, test_df
    """
    print("\nSplitting data...")
    
    test_size = params['test_size']
    shuffle = params['shuffle']
    
    if shuffle:
        # Random split (not recommended for time series)
        print(f"WARNING: Shuffling time series data!")
        df = df.sample(frac=1, random_state=params['random_state'])
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    else:
        # Temporal split - split by date for each ticker
        print(f"Temporal split with test_size={test_size}")
        
        train_dfs = []
        test_dfs = []
        
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].sort_values('Date')
            split_idx = int(len(ticker_df) * (1 - test_size))
            
            train_dfs.append(ticker_df.iloc[:split_idx])
            test_dfs.append(ticker_df.iloc[split_idx:])
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
    
    print(f"\nTrain set: {len(train_df)} records")
    print(f"  Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"  Tickers: {train_df['Ticker'].nunique()}")
    
    print(f"\nTest set: {len(test_df)} records")
    print(f"  Date range: {test_df['Date'].min()} to {test_df['Date'].max()}")
    print(f"  Tickers: {test_df['Ticker'].nunique()}")
    
    return train_df, test_df


def save_splits(train_df, test_df, output_dir='data/features'):
    """Save train and test splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nTrain data saved to {train_path}")
    print(f"Test data saved to {test_path}")


def main():
    """Main split function."""
    
    print("=" * 60)
    print("Data Splitting")
    print("=" * 60)
    
    # Load parameters
    params = load_params()
    
    # Load feature data
    df = load_features()
    
    # Split data
    train_df, test_df = split_data(df, params)
    
    # Save splits
    save_splits(train_df, test_df)
    
    print("\nData splitting complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
