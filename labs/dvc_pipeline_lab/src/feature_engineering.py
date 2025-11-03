"""
Feature engineering for stock market data.
Creates technical indicators and advanced features.
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


def load_params(params_path='params.yaml'):
    """Load parameters from YAML file."""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params['feature_engineering']


def load_processed_data(data_path='data/processed/stock_data_cleaned.csv'):
    """Load processed stock data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df)} records")
    return df


def add_moving_averages(df, params):
    """
    Add Moving Average features.
    
    Args:
        df: Input DataFrame
        params: Feature engineering parameters
    
    Returns:
        DataFrame with MA features
    """
    print("\nAdding Moving Averages...")
    
    short_window = params['ma_window_short']
    long_window = params['ma_window_long']
    
    # Calculate MAs for each ticker
    df[f'MA_{short_window}'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=short_window, min_periods=1).mean()
    )
    
    df[f'MA_{long_window}'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=long_window, min_periods=1).mean()
    )
    
    # MA crossover signal
    df['MA_Crossover'] = (df[f'MA_{short_window}'] > df[f'MA_{long_window}']).astype(int)
    
    print(f"Added MA_{short_window}, MA_{long_window}, MA_Crossover")
    
    return df


def add_rsi(df, params):
    """
    Add RSI (Relative Strength Index) feature.
    
    Args:
        df: Input DataFrame
        params: Feature engineering parameters
    
    Returns:
        DataFrame with RSI feature
    """
    print("\nAdding RSI...")
    
    period = params['rsi_period']
    
    # Calculate RSI for each ticker
    def calculate_rsi(group):
        rsi = RSIIndicator(close=group['Close'], window=period)
        return rsi.rsi()
    
    df['RSI'] = df.groupby('Ticker', group_keys=False).apply(calculate_rsi).values
    
    print(f"Added RSI_{period}")
    
    return df


def add_macd(df, params):
    """
    Add MACD (Moving Average Convergence Divergence) features.
    
    Args:
        df: Input DataFrame
        params: Feature engineering parameters
    
    Returns:
        DataFrame with MACD features
    """
    print("\nAdding MACD...")
    
    fast = params['macd_fast']
    slow = params['macd_slow']
    signal = params['macd_signal']
    
    # Calculate MACD for each ticker
    def calculate_macd(group):
        macd = MACD(
            close=group['Close'],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal
        )
        return pd.DataFrame({
            'MACD': macd.macd(),
            'MACD_Signal': macd.macd_signal(),
            'MACD_Diff': macd.macd_diff()
        })
    
    macd_features = df.groupby('Ticker', group_keys=False).apply(calculate_macd)
    df['MACD'] = macd_features['MACD'].values
    df['MACD_Signal'] = macd_features['MACD_Signal'].values
    df['MACD_Diff'] = macd_features['MACD_Diff'].values
    
    print(f"Added MACD, MACD_Signal, MACD_Diff")
    
    return df


def add_bollinger_bands(df, params):
    """
    Add Bollinger Bands features.
    
    Args:
        df: Input DataFrame
        params: Feature engineering parameters
    
    Returns:
        DataFrame with Bollinger Bands features
    """
    print("\nAdding Bollinger Bands...")
    
    window = params['bb_window']
    std_dev = params['bb_std']
    
    # Calculate Bollinger Bands for each ticker
    def calculate_bb(group):
        bb = BollingerBands(
            close=group['Close'],
            window=window,
            window_dev=std_dev
        )
        return pd.DataFrame({
            'BB_High': bb.bollinger_hband(),
            'BB_Mid': bb.bollinger_mavg(),
            'BB_Low': bb.bollinger_lband(),
            'BB_Width': bb.bollinger_wband(),
            'BB_Pct': bb.bollinger_pband()
        })
    
    bb_features = df.groupby('Ticker', group_keys=False).apply(calculate_bb)
    df['BB_High'] = bb_features['BB_High'].values
    df['BB_Mid'] = bb_features['BB_Mid'].values
    df['BB_Low'] = bb_features['BB_Low'].values
    df['BB_Width'] = bb_features['BB_Width'].values
    df['BB_Pct'] = bb_features['BB_Pct'].values
    
    print(f"Added BB_High, BB_Mid, BB_Low, BB_Width, BB_Pct")
    
    return df


def add_lag_features(df, lags=[1, 5, 10]):
    """
    Add lagged features.
    
    Args:
        df: Input DataFrame
        lags: List of lag periods
    
    Returns:
        DataFrame with lag features
    """
    print(f"\nAdding lag features for lags: {lags}...")
    
    for lag in lags:
        df[f'Close_Lag_{lag}'] = df.groupby('Ticker')['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df.groupby('Ticker')['Volume'].shift(lag)
    
    print(f"Added {len(lags) * 2} lag features")
    
    return df


def calculate_data_quality_metrics(df):
    """
    Calculate data quality metrics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with metrics
    """
    print("\nCalculating data quality metrics...")
    
    metrics = {
        'total_records': int(len(df)),
        'unique_tickers': int(df['Ticker'].nunique()),
        'date_range': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d')
        },
        'missing_values': {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        'tickers': df['Ticker'].unique().tolist(),
        'records_per_ticker': {k: int(v) for k, v in df['Ticker'].value_counts().to_dict().items()},
        'price_statistics': {
            ticker: {
                'mean': float(group['Close'].mean()),
                'std': float(group['Close'].std()),
                'min': float(group['Close'].min()),
                'max': float(group['Close'].max())
            }
            for ticker, group in df.groupby('Ticker')
        },
        'feature_count': len(df.columns)
    }
    
    return metrics


def save_features(df, output_path='data/features/stock_data_features.csv'):
    """Save engineered features."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nFeature data saved to {output_path}")
    print(f"Total features: {len(df.columns)}")


def save_metrics(metrics, output_path='metrics/data_quality.json'):
    """Save data quality metrics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")


def main():
    """Main feature engineering function."""
    
    print("=" * 60)
    print("Feature Engineering")
    print("=" * 60)
    
    # Load parameters
    params = load_params()
    
    # Load processed data
    df = load_processed_data()
    
    # Add features
    df = add_moving_averages(df, params)
    df = add_rsi(df, params)
    df = add_macd(df, params)
    df = add_bollinger_bands(df, params)
    df = add_lag_features(df)
    
    # Remove rows with NaN created by indicators
    initial_len = len(df)
    df = df.dropna()
    print(f"\nRemoved {initial_len - len(df)} rows with NaN from feature calculations")
    
    # Calculate metrics
    metrics = calculate_data_quality_metrics(df)
    
    # Save outputs
    save_features(df)
    save_metrics(metrics)
    
    print("\nFeature engineering complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
