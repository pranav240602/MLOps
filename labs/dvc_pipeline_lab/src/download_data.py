import os
import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime
from tqdm import tqdm


def load_params(params_path='params.yaml'):
    """Load parameters from YAML file."""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params['download']


def download_stock_data(ticker, start_date, end_date, interval='1d'):
    """
    Download stock data for a single ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        interval: Data interval (default: '1d' for daily)
    
    Returns:
        DataFrame with stock data
    """
    print(f"Downloading data for {ticker}...")
    
    try:
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            print(f"Warning: No data found for {ticker}")
            return None
        
        # Add ticker column
        df['Ticker'] = ticker
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        print(f"Successfully downloaded {len(df)} records for {ticker}")
        return df
        
    except Exception as e:
        print(f"Error downloading {ticker}: {str(e)}")
        return None


def main():
    """Main function to download all stock data."""
    
    # Load parameters
    params = load_params()
    tickers = params['tickers']
    start_date = params['start_date']
    end_date = params['end_date']
    interval = params['interval']
    
    print("=" * 60)
    print("Stock Data Download")
    print("=" * 60)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    print("=" * 60)
    
    # Create output directory
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    
    # Download data for each ticker
    all_data = []
    
    for ticker in tqdm(tickers, desc="Downloading stocks"):
        df = download_stock_data(ticker, start_date, end_date, interval)
        
        if df is not None:
            # Save individual stock data
            output_file = os.path.join(output_dir, f'{ticker}.csv')
            df.to_csv(output_file, index=False)
            print(f"Saved {ticker} data to {output_file}")
            
            all_data.append(df)
    
    # Combine all data into one file
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_file = os.path.join(output_dir, 'combined_stocks.csv')
        combined_df.to_csv(combined_file, index=False)
        print(f"\nCombined data saved to {combined_file}")
        print(f"Total records: {len(combined_df)}")
        print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    else:
        print("\nWarning: No data was downloaded!")
    
    print("\nDownload complete!")


if __name__ == '__main__':
    main()
