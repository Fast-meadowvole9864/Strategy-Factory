import os
import time
import argparse
import pandas as pd
import polars as pl
import ccxt
from tqdm import tqdm
from datetime import datetime

# Global settings
VAULT_1 = 'Vault_1_InSample'
VAULT_2 = 'Vault_2_OOS'
VAULT_3 = 'Vault_3_Holdout'

START_DATE_STR = '2020-02-01T00:00:00Z'
TIMEFRAMES = ['1h', '15m', '1m']
ASSETS = ['BTCUSDT.P', 'ETHUSDT.P', 'XRPUSDT.P', 'LTCUSDT.P']

# Initialize Binance Futures exchange
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
    }
})

def get_binance_symbol(raw_symbol: str) -> str:
    """Convert 'BTCUSDT.P' to 'BTC/USDT:USDT' for ccxt futures."""
    base = raw_symbol.replace('USDT.P', '')
    return f"{base}/USDT:USDT"

def get_vault_for_year(year: int) -> str:
    """Route data into the correct Vault based on the year."""
    if year <= 2023:
        return VAULT_1
    elif year == 2024:
        return VAULT_2
    else:
        return VAULT_3

def get_ms_from_timeframe(tf: str) -> int:
    """Convert string timeframe to milliseconds."""
    if tf == '1m': return 60000
    if tf == '15m': return 15 * 60000
    if tf == '1h': return 60 * 60000
    return 60000

def fetch_and_route_data(raw_symbol: str, timeframe: str, base_dir: str) -> None:
    """
    Fetches historical OHLCV data from Binance Futures, handles pagination, 
    and routes the data into Parquet files split by year into Vaults.
    """
    tf_ms = get_ms_from_timeframe(timeframe)
    start_since = exchange.parse8601(START_DATE_STR)
    request_symbol = raw_symbol.replace('.P', '')
    
    # Format folder and file names
    folder_name = raw_symbol.replace('USDT.P', '').lower() # e.g., 'btc'
    file_name = f"{raw_symbol.replace('.P', '').lower()}_{timeframe}.parquet" # e.g., 'btcusdt_15m.parquet'
    
    # 1. Load existing data across vaults to find where to resume
    existing_dfs = []
    for vault in [VAULT_1, VAULT_2, VAULT_3]:
        file_path = os.path.join(base_dir, vault, folder_name, timeframe, file_name)
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                if not df.empty:
                    existing_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    if existing_dfs:
        combined_df = pd.concat(existing_dfs, ignore_index=True)
        combined_df.drop_duplicates(subset=['timestamp'], inplace=True)
        combined_df.sort_values('timestamp', inplace=True)
        last_ts = int(combined_df['timestamp'].iloc[-1])
        since = last_ts + tf_ms
        all_ohlcv = combined_df.to_dict('records')
        print(f"[{raw_symbol} - {timeframe}] Resuming from {exchange.iso8601(since)}")
    else:
        since = start_since
        all_ohlcv = []
        print(f"[{raw_symbol} - {timeframe}] Starting fresh from {START_DATE_STR}")

    now = exchange.milliseconds()
    limit = 1000  # Binance futures maximum safe limit
    
    # Calculate total expected requests for tqdm
    if now > since:
        total_bars_expected = (now - since) // tf_ms
        total_requests = int(total_bars_expected // limit) + 1
    else:
        total_requests = 0

    new_data_fetched = False
    if total_requests > 0:
        pbar = tqdm(total=total_requests, desc=f"Fetching {raw_symbol} {timeframe}")
        while since < now:
            try:
                # Use fapiPublicGetKlines to get full 12 columns
                klines = exchange.fapiPublicGetKlines({'symbol': request_symbol, 'interval': timeframe, 'startTime': since, 'limit': limit})
                if not klines:
                    break
                
                all_ohlcv.extend([
                    {
                        'timestamp': int(x[0]),
                        'open': float(x[1]),
                        'high': float(x[2]),
                        'low': float(x[3]),
                        'close': float(x[4]),
                        'volume': float(x[5]),
                        'taker_buy_vol': float(x[9])
                    } for x in klines
                ])
                
                since = int(klines[-1][0]) + tf_ms
                new_data_fetched = True
                pbar.update(1)
                
                # Sleep to respect rate limits
                time.sleep(exchange.rateLimit / 1000.0)
                
                if len(klines) < limit:
                    break
                    
            except Exception as e:
                print(f"\nError fetching {request_symbol}: {e}")
                break
        pbar.close()
        
    if not all_ohlcv:
        print(f"[{raw_symbol} - {timeframe}] No data available.")
        return

    if not new_data_fetched and existing_dfs:
        print(f"[{raw_symbol} - {timeframe}] Data is already up to date.")
        return

    # Process and route
    df = pd.DataFrame(all_ohlcv)
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    
    # Ensure exact schema order
    expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_vol']
    df = df[expected_cols]
    
    # Convert timestamp to datetime to extract year
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['year'] = df['datetime'].dt.year
    df['vault'] = df['year'].apply(get_vault_for_year)
    
    # Route by vault
    for vault, group in df.groupby('vault'):
        vault_dir = os.path.join(base_dir, vault, folder_name, timeframe)
        os.makedirs(vault_dir, exist_ok=True)
        
        file_path = os.path.join(vault_dir, file_name)
        
        # Drop the temporary columns before saving
        save_df = group.drop(columns=['datetime', 'year', 'vault'])
        save_df.to_parquet(file_path, index=False)
        
    print(f"[{raw_symbol} - {timeframe}] Successfully routed and saved {len(df)} total rows.\n")

def run_downloader(target_timeframe: str = None) -> None:
    """Main execution orchestrator function."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    tfs_to_run = [target_timeframe] if target_timeframe else TIMEFRAMES
        
    print(f"Starting Binance Futures download for {len(ASSETS)} assets...")
    for asset in ASSETS:
        for tf in tfs_to_run:
            try:
                fetch_and_route_data(asset, tf, base_dir)
            except Exception as e:
                print(f"Critical failure on {asset} ({tf}): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance Futures data.")
    parser.add_argument("--timeframe", type=str, choices=TIMEFRAMES, help="Target a specific timeframe to download instead of all of them.")
    args = parser.parse_args()
    
    run_downloader(args.timeframe)
