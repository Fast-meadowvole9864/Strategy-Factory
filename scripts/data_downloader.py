import os
import sys
import time
import argparse
import pandas as pd
import ccxt
from tqdm import tqdm

# Global settings
VAULT_1 = 'Vault_1_InSample'
VAULT_2 = 'Vault_2_OOS'
VAULT_3 = 'Vault_3_Holdout'

DEFAULT_START_DATE_STR = '2020-02-01T00:00:00Z'
SUPPORTED_TIMEFRAMES = ['1h', '15m', '1m']
DEFAULT_TIMEFRAMES = ['1h', '15m', '1m']
DEFAULT_ASSETS = ['BTCUSDT.P', 'ETHUSDT.P', 'XRPUSDT.P', 'LTCUSDT.P']
DEFAULT_VAULTS = [VAULT_1, VAULT_2, VAULT_3]

# Initialize Binance Futures exchange
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
    }
})

def normalize_asset(raw_asset: str) -> str:
    """Normalize shorthand asset input into the repo's perpetual symbol format."""
    asset = raw_asset.strip().upper()
    if not asset:
        raise ValueError("Asset names cannot be empty.")

    core = asset[:-2] if asset.endswith(".P") else asset
    if core.endswith("USDT"):
        return f"{core}.P"
    if core.isalnum():
        return f"{core}USDT.P"

    raise ValueError(f"Unsupported asset format: {raw_asset}")

def normalize_date(raw_date: str, arg_name: str) -> str:
    """Accept YYYY-MM-DD or ISO8601 and normalize to a UTC ISO8601 string."""
    candidate = raw_date.strip()
    if len(candidate) == 10 and candidate[4] == "-" and candidate[7] == "-":
        candidate = f"{candidate}T00:00:00Z"

    parsed = exchange.parse8601(candidate)
    if parsed is None:
        raise ValueError(f"Invalid {arg_name}: {raw_date}. Use YYYY-MM-DD or ISO8601.")

    return exchange.iso8601(parsed)

def ensure_execution_timeframe(timeframes: list[str]) -> list[str]:
    """
    Stage 2 and Stage 3 require 1m data for execution simulation, so avoid
    silently building an incomplete vault layout.
    """
    ordered_unique = list(dict.fromkeys(timeframes))
    if "1m" in ordered_unique:
        return ordered_unique

    warning = (
        "Stage 2 and Stage 3 require 1m execution data to avoid intrabar trade assumptions."
    )

    if sys.stdin.isatty():
        reply = input(f"{warning} Add 1m to this download as well? [Y/n]: ").strip().lower()
        if reply in {"", "y", "yes"}:
            return ordered_unique + ["1m"]
        raise SystemExit("Download aborted because Stage 2 and Stage 3 require 1m data in the vaults.")

    print(f"{warning} Automatically adding 1m to the requested timeframes.")
    return ordered_unique + ["1m"]

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

def fetch_and_route_data(raw_symbol: str, timeframe: str, base_dir: str, start_date: str, end_date: str | None = None) -> None:
    """
    Fetches historical OHLCV data from Binance Futures, handles pagination, 
    and routes the data into Parquet files split by year into Vaults.
    """
    tf_ms = get_ms_from_timeframe(timeframe)
    start_since = exchange.parse8601(start_date)
    end_since = exchange.parse8601(end_date) if end_date else exchange.milliseconds()
    if end_since is None or start_since is None:
        raise ValueError("Could not parse the requested date range.")
    if end_since <= start_since:
        raise ValueError("The requested end date must be later than the start date.")

    request_symbol = raw_symbol.replace('.P', '')
    
    # Format folder and file names
    folder_name = raw_symbol.replace('USDT.P', '').lower() # e.g., 'btc'
    file_name = f"{raw_symbol.replace('.P', '').lower()}_{timeframe}.parquet" # e.g., 'btcusdt_15m.parquet'
    
    # 1. Load existing data across vaults to find where to resume
    existing_dfs = []
    for vault in DEFAULT_VAULTS:
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
        since = max(last_ts + tf_ms, start_since)
        all_ohlcv = combined_df.to_dict('records')
        print(f"[{raw_symbol} - {timeframe}] Resuming from {exchange.iso8601(since)}")
    else:
        since = start_since
        all_ohlcv = []
        print(f"[{raw_symbol} - {timeframe}] Starting fresh from {start_date}")

    fetch_until = min(exchange.milliseconds(), end_since)
    if since >= fetch_until:
        print(f"[{raw_symbol} - {timeframe}] Data is already up to date for the requested range.")
        return

    limit = 1000  # Binance futures maximum safe limit
    
    # Calculate total expected requests for tqdm
    if fetch_until > since:
        total_bars_expected = (fetch_until - since) // tf_ms
        total_requests = int(total_bars_expected // limit) + 1
    else:
        total_requests = 0

    new_data_fetched = False
    if total_requests > 0:
        pbar = tqdm(total=total_requests, desc=f"Fetching {raw_symbol} {timeframe}")
        while since < fetch_until:
            try:
                # Use fapiPublicGetKlines to get full 12 columns
                klines = exchange.fapiPublicGetKlines({
                    'symbol': request_symbol,
                    'interval': timeframe,
                    'startTime': since,
                    'endTime': fetch_until,
                    'limit': limit
                })
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
                
                if len(klines) < limit or since >= fetch_until:
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

def run_downloader(assets: list[str] | None = None, timeframes: list[str] | None = None, start_date: str = DEFAULT_START_DATE_STR, end_date: str | None = None) -> None:
    """Main execution orchestrator function."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    normalized_start_date = normalize_date(start_date, "--start-date")
    normalized_end_date = normalize_date(end_date, "--end-date") if end_date else None
    selected_assets = [normalize_asset(asset) for asset in (assets or DEFAULT_ASSETS)]
    selected_timeframes = ensure_execution_timeframe(timeframes or list(DEFAULT_TIMEFRAMES))

    print(f"Starting Binance Futures download for {len(selected_assets)} assets across {selected_timeframes}...")
    for asset in selected_assets:
        for tf in selected_timeframes:
            try:
                fetch_and_route_data(asset, tf, base_dir, normalized_start_date, normalized_end_date)
            except Exception as e:
                print(f"Critical failure on {asset} ({tf}): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance Futures data.")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS, help="Assets to download, e.g. BTC, BTCUSDT, or BTCUSDT.P")
    parser.add_argument("--timeframes", nargs="+", choices=SUPPORTED_TIMEFRAMES, help="One or more macro/execution timeframes to download.")
    parser.add_argument("--timeframe", type=str, choices=SUPPORTED_TIMEFRAMES, help="Legacy alias for a single timeframe.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE_STR, help="Start date in YYYY-MM-DD or ISO8601 format.")
    parser.add_argument("--end-date", help="Optional end date in YYYY-MM-DD or ISO8601 format. Defaults to now.")
    args = parser.parse_args()

    if args.timeframe and args.timeframes:
        parser.error("Use either --timeframe or --timeframes, not both.")

    chosen_timeframes = args.timeframes or ([args.timeframe] if args.timeframe else list(DEFAULT_TIMEFRAMES))
    
    run_downloader(args.assets, chosen_timeframes, args.start_date, args.end_date)
