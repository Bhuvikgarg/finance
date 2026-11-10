"""
Data ingestion module for historical market and benchmark data using yfinance.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import yfinance as yf


def clean_yfinance_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten multi-level columns if present and ensure clean standardized column names."""
    if isinstance(df.columns, pd.MultiIndex):
        # Extract the level corresponding to Price types (Open, High, Low, Close, Volume)
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

    # Standardize column naming
    rename_dict = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_dict)
    
    # Ensure Index is Datetime and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Drop any rows where close price is missing
    if "close" in df.columns:
        df = df.dropna(subset=["close"])
        
    return df


def fetch_ticker_data(
    ticker: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker with optional local caching.
    """
    clean_ticker_name = ticker.replace("^", "IDX_").replace(".", "_")
    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{clean_ticker_name}.csv"

    import time

    # Check if cache exists and is fresh
    if cache_file and cache_file.exists() and not force_download:
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            last_date = df.index[-1].date() if not df.empty else None
            today = pd.Timestamp.now().date()
            cache_age_seconds = time.time() - cache_file.stat().st_mtime
            
            # If cache has today's date or is less than 4 hours old, use it
            if last_date == today or cache_age_seconds < 14400:
                print(f"[DataLoader] Cache is fresh (last date: {last_date}, age: {cache_age_seconds/60:.1f}m). Loading {ticker} from cache.")
                return df
            else:
                print(f"[DataLoader] Cache for {ticker} is outdated (last date: {last_date}, today: {today}). Auto-refreshing...")
        except Exception as e:
            print(f"[DataLoader] Failed reading cache ({e}), re-downloading {ticker}...")

    print(f"[DataLoader] Downloading {ticker} ({start_date} to {end_date or 'latest'})...")
    try:
        raw_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if raw_df.empty:
            if cache_file and cache_file.exists():
                print(f"[DataLoader] Download returned empty; falling back to existing cache for {ticker}.")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            raise ValueError(f"No market data returned for ticker: {ticker}")

        df = clean_yfinance_df(raw_df, ticker)

        if cache_file:
            df.to_csv(cache_file)
            print(f"[DataLoader] Updated cache for {ticker} at {cache_file}")

        return df
    except Exception as err:
        if cache_file and cache_file.exists():
            print(f"[DataLoader] Download error ({err}); falling back to cached file for {ticker}.")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        raise err


def load_market_dataset(
    stock_ticker: str = "RELIANCE.NS",
    benchmark_ticker: str = "^NSEI",
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load target stock data alongside benchmark context (e.g., NIFTY 50).
    """
    stock_df = fetch_ticker_data(
        ticker=stock_ticker,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    bench_df = fetch_ticker_data(
        ticker=benchmark_ticker,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    return stock_df, bench_df


if __name__ == "__main__":
    raw_cache = Path(__file__).resolve().parent.parent / "data" / "raw"
    stock_df, bench_df = load_market_dataset(cache_dir=raw_cache)
    print(f"\nStock data shape: {stock_df.shape}")
    print(stock_df.head(2))
    print(f"\nBenchmark data shape: {bench_df.shape}")
    print(bench_df.head(2))
