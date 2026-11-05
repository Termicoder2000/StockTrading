from pathlib import Path
from urllib.parse import quote

import pandas as pd
import yfinance as yf


def load_close(ticker, start, end):
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if start_date >= end_date:
        raise ValueError("start must be earlier than end")

    ticker_key = quote(ticker.upper(), safe="")
    start_key = start_date.strftime("%Y-%m-%d")
    end_key = end_date.strftime("%Y-%m-%d")
    cache_path = Path(__file__).resolve().parent / f"{ticker_key}_{start_key}_{end_key}.csv"

    if cache_path.exists():
        print(f"Reading cached {ticker} data from '{cache_path.name}'")
        cached_data = pd.read_csv(cache_path, parse_dates=["Date"])
        if not {"Date", "Close"}.issubset(cached_data.columns):
            raise ValueError(f"Invalid cache schema in '{cache_path.name}'")
        if cached_data.empty or cached_data["Date"].duplicated().any():
            raise ValueError(f"Empty or duplicate dates in '{cache_path.name}'")

        close = cached_data.set_index("Date")["Close"].sort_index()
        close = pd.to_numeric(close, errors="raise").astype(float)
        if close.isna().any():
            raise ValueError(f"Missing close prices in '{cache_path.name}'")
        if close.index.min() < start_date or close.index.max() >= end_date:
            raise ValueError(f"Dates in '{cache_path.name}' do not match its filename")
        close.name = "Close"
        return close

    print(f"Downloading historical data for {ticker}...")
    downloaded_data = yf.download(ticker, start=start_key, end=end_key, progress=False)
    if downloaded_data.empty:
        raise ValueError(f"No data returned for {ticker} ({start_key} to {end_key})")

    if isinstance(downloaded_data.columns, pd.MultiIndex):
        close_data = downloaded_data["Close"]
        if isinstance(close_data, pd.DataFrame):
            if ticker in close_data.columns:
                close_data = close_data[ticker]
            elif close_data.shape[1] == 1:
                close_data = close_data.iloc[:, 0]
            else:
                raise ValueError(f"Could not select close prices for {ticker}")
    else:
        close_data = downloaded_data["Close"]

    close = close_data.dropna().sort_index().rename("Close")
    close.index.name = "Date"
    if close.empty:
        raise ValueError(f"No close prices returned for {ticker}")

    close.to_frame().to_csv(cache_path, date_format="%Y-%m-%d", float_format="%.17g")
    print(f"Saved downloaded data to '{cache_path.name}'")
    return close