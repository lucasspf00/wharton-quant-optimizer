"""CSV/online daily adjusted-close price adapters. Yahoo availability is external."""
from __future__ import annotations

import io
import pandas as pd
from optimizer import PortfolioError, validate_prices


def parse_tickers(raw: str) -> list[str]:
    # BRK.B notation is accepted, but Yahoo uses BRK-B.
    symbols = [s.strip().upper().replace(".", "-") for s in raw.replace("\n", ",").split(",")]
    result = [s for s in symbols if s]
    if len(result) < 2 or len(set(result)) != len(result):
        raise PortfolioError("Enter at least two distinct ticker symbols separated by commas.")
    return result


def load_wide_csv(source: io.BytesIO | str) -> pd.DataFrame:
    try:
        df = pd.read_csv(source, index_col=0)
    except Exception as exc:
        raise PortfolioError(f"Unable to parse the CSV file: {exc}") from exc
    return validate_prices(df)


def fetch_yahoo_prices(tickers: list[str], history_years: int = 5) -> pd.DataFrame:
    if not 2 <= history_years <= 15:
        raise PortfolioError("Historical period must be between 2 and 15 years.")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise PortfolioError("yfinance is not installed; install requirements.txt.") from exc
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=history_years)
    try:
        raw = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                          interval="1d", auto_adjust=True, progress=False, threads=False)
    except Exception as exc:
        raise PortfolioError(f"Yahoo Finance download failed: {exc}") from exc
    if raw is None or raw.empty:
        raise PortfolioError("Yahoo Finance returned no prices. Try a CSV file.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1)
        else:
            raise PortfolioError("Downloaded data does not contain closing prices.")
    else:
        if "Close" not in raw.columns or len(tickers) != 1:
            raise PortfolioError("Unexpected Yahoo Finance column layout.")
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.columns = [str(x).upper() for x in close.columns]
    missing = [x for x in tickers if x not in close.columns]
    if missing:
        raise PortfolioError("No price history for: " + ", ".join(missing))
    return validate_prices(close.loc[:, tickers])
