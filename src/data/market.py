from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_dirs, load_config, save_json


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _load_csv_market(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # tolerate Date/Close casing from third-party dumps
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    if "date" not in df.columns:
        raise ValueError(f"{path} missing date column")
    if "close" not in df.columns:
        raise ValueError(f"{path} missing close column")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in ("open", "high", "low", "volume"):
        if col not in df.columns:
            df[col] = pd.NA
    return df


def fetch_analyzer_kse100() -> pd.DataFrame | None:
    """Archival KSE-100 daily close from PSX Analyzer static dataset (2010–2026)."""
    import requests

    urls = [
        "https://psx-analyzer-apollo.vercel.app/download/macro_kse100.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "psx-event-sentiment/0.1"})
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df["source"] = f"psx-analyzer:{url}"
            return df
        except Exception:
            continue
    return None


def collect_market_data(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Collect KSE-100 history: local CSV → analyzer archive → Yahoo fallback."""
    import yfinance as yf

    cfg = load_config()
    paths = ensure_dirs()
    start = start or cfg["project"]["start_date"]
    ticker = cfg["market"]["yahoo_ticker"]
    provenance: dict = {}

    manual = paths["data_raw"] / "market" / "kse100.csv"
    if manual.exists():
        df = _load_csv_market(manual)
        provenance = {
            "source": "local:data/raw/market/kse100.csv",
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
            "rows": int(len(df)),
            "limitation": (
                "Local CSV used. OHLC/volume may be incomplete. "
                "Sector indices and advance/decline not included."
            ),
        }
    else:
        df = fetch_analyzer_kse100()
        if df is not None:
            provenance = {
                "source": str(df["source"].iloc[0]) if "source" in df.columns else "psx-analyzer",
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
                "rows": int(len(df)),
                "limitation": (
                    "PSX Analyzer archival composite (PSX portal + historical Yahoo ^KSE). "
                    "Volume/OHLC often unavailable for index series."
                ),
            }
            df.to_csv(paths["data_raw"] / "market" / "kse100.csv")
        else:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            raw = _flatten_columns(raw)
            if raw.empty:
                raise RuntimeError(
                    f"No market data from analyzer or {ticker}. "
                    "Supply a CSV under data/raw/market/kse100.csv"
                )
            df = raw.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Adj Close": "close",
                }
            )
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[keep].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            df = df.sort_index()
            df["source"] = f"yfinance:{ticker}"
            provenance = {
                "ticker": ticker,
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
                "rows": int(len(df)),
                "source": f"yfinance:{ticker}",
                "limitation": "Yahoo coverage may diverge from official PSX KSE-100.",
            }
            df.to_csv(paths["data_raw"] / "market" / "kse100_yahoo.csv")

    df["index_name"] = cfg["market"]["primary_index"]
    if "source" not in df.columns:
        df["source"] = provenance.get("source", "unknown")
    save_json(provenance, paths["data_raw"] / "market" / "provenance.json")
    return df


def load_or_collect_market() -> pd.DataFrame:
    paths = ensure_dirs()
    manual = paths["data_raw"] / "market" / "kse100.csv"
    local = paths["data_raw"] / "market" / "kse100_yahoo.csv"
    if manual.exists():
        df = _load_csv_market(manual)
        if "source" not in df.columns:
            df["source"] = "local_csv"
        return df
    if local.exists():
        return _load_csv_market(local)
    return collect_market_data()


def compute_market_metrics(prices: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg = load_config()
    paths = ensure_dirs()
    prices = prices if prices is not None else load_or_collect_market()
    df = prices.copy()
    if "close" not in df.columns:
        raise ValueError("Market frame requires a 'close' column")

    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_21d"] = df["close"].pct_change(21)
    df["log_ret"] = np.log(df["close"]).diff()
    df["vol_21d"] = df["ret_1d"].rolling(21).std() * np.sqrt(252)
    df["vol_63d"] = df["ret_1d"].rolling(63).std() * np.sqrt(252)
    if "volume" not in df.columns:
        df["volume"] = np.nan
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    vol_std = df["volume"].rolling(63).std()
    df["volume_z"] = np.where(vol_std > 0, (df["volume"] - df["volume"].rolling(63).mean()) / vol_std, np.nan)

    # Abnormal returns vs rolling mean (simple market model without beta peer)
    roll_mean = df["ret_1d"].rolling(63).mean()
    roll_std = df["ret_1d"].rolling(63).std()
    df["abnormal_ret"] = df["ret_1d"] - roll_mean
    df["abnormal_z"] = df["abnormal_ret"] / roll_std
    z = cfg["market"]["abnormal_z_threshold"]
    df["abnormal_session"] = df["abnormal_z"].abs() >= z
    df["abnormal_direction"] = np.where(
        df["abnormal_session"] & (df["abnormal_z"] > 0),
        "positive",
        np.where(df["abnormal_session"] & (df["abnormal_z"] < 0), "negative", "normal"),
    )

    # Drawdowns
    cummax = df["close"].cummax()
    df["drawdown"] = df["close"] / cummax - 1.0
    df["max_drawdown_252d"] = df["drawdown"].rolling(252).min()

    # Trend / breadth proxies (breadth unavailable → use price trend)
    df["trend_63d"] = df["close"] / df["close"].rolling(63).mean() - 1.0
    df["market_breadth"] = np.nan  # requires constituent A/D data
    df["advance_decline"] = np.nan

    out = paths["data_processed"] / "market_metrics.csv"
    df.to_csv(out)
    df.to_csv(paths["output"] / "market" / "market_metrics.csv")
    abn = df.loc[df["abnormal_session"]].copy()
    abn.to_csv(paths["output"] / "market" / "abnormal_sessions.csv")
    return df
