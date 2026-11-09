from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "Adj Close"] if c in df.columns]
    out = df[keep].copy()
    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def fetch_yahoo(ticker: str, start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = _flatten_yf(raw)
    out["ticker"] = ticker
    return out


def fetch_psx_eod(symbol: str = "KSE100") -> pd.DataFrame:
    """Best-effort PSX data portal timeseries fetch."""
    import requests

    url = f"https://dps.psx.com.pk/timeseries/eod/{symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; PSX-ESI-Research/0.1)",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://dps.psx.com.pk/",
    }
    try:
        r = requests.get(url, headers=headers, timeout=45)
        if r.status_code != 200:
            return pd.DataFrame()
        payload = r.json()
    except Exception:
        return pd.DataFrame()

    # Common shapes: {"data": [[ts, close, vol], ...]} or list of dicts
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        if isinstance(row, dict):
            records.append(row)
            continue
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            ts, close = row[0], row[1]
            vol = row[2] if len(row) > 2 else np.nan
            open_ = row[3] if len(row) > 3 else np.nan
            records.append({"date": ts, "Close": close, "Volume": vol, "Open": open_})
    df = pd.DataFrame(records)
    if df.empty:
        return df
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], unit="s", errors="coerce")
    if df[date_col].isna().all():
        df[date_col] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df = df.set_index(date_col).sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    rename = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename)
    if "Close" not in df.columns:
        # guess first numeric
        num = df.select_dtypes(include=[np.number]).columns
        if len(num):
            df = df.rename(columns={num[0]: "Close"})
    df["ticker"] = symbol
    return df


def stitch_kse100(yahoo_df: pd.DataFrame, psx_df: pd.DataFrame) -> pd.DataFrame:
    """Prefer Yahoo for older history and PSX for overlap/newer bars when available."""
    frames = []
    if yahoo_df is not None and not yahoo_df.empty:
        y = yahoo_df.copy()
        y["source"] = "yahoo"
        frames.append(y)
    if psx_df is not None and not psx_df.empty:
        p = psx_df.copy()
        p["source"] = "psx"
        frames.append(p)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames).sort_index()
    # Prefer PSX on overlapping dates
    all_df = all_df[~all_df.index.duplicated(keep="last")]
    return all_df


def _read_frame(path_parquet: Path, path_csv: Path) -> pd.DataFrame:
    if path_parquet.exists():
        try:
            return pd.read_parquet(path_parquet)
        except Exception:
            pass
    if path_csv.exists():
        df = pd.read_csv(path_csv, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    return pd.DataFrame()


def _write_frame(df: pd.DataFrame, path_parquet: Path, path_csv: Path) -> None:
    df.to_csv(path_csv)
    try:
        df.to_parquet(path_parquet)
    except Exception:
        pass


def load_or_fetch_market(market_dir: str | Path, start: str = "2000-01-01") -> dict[str, pd.DataFrame]:
    market_dir = Path(market_dir)
    market_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    kse_parquet = market_dir / "kse100.parquet"
    kse_csv = market_dir / "kse100.csv"
    kse = _read_frame(kse_parquet, kse_csv)
    if kse.empty:
        yahoo = fetch_yahoo("^KSE", start=start)
        if yahoo.empty:
            yahoo = fetch_yahoo("KSE100.KA", start=start)
        psx = fetch_psx_eod("KSE100")
        kse = stitch_kse100(yahoo, psx)
        if not kse.empty:
            _write_frame(kse, kse_parquet, kse_csv)
    out["KSE100"] = kse

    macros = {
        "USDPKR": "USDPKR=X",
        "OIL": "CL=F",
        "GOLD": "GC=F",
        "SP500": "^GSPC",
        "EM": "EEM",
    }
    for name, ticker in macros.items():
        path_p = market_dir / f"{name.lower()}.parquet"
        path_c = market_dir / f"{name.lower()}.csv"
        df = _read_frame(path_p, path_c)
        if df.empty:
            df = fetch_yahoo(ticker, start=start)
            if not df.empty:
                _write_frame(df, path_p, path_c)
        out[name] = df
    return out


def save_market_meta(market_dir: str | Path, series: dict[str, pd.DataFrame]) -> None:
    meta = {}
    for name, df in series.items():
        meta[name] = {
            "rows": int(len(df)),
            "start": None if df.empty else str(pd.Timestamp(df.index.min()).date()),
            "end": None if df.empty else str(pd.Timestamp(df.index.max()).date()),
            "columns": list(map(str, df.columns)),
        }
    Path(market_dir).mkdir(parents=True, exist_ok=True)
    Path(market_dir, "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
