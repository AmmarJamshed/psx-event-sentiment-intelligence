from __future__ import annotations

import pandas as pd

from src.utils import ensure_dirs, load_config, save_json


def _load_local_macro(paths) -> tuple[pd.DataFrame | None, dict]:
    provenance: dict = {}
    frames = []
    usdpkr = paths["data_raw"] / "macro" / "macro_usdpkr.csv"
    cpi = paths["data_raw"] / "macro" / "macro_cpi.csv"
    tbill = paths["data_raw"] / "macro" / "macro_tbill.csv"

    if usdpkr.exists():
        fx = pd.read_csv(usdpkr, parse_dates=["date"]).set_index("date").sort_index()
        frames.append(fx["close"].rename("usdpkr"))
        provenance["usdpkr"] = {"status": "ok", "source": str(usdpkr)}
    if cpi.exists():
        c = pd.read_csv(cpi, parse_dates=["date"]).set_index("date").sort_index()
        col = "cpi_yoy" if "cpi_yoy" in c.columns else c.columns[0]
        frames.append(c[col].rename("inflation_cpi"))
        provenance["inflation_cpi"] = {"status": "ok", "source": str(cpi), "note": "annual series forward-filled later"}
    if tbill.exists():
        t = pd.read_csv(tbill, parse_dates=["date"]).set_index("date").sort_index()
        col = "rate" if "rate" in t.columns else t.columns[0]
        frames.append(t[col].rename("policy_rate"))
        provenance["policy_rate"] = {"status": "ok", "source": str(tbill), "note": "annual/irregular; forward-filled"}

    if not frames:
        return None, provenance
    df = pd.concat(frames, axis=1).sort_index()
    return df, provenance


def collect_macro_data(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    cfg = load_config()
    paths = ensure_dirs()
    start = start or cfg["project"]["start_date"]
    tickers = cfg["macro"]

    local_df, provenance = _load_local_macro(paths)
    frames = []
    if local_df is not None:
        for col in local_df.columns:
            frames.append(local_df[col])

    for name, ticker in tickers.items():
        if name == "usdpkr" and any(getattr(s, "name", None) == "usdpkr" for s in frames):
            continue
        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            if raw.empty:
                provenance[name] = {"ticker": ticker, "status": "empty"}
                continue
            s = raw["Close"].rename(name)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            frames.append(s)
            provenance[name] = {
                "ticker": ticker,
                "status": "ok",
                "start": str(s.index.min().date()),
                "end": str(s.index.max().date()),
            }
        except Exception as exc:  # noqa: BLE001
            provenance[name] = {"ticker": ticker, "status": "error", "error": str(exc)}

    if not frames:
        raise RuntimeError("No macro series downloaded")

    df = pd.concat(frames, axis=1).sort_index()
    # Forward-fill sparse annual series onto daily calendar
    for sparse in ("inflation_cpi", "policy_rate"):
        if sparse in df.columns:
            df[sparse] = df[sparse].ffill()

    for col in list(df.columns):
        if col.endswith("_ret"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[f"{col}_ret"] = df[col].pct_change()

    if "fx_reserves" not in df.columns:
        df["fx_reserves"] = pd.NA
        provenance["fx_reserves"] = {
            "status": "unavailable",
            "note": "Supply official SBP reserves CSV under data/raw/macro to enrich controls.",
        }

    out = paths["data_raw"] / "macro" / "macro_panel.csv"
    df.to_csv(out)
    save_json(provenance, paths["data_raw"] / "macro" / "provenance.json")
    df.to_csv(paths["output"] / "market" / "macro.csv")
    return df
