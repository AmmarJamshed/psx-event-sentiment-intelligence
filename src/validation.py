from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.models import BASE_FEATURES, _available_features, _metrics
from src.utils import ensure_dirs, load_config, save_json


def expanding_window_validation(df: pd.DataFrame, target: str = "target_ret") -> pd.DataFrame:
    cfg = load_config()
    paths = ensure_dirs()
    feats = _available_features(df, BASE_FEATURES)
    data = df[feats + [target]].dropna()
    min_train = cfg["models"]["min_train"]
    rows = []
    # Step by ~21 trading days
    for end in range(min_train, len(data) - 21, 21):
        train = data.iloc[:end]
        test = data.iloc[end : end + 21]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train[feats])
        Xte = scaler.transform(test[feats])
        model = Ridge(alpha=1.0)
        model.fit(Xtr, train[target])
        pred = model.predict(Xte)
        m = _metrics(test[target], pred)
        rows.append({"window_end": str(train.index.max().date()), "test_start": str(test.index.min().date()), **m})
    out = pd.DataFrame(rows)
    out.to_csv(paths["output"] / "models" / "expanding_window_ridge.csv", index=False)
    return out


def rolling_window_validation(df: pd.DataFrame, target: str = "target_ret") -> pd.DataFrame:
    cfg = load_config()
    paths = ensure_dirs()
    feats = _available_features(df, BASE_FEATURES)
    data = df[feats + [target]].dropna()
    win = cfg["models"]["rolling_window"]
    rows = []
    for end in range(win, len(data) - 21, 21):
        train = data.iloc[end - win : end]
        test = data.iloc[end : end + 21]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train[feats])
        Xte = scaler.transform(test[feats])
        model = Ridge(alpha=1.0)
        model.fit(Xtr, train[target])
        pred = model.predict(Xte)
        m = _metrics(test[target], pred)
        rows.append({"window_end": str(train.index.max().date()), **m})
    out = pd.DataFrame(rows)
    out.to_csv(paths["output"] / "models" / "rolling_window_ridge.csv", index=False)
    return out


def residual_diagnostics(df: pd.DataFrame, target: str = "target_ret") -> dict:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tsa.stattools import adfuller

    paths = ensure_dirs()
    feats = _available_features(df, BASE_FEATURES)
    data = df[feats + [target]].dropna()
    X = sm.add_constant(data[feats])
    model = sm.OLS(data[target], X).fit()
    resid = model.resid

    diag = {
        "nobs": int(model.nobs),
        "rsquared": float(model.rsquared),
        "durbin_watson": float(sm.stats.stattools.durbin_watson(resid)),
    }
    try:
        bp = het_breuschpagan(resid, X)
        diag["breusch_pagan_pvalue"] = float(bp[1])
    except Exception as exc:  # noqa: BLE001
        diag["breusch_pagan_error"] = str(exc)
    try:
        lb = acorr_ljungbox(resid, lags=[10], return_df=True)
        diag["ljung_box_pvalue_lag10"] = float(lb["lb_pvalue"].iloc[0])
    except Exception as exc:  # noqa: BLE001
        diag["ljung_box_error"] = str(exc)
    try:
        adf = adfuller(data[target].dropna(), autolag="AIC")
        diag["adf_stat"] = float(adf[0])
        diag["adf_pvalue"] = float(adf[1])
    except Exception as exc:  # noqa: BLE001
        diag["adf_error"] = str(exc)

    # VIF multicollinearity
    try:
        vif_rows = []
        Xv = data[feats].dropna()
        for i, col in enumerate(feats):
            vif_rows.append({"feature": col, "vif": float(variance_inflation_factor(Xv.values, i))})
        vif_df = pd.DataFrame(vif_rows)
        vif_df.to_csv(paths["output"] / "models" / "vif.csv", index=False)
        diag["max_vif"] = float(vif_df["vif"].max())
    except Exception as exc:  # noqa: BLE001
        diag["vif_error"] = str(exc)

    save_json(diag, paths["output"] / "models" / "diagnostics.json")
    return diag


def crisis_vs_stable_performance(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Compare absolute returns / simple ridge IC around crisis vs non-crisis periods."""
    paths = ensure_dirs()
    crisis_cats = {"political", "geopolitical"}
    crisis_dates = set(
        pd.to_datetime(events.loc[events["category"].isin(crisis_cats), "info_date"]).dt.normalize()
    )
    # Mark +/- 5 trading days around crisis info dates as crisis regimes
    idx = df.index
    crisis = pd.Series(False, index=idx)
    for d in crisis_dates:
        if d in idx:
            loc = idx.get_loc(d)
            if isinstance(loc, slice):
                loc = loc.start
            lo = max(0, loc - 5)
            hi = min(len(idx), loc + 6)
            crisis.iloc[lo:hi] = True
    out = pd.DataFrame(
        {
            "regime": ["crisis", "stable"],
            "mean_abs_return": [
                df.loc[crisis, "psx_ret"].abs().mean() if "psx_ret" in df else np.nan,
                df.loc[~crisis, "psx_ret"].abs().mean() if "psx_ret" in df else np.nan,
            ],
            "mean_sentiment": [
                df.loc[crisis, "avg_sentiment"].mean() if "avg_sentiment" in df else np.nan,
                df.loc[~crisis, "avg_sentiment"].mean() if "avg_sentiment" in df else np.nan,
            ],
            "n_days": [int(crisis.sum()), int((~crisis).sum())],
        }
    )
    out.to_csv(paths["output"] / "models" / "crisis_vs_stable.csv", index=False)
    return out
