from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import ensure_dirs, load_config, save_json

warnings.filterwarnings("ignore", category=UserWarning)

BASE_FEATURES = [
    "prev_ret",
    "prev_vol",
    "hist_vol",
    "volume_z",
    "prev_trend",
    "event_severity",
    "political_instability",
    "geopolitical_risk",
    "economic_uncertainty",
    "election_period",
    "imf_announcement",
    "regulatory_announcement",
    "usdpkr_ret",
    "oil_ret",
    "gold_ret",
    "sp500_ret",
    "emerging_ret",
    "avg_sentiment",
    "neg_intensity",
    "pos_intensity",
    "fear_index",
    "news_volume",
    "public_sentiment",
]


def _available_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
    keep = []
    for c in cols:
        if c not in df.columns:
            continue
        # Drop columns that are entirely missing (e.g., historical sentiment)
        if df[c].notna().sum() < max(30, int(0.05 * len(df))):
            continue
        keep.append(c)
    return keep


def chronological_split(df: pd.DataFrame, target: str = "target_ret"):
    cfg = load_config()
    feats = _available_features(df, BASE_FEATURES)
    data = df[feats + [target]].dropna()
    n = len(data)
    if n < 200:
        raise RuntimeError(f"Insufficient overlapping observations for modelling: {n}")
    test_n = int(n * cfg["models"]["test_ratio"])
    val_n = int(n * cfg["models"]["val_ratio"])
    train_end = n - test_n - val_n
    val_end = n - test_n
    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]
    return train, val, test, feats


def _metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    directional = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    # Information coefficient
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        ic = 0.0
    else:
        ic = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional,
        "information_coefficient": ic,
    }


def _model_zoo(seed: int) -> dict[str, Any]:
    return {
        "ols": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=seed))]),
        "lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.0005, random_state=seed, max_iter=10000))]),
        "elastic_net": Pipeline(
            [("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=seed, max_iter=10000))]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


def fit_sklearn_models(df: pd.DataFrame, target: str = "target_ret") -> pd.DataFrame:
    cfg = load_config()
    paths = ensure_dirs()
    train, val, test, feats = chronological_split(df, target=target)
    seed = cfg["models"]["random_state"]
    rows = []
    importances: dict[str, Any] = {}

    for name, model in _model_zoo(seed).items():
        model.fit(train[feats], train[target])
        pred_test = model.predict(test[feats])
        pred_val = model.predict(val[feats])
        m_test = _metrics(test[target], pred_test)
        m_val = _metrics(val[target], pred_val)
        rows.append({"model": name, "target": target, "split": "test", **m_test, "n_features": len(feats), "features": ",".join(feats)})
        rows.append({"model": name, "target": target, "split": "validation", **m_val, "n_features": len(feats), "features": ",".join(feats)})

        # Feature importance / coefficients
        est = model.named_steps["model"] if isinstance(model, Pipeline) else model
        if hasattr(est, "coef_"):
            importances[name] = dict(zip(feats, est.coef_.tolist()))
        elif hasattr(est, "feature_importances_"):
            importances[name] = dict(zip(feats, est.feature_importances_.tolist()))

    # XGBoost optional
    try:
        from xgboost import XGBRegressor

        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            objective="reg:squarederror",
        )
        xgb.fit(train[feats], train[target])
        pred = xgb.predict(test[feats])
        rows.append({"model": "xgboost", "target": target, "split": "test", **_metrics(test[target], pred), "n_features": len(feats), "features": ",".join(feats)})
        importances["xgboost"] = dict(zip(feats, xgb.feature_importances_.tolist()))
    except Exception as exc:  # noqa: BLE001
        rows.append({"model": "xgboost", "target": target, "split": "test", "error": str(exc)})

    results = pd.DataFrame(rows)
    results.to_csv(paths["output"] / "models" / f"model_comparison_{target}.csv", index=False)
    save_json(importances, paths["output"] / "models" / f"feature_importance_{target}.json")
    save_json({"features_used": feats, "train_end": str(train.index.max().date()), "test_start": str(test.index.min().date())}, paths["output"] / "models" / f"split_{target}.json")
    return results


def fit_ols_statsmodels(df: pd.DataFrame, target: str = "target_ret") -> dict[str, Any]:
    import statsmodels.api as sm

    paths = ensure_dirs()
    feats = _available_features(df, BASE_FEATURES)
    data = df[feats + [target]].dropna()
    # Chronological train only for coefficient inference presentation
    cfg = load_config()
    cut = int(len(data) * (1 - cfg["models"]["test_ratio"]))
    train = data.iloc[:cut]
    X = sm.add_constant(train[feats])
    y = train[target]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    summary_path = paths["output"] / "models" / f"ols_hac_{target}.txt"
    summary_path.write_text(model.summary().as_text(), encoding="utf-8")
    coefs = (
        pd.DataFrame({"coef": model.params, "std_err": model.bse, "pvalue": model.pvalues, "t": model.tvalues})
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    coefs.to_csv(paths["output"] / "models" / f"ols_coefficients_{target}.csv", index=False)
    return {"rsquared": float(model.rsquared), "rsquared_adj": float(model.rsquared_adj), "nobs": int(model.nobs)}


def fit_var(df: pd.DataFrame, maxlags: int = 5) -> dict[str, Any]:
    """Small VAR on returns + available sentiment / FX."""
    from statsmodels.tsa.api import VAR

    paths = ensure_dirs()
    cols = [c for c in ["psx_ret", "usdpkr_ret", "sp500_ret", "avg_sentiment"] if c in df.columns and df[c].notna().sum() > 100]
    if "psx_ret" not in cols:
        cols = ["psx_ret"] if "psx_ret" in df.columns else []
    data = df[cols].dropna()
    if len(cols) < 2 or len(data) < 100:
        note = {"status": "skipped", "reason": "Need >=2 overlapping series with sufficient history (sentiment often missing historically)."}
        save_json(note, paths["output"] / "models" / "var_result.json")
        return note
    model = VAR(data)
    selected = model.fit(maxlags=maxlags, ic="aic")
    out = {
        "status": "ok",
        "k_ar": int(selected.k_ar),
        "aic": float(selected.aic),
        "variables": cols,
    }
    try:
        gc = {}
        for caused in cols:
            for causing in cols:
                if caused == causing:
                    continue
                test = selected.test_causality(caused, causing, kind="f")
                gc[f"{causing}-> {caused}"] = {"pvalue": float(test.pvalue), "conclusion": str(test.conclusion)}
        out["granger"] = gc
    except Exception as exc:  # noqa: BLE001
        out["granger_error"] = str(exc)
    save_json(out, paths["output"] / "models" / "var_result.json")
    return out


def fit_arimax(df: pd.DataFrame) -> dict[str, Any]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    paths = ensure_dirs()
    exog_cols = _available_features(df, ["event_severity", "usdpkr_ret", "sp500_ret", "avg_sentiment", "neg_intensity"])
    cols = ["psx_ret"] + exog_cols
    data = df[cols].dropna()
    if len(data) < 300:
        note = {"status": "skipped", "reason": f"Insufficient rows after dropna: {len(data)}"}
        save_json(note, paths["output"] / "models" / "arimax_result.json")
        return note
    cfg = load_config()
    cut = int(len(data) * (1 - cfg["models"]["test_ratio"]))
    train, test = data.iloc[:cut], data.iloc[cut:]
    try:
        model = SARIMAX(
            train["psx_ret"],
            exog=train[exog_cols],
            order=(1, 0, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        pred = res.forecast(steps=len(test), exog=test[exog_cols])
        m = _metrics(test["psx_ret"], pred)
        out = {"status": "ok", "exog": exog_cols, "test_metrics": m, "aic": float(res.aic)}
    except Exception as exc:  # noqa: BLE001
        out = {"status": "error", "error": str(exc)}
    save_json(out, paths["output"] / "models" / "arimax_result.json")
    return out
