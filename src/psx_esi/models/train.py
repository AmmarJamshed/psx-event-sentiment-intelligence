from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_end: str,
    valid_end: str,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    train_mask = X.index <= pd.Timestamp(train_end)
    valid_mask = (X.index > pd.Timestamp(train_end)) & (X.index <= pd.Timestamp(valid_end))
    test_mask = X.index > pd.Timestamp(valid_end)
    return {
        "train": (X.loc[train_mask], y.loc[train_mask]),
        "valid": (X.loc[valid_mask], y.loc[valid_mask]),
        "test": (X.loc[test_mask], y.loc[test_mask]),
    }


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    # Adjusted R2 approx with p estimated externally; here use simple r2
    direction = float(np.mean(np.sign(y_true) == np.sign(y_pred))) if len(y_true) else float("nan")
    # Information coefficient = rank correlation
    if len(y_true) > 2:
        ic = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    else:
        ic = float("nan")
    return {"r2": r2, "rmse": rmse, "mae": mae, "directional_accuracy": direction, "ic": ic}


def _fit_predict(model, Xtr, ytr, Xte):
    model.fit(Xtr, ytr)
    return model.predict(Xte), model


def train_regressors(
    X: pd.DataFrame,
    y: pd.Series,
    train_end: str = "2020-12-31",
    valid_end: str = "2022-12-31",
    random_state: int = 42,
) -> dict[str, Any]:
    splits = chronological_split(X, y, train_end, valid_end)
    Xtr, ytr = splits["train"]
    Xva, yva = splits["valid"]
    Xte, yte = splits["test"]

    scaler = StandardScaler()
    Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), index=Xtr.index, columns=Xtr.columns)
    Xva_s = pd.DataFrame(scaler.transform(Xva), index=Xva.index, columns=Xva.columns)
    Xte_s = pd.DataFrame(scaler.transform(Xte), index=Xte.index, columns=Xte.columns)

    specs = {
        "ols": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "lasso": Lasso(alpha=0.0005, random_state=random_state, max_iter=10000),
        "elasticnet": ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=random_state, max_iter=10000),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }
    try:
        from xgboost import XGBRegressor

        specs["xgboost"] = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    results: dict[str, Any] = {"splits": {k: int(len(v[0])) for k, v in splits.items()}, "models": {}}
    tree_models = {"random_forest", "gradient_boosting", "xgboost"}

    for name, model in specs.items():
        use_scaled = name not in tree_models
        xtr = Xtr_s if use_scaled else Xtr
        xva = Xva_s if use_scaled else Xva
        xte = Xte_s if use_scaled else Xte
        if len(xtr) < 100 or len(xte) < 20:
            results["models"][name] = {"error": "insufficient sample"}
            continue
        pred_va, fitted = _fit_predict(model, xtr, ytr, xva) if len(xva) else (np.array([]), model)
        pred_te, fitted = _fit_predict(fitted if len(xva) else model, xtr if not len(xva) else xtr, ytr, xte)
        # refit cleanly
        pred_te, fitted = _fit_predict(model, xtr, ytr, xte)
        pred_va = fitted.predict(xva) if len(xva) else np.array([])

        entry: dict[str, Any] = {
            "valid": _metrics(yva, pred_va) if len(xva) else {},
            "test": _metrics(yte, pred_te),
        }
        if hasattr(fitted, "coef_"):
            entry["coefficients"] = {c: float(v) for c, v in zip(X.columns, fitted.coef_)}
            entry["intercept"] = float(fitted.intercept_)
        if hasattr(fitted, "feature_importances_"):
            entry["feature_importance"] = {
                c: float(v) for c, v in zip(X.columns, fitted.feature_importances_)
            }
        results["models"][name] = entry

    # OLS statsmodels diagnostics on train
    try:
        import statsmodels.api as sm

        X_sm = sm.add_constant(Xtr_s)
        ols = sm.OLS(ytr, X_sm, missing="drop").fit()
        results["ols_diagnostics"] = {
            "summary_text": str(ols.summary()),
            "condition_number": float(ols.condition_number),
            "durbin_watson": float(sm.stats.stattools.durbin_watson(ols.resid)),
            "params": {k: float(v) for k, v in ols.params.items()},
            "pvalues": {k: float(v) for k, v in ols.pvalues.items()},
        }
        # Breusch-Pagan
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan

            bp = het_breuschpagan(ols.resid, ols.model.exog)
            results["ols_diagnostics"]["breusch_pagan_pvalue"] = float(bp[1])
        except Exception:
            pass
    except Exception as e:
        results["ols_diagnostics"] = {"error": str(e)}

    # Stationarity on target
    try:
        from statsmodels.tsa.stattools import adfuller

        adf = adfuller(ytr.dropna(), maxlag=20, autolag="AIC")
        results["stationarity_adf"] = {
            "adf_stat": float(adf[0]),
            "pvalue": float(adf[1]),
            "usedlag": int(adf[2]),
        }
    except Exception as e:
        results["stationarity_adf"] = {"error": str(e)}

    # Simple Granger: does sentiment_avg Granger-cause returns?
    try:
        from statsmodels.tsa.stattools import grangercausalitytests

        if "sentiment_avg" in X.columns and X["sentiment_avg"].abs().sum() > 0:
            gdf = pd.concat([y.rename("ret"), X["sentiment_avg"]], axis=1).dropna()
            if len(gdf) > 200:
                g = grangercausalitytests(gdf[["ret", "sentiment_avg"]], maxlag=5, verbose=False)
                results["granger_sentiment_to_returns"] = {
                    str(lag): float(val[0]["ssr_ftest"][1]) for lag, val in g.items()
                }
            else:
                results["granger_sentiment_to_returns"] = {
                    "note": "Insufficient overlapping sentiment observations for Granger test."
                }
        else:
            results["granger_sentiment_to_returns"] = {
                "note": "Sentiment series mostly missing/zero; Granger not informative."
            }
    except Exception as e:
        results["granger_sentiment_to_returns"] = {"error": str(e)}

    # VAR lite on returns + event severity
    try:
        from statsmodels.tsa.api import VAR

        var_cols = [c for c in ["prev_return", "event_severity_max", "usdpkr_return"] if c in X.columns]
        vdf = X[var_cols].copy()
        vdf["return_t"] = y.shift(1)  # align carefully: use contemporaneous return in VAR system
        # Better: use market return_t from index alignment
        vdf = pd.concat([y.rename("y"), X[var_cols]], axis=1).dropna()
        if len(vdf) > 300:
            model = VAR(vdf)
            res = model.fit(maxlags=5, ic="aic")
            results["var"] = {
                "selected_lags": int(res.k_ar),
                "aic": float(res.aic),
                "variable_names": list(vdf.columns),
            }
    except Exception as e:
        results["var"] = {"error": str(e)}

    return results


def crisis_vs_stable_performance(
    panel: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "ridge",
) -> dict[str, Any]:
    """Compare predictive fit in high vs low political/economic event intensity regimes."""
    if panel.empty:
        return {}
    intensity = panel.reindex(X.index)[["political_instability", "economic_uncertainty", "geopolitical_risk"]].sum(axis=1)
    crisis = intensity >= 1
    out = {}
    for label, mask in [("crisis_days", crisis), ("stable_days", ~crisis)]:
        Xm, ym = X.loc[mask], y.loc[mask]
        if len(Xm) < 50:
            out[label] = {"n": int(len(Xm)), "note": "too few observations"}
            continue
        # simple in-sample ridge for regime comparison (explanatory, not OOS claim)
        mid = int(len(Xm) * 0.7)
        model = Ridge(alpha=1.0)
        model.fit(Xm.iloc[:mid], ym.iloc[:mid])
        pred = model.predict(Xm.iloc[mid:])
        out[label] = {"n": int(len(Xm)), **_metrics(ym.iloc[mid:], pred)}
    return out


def save_results(results: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return str(o)

    path.write_text(json.dumps(results, indent=2, default=_default), encoding="utf-8")
