"""
qualtrics_logistic_relative_weights.py

Implementa Logistic Relative Weights siguiendo Tonidandel & LeBreton (2009)
para regresión logística y un chequeo opcional tipo Shapley/LMG.

Uso de ejemplo:

    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from qualtrics_logistic_relative_weights import (
        fit_logistic, logistic_relative_weights_TL2009, shapley_logistic_importance
    )

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    model = fit_logistic(X, y)
    rw = logistic_relative_weights_TL2009(X, y, r2_type="nagelkerke")
    print(rw.sort_values("relative_weight_pct", ascending=False))

Notas:
- Los predictores se estandarizan automáticamente.
- Los pesos relativos se escalan al pseudo-R² seleccionado (McFadden o Nagelkerke).
- Shapley/LMG distribuye pseudo-R² promedio sobre todas las permutaciones de predictores (costoso si p>12).
"""

import numpy as np
import pandas as pd
from scipy.special import logit
from numpy.linalg import eig
import statsmodels.api as sm
from itertools import combinations

def _ensure_dataframe(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(np.asarray(X))

def _standardize_df(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0,1.0)
    Z = (X - mu) / sd
    return Z, mu, sd

def fit_logistic(X, y):
    X = _ensure_dataframe(X)
    y = pd.Series(y).astype(float)
    X_std, _, _ = _standardize_df(X)
    X_sm = sm.add_constant(X_std, has_constant="add")
    model = sm.Logit(y, X_sm).fit(disp=False, maxiter=1000)
    return model

def _pseudo_r2(model, r2_type="nagelkerke"):
    llf = model.llf
    llnull = model.llnull
    n = model.nobs
    if r2_type == "mcfadden":
        return 1.0 - (llf / llnull)
    elif r2_type == "nagelkerke":
        cs = 1.0 - np.exp((llnull - llf)*(2.0/n))
        max_cs = 1.0 - np.exp(llnull*(2.0/n))
        if max_cs <=0: return np.nan
        return cs / max_cs
    else:
        raise ValueError("r2_type must be 'mcfadden' or 'nagelkerke'")

def _johnson_relative_weights(Rxx, rxy):
    eigvals, eigvecs = eig(Rxx)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    eigvals[eigvals<0]=0.0
    sqrt_lambda = np.diag(np.sqrt(eigvals))
    A = eigvecs @ sqrt_lambda @ eigvecs.T
    A_inv = np.linalg.pinv(A)
    beta_z = A_inv @ rxy
    raw_weights = (A @ beta_z)**2
    return np.real(raw_weights)

def logistic_relative_weights_TL2009(X, y, r2_type="nagelkerke"):
    X = _ensure_dataframe(X)
    y = pd.Series(y).astype(float)
    cols = X.columns
    X_std, _, _ = _standardize_df(X)
    X_sm = sm.add_constant(X_std, has_constant="add")
    model = sm.Logit(y, X_sm).fit(disp=False, maxiter=1000)
    beta = model.params.reindex(["const", *cols]).values
    eta = X_sm.values @ beta
    p = 1.0/(1.0+np.exp(-eta))
    eps = 1e-9
    p = np.clip(p, eps, 1-eps)
    y_star = logit(p)
    Rxx = np.corrcoef(X_std.values, rowvar=False)
    rxy = np.array([np.corrcoef(X_std[c], y_star)[0,1] for c in cols])
    raw_rw = _johnson_relative_weights(Rxx, rxy)
    raw_rw = np.maximum(raw_rw, 0)
    total_r2 = _pseudo_r2(model, r2_type=r2_type)
    sum_raw = raw_rw.sum()
    if sum_raw>0 and np.isfinite(total_r2):
        scaled_rw = raw_rw*(total_r2/sum_raw)
    else:
        scaled_rw = np.full_like(raw_rw, np.nan)
    out = pd.DataFrame({
        "predictor": cols,
        "raw_weight": raw_rw,
        f"scaled_weight_{r2_type}": scaled_rw,
    })
    total = out[f"scaled_weight_{r2_type}"].sum(skipna=True)
    out["relative_weight_pct"] = 100*out[f"scaled_weight_{r2_type}"]/total if total and np.isfinite(total) else np.nan
    return out.sort_values("relative_weight_pct", ascending=False).reset_index(drop=True)

def _fit_logit_subset(X_df, y):
    X_sm = sm.add_constant(X_df, has_constant="add")
    try:
        m = sm.Logit(y, X_sm).fit(disp=False, maxiter=1000)
        return m
    except Exception:
        return None

def shapley_logistic_importance(X, y, r2_type="nagelkerke", max_predictors=None):
    X = _ensure_dataframe(X)
    y = pd.Series(y).astype(float)
    cols = list(X.columns)
    p = len(cols)
    if max_predictors is not None and p>max_predictors:
        raise ValueError(f"Too many predictors ({p}) for Shapley with max_predictors={max_predictors}.")
    null_model = sm.Logit(y, sm.add_constant(pd.DataFrame(index=X.index))).fit(disp=False, maxiter=1000)
    r2_cache = {(): _pseudo_r2(null_model, r2_type=r2_type)}
    def subset_r2(subset):
        key = tuple(sorted(subset))
        if key in r2_cache:
            return r2_cache[key]
        X_sub = X[list(key)]
        m = _fit_logit_subset(X_sub, y)
        if m is None:
            r2_cache[key] = np.nan
        else:
            r2_cache[key] = _pseudo_r2(m, r2_type=r2_type)
        return r2_cache[key]
    shapley = {c:0.0 for c in cols}
    from math import comb
    for j, cj in enumerate(cols):
        others = [c for c in cols if c!=cj]
        for k in range(0,len(others)+1):
            for S in combinations(others,k):
                r2_S = subset_r2(S)
                r2_Sj = subset_r2(S + (cj,))
                if np.isfinite(r2_S) and np.isfinite(r2_Sj):
                    contrib = r2_Sj - r2_S
                    weight = 1.0 / (p * comb(len(others),k))
                    shapley[cj] += weight * contrib
    out = pd.DataFrame({
        "predictor": cols,
        f"shapley_weight_{r2_type}":[shapley[c] for c in cols],
    })
    total = out[f"shapley_weight_{r2_type}"].sum(skipna=True)
    out["shapley_weight_pct"] = 100*out[f"shapley_weight_{r2_type}"]/total if total and np.isfinite(total) else np.nan
    return out.sort_values("shapley_weight_pct", ascending=False).reset_index(drop=True)