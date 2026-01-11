## Causal Models
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import pandas as pd

def dr_learner(X, y, w, n_splits=3, clip=(0.05,0.95), mu_model=None, prop_model=None, final_model=None, eps=1e-6, random_state=0):
    """
    X: array (n_samples, n_features)
    y: array (n_samples,)
    w: binary treatment array (n_samples,)

    Robust DR learner with:
      - default propensity model = RandomForestClassifier
      - denominator clamped to max(e*(1-e), eps) to avoid numeric blowups

    returns: tau_hat array (n_samples,) and diagnostics if requested
    """
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.asarray(w)
    assert X.shape[0] == y.shape[0] == w.shape[0]
    
    # get outcome, treatment and controls from data
    n = len(y)

    mu0_all = np.zeros(n)
    mu1_all = np.zeros(n)
    e_all = np.zeros(n)

    # default propensity model: RandomForestClassifier unless user provided one
    if prop_model is None:
        base_prop = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
    else:
        base_prop = prop_model

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X, w):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        w_tr = w[train_idx]   
        # Strong RF models for nuisance functions
        mu0 = RandomForestRegressor(
            n_estimators=200, n_jobs=-1, max_depth=None, min_samples_leaf=5, random_state=random_state
        )
        mu1 = RandomForestRegressor(
            n_estimators=200, n_jobs=-1, max_depth=None, min_samples_leaf=5, random_state=random_state
        )
        mu0.fit(X_tr[w_tr == 0], y_tr[w_tr == 0])
        mu1.fit(X_tr[w_tr == 1], y_tr[w_tr == 1])
        mu0_all[test_idx] = mu0.predict(X_te)
        mu1_all[test_idx] = mu1.predict(X_te)

        # Propensity: clone base_prop to get a fresh estimator per fold when needed
        try:
            e_model = clone(base_prop)
        except Exception:
            e_model = base_prop
        e_model.fit(X_tr, w_tr)
        probs = e_model.predict_proba(X_te)[:, 1]
        e_all[test_idx] = np.clip(probs, clip[0], clip[1])

    # Doubly robust pseudo-outcome (orthogonal score)
    mu_t = mu0_all * (1 - w) + mu1_all * w
    denom = np.maximum(e_all * (1 - e_all), eps)
    phi = (mu1_all - mu0_all) + (w - e_all) * (y - mu_t) / denom
    
    # Clip extreme values for numerical stability
    phi = np.clip(phi, -10, 10)

    # Final CATE model (rich RF)
    tau_model = RandomForestRegressor(
        n_estimators=300, n_jobs=-1, max_depth=None, min_samples_leaf=5, random_state=random_state
    )
    tau_model.fit(X, phi)
    ol_tau_hat = tau_model.predict(X)
    return ol_tau_hat

