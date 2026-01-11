## Causal Models
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import pandas as pd

def dr_learner(data:pd.DataFrame) -> np.array:
    # get outcome, treatment and controls from data
    y = data.iloc[:,3].to_numpy()
    t = data.iloc[:,4].to_numpy()
    X = data.iloc[:,5:].to_numpy()

    n = len(y)
    n_split = 3

    mu0_all = np.zeros(n)
    mu1_all = np.zeros(n)
    e_all = np.zeros(n)

    kf = KFold(n_splits=n_split, shuffle=True, random_state=0)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        t_tr = t[train_idx]

        # Strong RF models for nuisance functions
        mu0 = RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=5, random_state=0
        )
        mu1 = RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=5, random_state=0
        )
        mu0.fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
        mu1.fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
        mu0_all[test_idx] = mu0.predict(X_te)
        mu1_all[test_idx] = mu1.predict(X_te)

        # Propensity
        e_model = LogisticRegression(max_iter=500)
        e_model.fit(X_tr, t_tr)
        e_all[test_idx] = np.clip(e_model.predict_proba(X_te)[:, 1], 0.05, 0.95)

    # Doubly robust pseudo-outcome (orthogonal score)
    mu_t = mu0_all * (1 - t) + mu1_all * t
    phi = (mu1_all - mu0_all) + (t - e_all) * (y - mu_t) / (e_all * (1 - e_all))
    
    # Clip extreme values for numerical stability
    phi = np.clip(phi, -10, 10)

    # Final CATE model (rich RF)
    tau_model = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=5, random_state=0
    )
    tau_model.fit(X, phi)
    ol_tau_hat = tau_model.predict(X)
    return ol_tau_hat

