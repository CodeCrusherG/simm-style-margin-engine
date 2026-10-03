"""Simulated risk-factor history, full-revaluation 10-day P&L and IM back-test.

NOTE: tenor-level rate curves and CDS spread histories are not freely
available, so the factor history is SIMULATED (fat-tailed, correlated, with a
stress regime in the final part of the sample). The back-test mechanics are
real; plug in real factor histories to get real results.
"""
from copy import deepcopy

import numpy as np
import pandas as pd

from pricers import IR_TENORS, portfolio_pv

MPOR = 10


def simulate_factors(n_days=2500, seed=7, stress_start=0.75):
    rng = np.random.default_rng(seed)
    scale = np.ones(n_days)
    scale[int(stress_start * n_days):] = 2.2
    t4 = lambda *shape: rng.standard_t(4, shape) / np.sqrt(2)

    level, slope = t4(n_days) * 4.0 * scale, t4(n_days) * 1.5 * scale
    lt = np.log(IR_TENORS)
    shape = (lt - lt.mean()) / lt.std()
    ir = {c: b * (level[:, None] + slope[:, None] * shape[None, :])
             + t4(n_days, 12) * 0.6 * scale[:, None]
          for c, b in (("USD", 1.0), ("INR", 0.8))}

    vols = np.array([0.28, 0.24]) / np.sqrt(252)
    L = np.linalg.cholesky(np.array([[1.0, 0.6], [0.6, 1.0]]))
    eq = (t4(n_days, 2) @ L.T) * vols * scale[:, None]
    eqvol = -30 * eq + t4(n_days, 2) * 0.4
    mkt_ret = eq.mean(axis=1)
    cds = np.column_stack([-120 * mkt_ret + t4(n_days) * 1.5 * scale,
                           -200 * mkt_ret + t4(n_days) * 2.5 * scale])
    fx = t4(n_days, 1) * 0.06 / np.sqrt(252) * scale[:, None]
    return {"ir": ir, "eq": eq, "eqvol": eqvol, "cds": cds, "fx": fx}


def window_sums(f, mpor=MPOR):
    n = (len(f["eq"]) // mpor) * mpor
    cut = lambda a: a[:n].reshape(n // mpor, mpor, *a.shape[1:]).sum(axis=1)
    return {"ir": {c: cut(a) for c, a in f["ir"].items()},
            **{k: cut(f[k]) for k in ("eq", "eqvol", "cds", "fx")}}


def apply_shock(m, w, i):
    new = deepcopy(m)
    for c in new["curves"]:
        new["curves"][c] = m["curves"][c] + w["ir"][c][i] * 1e-4
    for k, n in enumerate(m["spot"]):
        new["spot"][n] = m["spot"][n] * np.exp(w["eq"][i, k])
        new["vol"][n] = max(m["vol"][n] + w["eqvol"][i, k] * 0.01, 0.05)
    for k, n in enumerate(m["cds"]):
        new["cds"][n] = max(m["cds"][n] + w["cds"][i, k] * 1e-4, 1e-4)
    new["fx"]["USDINR"] = m["fx"]["USDINR"] * np.exp(w["fx"][i, 0])
    return new


def window_pnl(trades, m, w):
    base = portfolio_pv(trades, m)
    n = len(w["eq"])
    return np.array([portfolio_pv(trades, apply_shock(m, w, i)) - base
                     for i in range(n)])


def kupiec(n_exc, n_obs, alpha=0.99):
    from scipy import stats
    p = 1 - alpha
    if n_exc == 0:
        lr = -2 * n_obs * np.log(1 - p)
    else:
        ph = n_exc / n_obs
        lr = -2 * ((n_obs - n_exc) * np.log(1 - p) + n_exc * np.log(p)
                   - (n_obs - n_exc) * np.log(1 - ph) - n_exc * np.log(ph))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def backtest_im(pnl, simm_im, calib_frac=0.6, alpha=0.99):
    """Out-of-sample: historical-VaR IM calibrated on early windows, SIMM-style
    IM static; both tested on the later windows (which contain the stress)."""
    k = int(len(pnl) * calib_frac)
    calib, test = pnl[:k], pnl[k:]
    hvar_im = -np.quantile(calib, 1 - alpha)
    rows = []
    for name, im in (("SIMM-style IM", simm_im), ("Historical-VaR IM (calibrated)", hvar_im)):
        exc = int((test < -im).sum())
        lr, p = kupiec(exc, len(test), alpha)
        rows.append({"Model": name, "IM": im, "Test windows": len(test),
                     "Exceptions": exc, "Expected": round(len(test) * (1 - alpha), 1),
                     "Worst loss / IM": float(-test.min() / im),
                     "Kupiec p": round(p, 4)})
    return pd.DataFrame(rows), hvar_im
