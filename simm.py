"""SIMM-style initial margin aggregation.

Follows the structure of the ISDA SIMM methodology:
  weighted sensitivity  WS = RW * s
  intra-bucket          K_b = sqrt(sum_ij rho_ij WS_i WS_j)
  inter-bucket          sqrt(sum K_b^2 + sum_{b!=c} gamma S_b S_c),  S_b = clip(sum WS, -K_b, K_b)
  risk classes          IM_pc = sqrt(sum_rs psi_rs IM_r IM_s)   (per product class)
  total                 sum over product classes

IMPORTANT: the parameter values below (risk weights, correlations, gammas,
psi) are ILLUSTRATIVE PLACEHOLDERS, not the ISDA-calibrated set. Replace them
with the official published parameters for a given SIMM version before using
the numbers for anything real. Curvature and base-correlation margin are not
implemented.
"""
import numpy as np

from pricers import IR_TENORS

# ---- illustrative parameters ------------------------------------------------
IR_RW = np.array([77, 77, 77, 64, 58, 49, 47, 47, 45, 45, 48, 56], dtype=float)
IR_THETA, IR_RHO_FLOOR = 0.03, 0.20
DELTA_RW = {"CreditQ": 80.0, "Equity": 25.0, "FX": 7.4}
VEGA_RW = {"Equity": 25.0}
GAMMA = {"IR": 0.27, "CreditQ": 0.30, "Equity": 0.15, "FX": 0.50}
RISK_CLASSES = ["IR", "CreditQ", "Equity", "FX"]
PSI = np.array([[1.00, 0.20, 0.20, 0.25],
                [0.20, 1.00, 0.30, 0.20],
                [0.20, 0.30, 1.00, 0.20],
                [0.25, 0.20, 0.20, 1.00]])


def _rho(rc, a, b):
    if rc == "IR":
        ta, tb = IR_TENORS[a], IR_TENORS[b]
        return max(np.exp(-IR_THETA * abs(ta - tb) / min(ta, tb)), IR_RHO_FLOOR)
    return 1.0


def _bucket(rc, d):
    keys = list(d)
    w = np.array([d[k] for k in keys])
    R = np.array([[1.0 if i == j else _rho(rc, a, b)
                   for j, b in enumerate(keys)] for i, a in enumerate(keys)])
    return float(np.sqrt(max(w @ R @ w, 0.0))), float(w.sum())


def aggregate(rc, ws_by_bucket):
    """Combine buckets of weighted sensitivities into one margin number."""
    Ks, sums = {}, {}
    for b, d in ws_by_bucket.items():
        Ks[b], sums[b] = _bucket(rc, d)
    S = {b: max(min(sums[b], Ks[b]), -Ks[b]) for b in Ks}
    total = sum(K**2 for K in Ks.values())
    names = list(Ks)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                total += GAMMA[rc] * S[a] * S[b]
    return float(np.sqrt(max(total, 0.0)))


def _risk_weight(rc, typ, factor):
    if typ == "vega":
        return VEGA_RW[rc]
    return IR_RW[int(factor)] if rc == "IR" else DELTA_RW[rc]


def risk_class_margin(sens, rc):
    """Delta + vega margin for one risk class from a sensitivity DataFrame."""
    total = 0.0
    for typ in ("delta", "vega"):
        sub = sens[(sens.risk_class == rc) & (sens.type == typ)]
        if sub.empty:
            continue
        ws = {}
        for r in sub.itertuples():
            ws.setdefault(r.bucket, {})[r.factor] = _risk_weight(rc, typ, r.factor) * r.sens
        total += aggregate(rc, ws)
    return total


def simm_im(sens):
    """Total SIMM-style IM and a breakdown by product class and risk class."""
    breakdown, total = {}, 0.0
    for pc, sub in sens.groupby("product_class"):
        im_r = np.array([risk_class_margin(sub, rc) for rc in RISK_CLASSES])
        im_pc = float(np.sqrt(max(im_r @ PSI @ im_r, 0.0)))
        breakdown[pc] = {"total": im_pc, **dict(zip(RISK_CLASSES, im_r))}
        total += im_pc
    return total, breakdown
