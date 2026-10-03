"""Bump-and-reprice sensitivities (the inputs to SIMM).

Units follow the SIMM convention used here:
  IR delta  : PV change per +1bp in a zero-rate pillar
  Credit    : PV change per +1bp in the CDS spread
  Equity    : PV change per +1% in spot;  vega: per +1 vol point
  FX        : PV change per +1% in spot
"""
from copy import deepcopy

import pandas as pd


def trade_sensitivities(trade, m):
    pv0 = trade.pv(m)
    out = []

    def add(rc, typ, bucket, factor, mm):
        s = trade.pv(mm) - pv0
        if abs(s) > 1e-9:
            out.append((trade.product_class, rc, typ, bucket, factor, s))

    for ccy, curve in m["curves"].items():
        for k in range(len(curve)):
            mm = deepcopy(m)
            mm["curves"][ccy][k] += 1e-4
            add("IR", "delta", ccy, k, mm)
    for n in m["spot"]:
        mm = deepcopy(m); mm["spot"][n] *= 1.01
        add("Equity", "delta", n, 0, mm)
        mm = deepcopy(m); mm["vol"][n] += 0.01
        add("Equity", "vega", n, 0, mm)
    for n in m["cds"]:
        mm = deepcopy(m); mm["cds"][n] += 1e-4
        add("CreditQ", "delta", n, 0, mm)
    for p in m["fx"]:
        mm = deepcopy(m); mm["fx"][p] *= 1.01
        add("FX", "delta", p, 0, mm)
    return out


def portfolio_sensitivities(trades, m):
    rows = [r for t in trades for r in trade_sensitivities(t, m)]
    df_ = pd.DataFrame(rows, columns=["product_class", "risk_class", "type",
                                      "bucket", "factor", "sens"])
    return df_.groupby(["product_class", "risk_class", "type", "bucket", "factor"],
                       as_index=False)["sens"].sum()
