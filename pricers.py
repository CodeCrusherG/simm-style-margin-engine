"""Trade pricers and market data for the SIMM-style margin engine.

Products: interest rate swap, fixed-rate bond, equity option (Black-Scholes),
CDS (flat hazard rate / credit triangle), FX forward.
Simplifications: annual fixed legs (integer maturities for IRS/bond), flat
hazard rate per name, zero-coupon curves interpolated on SIMM-style tenors.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

IR_TENORS = np.array([2 / 52, 1 / 12, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30])
RECOVERY = 0.40


def df(curve, t):
    """Discount factor from a zero-rate curve defined on IR_TENORS."""
    t = np.asarray(t, dtype=float)
    return np.exp(-np.interp(t, IR_TENORS, curve) * t)


def bs_price(S, K, T, r, vol, call=True, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def base_market():
    usd = np.array([0.052, 0.052, 0.0515, 0.050, 0.047, 0.044,
                    0.042, 0.041, 0.042, 0.043, 0.0435, 0.044])
    return {
        "curves": {"USD": usd, "INR": usd + 0.02},
        "spot": {"EQ_A": 100.0, "EQ_B": 250.0},
        "vol": {"EQ_A": 0.28, "EQ_B": 0.24},
        "cds": {"CORP_X": 0.012, "CORP_Y": 0.025},   # decimal spreads
        "fx": {"USDINR": 83.0},
    }


@dataclass
class IRS:
    ccy: str
    notional: float
    fixed: float
    maturity: int
    payer: bool = True            # pay fixed, receive floating
    product_class = "RatesFX"

    def pv(self, m):
        c = m["curves"][self.ccy]
        pay = np.arange(1, int(self.maturity) + 1, dtype=float)
        v = self.notional * ((1 - df(c, self.maturity)) - self.fixed * df(c, pay).sum())
        return v if self.payer else -v


@dataclass
class Bond:
    ccy: str
    notional: float
    coupon: float
    maturity: int
    product_class = "RatesFX"

    def pv(self, m):
        c = m["curves"][self.ccy]
        pay = np.arange(1, int(self.maturity) + 1, dtype=float)
        return self.notional * (self.coupon * df(c, pay).sum() + df(c, self.maturity))


@dataclass
class EQOption:
    name: str
    strike: float
    expiry: float
    call: bool
    qty: float                    # signed: negative = short
    product_class = "Equity"

    def pv(self, m):
        r = float(np.interp(self.expiry, IR_TENORS, m["curves"]["USD"]))
        return self.qty * bs_price(m["spot"][self.name], self.strike, self.expiry,
                                   r, m["vol"][self.name], self.call)


@dataclass
class CDS:
    name: str
    notional: float
    contract_spread: float
    maturity: int
    buyer: bool = True            # protection buyer
    product_class = "Credit"

    def pv(self, m):
        lam = m["cds"][self.name] / (1 - RECOVERY)
        c = m["curves"]["USD"]
        dt = 0.25
        t = np.arange(1, int(self.maturity / dt) + 1) * dt
        Q, Qp = np.exp(-lam * t), np.exp(-lam * (t - dt))
        protection = (1 - RECOVERY) * np.sum(df(c, t - dt / 2) * (Qp - Q))
        premium = self.contract_spread * np.sum(df(c, t) * Q * dt)
        v = self.notional * (protection - premium)
        return v if self.buyer else -v


@dataclass
class FXFwd:
    pair: str                     # e.g. USDINR (INR per USD)
    notional_usd: float
    strike: float
    maturity: float
    buy_usd: bool = True
    product_class = "RatesFX"

    def pv(self, m):
        S = m["fx"][self.pair]
        usd, inr = df(m["curves"]["USD"], self.maturity), df(m["curves"]["INR"], self.maturity)
        v = self.notional_usd * (usd - self.strike * inr / S)
        return v if self.buy_usd else -v


def portfolio_pv(trades, m):
    return float(sum(t.pv(m) for t in trades))


def demo_portfolio():
    return [
        IRS("USD", 50e6, 0.042, 10, payer=True),
        IRS("USD", 30e6, 0.041, 5, payer=False),
        Bond("USD", 20e6, 0.045, 7),
        EQOption("EQ_A", 100.0, 1.0, True, 20_000),
        EQOption("EQ_B", 240.0, 0.5, False, -5_000),
        CDS("CORP_X", 10e6, 0.010, 5, buyer=True),
        CDS("CORP_Y", 8e6, 0.024, 5, buyer=False),
        FXFwd("USDINR", 5e6, 83.5, 1.0, buy_usd=True),
    ]
