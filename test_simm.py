"""Unit tests. Run with:  python -m unittest -v"""
import unittest

import numpy as np
import pandas as pd

import im_backtest as bt
import simm
from pricers import (CDS, IRS, Bond, EQOption, base_market, bs_price, df,
                     demo_portfolio, IR_TENORS)
from sensitivities import portfolio_sensitivities


def flat_market(r=0.04):
    m = base_market()
    m["curves"] = {"USD": np.full(12, r), "INR": np.full(12, r + 0.02)}
    return m


class TestPricers(unittest.TestCase):
    def test_put_call_parity(self):
        S, K, T, r, v = 100, 105, 1.0, 0.04, 0.25
        c, p = bs_price(S, K, T, r, v, True), bs_price(S, K, T, r, v, False)
        self.assertAlmostEqual(c - p, S - K * np.exp(-r * T), places=8)

    def test_irs_zero_at_par_rate(self):
        m = base_market()
        c = m["curves"]["USD"]
        pay = np.arange(1, 6, dtype=float)
        par = (1 - df(c, 5)) / df(c, pay).sum()
        self.assertAlmostEqual(IRS("USD", 1e6, par, 5).pv(m), 0.0, places=4)

    def test_payer_receiver_are_opposites(self):
        m = base_market()
        a, b = IRS("USD", 1e6, 0.04, 5, True), IRS("USD", 1e6, 0.04, 5, False)
        self.assertAlmostEqual(a.pv(m), -b.pv(m), places=6)

    def test_bond_at_par_on_flat_curve(self):
        r, T = 0.04, 5
        m = flat_market(r)
        pay = np.arange(1, T + 1, dtype=float)
        par = (1 - np.exp(-r * T)) / np.exp(-r * pay).sum()
        self.assertAlmostEqual(Bond("USD", 1e6, par, T).pv(m) / 1e6, 1.0, places=6)

    def test_cds_near_zero_at_market_spread(self):
        m = base_market()
        s = m["cds"]["CORP_X"]
        self.assertLess(abs(CDS("CORP_X", 1e6, s, 5).pv(m)) / 1e6, 5e-4)

    def test_cds_buyer_gains_when_spreads_widen(self):
        m = base_market()
        buyer = CDS("CORP_X", 1e6, 0.012, 5, True)
        v0 = buyer.pv(m)
        m["cds"]["CORP_X"] += 0.005
        self.assertGreater(buyer.pv(m), v0)


class TestSensitivities(unittest.TestCase):
    def test_payer_swap_has_positive_pv01(self):
        m = base_market()
        s = portfolio_sensitivities([IRS("USD", 10e6, 0.04, 5, True)], m)
        self.assertGreater(s[s.risk_class == "IR"].sens.sum(), 0)

    def test_receiver_swap_has_negative_pv01(self):
        m = base_market()
        s = portfolio_sensitivities([IRS("USD", 10e6, 0.04, 5, False)], m)
        self.assertLess(s[s.risk_class == "IR"].sens.sum(), 0)

    def test_equity_delta_matches_black_scholes_delta(self):
        from scipy.stats import norm
        m = base_market()
        opt = EQOption("EQ_A", 100.0, 1.0, True, 1000)
        s = portfolio_sensitivities([opt], m)
        got = s[(s.risk_class == "Equity") & (s.type == "delta")].sens.iloc[0]
        r = float(np.interp(1.0, IR_TENORS, m["curves"]["USD"]))
        d1 = (np.log(1.0) + (r + 0.5 * 0.28**2)) / 0.28
        expected = 1000 * norm.cdf(d1) * 100.0 * 0.01
        self.assertAlmostEqual(got / expected, 1.0, delta=0.02)

    def test_long_option_has_positive_vega(self):
        m = base_market()
        s = portfolio_sensitivities([EQOption("EQ_A", 100.0, 1.0, True, 1000)], m)
        self.assertGreater(s[s.type == "vega"].sens.iloc[0], 0)


class TestSIMM(unittest.TestCase):
    def test_single_equity_sensitivity(self):
        sens = pd.DataFrame([("Equity", "Equity", "delta", "EQ_A", 0, -1000.0)],
                            columns=["product_class", "risk_class", "type",
                                     "bucket", "factor", "sens"])
        total, _ = simm.simm_im(sens)
        self.assertAlmostEqual(total, simm.DELTA_RW["Equity"] * 1000.0, places=6)

    def test_offsetting_trades_give_zero_margin(self):
        m = base_market()
        trades = [IRS("USD", 10e6, 0.04, 5, True), IRS("USD", 10e6, 0.04, 5, False)]
        total, _ = simm.simm_im(portfolio_sensitivities(trades, m))
        self.assertAlmostEqual(total, 0.0, places=3)

    def test_diversification_across_buckets(self):
        rows = [("Equity", "Equity", "delta", "EQ_A", 0, 1000.0),
                ("Equity", "Equity", "delta", "EQ_B", 0, -1000.0)]
        sens = pd.DataFrame(rows, columns=["product_class", "risk_class", "type",
                                           "bucket", "factor", "sens"])
        total, _ = simm.simm_im(sens)
        self.assertLess(total, 2 * simm.DELTA_RW["Equity"] * 1000.0)

    def test_subadditivity_within_product_class(self):
        m = base_market()
        a = [IRS("USD", 20e6, 0.042, 10, True)]
        b = [IRS("USD", 15e6, 0.041, 3, False), Bond("USD", 5e6, 0.045, 7)]
        im = lambda t: simm.simm_im(portfolio_sensitivities(t, m))[0]
        self.assertLessEqual(im(a + b), im(a) + im(b) + 1e-6)

    def test_im_scales_linearly_with_notional(self):
        m = base_market()
        im = lambda n: simm.simm_im(portfolio_sensitivities(
            [IRS("USD", n, 0.04, 5, True)], m))[0]
        self.assertAlmostEqual(im(20e6) / im(10e6), 2.0, places=4)


class TestBacktest(unittest.TestCase):
    def test_window_sums_shape(self):
        f = bt.simulate_factors(n_days=500)
        w = bt.window_sums(f)
        self.assertEqual(w["eq"].shape, (50, 2))
        self.assertEqual(w["ir"]["USD"].shape, (50, 12))

    def test_zero_shock_gives_zero_pnl(self):
        m, trades = base_market(), demo_portfolio()
        w = bt.window_sums(bt.simulate_factors(n_days=100))
        w = {"ir": {c: np.zeros_like(a) for c, a in w["ir"].items()},
             **{k: np.zeros_like(w[k]) for k in ("eq", "eqvol", "cds", "fx")}}
        self.assertTrue(np.allclose(bt.window_pnl(trades, m, w), 0.0, atol=1e-6))

    def test_kupiec_calibrated_not_rejected(self):
        _, p = bt.kupiec(1, 100, 0.99)
        self.assertGreater(p, 0.9)


if __name__ == "__main__":
    unittest.main()
