"""
Hi PRIOR Momentum vs Broad Market Benchmarks
==============================================
Compares four cumulative-return curves:

  1. FF Hi PRIOR Equal-Weighted   – top momentum decile, EW (Ken French)
  2. FF Hi PRIOR Value-Weighted   – top momentum decile, VW (Ken French)
  3. S&P 500 Total Return         – IVV ETF (iShares Core S&P 500) via Yahoo Finance
  4. Russell 3000 Total Return    – IWV ETF (iShares Russell 3000) via Yahoo Finance

Both ETFs are from iShares (BlackRock) and pay dividends quarterly (distributing).
Total return is captured via yfinance Adj Close, which retroactively adjusts all
historical prices for dividends and splits — pct_change() on Adj Close equals
price appreciation + reinvested dividends.

Chart starts at the later of the two ETF inception dates:
  IVV: 2000-05-15   IWV: 2000-07-26  →  common start ~2000-07
"""

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from utils import parse_ff_monthly_section, compute_stats, stats_label


FF_FILE = "10_Portfolios_Prior_12_2.csv"
START   = "1990-01-01"
END     = "2025-12-31"

# Both iShares ETFs — consistent fund family, similar methodology
SP500_TICKER   = "IVV"   # iShares Core S&P 500 ETF,  inception 2000-05-15, ER 0.03%
RUSSELL_TICKER = "IWV"   # iShares Russell 3000 ETF,  inception 2000-07-26, ER 0.20%


# ── 1. Fama-French Hi PRIOR EW & VW ───────────────────────────────────────────

ff_ew = parse_ff_monthly_section(FF_FILE, "Average Equal Weighted Returns -- Monthly")
ff_vw = parse_ff_monthly_section(FF_FILE, "Value Weight Returns -- Monthly")

hi_ew = ff_ew["Hi PRIOR"].loc["1990":"2025"]
hi_vw = ff_vw["Hi PRIOR"].loc["1990":"2025"]


# ── 2. Benchmark monthly returns via yfinance ─────────────────────────────────

def load_monthly_returns(ticker: str) -> pd.Series:
    """
    Download ETF total-return series via yfinance.
    Adj Close is used explicitly (auto_adjust=False) — reliable across all
    yfinance versions and correctly accounts for dividends and splits.
    """
    df  = yf.download(ticker, start=START, end=END, progress=False,
                      multi_level_index=False, auto_adjust=False)
    raw = df["Adj Close"]
    monthly = raw.resample("ME").last()
    return monthly.pct_change().dropna().rename(ticker)

sp500_ret   = load_monthly_returns(SP500_TICKER)
russell_ret = load_monthly_returns(RUSSELL_TICKER)


# ── 3. Align to common start & plot ───────────────────────────────────────────

all_series = {
    "FF Hi PRIOR EW":      hi_ew,
    "FF Hi PRIOR VW":      hi_vw,
    "S&P 500 (IVV)":       sp500_ret,
    "Russell 3000 (IWV)":  russell_ret,
}

common_start = max(s.dropna().index[0] for s in all_series.values())

def to_cum(ret: pd.Series) -> pd.Series:
    c = (1 + ret[ret.index >= common_start]).cumprod()
    return c / c.iloc[0]

colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

fig, ax = plt.subplots(figsize=(14, 8))

for (name, ret), color in zip(all_series.items(), colors):
    cum = to_cum(ret)
    s   = compute_stats(ret[ret.index >= common_start])
    ax.plot(cum.index, cum.values, color=color, linewidth=1.5,
            label=stats_label(name, s))

ax.set_title(
    "Hi PRIOR Momentum vs Broad Market\n"
    "Fama-French EW & VW  ·  IVV (S&P 500) & IWV (Russell 3000) via Yahoo Finance"
)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (normalized to 1.0)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
