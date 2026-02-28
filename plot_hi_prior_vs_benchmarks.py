"""
Hi PRIOR Momentum vs Broad Market Benchmarks
==============================================
Compares three cumulative-return curves:

  1. FF Hi PRIOR Equal-Weighted   – top momentum decile, EW (Ken French)
  2. FF Hi PRIOR Value-Weighted   – top momentum decile, VW (Ken French)
  3. S&P 500 Total Return         – SPY ETF (SPDR S&P 500) via Yahoo Finance

SPY inception: 1993-01-22 — the longest-running S&P 500 ETF.
Total return is captured via yfinance Adj Close, which retroactively adjusts all
historical prices for dividends and splits — pct_change() on Adj Close equals
price appreciation + reinvested dividends.
"""

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from utils import parse_ff_monthly_section, compute_stats, stats_label


FF_FILE = "10_Portfolios_Prior_12_2.csv"
START   = "1990-01-01"
END     = "2025-12-31"

SP500_TICKER = "SPY"   # SPDR S&P 500 ETF, inception 1993-01-22, ER 0.0945%


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

sp500_ret = load_monthly_returns(SP500_TICKER)


# ── 3. Align to common start & plot ───────────────────────────────────────────

all_series = {
    "FF Hi PRIOR EW": hi_ew,
    "FF Hi PRIOR VW": hi_vw,
    "S&P 500 (SPY)":  sp500_ret,
}

common_start = max(s.dropna().index[0] for s in all_series.values())

def to_cum(ret: pd.Series) -> pd.Series:
    c = (1 + ret[ret.index >= common_start]).cumprod()
    return c / c.iloc[0]

colors = ["steelblue", "darkorange", "forestgreen"]

fig, ax = plt.subplots(figsize=(14, 8))

for (name, ret), color in zip(all_series.items(), colors):
    cum = to_cum(ret)
    s   = compute_stats(ret[ret.index >= common_start])
    ax.plot(cum.index, cum.values, color=color, linewidth=1.5,
            label=stats_label(name, s))

ax.set_title(
    "Hi PRIOR Momentum vs Broad Market\n"
    "Fama-French EW & VW  ·  SPY (S&P 500) via Yahoo Finance"
)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (normalized to 1.0)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
