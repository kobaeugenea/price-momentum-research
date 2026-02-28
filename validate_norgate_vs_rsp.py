"""
Validate Norgate Data Quality: Equal-Weight S&P 500 vs RSP ETF
==============================================================
Builds a simulated equal-weight S&P 500 index from the Norgate parquet
(using survivorship-bias-free constituent history) and compares it to the
Invesco S&P 500 Equal Weight ETF (RSP) downloaded via yfinance.

A close match between the two curves confirms that:
  - Norgate's index-constituent flags are accurate
  - Price data and total-return adjustments are correct
  - Delisted stocks are handled properly (no survivorship bias)

Methodology:
  - Quarterly rebalancing (last trading day of each quarter)
  - Equal weight among all S&P 500 constituents with a valid price on rebalance date
  - RSP expense ratio (0.20% p.a.) added back to make the comparison fair
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


PARQUET = "survivorship_bias_free_db.parquet"


# ── 1. RSP from Yahoo Finance ──────────────────────────────────────────────────

rsp_raw = yf.download(
    "RSP",
    start="1900-01-01",
    progress=False,
    multi_level_index=False,
)["Close"].dropna()

rsp_start, rsp_end = rsp_raw.index[0], rsp_raw.index[-1]


# ── 2. Load Norgate parquet (S&P 500 constituents only) ───────────────────────

df = pd.read_parquet(PARQUET, columns=["Close", "Symbol", "In_S&P_500"])
df = df.reset_index()
df["Date"]   = pd.to_datetime(df["Date"])
df["Symbol"] = df["Symbol"].astype("category")

# Keep only S&P 500 members within the RSP date range
df = df[(df["In_S&P_500"] == True) &
        (df["Date"] >= rsp_start) &
        (df["Date"] <= rsp_end)].copy()


# ── 3. Pivot to wide price matrix (dates × symbols) ───────────────────────────

prices_wide = df.pivot_table(index="Date", columns="Symbol", values="Close").sort_index()
all_dates   = prices_wide.index


# ── 4. Equal-weight portfolio with quarterly rebalancing ──────────────────────

date_series     = pd.Series(all_dates, index=all_dates)
rebalance_dates = date_series.groupby(date_series.dt.to_period("Q")).last().values

period_returns = []
for i in range(len(rebalance_dates) - 1):
    rb_start = rebalance_dates[i]
    rb_end   = rebalance_dates[i + 1]

    # Stocks with a valid price on the rebalance date become constituents
    base_prices = prices_wide.loc[rb_start].dropna()
    cons = base_prices.index.tolist()
    if not cons:
        continue

    period_mask   = (all_dates >= rb_start) & (all_dates <= rb_end)
    period_prices = prices_wide.loc[period_mask, cons].ffill()

    # Re-check validity after forward-fill
    base = period_prices.iloc[0].dropna()
    period_prices = period_prices[base.index]
    if period_prices.empty:
        continue

    # Equal-weight: normalise each stock to 1.0 at rb_start, then average
    port_val = (period_prices / period_prices.iloc[0]).mean(axis=1)
    period_returns.append(port_val.pct_change().dropna())

all_returns = pd.concat(period_returns).sort_index()
all_returns = all_returns[~all_returns.index.duplicated(keep="first")]

sim_rsp = (1 + all_returns).cumprod().rename("SIM (Norgate, EW S&P 500)")


# ── 5. RSP cumulative return (expense ratio added back for fair comparison) ───

# RSP charges 0.20% p.a.; add it back so we compare gross returns
rsp_gross = rsp_raw.pct_change() + 0.002 / 252
rsp_cum   = (1 + rsp_gross).cumprod().rename("RSP (gross of 0.20% ER)")


# ── 6. Align & normalise to 1.0 at common start ───────────────────────────────

common_start = max(sim_rsp.dropna().index[0], rsp_cum.dropna().index[0])
sim_rsp = sim_rsp[sim_rsp.index >= common_start] / sim_rsp[sim_rsp.index >= common_start].iloc[0]
rsp_cum = rsp_cum[rsp_cum.index >= common_start] / rsp_cum[rsp_cum.index >= common_start].iloc[0]


# ── 7. Plot ────────────────────────────────────────────────────────────────────

plt.figure(figsize=(13, 7))
sim_rsp.plot(label="SIM (Norgate, EW S&P 500, quarterly rebalance)")
rsp_cum.plot(label="RSP ETF (gross of 0.20% ER)")
plt.title("Norgate Data Validation: Equal-Weight S&P 500 Simulation vs RSP")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (normalized to 1.0)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
