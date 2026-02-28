"""
Fama-French 12-2 Momentum Backtest
====================================
Simulates the Hi PRIOR (top-decile momentum) equal-weight portfolio and
compares two breakpoint choices against the Fama-French reference series.

Curves:
  1. SIM – S&P 500 breakpoints  (closer to FF methodology)
  2. SIM – Russell 3000 breakpoints
  3. FF Hi PRIOR Equal-Weighted  (Ken French data library reference)

Universe   : all stocks with Major_Exchange_Listed == 1 on the formation date
Breakpoints: S&P 500 OR Russell 3000 percentiles (10th … 90th)
Momentum   : price(t-2) / price(t-12) – 1  (skips most-recent month)
Rebalancing: monthly, end-of-month
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import parse_ff_monthly_section, compute_stats, stats_label


# ── Configuration ─────────────────────────────────────────────────────────────

FF_FILE  = "10_Portfolios_Prior_12_2.csv"
PARQUET  = "survivorship_bias_free_db.parquet"
START    = "1990"
END      = "2025"
PCTILES  = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])   # 9 cuts → 10 deciles


# ── 1. Fama-French reference ───────────────────────────────────────────────────

ff          = parse_ff_monthly_section(FF_FILE, "Average Equal Weighted Returns -- Monthly")
ff_hi_prior = ff["Hi PRIOR"].loc[START:END]


# ── 2. Load parquet ────────────────────────────────────────────────────────────

df = pd.read_parquet(
    PARQUET,
    columns=["Close", "Symbol", "In_S&P_500", "In_Russell_3000", "Major_Exchange_Listed"],
)
df = df.reset_index()
df["Symbol"] = df["Symbol"].astype(str)


# ── 3. End-of-month price grid ─────────────────────────────────────────────────

all_dates    = pd.to_datetime(df["Date"].unique())
date_series  = pd.Series(all_dates, index=all_dates).sort_index()
eom_dates    = set(date_series.groupby(date_series.dt.to_period("M")).last().values)

df_eom = df[df["Date"].isin(eom_dates)].copy()

monthly_prices    = df_eom.pivot_table(index="Date", columns="Symbol", values="Close").sort_index()
monthly_sp500     = df_eom.pivot_table(index="Date", columns="Symbol", values="In_S&P_500"    ).fillna(0).astype(bool).sort_index()
monthly_r3000     = df_eom.pivot_table(index="Date", columns="Symbol", values="In_Russell_3000").fillna(0).astype(bool).sort_index()
monthly_maj_exch  = df_eom.pivot_table(index="Date", columns="Symbol", values="Major_Exchange_Listed").fillna(0).astype(bool).sort_index()

print(f"Grid: {monthly_prices.shape[0]} months × {monthly_prices.shape[1]} symbols")


# ── 4. Momentum signal & forward returns ──────────────────────────────────────

# Momentum at formation date t: return from t-12 to t-2 (skips t-1)
momentum   = monthly_prices.shift(2) / monthly_prices.shift(12) - 1
monthly_rets = monthly_prices.pct_change()
next_rets    = monthly_rets.shift(-1)   # return earned in month t+1


# ── 5. Simulation function ────────────────────────────────────────────────────

def run_simulation(
    monthly_prices: pd.DataFrame,
    momentum: pd.DataFrame,
    next_rets: pd.DataFrame,
    monthly_maj_exch: pd.DataFrame,
    bp_matrix: pd.DataFrame,
    label: str,
) -> pd.Series:
    """
    Run Hi PRIOR equal-weight backtest.

    bp_matrix : boolean DataFrame – True where a stock belongs to the
                breakpoint universe (e.g. S&P 500 or Russell 3000).
    Returns a monthly return Series.
    """
    all_months = monthly_prices.index
    returns    = {}

    for i in range(12, len(all_months) - 1):
        formation_date = all_months[i]
        return_date    = all_months[i + 1]

        mom          = momentum.loc[formation_date]
        bp_m         = bp_matrix.loc[formation_date]
        maj_exch_m   = monthly_maj_exch.loc[formation_date]
        next_ret     = next_rets.loc[formation_date]

        # Breakpoints from the chosen index universe
        bp_mom = mom[bp_m].dropna()
        if len(bp_mom) < 50:
            continue
        breakpoints = np.percentile(bp_mom.values, PCTILES)

        # Investment universe: major-exchange-listed stocks with valid momentum
        universe_mom = mom[maj_exch_m].dropna()
        if len(universe_mom) < 100:
            continue

        deciles   = np.digitize(universe_mom.values, breakpoints) + 1  # 1-based
        hi_stocks = universe_mom.index[deciles == 10]
        valid_ret = next_ret[hi_stocks].dropna()
        if len(valid_ret) > 0:
            returns[return_date] = valid_ret.mean()

    return pd.Series(returns, name=label).sort_index()


# ── 6. Run both simulations ────────────────────────────────────────────────────

sim_sp500 = run_simulation(
    monthly_prices, momentum, next_rets, monthly_maj_exch,
    monthly_sp500, "SIM – S&P 500 BP",
).loc[START:END]

sim_r3000 = run_simulation(
    monthly_prices, momentum, next_rets, monthly_maj_exch,
    monthly_r3000, "SIM – Russell 3000 BP",
).loc[START:END]

print(f"SIM S&P 500 BP   : {len(sim_sp500)} months")
print(f"SIM Russell 3000 : {len(sim_r3000)} months")


# ── 7. Cumulative returns (normalised to 1.0 at common start) ─────────────────

def to_cum(ret: pd.Series) -> pd.Series:
    cum = (1 + ret).cumprod()
    return cum / cum.iloc[0]

all_series = {"SIM – S&P 500 BP": sim_sp500, "SIM – Russell 3000 BP": sim_r3000, "FF Hi PRIOR EW": ff_hi_prior}
common_start = max(s.dropna().index[0] for s in all_series.values())

cum_series = {name: to_cum(s[s.index >= common_start]) for name, s in all_series.items()}


# ── 8. Plot ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 8))

colors = ["steelblue", "darkorange", "forestgreen"]
for (name, cum), color in zip(cum_series.items(), colors):
    s = compute_stats(all_series[name][all_series[name].index >= common_start])
    cum.plot(ax=ax, label=stats_label(name, s), color=color, linewidth=1.5)

ax.set_title(
    "Momentum Hi PRIOR: Simulation vs Fama-French Reference\n"
    "12-2, Equal-Weight, Universe = Major Exchange Listed"
)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (normalized to 1.0)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
