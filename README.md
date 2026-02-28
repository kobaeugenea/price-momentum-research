# US Momentum Factor Research

Replication and analysis of the Fama-French 12-2 price momentum factor on the full US equity universe (survivorship-bias-free), using Norgate Data as the data source.

> **Note:** The majority of the code in this repository was generated with the assistance of [Claude](https://claude.ai) (Anthropic).

---

## Project Overview

The goal is to closely replicate the [Ken French momentum portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) from raw price data, and to explore how different implementation choices (universe definition, breakpoint source) affect the results.

Key findings:
- Using **major-exchange-listed stocks** as the investment universe (filtering out OTC/Pink Sheets) produces results that track the Fama-French equal-weight Hi PRIOR reference closely over 30+ years, regardless of which breakpoint universe is used.
- Both S&P 500 and Russell 3000 breakpoints replicate the FF series well across most of the sample. The divergences are largely concentrated in the most recent years: S&P 500 breakpoints tend to slightly overstate the Hi PRIOR return relative to FF, while Russell 3000 breakpoints tend to slightly understate it. A plausible explanation is the growing concentration of the equity market in a handful of mega-cap technology stocks: because these dominate the S&P 500's momentum distribution, they shift its decile boundaries in a way that is different from the broader Russell 3000 universe — nudging the composition of the top decile in opposite directions and causing the two simulations to bracket the FF reference.

---

## Articles

Detailed write-ups on the methodology and findings:

- [On Price Momentum (Part 1)](https://medium.com/@kobaeugenea/on-price-momentum-part-1-3c777b546420) — English (Medium)
- [О ценовом моментуме (часть 1)](https://habr.com/ru/articles/1004740/) — Russian (Habr)

---

## Data Requirements

- **[Norgate Data](https://norgatedata.com/)** subscription — US Equities + US Equities Delisted databases.
- **[Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)** — download `10_Portfolios_Prior_12_2_CSV.zip`, extract, and place `10_Portfolios_Prior_12_2.csv` in the project root.
- **[yfinance](https://github.com/ranaroussi/yfinance)** — used for benchmark prices (`SPY`, `RSP`). Install via `pip install yfinance`.

---

## File Descriptions

### Data Pipeline

| File | Description |
|------|-------------|
| `build_database.py` | **Run first.** Downloads price history for all active and delisted US equities from Norgate, enriches each row with GICS sector, S&P 500 / Russell 3000 index-membership flags, and `Major_Exchange_Listed` (1 = NYSE/Nasdaq/etc., 0 = OTC), and saves to `survivorship_bias_free_db.parquet`. |

### Backtesting

| File | Description |
|------|-------------|
| `momentum_backtest.py` | Simulates the Hi PRIOR (top-decile momentum) equal-weight portfolio. Runs two variants — S&P 500 breakpoints and Russell 3000 breakpoints — and plots both against the FF reference. Displays CAGR, Sharpe, and Max Drawdown in the legend. |

### Visualisation

| File | Description |
|------|-------------|
| `plot_deciles_ew.py` | Plots cumulative returns for all 10 **equal-weighted** momentum deciles (Lo PRIOR → Hi PRIOR) using Fama-French reference data. |
| `plot_deciles_vw.py` | Same as above but for **value-weighted** deciles. |
| `plot_hi_prior_vs_benchmarks.py` | Compares Hi PRIOR EW and VW (from FF) against `SPY` (SPDR S&P 500 ETF, inception 1993-01-22) downloaded via yfinance. Total return is captured via `Adj Close`. |

### Validation

| File | Description |
|------|-------------|
| `validate_norgate_vs_rsp.py` | Validates Norgate data quality by comparing a simulated equal-weight S&P 500 (quarterly rebalance, survivorship-bias-free) against the RSP ETF from yfinance. A close match confirms that constituent flags, prices, and delisted-stock handling are correct. |

### Utilities

| File | Description |
|------|-------------|
| `utils.py` | Shared helpers: `parse_ff_monthly_section()` for reading Ken French CSV files, `compute_stats()` for CAGR / Sharpe / Max Drawdown, and `stats_label()` for legend formatting. |

---

## How to Run

```bash
# 1. Install dependencies
pip install norgatedata pandas pyarrow matplotlib tqdm yfinance

# 2. Build the database (takes ~6 hours depending on Norgate API speed)
python build_database.py

# 3. Run the momentum backtest
python momentum_backtest.py

# 4. Generate charts
python plot_deciles_ew.py
python plot_deciles_vw.py
python plot_hi_prior_vs_benchmarks.py

# 5. (Optional) Validate Norgate data against RSP ETF
python validate_norgate_vs_rsp.py
```

---

## Methodology

- **Momentum signal:** `price(t-2) / price(t-12) – 1` — skips the most-recent month to avoid short-term reversal bias.
- **Breakpoints:** Computed from S&P 500 or Russell 3000 constituents at formation date (10th–90th percentiles, 9 cuts → 10 deciles).
- **Universe:** All stocks with `Major_Exchange_Listed == 1` (NYSE, Nasdaq, NYSE American, NYSE Arca, Cboe BZX, IEX) — excludes OTC/Pink Sheets.
- **Weighting:** Equal-weight within each decile.
- **Rebalancing:** Monthly, end-of-month.
- **Note on FF methodology:** The official Fama-French breakpoints use NYSE-listed stocks only. S&P 500 breakpoints are a practical proxy that produces very similar results; Russell 3000 breakpoints diverge because the inclusion of small-cap NASDAQ stocks shifts the momentum distribution.

---

## Caveats for Live Trading

| Factor | Estimated impact | Basis |
|--------|-----------------|-------|
| Momentum crashes (2009, 2020) | Drawdowns of 40–50%+ | Directly observed in backtest |
| Equal-weight → implicit small-cap bias | Overstates returns vs a cap-weighted implementation | Inferred from comparing FF EW vs VW series |
| Transaction costs (commissions + bid-ask) | −1 to −3% CAGR | Estimated — not modelled in backtest |
| Market impact on small/mid caps | −0.5 to −2% CAGR | Estimated — not modelled in backtest |
| Strategy capacity | ~$5–50M with equal-weight small caps | Estimated — not tested empirically |

---

## License

Data is subject to Norgate Data and Ken French data library terms of use. Code is released under the MIT License.
