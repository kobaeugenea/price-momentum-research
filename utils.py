"""Shared utilities: Fama-French CSV parsing and portfolio statistics."""

import numpy as np
import pandas as pd


# ── Fama-French data ──────────────────────────────────────────────────────────

def parse_ff_monthly_section(filepath: str, section_name: str) -> pd.DataFrame:
    """
    Parse a named section from a Ken French data library CSV.
    Returns a DataFrame indexed by month-end timestamps, values as decimals.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    start_idx = next((i for i, l in enumerate(lines) if section_name in l), None)
    if start_idx is None:
        raise ValueError(f"Section '{section_name}' not found in {filepath}")

    col_header = lines[start_idx + 1].strip().split(",")
    col_header[0] = "Date"

    data_lines = []
    for line in lines[start_idx + 2:]:
        stripped = line.strip()
        if not stripped:
            break
        data_lines.append(stripped.split(","))

    df = pd.DataFrame(data_lines, columns=col_header)
    df["Date"] = (
        pd.to_datetime(df["Date"].str.strip(), format="%Y%m")
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )
    df = df.set_index("Date").apply(pd.to_numeric, errors="coerce")
    df.replace([-99.99, -999.0], np.nan, inplace=True)
    df /= 100
    return df


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(monthly_ret: pd.Series) -> dict:
    """Compute CAGR, annualised Sharpe (rf=0), and max drawdown from monthly returns."""
    ret = monthly_ret.dropna()
    n_years = len(ret) / 12
    cagr   = (1 + ret).prod() ** (1 / n_years) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(12)
    cum    = (1 + ret).cumprod()
    max_dd = ((cum / cum.cummax()) - 1).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd}


def stats_label(name: str, s: dict) -> str:
    """Format a legend entry with key performance statistics."""
    return f"{name}  CAGR={s['CAGR']:.1%}  Sharpe={s['Sharpe']:.2f}  MaxDD={s['MaxDD']:.1%}"
