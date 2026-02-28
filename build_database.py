"""
Build Survivorship-Bias-Free US Equity Database
================================================
Downloads price history for all active and delisted US equities from Norgate Data,
enriches each row with sector, index-membership, and major-exchange-listing flags,
and saves the result to a single Parquet file.

Output columns (beyond standard OHLCV + Turnover from Norgate):
  Symbol                 – ticker
  Security_Name          – full company name
  Sector                 – GICS level-1 sector name
  In_S&P_500             – 1 if S&P 500 constituent on that date, else 0
  In_Russell_3000        – 1 if Russell 3000 constituent on that date, else 0
  Major_Exchange_Listed  – 1 if listed on NYSE / Nasdaq / NYSE Arca / Cboe BZX / IEX,
                           0 if OTC / Pink Sheet

Requirements: pip install norgatedata pandas pyarrow tqdm
"""

import norgatedata
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm


# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE  = pd.Timestamp("1990-01-01")
END_DATE    = pd.Timestamp("2026-01-01")
OUTPUT_FILE = "survivorship_bias_free_db.parquet"
INDEX_LIST  = ["S&P 500", "Russell 3000"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)


# ── Price download ─────────────────────────────────────────────────────────────

def download_stock_data(symbol: str) -> pd.DataFrame | None:
    try:
        df = norgatedata.price_timeseries(
            symbol,
            stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.TOTALRETURN,
            padding_setting=norgatedata.PaddingType.NONE,
            start_date=START_DATE,
            end_date=END_DATE,
            timeseriesformat="pandas-dataframe",
        )
        if df is None or df.empty:
            return None
        df["Symbol"] = symbol
        return df
    except Exception as e:
        logging.warning(f"Failed {symbol}: {e}")
        return None


def get_all_symbols() -> tuple[list, list]:
    logging.info("Fetching symbol lists …")
    active   = norgatedata.database_symbols("US Equities")
    delisted = norgatedata.database_symbols("US Equities Delisted")
    logging.info(f"Active: {len(active):,}   Delisted: {len(delisted):,}")
    return active, delisted


def filter_equities(symbols: list, label: str) -> list:
    """Keep only symbols whose Norgate subtype1 == 'Equity'."""
    result = []
    for s in tqdm(symbols, desc=f"Filtering {label}", unit="symbol"):
        try:
            if norgatedata.subtype1(s) == "Equity":
                result.append(s)
        except Exception:
            continue
    return result


def download_all(symbols: list, label: str) -> pd.DataFrame:
    all_data = []
    for s in tqdm(symbols, desc=f"Downloading {label}", unit="symbol"):
        df = download_stock_data(s)
        if df is not None:
            all_data.append(df)
    return pd.concat(all_data) if all_data else pd.DataFrame()


# ── Enrichment helpers ────────────────────────────────────────────────────────

def add_security_names(df: pd.DataFrame) -> pd.DataFrame:
    names = {}
    for s in tqdm(df["Symbol"].unique(), desc="Security names", unit="symbol"):
        try:
            names[s] = norgatedata.security_name(s)
        except Exception:
            names[s] = None
    df["Security_Name"] = df["Symbol"].map(names)
    return df


def add_sector_info(df: pd.DataFrame) -> pd.DataFrame:
    sectors = {}
    for s in tqdm(df["Symbol"].unique(), desc="GICS sectors", unit="symbol"):
        try:
            sectors[s] = norgatedata.classification_at_level(s, "GICS", "Name", level=1)
        except Exception:
            sectors[s] = None
    df["Sector"] = df["Symbol"].map(sectors)
    return df


def add_index_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-varying index-membership columns for every index in INDEX_LIST."""
    result = []
    for symbol in tqdm(df["Symbol"].unique(), desc="Index flags", unit="symbol"):
        temp = df[df["Symbol"] == symbol].copy()
        for index_name in INDEX_LIST:
            col = f"In_{index_name.replace(' ', '_')}"
            try:
                idx_df = norgatedata.index_constituent_timeseries(
                    symbol, index_name,
                    pandas_dataframe=temp.copy(),
                    padding_setting=norgatedata.PaddingType.NONE,
                    timeseriesformat="pandas-dataframe",
                )
                temp[col] = idx_df["Index Constituent"]
            except Exception:
                temp[col] = 0
        result.append(temp)
    return pd.concat(result)


def add_major_exchange_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Major_Exchange_Listed column.
      1 = listed on NYSE, Nasdaq, NYSE American, NYSE Arca, Cboe BZX, or IEX
      0 = OTC / Pink Sheet
    """
    all_flags = []
    ok = failed = 0
    pbar = tqdm(
        df["Symbol"].unique(),
        desc="Major Exchange Listed",
        unit="symbol",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]  "
            "ok={postfix[0]} failed={postfix[1]}"
        ),
        postfix=[0, 0],
    )
    for symbol in pbar:
        try:
            ra = norgatedata.major_exchange_listed_timeseries(
                symbol, timeseriesformat="numpy-recarray"
            )
            if ra is None or len(ra) == 0:
                raise ValueError("empty response")
            value_field = [f for f in ra.dtype.names if f.lower() != "date"][0]
            all_flags.append(pd.DataFrame({
                "Date":                 pd.to_datetime(ra["Date"]),
                "Symbol":               symbol,
                "Major_Exchange_Listed": ra[value_field].astype(np.int8),
            }))
            ok += 1
        except Exception as e:
            tqdm.write(f"[WARN] {symbol}: {e}")
            failed += 1
        pbar.postfix[0] = ok
        pbar.postfix[1] = failed
        pbar.update(0)

    if not all_flags:
        logging.warning("No major-exchange data fetched — column will be all zeros.")
        df["Major_Exchange_Listed"] = np.int8(0)
        return df

    flags_df = pd.concat(all_flags, ignore_index=True)

    had_date_index = df.index.name == "Date" or isinstance(df.index, pd.DatetimeIndex)
    if had_date_index:
        df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    merged = df.merge(flags_df, on=["Date", "Symbol"], how="left")
    merged["Major_Exchange_Listed"] = merged["Major_Exchange_Listed"].fillna(0).astype(np.int8)

    if had_date_index:
        merged = merged.set_index("Date")
    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def enrich(data: pd.DataFrame) -> pd.DataFrame:
    data = add_security_names(data)
    data = add_sector_info(data)
    data = add_index_flags(data)
    data = add_major_exchange_flag(data)
    return data


def main():
    logging.info("Starting survivorship-bias-free database build …")

    active_symbols, delisted_symbols = get_all_symbols()

    active_eq   = filter_equities(active_symbols,   "ACTIVE")
    delisted_eq = filter_equities(delisted_symbols, "DELISTED")

    active_data   = download_all(active_eq,   "ACTIVE")
    delisted_data = download_all(delisted_eq, "DELISTED")

    if not active_data.empty:
        active_data = enrich(active_data)
    if not delisted_data.empty:
        delisted_data = enrich(delisted_data)

    final_df = pd.concat([active_data, delisted_data])
    final_df.sort_index(inplace=True)

    logging.info(f"Saving {len(final_df):,} rows to {OUTPUT_FILE} …")
    final_df.to_parquet(OUTPUT_FILE, engine="pyarrow")
    logging.info("Done ✅")


if __name__ == "__main__":
    main()
