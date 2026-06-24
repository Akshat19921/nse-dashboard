"""
Market (universe) registry. All markets are served by filtering ONE data file
(nifty500.xlsx) down to the market's symbol list.

  Nifty 100            -> nifty100_symbols.csv          (subset)
  Nifty LargeMidcap250 -> nifty_largemidcap250_symbols.csv (subset)
  Nifty 500            -> nifty500_symbols.csv          (the full superset)
"""
import os
import pandas as pd

DATA_XLSX = "market_data.xlsx"  # single workbook holding all symbols' 5y history
HISTORY_SHEET = "Price History"

MARKETS = {
    "nifty100":       {"name": "Nifty 100",             "symbols": "nifty100_symbols.csv"},
    "largemidcap250": {"name": "Nifty LargeMidcap 250", "symbols": "nifty_largemidcap250_symbols.csv"},
    "nifty500":       {"name": "Nifty 500",             "symbols": "nifty500_symbols.csv"},
}
ORDER = ["nifty100", "largemidcap250", "nifty500"]
DEFAULT = "nifty100"


def label(key):
    return MARKETS[key]["name"]


def choices():
    return [(k, MARKETS[k]["name"]) for k in ORDER]


def market_symbols(key):
    """List of '.NS' symbols for a market (empty if its CSV is missing)."""
    f = MARKETS.get(key, {}).get("symbols")
    if not f or not os.path.exists(f):
        return []
    df = pd.read_csv(f)
    return df["Symbol"].dropna().astype(str).str.strip().tolist()


def market_names(key):
    """dict symbol -> display name."""
    f = MARKETS.get(key, {}).get("symbols")
    if not f or not os.path.exists(f):
        return {}
    df = pd.read_csv(f)
    if "Name" in df.columns:
        return dict(zip(df["Symbol"].astype(str), df["Name"].astype(str)))
    return {}
