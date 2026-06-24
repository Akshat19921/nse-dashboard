"""
Build ONE Excel workbook for the whole Nifty 500 universe (5y daily).
Nifty 100 / LargeMidcap 250 / Nifty 500 are all served by FILTERING this file.

    python build_market_excel.py

Output: market_data.xlsx  (sheets: Price History [all symbols stacked], Symbols)
"""
import sys
import pandas as pd
import yfinance as yf

SYMS = "nifty500_symbols.csv"
OUT = "market_data.xlsx"
PERIOD = "5y"

def main():
    syms = pd.read_csv(SYMS)["Symbol"].dropna().astype(str).str.strip().tolist()
    frames, ok, fail = [], 0, []
    for i, s in enumerate(syms, 1):
        try:
            d = yf.download(s, period=PERIOD, interval="1d", auto_adjust=False,
                            progress=False, threads=False)
            if d is None or d.empty:
                fail.append(s); continue
            if hasattr(d.columns, "get_level_values"):
                d.columns = d.columns.get_level_values(0)
            d = d.reset_index().rename(columns={"Adj Close": "AdjClose"})
            keep = [c for c in ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"] if c in d.columns]
            d = d[keep].dropna(subset=["Open", "High", "Low", "Close"])
            d.insert(0, "Symbol", s)
            frames.append(d); ok += 1
        except Exception:
            fail.append(s)
        if i % 25 == 0:
            print(f"{i}/{len(syms)} ok={ok} fail={len(fail)}", flush=True)
    if not frames:
        print("no data downloaded", flush=True); sys.exit(1)
    hist = pd.concat(frames, ignore_index=True)
    syms_df = pd.read_csv(SYMS)
    with pd.ExcelWriter(OUT, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xl:
        hist.to_excel(xl, sheet_name="Price History", index=False)
        syms_df.to_excel(xl, sheet_name="Symbols", index=False)
    print(f"DONE rows={len(hist)} ok={ok} fail={len(fail)} -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
