"""
Strategy backtester for the NSE Nifty 100 universe.

Backtests ANY strategy defined in strategies.py (the same ones the dashboard
screens). Each strategy supplies full-series entry/exit signals; this file runs
a shared portfolio engine over them.

Engine assumptions (from the playbook):
  - Signals on the daily close; execution on the NEXT day's open.
  - Equal weight, max 10% of starting capital per stock (=> max 10 positions).
  - Costs (transaction + slippage): 0.10% per side.
  - Exit = the strategy's exit rule, OR an entry-price stop (strategy 'stop_pct').

Usage:
    pip install pandas numpy openpyxl
    python backtest.py                       # default strategy (1% Club)
    python backtest.py --strategy s2_supertrend
    python backtest.py --strategy all        # backtest every strategy

Data source: nifty100.xlsx [Price History] (falls back to ./data/*.csv).
Output: backtest_results/<strategy_key>/trades.csv + equity_curve.csv
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

import strategies
import markets

# =====================================================
# CONFIG
# =====================================================
EXCEL_FILE = "nifty100.xlsx"
HISTORY_SHEET = "Price History"
DATA_FOLDER = "data"
RESULTS_FOLDER = "backtest_results"

CAPITAL = 100_000.0          # ₹1,00,000 (1 lakh)
MAX_WEIGHT = 0.10            # max 10% per stock
PER_TRADE_ALLOC = CAPITAL * MAX_WEIGHT
MAX_POSITIONS = int(round(1 / MAX_WEIGHT))   # 10

COST_PER_SIDE = 0.001        # 0.10% per side
MIN_BARS = 260               # need ~1y before 52w/200-day rules are valid

TRADING_DAYS = 252


# =====================================================
# DATA LOADING (raw OHLCV; indicators come from the strategy)
# =====================================================
def load_all_data(market=None):
    """Load price history, optionally filtered to a market's symbol list.
    Prefers the single nifty500.xlsx workbook; falls back to nifty100.xlsx / data/."""
    sym_set = set(markets.market_symbols(market)) if market else None
    src = markets.DATA_XLSX if os.path.exists(markets.DATA_XLSX) else (
        EXCEL_FILE if os.path.exists(EXCEL_FILE) else None)
    if src:
        print(f"  reading {src} [{markets.HISTORY_SHEET}]...")
        raw = pd.read_excel(src, sheet_name=markets.HISTORY_SHEET, parse_dates=["Date"])
        if sym_set:
            raw = raw[raw["Symbol"].isin(sym_set)]
        return _clean({sym: g.drop(columns=["Symbol"]) for sym, g in raw.groupby("Symbol")})
    files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    frames = {}
    for f in files:
        s = os.path.splitext(os.path.basename(f))[0]
        if sym_set and s not in sym_set:
            continue
        frames[s] = pd.read_csv(f, parse_dates=["Date"])
    return _clean(frames)


def _clean(frames):
    out = {}
    for sym, df in frames.items():
        if not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
            continue
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_BARS:
            continue
        out[sym] = df.set_index("Date")
    return out


def prepare(data):
    """Per-symbol frames with common indicators (ATR, 200SMA, 6-mo return, momentum)."""
    out = {}
    for sym, df in data.items():
        cols = ["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])
        d = df[cols].copy()
        d["atr"] = strategies._atr(df, 22).values
        d["sma200"] = strategies._sma(df["Close"], 200).values   # 200 DMA trend filter
        d["r126"] = (df["Close"] / df["Close"].shift(126) - 1).values
        d["mom"] = (df["Close"] / df["Close"].shift(20)).values
        out[sym] = d
    return out


def attach_signals(prepared, strat):
    """Add the strategy's entry/exit boolean columns (for signal strategies)."""
    sig_fn = strat["signals"]
    out = {}
    for sym, d in prepared.items():
        try:
            sig = sig_fn(d.reset_index()).set_index(d.index)
        except Exception:
            continue
        d = d.copy()
        d["entry"] = sig["entry"].values
        d["exit"] = sig["exit"].values
        out[sym] = d
    return out


# =====================================================
# PORTFOLIO ENGINE (next-day-open execution, shared cash)
# =====================================================
def backtest(data, stop_pct, atr_trail=None):
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in data.values()])))
    cash = CAPITAL
    positions = {}
    pending_buys, pending_sells = [], []
    trades, equity_curve = [], []

    def price(sym, date, col):
        df = data[sym]
        if date in df.index:
            v = df.at[date, col]
            return float(v) if pd.notna(v) else None
        return None

    for i, date in enumerate(all_dates):
        # 1. execute pending orders at today's open
        for sym in pending_sells:
            if sym not in positions:
                continue
            o = price(sym, date, "Open")
            if o is None:
                continue
            pos = positions.pop(sym)
            proceeds = pos["qty"] * o * (1 - COST_PER_SIDE)
            cash += proceeds
            cost_basis = pos["qty"] * pos["entry_price"] * (1 + COST_PER_SIDE)  # incl. buy cost
            pnl = proceeds - cost_basis
            trades.append({
                "Symbol": sym, "EntryDate": pos["entry_date"],
                "EntryPrice": round(pos["entry_price"], 4),
                "ExitDate": date, "ExitPrice": round(o, 4), "Qty": pos["qty"],
                "PnL": round(pnl, 2),
                "ReturnPct": round(pnl / cost_basis * 100, 2),  # net of both-side costs
                "ExitReason": pos.get("exit_reason", ""),
                "HoldingDays": (date - pos["entry_date"]).days,
            })
        for sym in pending_buys:
            if sym in positions or len(positions) >= MAX_POSITIONS:
                continue
            o = price(sym, date, "Open")
            if o is None or o <= 0:
                continue
            alloc = min(PER_TRADE_ALLOC, cash)
            if alloc < o * (1 + COST_PER_SIDE):
                continue
            qty = int(alloc // (o * (1 + COST_PER_SIDE)))
            if qty <= 0:
                continue
            cash -= qty * o * (1 + COST_PER_SIDE)
            positions[sym] = {"entry_price": o, "qty": qty, "entry_date": date, "peak": o}
        pending_buys, pending_sells = [], []

        # 2. mark to market
        holdings = 0.0
        for sym, pos in positions.items():
            c = price(sym, date, "Close")
            if c is not None:
                pos["peak"] = max(pos["peak"], c)
            holdings += pos["qty"] * (c if c is not None else pos["entry_price"])
        equity_curve.append({"Date": date, "Cash": round(cash, 2),
                             "Holdings": round(holdings, 2),
                             "Equity": round(cash + holdings, 2),
                             "OpenPositions": len(positions)})

        # 3. signals on today's close (executed next open)
        if i + 1 >= len(all_dates):
            continue
        for sym, pos in list(positions.items()):
            c = price(sym, date, "Close")
            if c is None:
                continue
            ex = bool(data[sym].at[date, "exit"]) if date in data[sym].index else False
            stop_hit = stop_pct is not None and c <= pos["entry_price"] * (1 - stop_pct)
            trail_hit = False
            if atr_trail is not None and date in data[sym].index:
                atr = data[sym].at[date, "atr"]
                if pd.notna(atr):
                    trail_hit = c <= pos["peak"] - atr_trail * atr
            if ex or stop_hit or trail_hit:
                pos["exit_reason"] = ("Trail" if trail_hit and not ex else
                                      "Stop" if stop_hit and not ex else "Exit rule")
                pending_sells.append(sym)
        free = MAX_POSITIONS - (len(positions) - len(pending_sells))
        if free > 0:
            cands = [sym for sym, df in data.items()
                     if sym not in positions and sym not in pending_sells
                     and date in df.index and bool(df.at[date, "entry"])]

            def _mom(s):
                m = data[s].at[date, "mom"] if date in data[s].index else None
                return float(m) if m is not None and pd.notna(m) else 0.0

            cands.sort(key=_mom, reverse=True)
            pending_buys.extend(cands[:free])

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


# =====================================================
# ROTATION ENGINE (cross-sectional monthly rebalance)
# =====================================================
def backtest_rotation(data, topn=10, lookback=126):
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in data.values()])))
    cash = CAPITAL
    positions = {}
    trades, equity_curve = [], []
    last_period = None

    def price(sym, date, col):
        df = data[sym]
        if date in df.index:
            v = df.at[date, col]
            return float(v) if pd.notna(v) else None
        return None

    for date in all_dates:
        period = (date.year, date.month)
        if period != last_period:        # rebalance on first trading day of the month
            last_period = period
            elig = []
            for sym, df in data.items():
                if date in df.index and pd.notna(df.at[date, "r126"]) and pd.notna(df.at[date, "sma200"]) \
                        and df.at[date, "Close"] > df.at[date, "sma200"]:
                    elig.append((sym, float(df.at[date, "r126"])))
            elig.sort(key=lambda x: x[1], reverse=True)
            target = {s for s, _ in elig[:topn]}
            for sym in list(positions):           # sell what dropped out
                if sym not in target:
                    o = price(sym, date, "Open") or price(sym, date, "Close")
                    if o is None:
                        continue
                    pos = positions.pop(sym)
                    proceeds = pos["qty"] * o * (1 - COST_PER_SIDE)
                    cash += proceeds
                    cost_basis = pos["qty"] * pos["entry_price"] * (1 + COST_PER_SIDE)
                    pnl = proceeds - cost_basis
                    trades.append({
                        "Symbol": sym, "EntryDate": pos["entry_date"],
                        "EntryPrice": round(pos["entry_price"], 4), "ExitDate": date,
                        "ExitPrice": round(o, 4), "Qty": pos["qty"],
                        "PnL": round(pnl, 2),
                        "ReturnPct": round(pnl / cost_basis * 100, 2),
                        "ExitReason": "Rotation", "HoldingDays": (date - pos["entry_date"]).days})
            for sym in target:                     # buy new entrants
                if sym in positions:
                    continue
                o = price(sym, date, "Open") or price(sym, date, "Close")
                if o is None or o <= 0:
                    continue
                alloc = min(CAPITAL / topn, cash)
                if alloc < o * (1 + COST_PER_SIDE):
                    continue
                qty = int(alloc // (o * (1 + COST_PER_SIDE)))
                if qty <= 0:
                    continue
                cash -= qty * o * (1 + COST_PER_SIDE)
                positions[sym] = {"entry_price": o, "qty": qty, "entry_date": date}
        holdings = sum(pos["qty"] * (price(s, date, "Close") or pos["entry_price"])
                       for s, pos in positions.items())
        equity_curve.append({"Date": date, "Cash": round(cash, 2),
                             "Holdings": round(holdings, 2),
                             "Equity": round(cash + holdings, 2),
                             "OpenPositions": len(positions)})
    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


# =====================================================
# PERFORMANCE
# =====================================================
def performance(trades, equity):
    if equity.empty:
        return {"Note": "No equity curve."}
    eq = equity.set_index("Date")["Equity"]
    start, end = float(eq.iloc[0]), float(eq.iloc[-1])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    daily = eq.pct_change().dropna()
    sharpe = (daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if daily.std() > 0 else 0.0
    max_dd = (eq / eq.cummax() - 1).min()
    perf = {
        "Start Equity": round(start, 2), "End Equity": round(end, 2),
        "Total Return %": round((end / start - 1) * 100, 2),
        "CAGR %": round(((end / start) ** (1 / years) - 1) * 100, 2),
        "Max Drawdown %": round(max_dd * 100, 2),
        "Sharpe (ann.)": round(sharpe, 2),
        "Period": f"{eq.index[0].date()} -> {eq.index[-1].date()}",
    }
    if not trades.empty:
        wins = trades[trades["PnL"] > 0]; losses = trades[trades["PnL"] <= 0]
        gl = abs(losses["PnL"].sum())
        perf.update({
            "Total Trades": len(trades),
            "Win Rate %": round(len(wins) / len(trades) * 100, 2),
            "Avg Win %": round(wins["ReturnPct"].mean(), 2) if len(wins) else 0.0,
            "Avg Loss %": round(losses["ReturnPct"].mean(), 2) if len(losses) else 0.0,
            "Avg Holding Days": round(trades["HoldingDays"].mean(), 1),
            "Profit Factor": round(wins["PnL"].sum() / gl, 2) if gl > 0 else float("inf"),
        })
    else:
        perf["Total Trades"] = 0
    return perf


def run_one(key, data, market):
    strat = strategies.STRATEGIES[key]
    print(f"\n=== [{market}] Backtesting: {strategies.label(key)} ===")
    prepared = prepare(data)
    if strat.get("type") == "rotation":
        trades, equity = backtest_rotation(prepared, strat.get("topn", 10),
                                           strat.get("lookback", 126))
    else:
        sig_data = attach_signals(prepared, strat)
        if not sig_data:
            print("  no signals computed; skipping.")
            return
        trades, equity = backtest(sig_data, strat.get("stop_pct"), strat.get("atr_trail"))
    out_dir = os.path.join(RESULTS_FOLDER, market, key)
    os.makedirs(out_dir, exist_ok=True)
    trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    equity.to_csv(os.path.join(out_dir, "equity_curve.csv"), index=False)
    perf = performance(trades, equity)
    for k, v in perf.items():
        print(f"  {k:<20}: {v}")
    print(f"  saved -> {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=strategies.DEFAULT_KEY,
                    help="strategy key, or 'all'. Options: " + ", ".join(strategies.ORDER))
    ap.add_argument("--market", default=markets.DEFAULT,
                    help="market key, or 'all'. Options: " + ", ".join(markets.ORDER))
    ap.add_argument("--years", type=int, default=0,
                    help="backtest only the last N years (1/3/5/10). 0 = all available.")
    args = ap.parse_args()

    mkts = markets.ORDER if args.market == "all" else [args.market]
    keys = strategies.ORDER if args.strategy == "all" else [args.strategy]

    for mk in mkts:
        print(f"\n##### MARKET: {markets.label(mk)} #####")
        data = load_all_data(mk)
        if not data:
            print(f"  No data for {mk}. Run build_market_excel.py first.")
            continue
        if args.years and args.years > 0:
            gmax = max(df.index.max() for df in data.values())
            cutoff = gmax - pd.Timedelta(days=int(round(args.years * 365.25)))
            data = {s: df[df.index >= cutoff] for s, df in data.items()}
            data = {s: df for s, df in data.items() if len(df) >= 30}
        print(f"  Loaded {len(data)} symbols.")
        for key in keys:
            if key not in strategies.STRATEGIES:
                print(f"  Unknown strategy '{key}'.")
                continue
            run_one(key, data, mk)


if __name__ == "__main__":
    main()
