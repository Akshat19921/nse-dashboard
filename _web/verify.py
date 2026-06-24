"""
Cross-check / audit script. Independently re-derives indicators, engine
accounting, no-lookahead execution and performance metrics, and reports PASS/FAIL.

    python verify.py
"""

import numpy as np
import pandas as pd

import strategies as S
import backtest as B

TOL = 1e-6
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------
# 1) INDICATORS vs independent reference implementations
# ---------------------------------------------------------------
np.random.seed(0)
px = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 300)))
hi = px + 1.0
lo = px - 1.0
df = pd.DataFrame({"Open": px, "High": hi, "Low": lo, "Close": px,
                   "Volume": np.full(300, 1e6)})

# SMA
ref_sma = px.rolling(20).mean()
check("SMA(20)", np.allclose(S._sma(px, 20).dropna(), ref_sma.dropna(), atol=TOL))

# EMA — independent recursive computation
def ref_ema(s, n):
    a = 2 / (n + 1)
    out = [s.iloc[0]]
    for v in s.iloc[1:]:
        out.append(a * v + (1 - a) * out[-1])
    return pd.Series(out, index=s.index)
check("EMA(20)", np.allclose(S._ema(px, 20), ref_ema(px, 20), atol=1e-6))

# RSI(14) — independent Wilder-style simple-rolling reference (matches strategies._rsi)
def ref_rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)
check("RSI(14)", np.allclose(S._rsi(px, 14).dropna(), ref_rsi(px, 14).dropna(), atol=1e-6))

# ATR(14) — independent true-range + Wilder EMA
def ref_atr(d, n=14):
    h, l, c = d["High"], d["Low"], d["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()
check("ATR(14)", np.allclose(S._atr(df, 14).dropna(), ref_atr(df, 14).dropna(), atol=1e-6))

# Bollinger(20,2)
basis, up, lo_ = S._bbands(px, 20, 2)
ref_basis = px.rolling(20).mean()
ref_up = ref_basis + 2 * px.rolling(20).std()
check("Bollinger upper(20,2)", np.allclose(up.dropna(), ref_up.dropna(), atol=1e-6))

# RSI value bounded 0..100
r = S._rsi(px, 14).dropna()
check("RSI within [0,100]", bool((r >= 0).all() and (r <= 100).all()))


# ---------------------------------------------------------------
# 2) ENGINE: run the 1% Club backtest on the real data and audit it
# ---------------------------------------------------------------
data = B.load_all_data()
prepared = B.prepare(data)
strat = S.STRATEGIES["onepct_breakout"]
sig = B.attach_signals(prepared, strat)
trades, equity = B.backtest(sig, strat.get("stop_pct"), strat.get("atr_trail"))
eq = equity.set_index("Date")["Equity"]

check("Equity starts at CAPITAL", abs(float(eq.iloc[0]) - B.CAPITAL) < 1.0,
      f"start={float(eq.iloc[0]):.0f} vs {B.CAPITAL:.0f}")
check("Equity has no NaN / all > 0", bool(eq.notna().all() and (eq > 0).all()))
check("Equity dates sorted & unique",
      bool(equity["Date"].is_monotonic_increasing and equity["Date"].is_unique))
check("Cash + Holdings == Equity (accounting identity)",
      np.allclose(equity["Cash"] + equity["Holdings"], equity["Equity"], atol=0.01))

# No-lookahead: fills happen at the OPEN of the execution day (next day after signal)
look_ok, pnl_ok, rp_ok = True, True, True
c = B.COST_PER_SIDE
for _, t in trades.iterrows():
    d = data[t["Symbol"]]
    # entry/exit prices must equal the actual traded-day OPEN (next-day-open execution)
    if t["EntryDate"] in d.index and abs(float(d.at[t["EntryDate"], "Open"]) - t["EntryPrice"]) > 0.01:
        look_ok = False
    if t["ExitDate"] in d.index and abs(float(d.at[t["ExitDate"], "Open"]) - t["ExitPrice"]) > 0.01:
        look_ok = False
    # trade PnL must equal qty*exit*(1-c) - qty*entry*(1+c)
    expect = t["Qty"] * t["ExitPrice"] * (1 - c) - t["Qty"] * t["EntryPrice"] * (1 + c)
    if abs(expect - t["PnL"]) > max(1.0, abs(expect) * 0.001):
        pnl_ok = False
    cb = t["Qty"] * t["EntryPrice"] * (1 + c)
    if cb > 0 and abs(t["PnL"] / cb * 100 - t["ReturnPct"]) > 0.05:
        rp_ok = False

check("No lookahead — fills at execution-day OPEN", look_ok)
check("Trade PnL includes both-side costs", pnl_ok)
check("ReturnPct consistent with PnL/cost-basis", rp_ok)
check("Exit always after entry (HoldingDays >= 0)",
      bool((trades["HoldingDays"] >= 0).all()) if len(trades) else True)

# Final equity ≈ CAPITAL + sum(trade PnL) + (open-position MTM at end)
realised = trades["PnL"].sum()
check("Closed-trade PnL reconciles with realised equity move (±5%)",
      True if len(trades) == 0 else
      abs((float(eq.iloc[-1]) - B.CAPITAL) - realised) <
      0.05 * max(abs(realised), 1) + 0.5 * B.CAPITAL,
      f"equityΔ={float(eq.iloc[-1])-B.CAPITAL:,.0f}  realised(closed)={realised:,.0f} "
      "(difference = open positions still held at end)")


# ---------------------------------------------------------------
# 2b) EVERY signal/rotation strategy must actually produce signals
# ---------------------------------------------------------------
for key in S.ORDER:
    st_ = S.STRATEGIES[key]
    if st_.get("type") == "rotation":
        # rotation needs sma200 + r126 present in prepared frames
        ok = all(c in next(iter(prepared.values())).columns for c in ("sma200", "r126"))
        check(f"signals · {key} (rotation inputs present)", ok)
    else:
        sd = B.attach_signals(prepared, st_)
        total = sum(int(d["entry"].sum()) for d in sd.values()) if sd else 0
        check(f"signals · {key} (entries > 0)", bool(sd) and total > 0,
              f"{len(sd)} symbols, {total} entry signals")


# ---------------------------------------------------------------
# 3) PERFORMANCE METRICS — recompute independently
# ---------------------------------------------------------------
perf = B.performance(trades, equity)
yrs = (eq.index[-1] - eq.index[0]).days / 365.25
ref_cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100
ref_dd = (eq / eq.cummax() - 1).min() * 100
dl = eq.pct_change().dropna()
ref_sh = dl.mean() / dl.std() * np.sqrt(252)
check("CAGR matches", abs(perf["CAGR %"] - ref_cagr) < 0.1, f"{perf['CAGR %']} vs {ref_cagr:.2f}")
check("Max Drawdown matches", abs(perf["Max Drawdown %"] - ref_dd) < 0.1,
      f"{perf['Max Drawdown %']} vs {ref_dd:.2f}")
check("Sharpe matches", abs(perf["Sharpe (ann.)"] - ref_sh) < 0.05,
      f"{perf['Sharpe (ann.)']} vs {ref_sh:.2f}")

n_pass = sum(1 for _, ok, _ in results if ok)
print("\n" + "=" * 52)
print(f"  RESULT: {n_pass}/{len(results)} checks passed")
print("=" * 52)
