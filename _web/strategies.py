"""
Strategy registry for the screener dashboard.

Each strategy is a plain dict with:
  key, name, author, description (markdown), min_bars, sort_col, sort_asc,
  tv_studies (chart overlays), and evaluate(df) -> dict | None.

`df` is a daily OHLCV DataFrame (columns Open, High, Low, Close, Volume).
Every evaluate() returns the keys 'Selected' (qualifies / on watchlist) and
'ENTRY SIGNAL' (trigger fired today) so the dashboard can filter consistently.

These are SCREENERS — they flag entry candidates. Stops, targets and exits are
order rules you apply per name (see the playbook). Educational use only.

To add a strategy: write an evaluate() and append an entry to STRATEGIES + ORDER.
"""

import numpy as np
import pandas as pd

WEEK_52 = 252


# ---------- indicator helpers ----------
def _sma(s, n):
    return s.rolling(n).mean()


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _true_range(df):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(df, n=10):
    return _true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def _adx(df, n=14):
    h, l = df["High"], df["Low"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = _true_range(df).ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _supertrend(df, period=10, mult=3.0):
    """Returns (supertrend_line, direction) where direction 1 = up, -1 = down."""
    c = df["Close"].values
    hl2 = ((df["High"] + df["Low"]) / 2).values
    atr = _atr(df, period).values
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    fu = upper.copy()
    fl = lower.copy()
    direction = np.ones(len(c), dtype=int)
    for i in range(1, len(c)):
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if c[i] > fu[i - 1]:
            direction[i] = 1
        elif c[i] < fl[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    st_line = np.where(direction == 1, fl, fu)
    return pd.Series(st_line, index=df.index), pd.Series(direction, index=df.index)


def _bbands(close, n=20, k=2.0):
    basis = close.rolling(n).mean()
    dev = k * close.rolling(n).std()
    return basis, basis + dev, basis - dev


def _monthly(df):
    """Resample a daily OHLC frame (DatetimeIndex) to monthly candles (month-end)."""
    o = df["Open"].resample("ME").first()
    h = df["High"].resample("ME").max()
    l = df["Low"].resample("ME").min()
    c = df["Close"].resample("ME").last()
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}).dropna()


# =====================================================
# STRATEGY 0 — 1% Club "Strong Stock Breakout"  (also backtested)
# =====================================================
def eval_onepct(df):
    if len(df) < WEEK_52 + 1:
        return None
    c = df["Close"]
    sma50 = _sma(c, 50); sma150 = _sma(c, 150); ema220 = _ema(c, 220)
    high52 = c.rolling(WEEK_52).max(); low52 = df["Low"].rolling(WEEK_52).min()
    dipped = (df["Low"] < ema220).rolling(90).max().fillna(0).astype(bool)
    close = float(c.iloc[-1])
    f1 = float(sma150.iloc[-1]) > float(ema220.iloc[-1])
    f2 = close > float(sma50.iloc[-1])
    f3 = float(sma50.iloc[-1]) > float(sma150.iloc[-1])
    f4 = close > 1.25 * float(low52.iloc[-1])
    f5 = bool(dipped.iloc[-1])
    new_high = close >= float(high52.iloc[-1])
    selected = f1 and f2 and f3 and f4 and f5
    return {
        "Close": round(close, 2), "SMA50": round(float(sma50.iloc[-1]), 2),
        "SMA150": round(float(sma150.iloc[-1]), 2), "EMA220": round(float(ema220.iloc[-1]), 2),
        "52W High": round(float(high52.iloc[-1]), 2), "52W Low": round(float(low52.iloc[-1]), 2),
        "% above 52W low": round((close / float(low52.iloc[-1]) - 1) * 100, 1),
        "1 SMA150>EMA220": f1, "2 Close>SMA50": f2, "3 SMA50>SMA150": f3,
        "4 >25% above low": f4, "5 Dipped<EMA220 90d": f5,
        "New 52W High": new_high, "Selected": selected,
        "ENTRY SIGNAL": selected and new_high,
    }


# =====================================================
# STRATEGY 1 — RSI Pullback Inside an Uptrend
# =====================================================
def eval_s1_rsi_pullback(df):
    if len(df) < 210:
        return None
    c, o, v = df["Close"], df["Open"], df["Volume"]
    ema20 = _ema(c, 20); ema50 = _ema(c, 50); ema200 = _ema(c, 200)
    rsi = _rsi(c, 14); vsma = _sma(v, 20)
    close = float(c.iloc[-1])
    uptrend = close > float(ema50.iloc[-1]) and float(ema50.iloc[-1]) > float(ema200.iloc[-1])
    cross = float(rsi.iloc[-1]) > 40 and float(rsi.iloc[-2]) <= 40
    bullish = close > float(o.iloc[-1])
    vol_ok = float(v.iloc[-1]) > float(vsma.iloc[-1])
    selected = uptrend
    entry = uptrend and cross and bullish and vol_ok
    return {
        "Close": round(close, 2), "EMA20": round(float(ema20.iloc[-1]), 2),
        "EMA50": round(float(ema50.iloc[-1]), 2), "EMA200": round(float(ema200.iloc[-1]), 2),
        "RSI(14)": round(float(rsi.iloc[-1]), 1),
        "Vol vs 20d avg": round(float(v.iloc[-1]) / float(vsma.iloc[-1]), 2) if float(vsma.iloc[-1]) else 0,
        "Uptrend (>50>200)": uptrend, "RSI crossed >40": cross,
        "Bullish day": bullish, "Vol>avg": vol_ok,
        "Selected": selected, "ENTRY SIGNAL": entry,
    }


# =====================================================
# STRATEGY 2 — Supertrend (10,3) + ADX Trend Rider
# =====================================================
def eval_s2_supertrend(df):
    if len(df) < 210:
        return None
    c = df["Close"]
    ema50 = _ema(c, 50); ema200 = _ema(c, 200)
    adx = _adx(df, 14)
    st_line, direction = _supertrend(df, 10, 3.0)
    close = float(c.iloc[-1])
    dir_now = int(direction.iloc[-1]); dir_prev = int(direction.iloc[-2])
    flip_up = dir_now == 1 and dir_prev == -1
    adx_ok = float(adx.iloc[-1]) > 25
    trend = float(ema50.iloc[-1]) > float(ema200.iloc[-1])
    selected = trend and adx_ok and dir_now == 1
    entry = trend and adx_ok and flip_up
    return {
        "Close": round(close, 2), "Supertrend": round(float(st_line.iloc[-1]), 2),
        "ST direction": "UP" if dir_now == 1 else "DOWN",
        "ADX(14)": round(float(adx.iloc[-1]), 1),
        "EMA50": round(float(ema50.iloc[-1]), 2), "EMA200": round(float(ema200.iloc[-1]), 2),
        "ADX>25": adx_ok, "EMA50>EMA200": trend, "ST flip up today": flip_up,
        "Selected": selected, "ENTRY SIGNAL": entry,
    }


# =====================================================
# STRATEGY 3 — Bollinger Band Squeeze Breakout + Volume
# =====================================================
def eval_s3_squeeze(df):
    if len(df) < 210:
        return None
    c, v = df["Close"], df["Volume"]
    ema200 = _ema(c, 200); vsma = _sma(v, 20)
    basis, upper, lower = _bbands(c, 20, 2.0)
    bandwidth = (upper - lower) / basis
    close = float(c.iloc[-1])
    rvol = float(v.iloc[-1]) / float(vsma.iloc[-1]) if float(vsma.iloc[-1]) else 0
    breakout = close > float(upper.iloc[-1]) and float(c.iloc[-2]) <= float(upper.iloc[-2])
    vol_spike = rvol > 2
    trend = close > float(ema200.iloc[-1])
    # squeeze = bandwidth at its narrowest over the last ~6 months (120 bars)
    look = bandwidth.tail(120)
    squeeze_now = float(bandwidth.iloc[-1]) <= float(look.min())
    squeeze_recent = bool((bandwidth.tail(10) <= look.min() * 1.05).any())
    selected = trend and squeeze_recent
    entry = breakout and vol_spike and trend
    return {
        "Close": round(close, 2), "Upper BB": round(float(upper.iloc[-1]), 2),
        "Lower BB": round(float(lower.iloc[-1]), 2),
        "Bandwidth %": round(float(bandwidth.iloc[-1]) * 100, 2),
        "RVOL": round(rvol, 2), "EMA200": round(float(ema200.iloc[-1]), 2),
        "In squeeze": squeeze_now, "Squeeze recent": squeeze_recent,
        "Breakout↑": breakout, "Vol>2x": vol_spike, "Above EMA200": trend,
        "Selected": selected, "ENTRY SIGNAL": entry,
    }


# =====================================================
# STRATEGY 4 — Momentum Rotation (cross-sectional; latest-bar screen)
# =====================================================
def eval_momentum(df):
    if len(df) < 210:
        return None
    c = df["Close"]
    sma200 = _sma(c, 200)
    r126 = c / c.shift(126) - 1
    close = float(c.iloc[-1])
    above = close > float(sma200.iloc[-1])
    return {
        "Close": round(close, 2),
        "SMA200": round(float(sma200.iloc[-1]), 2),
        "6M Return %": round(float(r126.iloc[-1]) * 100, 1),
        "Above 200DMA": above,
        "Selected": above,        # eligible; ENTRY (top-N) is ranked cross-sectionally
        "ENTRY SIGNAL": False,
    }


# =====================================================
# STRATEGY 5 — VM RSI 40 Support (MONTHLY bounce off the RSI-40 floor)
# =====================================================
RSI_FLOOR_LO, RSI_FLOOR_HI = 38, 42   # support band on the SUPPORT month
RSI_BULL_PRIOR = 65                   # RSI must have tagged ~70+ earlier in the up-leg


def eval_vm_rsi40(df):
    """Monthly screen: a stock bouncing off its RSI-40 floor inside a bull range."""
    if len(df) < 400:
        return None
    m = _monthly(df)
    if len(m) < 16:
        return None
    mr = _rsi(m["Close"], 14)
    msma = _sma(m["Close"], 20)
    prev = mr.shift(1)
    prior_peak = mr.shift(1).rolling(12).max()   # highest monthly RSI in the prior 12 months

    cur = float(mr.iloc[-1])
    pr = float(prev.iloc[-1]) if pd.notna(prev.iloc[-1]) else float("nan")
    close = float(m["Close"].iloc[-1]); op = float(m["Open"].iloc[-1])
    sma = float(msma.iloc[-1]) if pd.notna(msma.iloc[-1]) else float("nan")
    peak = float(prior_peak.iloc[-1]) if pd.notna(prior_peak.iloc[-1]) else float("nan")

    above_sma = pd.notna(sma) and close > sma
    bull_ok = pd.notna(peak) and peak >= RSI_BULL_PRIOR
    at_floor = (RSI_FLOOR_LO <= cur <= RSI_FLOOR_HI) and above_sma           # watchlist now
    green = close > op
    entry = (pd.notna(pr) and RSI_FLOOR_LO <= pr <= RSI_FLOOR_HI
             and cur > 40 and cur > pr and green and above_sma and bull_ok)  # confirmed bounce
    return {
        "Close": round(close, 2),
        "Monthly RSI": round(cur, 1),
        "Prev-Mo RSI": round(pr, 1) if pd.notna(pr) else None,
        "Monthly SMA20": round(sma, 2) if pd.notna(sma) else None,
        "Prior peak RSI (12m)": round(peak, 1) if pd.notna(peak) else None,
        "Above mthly SMA20": above_sma,
        "Bull range (≥65 prior)": bull_ok,
        "Green month": green,
        "At floor 38-42": bool(RSI_FLOOR_LO <= cur <= RSI_FLOOR_HI),
        "Selected": bool(at_floor),
        "ENTRY SIGNAL": bool(entry),
    }


def sig_vm_rsi40(df):
    """Full-series MONTHLY entry/exit, mapped onto the LAST trading day of each month
    (signal on the monthly close → engine executes at the next session's open)."""
    d = df
    idx = pd.to_datetime(d["Date"]) if "Date" in d.columns else pd.to_datetime(d.index)
    s = pd.DataFrame({"Open": d["Open"].values, "High": d["High"].values,
                      "Low": d["Low"].values, "Close": d["Close"].values},
                     index=pd.DatetimeIndex(idx))
    m = _monthly(s)
    mr = _rsi(m["Close"], 14); msma = _sma(m["Close"], 20); prev = mr.shift(1)
    prior_peak = mr.shift(1).rolling(12).max()

    entry_m = ((prev >= RSI_FLOOR_LO) & (prev <= RSI_FLOOR_HI) & (mr > 40) & (mr > prev)
               & (m["Close"] > m["Open"]) & (m["Close"] > msma)
               & (prior_peak >= RSI_BULL_PRIOR)).fillna(False)
    # invalidation: a decisive monthly close below ~37 RSI, OR two straight closes under 38
    exit_m = ((mr < 37) | ((mr < 38) & (mr.shift(1) < 38))).fillna(False)

    em = entry_m.copy(); xm = exit_m.copy()
    em.index = m.index.to_period("M"); xm.index = m.index.to_period("M")

    dper = pd.PeriodIndex(pd.DatetimeIndex(idx), freq="M")
    pos = pd.Series(np.arange(len(idx)), index=dper)
    last_pos = pos.groupby(level=0).max()          # period -> last daily row of that month

    entry = np.zeros(len(idx), dtype=bool)
    exit_ = np.zeros(len(idx), dtype=bool)
    for per, p in last_pos.items():
        if per in em.index and bool(em.loc[per]):
            entry[p] = True
        if per in xm.index and bool(xm.loc[per]):
            exit_[p] = True
    return pd.DataFrame({"entry": entry, "exit": exit_})


# =====================================================
# FULL-SERIES SIGNALS (for the backtester)
# Each returns a DataFrame with boolean 'entry' and 'exit' columns aligned to df.
# Signals are computed on the close; the engine executes on the next open.
# 'exit' here is the price-independent rule; an entry-price stop (stop_pct in the
# registry) is applied by the engine on top of it.
# =====================================================
def sig_onepct(df):
    c = df["Close"]
    sma50 = _sma(c, 50); sma150 = _sma(c, 150); ema220 = _ema(c, 220)
    high52 = c.rolling(WEEK_52).max(); low52 = df["Low"].rolling(WEEK_52).min()
    dipped = (df["Low"] < ema220).rolling(90).max().fillna(0).astype(bool)
    sel = (sma150 > ema220) & (c > sma50) & (sma50 > sma150) & (c > 1.25 * low52) & dipped
    entry = (sel & (c >= high52)).fillna(False)
    exit_ = (c < ema220).fillna(False)
    return pd.DataFrame({"entry": entry, "exit": exit_})


def sig_s1_rsi_pullback(df):
    c = df["Close"]; o = df["Open"]; v = df["Volume"]
    ema50 = _ema(c, 50); ema200 = _ema(c, 200); rsi = _rsi(c, 14); vsma = _sma(v, 20)
    uptrend = (c > ema50) & (ema50 > ema200)
    cross = (rsi > 40) & (rsi.shift(1) <= 40)
    entry = (uptrend & cross & (c > o) & (v > vsma)).fillna(False)
    exit_ = ((rsi > 70) | (c < ema50)).fillna(False)
    return pd.DataFrame({"entry": entry, "exit": exit_})


def sig_s2_supertrend(df):
    c = df["Close"]; ema50 = _ema(c, 50); ema200 = _ema(c, 200); adx = _adx(df, 14)
    st_line, direction = _supertrend(df, 10, 3.0)
    flip_up = (direction == 1) & (direction.shift(1) == -1)
    entry = ((ema50 > ema200) & (adx > 25) & flip_up).fillna(False)
    exit_ = (c < st_line).fillna(False)
    return pd.DataFrame({"entry": entry, "exit": exit_})


def sig_s3_squeeze(df):
    c = df["Close"]; v = df["Volume"]; ema200 = _ema(c, 200); vsma = _sma(v, 20)
    basis, upper, lower = _bbands(c, 20, 2.0)
    breakout = (c > upper) & (c.shift(1) <= upper.shift(1))
    entry = (breakout & (v > 2 * vsma) & (c > ema200)).fillna(False)
    exit_ = (c < basis).fillna(False)
    return pd.DataFrame({"entry": entry, "exit": exit_})


# =====================================================
# REGISTRY
# =====================================================
_EMA = "MAExp@tv-basicstudies"
_SMA = "MASimple@tv-basicstudies"
_RSI = "RSI@tv-basicstudies"
_BB = "BB@tv-basicstudies"
_ST = "Supertrend@tv-basicstudies"
_ADX = "ADX@tv-basicstudies"

STRATEGIES = {
    "onepct_breakout": {
        "key": "onepct_breakout", "name": "Strong Stock Breakout", "author": "1% Club",
        "min_bars": WEEK_52 + 1, "sort_col": "% above 52W low", "sort_asc": False,
        "evaluate": eval_onepct, "signals": sig_onepct, "stop_pct": 0.15,
        "tv_studies": [{"id": _SMA, "inputs": {"length": 50}},
                       {"id": _SMA, "inputs": {"length": 150}},
                       {"id": _EMA, "inputs": {"length": 220}}],
        "description": (
            "**Selection (all true):** 1) SMA150>EMA220  2) Close>SMA50  3) SMA50>SMA150  "
            "4) Close>1.25×52w-low  5) Low dipped below EMA220 in last 90d.\n\n"
            "**Entry:** new 52-week high on a closing basis. (This is the backtested strategy.)"
        ),
    },
    "s1_rsi_pullback": {
        "key": "s1_rsi_pullback", "name": "RSI Pullback in Uptrend", "author": "Playbook",
        "min_bars": 210, "sort_col": "RSI(14)", "sort_asc": True,
        "evaluate": eval_s1_rsi_pullback, "signals": sig_s1_rsi_pullback, "stop_pct": 0.10,
        "tv_studies": [{"id": _EMA, "inputs": {"length": 50}},
                       {"id": _EMA, "inputs": {"length": 200}},
                       {"id": _RSI, "inputs": {"length": 14}}],
        "description": (
            "Buy a dip inside a confirmed uptrend.\n\n"
            "**Watchlist (Selected):** Close>EMA50 and EMA50>EMA200.\n"
            "**Entry trigger:** RSI(14) crosses back above 40 + bullish candle + volume>20d avg.\n"
            "**Stop:** below swing low / 50 EMA. **Exit:** RSI>70 or close<EMA50. Hold: days–weeks."
        ),
    },
    "s3_squeeze": {
        "key": "s3_squeeze", "name": "Bollinger Squeeze Breakout", "author": "Playbook",
        "min_bars": 210, "sort_col": "RVOL", "sort_asc": False,
        "evaluate": eval_s3_squeeze, "signals": sig_s3_squeeze, "stop_pct": 0.12,
        "tv_studies": [{"id": _BB, "inputs": {"length": 20}},
                       {"id": _EMA, "inputs": {"length": 200}}],
        "description": (
            "Catch the move as a quiet stock expands out of a squeeze.\n\n"
            "**Watchlist (Selected):** above EMA200 and recently in a 6-month-tight squeeze.\n"
            "**Entry trigger:** close above the upper band with RVOL>2 (volume ≥ 2× 20d avg).\n"
            "**Stop:** breakout candle low / back inside the 20 SMA midline. Hold: days–weeks.\n"
            "_No volume = fake-out; skip._"
        ),
    },
    "vm_rsi40_support": {
        "key": "vm_rsi40_support", "name": "VM RSI 40 Support", "author": "VM",
        "min_bars": 400, "sort_col": "Monthly RSI", "sort_asc": True,
        "evaluate": eval_vm_rsi40, "signals": sig_vm_rsi40, "stop_pct": 0.15,
        "tv_studies": [{"id": _SMA, "inputs": {"length": 20}},
                       {"id": _RSI, "inputs": {"length": 14}}],
        "description": (
            "**Monthly** strategy — buy the bounce off the RSI-40 floor inside a bull range.\n\n"
            "**Watchlist (Selected):** monthly RSI(14) sitting in the **38–42** band AND "
            "monthly close > monthly 20-SMA (stocks testing the floor right now).\n"
            "**Entry trigger:** the *support* month (last month) had RSI 38–42, the current "
            "month closes with RSI back **above 40 and rising**, a **green** candle, still "
            "above the monthly 20-SMA — and RSI had tagged **≥65** earlier in the up-leg "
            "(so 40 is genuine support, not a falling knife).\n"
            "**Invalidation / exit:** a decisive monthly close with RSI **below ~37**, or "
            "**two straight** monthly closes under 38 → the floor broke; skip/exit. "
            "A 15% hard stop from entry is the backstop.\n"
            "_38 is the edge of tolerance; the healthy centre of gravity is 39–41._"
        ),
    },
}

ORDER = ["onepct_breakout", "s1_rsi_pullback", "s3_squeeze", "vm_rsi40_support"]
DEFAULT_KEY = "onepct_breakout"


def label(key):
    s = STRATEGIES[key]
    return f"{s['name']} — {s['author']}"


def choices():
    return [(k, label(k)) for k in ORDER]


# =====================================================
# POSITION ADVICE — HOLD / SELL + protective stop level per strategy
# =====================================================
def position_advice(key, df, entry_price):
    """For an existing holding, return {Close, Action(HOLD/SELL), Stop level, Reason}
    using the selected strategy's exit rule + its natural protective stop."""
    if df is None or len(df) < 60:
        return None
    strat = STRATEGIES[key]
    c = df["Close"]
    close = float(c.iloc[-1])
    entry = float(entry_price) if entry_price else 0.0

    exit_now, reason = False, ""
    sig = strat.get("signals")
    if sig is not None:
        try:
            exit_now = bool(sig(df.reset_index())["exit"].iloc[-1])
            if exit_now:
                reason = "strategy exit rule triggered"
        except Exception:
            pass
    else:  # rotation
        if close < float(_sma(c, 200).iloc[-1]):
            exit_now, reason = True, "below 200 DMA (drops out of momentum set)"

    # hard entry-based stop (strategies that define one)
    sp = strat.get("stop_pct")
    hard = entry * (1 - sp) if (sp and entry) else None
    if hard and close <= hard:
        exit_now, reason = True, f"hit {int(sp*100)}% stop from your buy price"

    # protective stop LEVEL (the price to exit at) by strategy
    stop = None
    if key in ("onepct_breakout", "onepct_trailing"):
        ema220 = float(_ema(c, 220).iloc[-1])
        cands = [ema220] + ([hard] if hard else [])
        if key == "onepct_trailing":
            atr = float(_atr(df, 22).iloc[-1])
            peak = float(df["High"].tail(22).max())
            cands.append(peak - 3 * atr)            # chandelier trail
        stop = max(cands)
        reason = reason or ("below 220 EMA" if close < ema220 else "trail / 220-EMA stop")
    elif key == "s1_rsi_pullback":
        stop = float(_ema(c, 50).iloc[-1])
        reason = reason or "exit on close < 50 EMA (or RSI > 70)"
    elif key == "s2_supertrend":
        st_line, _ = _supertrend(df, 10, 3.0)
        stop = float(st_line.iloc[-1])
        reason = reason or "trail the Supertrend line"
    elif key == "s3_squeeze":
        basis, _, _ = _bbands(c, 20, 2.0)
        stop = float(basis.iloc[-1])
        reason = reason or "exit on close < 20-SMA midline"
    elif key == "momentum_rotation":
        stop = float(_sma(c, 200).iloc[-1])
        reason = reason or "hold while above 200 DMA & in top-N"
    elif key == "vm_rsi40_support":
        try:
            m = _monthly(df if isinstance(df.index, pd.DatetimeIndex)
                         else df.set_index(pd.to_datetime(df["Date"])))
            stop = float(_sma(m["Close"], 20).iloc[-1])     # monthly 20-SMA = floor
        except Exception:
            stop = hard
        reason = reason or "exit if monthly RSI<37 (or 2 closes <38) / below monthly 20-SMA"

    return {
        "Close": round(close, 2),
        "Action": "SELL" if exit_now else "HOLD",
        "Stop level": round(stop, 2) if stop is not None else None,
        "Reason": reason,
    }
