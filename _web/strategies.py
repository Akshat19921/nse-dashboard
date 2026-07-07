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


def _macd_signal(close, fast=12, slow=26, sig=9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return line.ewm(span=sig, adjust=False).mean()


def _heikin(df):
    """Heikin-Ashi (ha_open, ha_close) for a frame indexed in time order."""
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    o = np.empty(len(df))
    if len(df) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    o[0] = (float(df["Open"].iloc[0]) + float(df["Close"].iloc[0])) / 2.0
    hc = ha_close.values
    for i in range(1, len(o)):
        o[i] = (o[i - 1] + hc[i - 1]) / 2.0
    return pd.Series(o, index=df.index), ha_close


def _weekly_ha_bull(df):
    """Daily-aligned boolean: was the last COMPLETED weekly Heikin-Ashi candle bullish?"""
    w = pd.DataFrame({"Open": df["Open"].resample("W").first(),
                      "High": df["High"].resample("W").max(),
                      "Low":  df["Low"].resample("W").min(),
                      "Close": df["Close"].resample("W").last()}).dropna()
    if len(w) < 2:
        return pd.Series(False, index=df.index)
    ho, hc = _heikin(w)
    bull = (hc > ho).shift(1).fillna(False)          # prior completed week (no lookahead)
    return bull.reindex(df.index, method="ffill").fillna(False)


def _as_dt(df):
    """Return df indexed by a DatetimeIndex (handles a 'Date' column or existing index)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    idx = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.to_datetime(df.index)
    return df.set_index(pd.DatetimeIndex(idx))


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
# STRATEGY 6 — 200-DMA Pullback (buy dips to a rising 200-DMA in an uptrend)
# =====================================================
def eval_dma_pullback(df):
    if len(df) < 210:
        return None
    c = df["Close"]
    s50 = _sma(c, 50); s200 = _sma(c, 200); rsi = _rsi(c, 14)
    close = float(c.iloc[-1]); sma50 = float(s50.iloc[-1]); sma200 = float(s200.iloc[-1])
    r = float(rsi.iloc[-1])
    rising = pd.notna(s200.iloc[-21]) and sma200 > float(s200.iloc[-21])
    above200 = close > sma200
    near50 = close <= sma50 * 1.02
    cross = r > 40 and float(rsi.iloc[-2]) <= 40
    selected = above200 and rising
    entry = selected and near50 and cross
    return {
        "Close": round(close, 2), "SMA50": round(sma50, 2), "SMA200": round(sma200, 2),
        "RSI(14)": round(r, 1), "% above 200DMA": round((close / sma200 - 1) * 100, 1),
        "200DMA rising": rising, "Above 200DMA": above200, "Near 50DMA": near50,
        "RSI crossed >40": cross, "Selected": selected, "ENTRY SIGNAL": entry,
    }


def sig_dma_pullback(df):
    c = df["Close"]; s50 = _sma(c, 50); s200 = _sma(c, 200); rsi = _rsi(c, 14)
    rising = s200 > s200.shift(20)
    near = (c > s200) & (c <= s50 * 1.02) & rising
    recover = (rsi > 40) & (rsi.shift(1) <= 40)
    entry = (near & recover).fillna(False)
    exit_ = (c < s200).fillna(False)
    return pd.DataFrame({"entry": entry, "exit": exit_})


# =====================================================
# STRATEGY 7 — Minervini Trend Template (SEPA) + 20-day-high breakout
# =====================================================
def eval_minervini(df):
    if len(df) < WEEK_52 + 1:
        return None
    c = df["Close"]
    s50 = _sma(c, 50); s150 = _sma(c, 150); s200 = _sma(c, 200)
    hi52 = c.rolling(WEEK_52).max(); lo52 = df["Low"].rolling(WEEK_52).min()
    close = float(c.iloc[-1])
    sma50 = float(s50.iloc[-1]); sma150 = float(s150.iloc[-1]); sma200 = float(s200.iloc[-1])
    lo = float(lo52.iloc[-1]); hi = float(hi52.iloc[-1])
    rising = pd.notna(s200.iloc[-21]) and sma200 > float(s200.iloc[-21])
    trend = close > sma50 and sma50 > sma150 and sma150 > sma200 and rising
    rng = close >= 1.30 * lo and close >= 0.75 * hi
    hi20 = float(c.rolling(20).max().iloc[-1])
    new_high = close >= hi20
    selected = trend and rng
    entry = selected and new_high
    return {
        "Close": round(close, 2), "SMA50": round(sma50, 2), "SMA150": round(sma150, 2),
        "SMA200": round(sma200, 2), "52W High": round(hi, 2), "52W Low": round(lo, 2),
        "% above 52W low": round((close / lo - 1) * 100, 1),
        "% below 52W high": round((close / hi - 1) * 100, 1),
        "Trend stack (50>150>200↑)": trend, "In 25% range": rng, "New 20D high": new_high,
        "Selected": selected, "ENTRY SIGNAL": entry,
    }


def sig_minervini(df):
    c = df["Close"]; s50 = _sma(c, 50); s150 = _sma(c, 150); s200 = _sma(c, 200)
    hi52 = c.rolling(WEEK_52).max(); lo52 = df["Low"].rolling(WEEK_52).min()
    trend = (c > s50) & (s50 > s150) & (s150 > s200) & (s200 > s200.shift(20))
    rng = (c >= 1.30 * lo52) & (c >= 0.75 * hi52)
    breakout = c >= c.rolling(20).max()
    entry = (trend & rng & breakout).fillna(False)
    exit_ = (c < s150).fillna(False)
    return pd.DataFrame({"entry": entry, "exit": exit_})


# =====================================================
# STRATEGY 8 & 9 — KISS / "Swing Systematic" (trend + reversal LONG)
#   55-EMA(High/Low) band · MACD(12,26,9) signal vs zero · Heikin-Ashi colour
#   (daily timeframe; weekly Heikin-Ashi = trend). Dhan Signal MA dropped.
# =====================================================
def _kiss_frames(df):
    d = _as_dt(df)
    c = d["Close"]
    ema_hi = d["High"].ewm(span=55, adjust=False).mean()
    ema_lo = d["Low"].ewm(span=55, adjust=False).mean()
    sig = _macd_signal(c)
    ha_o, ha_c = _heikin(d)
    wbull = _weekly_ha_bull(d)
    return d, c, ema_hi, ema_lo, sig, (ha_c > ha_o), wbull


def eval_kiss_trend(df):
    if len(df) < 120:
        return None
    d, c, ema_hi, ema_lo, sig, ha_bull, wbull = _kiss_frames(df)
    close = float(c.iloc[-1]); ehi = float(ema_hi.iloc[-1]); elo = float(ema_lo.iloc[-1])
    s0 = float(sig.iloc[-1]); s1 = float(sig.iloc[-2]); hb = bool(ha_bull.iloc[-1]); wb = bool(wbull.iloc[-1])
    above = close > ehi
    selected = above and s0 > 0 and hb and wb
    entry = (s0 > 0 and s1 <= 0) and above and hb and wb
    return {
        "Close": round(close, 2), "55EMA Hi": round(ehi, 2), "55EMA Lo": round(elo, 2),
        "MACD signal": round(s0, 2), "Above band": above, "MACD>0": s0 > 0,
        "HA bullish": hb, "Weekly up": wb, "Selected": selected, "ENTRY SIGNAL": bool(entry),
    }


def sig_kiss_trend(df):
    d, c, ema_hi, ema_lo, sig, ha_bull, wbull = _kiss_frames(df)
    above = c > ema_hi
    cross_up = (sig > 0) & (sig.shift(1) <= 0)
    entry = (cross_up & above & ha_bull & wbull).fillna(False)
    exit_ = (((sig < 0) & (sig.shift(1) >= 0)) | (c < ema_lo)).fillna(False)
    return pd.DataFrame({"entry": entry.values, "exit": exit_.values})


def eval_kiss_reversal(df):
    if len(df) < 120:
        return None
    d, c, ema_hi, ema_lo, sig, ha_bull, wbull = _kiss_frames(df)
    close = float(c.iloc[-1]); elo = float(ema_lo.iloc[-1])
    s0 = float(sig.iloc[-1]); s1 = float(sig.iloc[-2]); hb = bool(ha_bull.iloc[-1])
    near = elo * 0.95 <= close <= elo * 1.03
    turn_up = s0 < 0 and s0 > s1
    selected = (s0 < 0) and near
    entry = turn_up and near and hb
    return {
        "Close": round(close, 2), "55EMA Lo": round(elo, 2), "MACD signal": round(s0, 2),
        "Near lower band": near, "MACD turning up (<0)": turn_up, "HA bullish": hb,
        "Selected": selected, "ENTRY SIGNAL": bool(entry),
    }


def sig_kiss_reversal(df):
    d, c, ema_hi, ema_lo, sig, ha_bull, wbull = _kiss_frames(df)
    near = (c >= ema_lo * 0.95) & (c <= ema_lo * 1.03)
    turn_up = (sig < 0) & (sig > sig.shift(1))
    entry = (turn_up & near & ha_bull).fillna(False)
    exit_ = (((sig < 0) & (sig.shift(1) >= 0)) | (c < ema_lo * 0.95)).fillna(False)
    return pd.DataFrame({"entry": entry.values, "exit": exit_.values})


# =====================================================
# STRATEGY 10-12 — short SWING systems (1-2 week hold)
# =====================================================
def _willr(df, n=10):
    hh = df["High"].rolling(n).max(); ll = df["Low"].rolling(n).min()
    return -100 * (hh - df["Close"]) / (hh - ll).replace(0, np.nan)


def eval_bb_reversion(df):
    if len(df) < 210:
        return None
    c = df["Close"]; s200 = _sma(c, 200); basis, upper, lower = _bbands(c, 20, 2.0)
    close = float(c.iloc[-1]); lo = float(lower.iloc[-1]); mid = float(basis.iloc[-1])
    up = close > float(s200.iloc[-1])
    below = close < lo
    return {"Close": round(close, 2), "Lower BB": round(lo, 2), "Mid (20SMA)": round(mid, 2),
            "200SMA": round(float(s200.iloc[-1]), 2), "Uptrend": up, "Below lower band": below,
            "Selected": up, "ENTRY SIGNAL": bool(up and below)}


def sig_bb_reversion(df):
    c = df["Close"]; s200 = _sma(c, 200); basis, upper, lower = _bbands(c, 20, 2.0)
    entry = ((c > s200) & (c < lower)).fillna(False)
    exit_ = (c >= basis).fillna(False)
    return pd.DataFrame({"entry": entry, "exit": exit_})


def eval_willr(df):
    if len(df) < 210:
        return None
    c = df["Close"]; s200 = _sma(c, 200); wr = _willr(df, 10)
    close = float(c.iloc[-1]); w = float(wr.iloc[-1]); up = close > float(s200.iloc[-1])
    return {"Close": round(close, 2), "Williams %R": round(w, 1),
            "200SMA": round(float(s200.iloc[-1]), 2), "Uptrend": up, "Oversold (<-90)": w < -90,
            "Selected": up, "ENTRY SIGNAL": bool(up and w < -90)}


def sig_willr(df):
    c = df["Close"]; s200 = _sma(c, 200); wr = _willr(df, 10)
    entry = ((c > s200) & (wr < -90)).fillna(False)
    exit_ = (wr > -30).fillna(False)
    return pd.DataFrame({"entry": entry, "exit": exit_})


def eval_mom20(df):
    if len(df) < 210:
        return None
    c = df["Close"]; s50 = _sma(c, 50); hi20 = c.rolling(20).max()
    v = df["Volume"] if "Volume" in df.columns else None
    vavg = _sma(v, 20) if v is not None else None
    close = float(c.iloc[-1]); newhi = close >= float(hi20.iloc[-1])
    trend = close > float(s50.iloc[-1])
    vol_ok = bool(v is not None and float(v.iloc[-1]) > float(vavg.iloc[-1])) if v is not None else True
    return {"Close": round(close, 2), "20D High": round(float(hi20.iloc[-1]), 2),
            "50SMA": round(float(s50.iloc[-1]), 2), "New 20D high": newhi,
            "Above 50SMA": trend, "Vol>avg": vol_ok,
            "Selected": trend, "ENTRY SIGNAL": bool(newhi and trend and vol_ok)}


def sig_mom20(df):
    c = df["Close"]; s50 = _sma(c, 50); hi20 = c.rolling(20).max()
    if "Volume" in df.columns:
        v = df["Volume"]; vol_ok = v > _sma(v, 20)
    else:
        vol_ok = pd.Series(True, index=c.index)
    entry = ((c >= hi20) & (c > s50) & vol_ok).fillna(False)
    exit_ = (c < s50).fillna(False)
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
    "dma_pullback": {
        "key": "dma_pullback", "name": "200-DMA Pullback", "author": "Trend Pullback",
        "min_bars": 210, "sort_col": "RSI(14)", "sort_asc": True,
        "evaluate": eval_dma_pullback, "signals": sig_dma_pullback, "stop_pct": 0.10,
        "tv_studies": [{"id": _SMA, "inputs": {"length": 50}},
                       {"id": _SMA, "inputs": {"length": 200}},
                       {"id": _RSI, "inputs": {"length": 14}}],
        "description": (
            "Buy a healthy **dip to a rising 200-DMA** inside a long-term uptrend, then "
            "hold for the next leg (position trade, weeks–months).\n\n"
            "**Watchlist (Selected):** Close above the 200-DMA **and** the 200-DMA is rising "
            "(higher than 20 days ago) — i.e. a genuine uptrend.\n"
            "**Entry trigger:** price has pulled back near the 50-DMA (≤2% above it) and "
            "**RSI(14) crosses back above 40** (momentum turning up from the dip).\n"
            "**Exit / stop:** close **below the 200-DMA** (trend broken), or a 10% hard stop "
            "from entry. Backtest: +140% / 5y, +11% last yr, CAGR 19%, Sharpe 1.41 on LMC250."
        ),
    },
    "minervini": {
        "key": "minervini", "name": "Minervini Trend Template", "author": "Minervini (SEPA)",
        "min_bars": WEEK_52 + 1, "sort_col": "% above 52W low", "sort_asc": False,
        "evaluate": eval_minervini, "signals": sig_minervini, "stop_pct": 0.12,
        "tv_studies": [{"id": _SMA, "inputs": {"length": 50}},
                       {"id": _SMA, "inputs": {"length": 150}},
                       {"id": _SMA, "inputs": {"length": 200}}],
        "description": (
            "Mark Minervini's **Trend Template** (from *Trade Like a Stock Market Wizard*): "
            "own only true market leaders in a confirmed Stage-2 uptrend. Long holds (months+).\n\n"
            "**Watchlist (Selected) — the template, all true:** Close>50-DMA, 50>150>200-DMA, "
            "200-DMA rising, price ≥30% above its 52-week low and within 25% of its 52-week high.\n"
            "**Entry trigger:** while the template holds, a **new 20-day high** (fresh breakout).\n"
            "**Exit / stop:** close below the **150-DMA (30-week)**, or a 12% hard stop from entry.\n"
            "Backtest: +183% / 5y, CAGR 23%, Sharpe 1.68 on LMC250 — best long-horizon compounder "
            "(last-yr +5.8% only because mid-caps were broadly soft that window)."
        ),
    },
    "kiss_trend": {
        "key": "kiss_trend", "name": "KISS Swing — Trend LONG", "author": "Swing Systematic",
        "min_bars": 120, "sort_col": "MACD signal", "sort_asc": False,
        "evaluate": eval_kiss_trend, "signals": sig_kiss_trend, "stop_pct": 0.06,
        "tv_studies": [{"id": _EMA, "inputs": {"length": 55}},
                       {"id": "MACD@tv-basicstudies"}],
        "description": (
            "Trend-following swing (the masterclass's main engine), **daily approximation** "
            "of his 1H/4H + weekly setup — equity only.\n\n"
            "**Watchlist (Selected):** price **above** the 55-EMA(High/Low) band, MACD(12,26,9) "
            "signal **above zero**, daily Heikin-Ashi **green**, and the weekly Heikin-Ashi **up**.\n"
            "**Entry trigger:** the fresh event — MACD signal **crosses above zero** while all the "
            "above hold (confirmation, not FOMO).\n"
            "**Exit / stop:** MACD signal crosses back **below zero**, or close **below the lower "
            "band**; 6% protective stop. He targets 1:3 RR and a weekly-close time exit.\n"
            "_Trade-level 5y test (LMC250, daily proxy): ~1,032 trades, 48% win, profit-factor 1.31._\n"
            "_Note: real version uses 1H/4H candles we don't have — treat as an approximation._"
        ),
    },
    "kiss_reversal": {
        "key": "kiss_reversal", "name": "KISS Swing — Reversal LONG", "author": "Swing Systematic",
        "min_bars": 120, "sort_col": "MACD signal", "sort_asc": True,
        "evaluate": eval_kiss_reversal, "signals": sig_kiss_reversal, "stop_pct": 0.06,
        "tv_studies": [{"id": _EMA, "inputs": {"length": 55}},
                       {"id": "MACD@tv-basicstudies"}],
        "description": (
            "Counter-trend reversal swing (his ~30% allocation, smaller size). **Daily proxy**, "
            "equity only.\n\n"
            "**Watchlist (Selected):** MACD signal still **below zero** and price pushing into the "
            "**lower-band** area (the turning point). Trend filters are intentionally ignored here.\n"
            "**Entry trigger:** MACD signal **turns up while below zero** near the lower band with a "
            "green Heikin-Ashi candle — deploy partial size, add as MACD clears zero.\n"
            "**Exit / stop:** MACD signal crosses below zero, or close below the lower band; 6% stop. "
            "1:3 RR target.\n"
            "_Trade-level 5y test (LMC250, daily proxy): ~5,742 trades, 43% win, profit-factor 1.11 "
            "(high-frequency, thin edge — the riskier book)._"
        ),
    },
    "bb_reversion": {
        "key": "bb_reversion", "name": "Bollinger Mean-Reversion", "author": "Connors/Bollinger",
        "min_bars": 210, "sort_col": "Close", "sort_asc": True,
        "evaluate": eval_bb_reversion, "signals": sig_bb_reversion, "stop_pct": 0.05,
        "tv_studies": [{"id": _BB, "inputs": {"length": 20}}, {"id": _SMA, "inputs": {"length": 200}}],
        "description": (
            "Short-swing mean-reversion (~5-day hold). Buy panic dips in an uptrend, sell the "
            "snap-back to the mean.\n\n"
            "**Watchlist (Selected):** Close above the 200-SMA (uptrend intact).\n"
            "**Entry trigger:** Close drops **below the lower Bollinger Band** (20, 2σ).\n"
            "**Target:** the **20-SMA / mid-band** (or +6%). **Stop:** −5%. Max ~10 days.\n"
            "_5y test (LMC250, after 0.25% costs): ~2,358 trades, 54% win, ~5-day hold, "
            "expectancy +0.20%/trade, PF 1.10 — the only short-swing system that stayed net-positive._"
        ),
    },
    "willr": {
        "key": "willr", "name": "Williams %R Bounce", "author": "Larry Williams",
        "min_bars": 210, "sort_col": "Williams %R", "sort_asc": True,
        "evaluate": eval_willr, "signals": sig_willr, "stop_pct": 0.04,
        "tv_studies": [{"id": _SMA, "inputs": {"length": 200}}],
        "description": (
            "Short-swing oversold bounce inside an uptrend (~4-day hold).\n\n"
            "**Watchlist (Selected):** Close above the 200-SMA.\n"
            "**Entry trigger:** **Williams %R(10) < −90** (deeply oversold).\n"
            "**Target:** %R recovers **above −30** (or +6%). **Stop:** −4%. Max ~10 days.\n"
            "_5y test (after costs): ~5,480 trades, 48% win, ~4-day hold, expectancy ≈ break-even "
            "(PF 0.97) — marginal; include for comparison._"
        ),
    },
    "mom20": {
        "key": "mom20", "name": "20-Day-High Momentum Swing", "author": "Breakout (ORB/Darvas)",
        "min_bars": 210, "sort_col": "Close", "sort_asc": False,
        "evaluate": eval_mom20, "signals": sig_mom20, "stop_pct": 0.04,
        "tv_studies": [{"id": _SMA, "inputs": {"length": 50}}, {"id": _SMA, "inputs": {"length": 200}}],
        "description": (
            "Short momentum-breakout swing (~5-day hold).\n\n"
            "**Watchlist (Selected):** Close above the 50-SMA.\n"
            "**Entry trigger:** new **20-day high** with above-average volume.\n"
            "**Target:** **+8%**. **Stop:** −4% (also exits on a close below the 50-SMA). Max ~10 days.\n"
            "_5y test (after costs): ~9,680 trades, 39% win, ~5-day hold, expectancy −0.13% (PF 0.94) "
            "— low win-rate breakout; the big winners didn't quite cover the chop in this window._"
        ),
    },
}

ORDER = ["onepct_breakout", "dma_pullback", "minervini", "s3_squeeze",
         "kiss_trend", "bb_reversion", "willr", "mom20"]
DEFAULT_KEY = "onepct_breakout"


def label(key):
    s = STRATEGIES[key]
    return f"{s['name']} — {s['author']}"


def choices():
    return [(k, label(k)) for k in ORDER]


# ---- trading STYLE bracket: Long Term (position, months–years) vs Swing (days–weeks) ----
STYLES = ["Long Term", "Swing"]
STYLE = {
    "onepct_breakout": "Long Term",
    "dma_pullback": "Long Term",
    "minervini": "Long Term",
    "s3_squeeze": "Swing",
    "kiss_trend": "Swing",
    "kiss_reversal": "Swing",
    "bb_reversion": "Swing",
    "willr": "Swing",
    "mom20": "Swing",
}
STYLE_DEFAULT = {"Long Term": "onepct_breakout", "Swing": "kiss_trend"}


def style_of(key):
    return STYLE.get(key, "Long Term")


def choices_by_style(style):
    return [(k, label(k)) for k in ORDER if STYLE.get(k) == style]


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
    elif key == "dma_pullback":
        sma200 = float(_sma(c, 200).iloc[-1])
        stop = max(sma200, hard) if hard else sma200       # exit below the 200-DMA
        reason = reason or ("below 200 DMA — trend broken" if close < sma200
                            else "hold while above the 200 DMA")
    elif key == "minervini":
        sma150 = float(_sma(c, 150).iloc[-1])
        stop = max(sma150, hard) if hard else sma150       # exit below 150-DMA (30-week)
        reason = reason or ("below 150 DMA — Stage-2 broken" if close < sma150
                            else "hold while above the 150 DMA")
    elif key in ("kiss_trend", "kiss_reversal"):
        band_lo = float(df["Low"].ewm(span=55, adjust=False).mean().iloc[-1])
        stop = max(band_lo, hard) if hard else band_lo     # exit below the 55-EMA lower band
        reason = reason or ("MACD signal turned down / below lower band"
                            if close < band_lo else "hold while MACD>0 and above the band")
    elif key == "bb_reversion":
        mid = float(_bbands(c, 20, 2.0)[0].iloc[-1])
        stop = hard
        reason = reason or (f"SELL at the 20-SMA target ({mid:.0f}) or -5% stop")
    elif key == "willr":
        stop = hard
        reason = reason or "SELL when %R recovers above -30, or -4% stop"
    elif key == "mom20":
        sma50 = float(_sma(c, 50).iloc[-1])
        stop = max(sma50, hard) if hard else sma50
        reason = reason or ("below 50-SMA — momentum lost" if close < sma50
                            else "+8% target; trail the 50-SMA, -4% stop")

    return {
        "Close": round(close, 2),
        "Action": "SELL" if exit_now else "HOLD",
        "Stop level": round(stop, 2) if stop is not None else None,
        "Reason": reason,
    }
