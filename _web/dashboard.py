"""
Live Streamlit dashboard for the Nifty 100 "Strong Stock Breakout" screen.

Fetches recent daily data from Yahoo Finance for every Nifty 100 symbol,
computes the strategy's selection + breakout conditions on the latest close,
and shows which stocks currently qualify.

Usage:
    pip install streamlit yfinance pandas numpy
    streamlit run dashboard.py
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import altair as alt

import strategies
import markets
S = strategies  # alias for the indicator helpers (S._ema, S._adx, ...)

try:
    import kite_helper          # optional Zerodha Kite Connect integration
except Exception:
    kite_helper = None

try:
    import portfolio_ocr        # optional screenshot → holdings OCR
except Exception:
    portfolio_ocr = None


@st.cache_resource
def kite_client():
    return kite_helper.get_kite() if kite_helper else None

# =====================================================
# CONFIG  (keep in sync with backtest.py)
# =====================================================
SYMBOLS_CSV = "nifty100_symbols.csv"
SYMBOL_COLUMN = "Symbol"
NAME_COLUMN = "Name"

EXCEL_FILE = "nifty100.xlsx"               # consolidated workbook
SNAPSHOT_SHEET = "Live Snapshot"

RESULTS_FOLDER = "backtest_results"
EQUITY_FILE = os.path.join(RESULTS_FOLDER, "equity_curve.csv")
TRADES_FILE = os.path.join(RESULTS_FOLDER, "trades.csv")

POSITIONS_FILE = "my_positions.csv"   # your saved holdings (persists across refresh)

PERIOD = "2y"      # enough for 52w + 220EMA on the latest close
INTERVAL = "1d"

SMA_FAST = 50
SMA_SLOW = 150
EMA_LONG = 220
WEEK_52 = 252
DIP_LOOKBACK = 90

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="NSE Strategy Screener", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")


def inject_css():
    st.markdown("""
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
      /* hero header */
      .hero { background: linear-gradient(100deg,#1d4ed8 0%,#2563eb 55%,#3b82f6 100%);
              color:#fff; padding:18px 22px; border-radius:14px; margin-bottom:14px;
              box-shadow:0 6px 18px rgba(37,99,235,.18); }
      .hero h1 { color:#fff; font-size:1.55rem; margin:0; font-weight:700; letter-spacing:.2px; }
      .hero p  { color:#e0ecff; margin:.25rem 0 0; font-size:.86rem; }
      /* KPI cards */
      .kpis { display:flex; gap:12px; margin:2px 0 12px; flex-wrap:wrap; }
      .kpi { flex:1; min-width:170px; background:#fff; border:1px solid #e6eaf1;
             border-radius:12px; padding:12px 16px; box-shadow:0 1px 3px rgba(16,24,40,.05); }
      .kpi .lbl { font-size:.72rem; color:#64748b; text-transform:uppercase; letter-spacing:.4px; font-weight:600; }
      .kpi .val { font-size:1.45rem; font-weight:700; margin-top:2px; }
      .kpi .sub { font-size:.78rem; margin-top:1px; }
      .pos{color:#059669;} .neg{color:#dc2626;} .neu{color:#475569;}
      .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; }
      .pill.green{background:#dcfce7;color:#166534;} .pill.red{background:#fee2e2;color:#991b1b;}
      .pill.amber{background:#fef9c3;color:#854d0e;}
      /* tabs */
      .stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid #e6eaf1; }
      .stTabs [data-baseweb="tab"] { padding:8px 14px; border-radius:8px 8px 0 0; font-weight:600; }
      .stTabs [aria-selected="true"] { background:#eff4ff; color:#1d4ed8; }
      /* metric tiles */
      div[data-testid="stMetric"] { background:#fff; border:1px solid #e6eaf1; border-radius:12px;
             padding:12px 14px; box-shadow:0 1px 3px rgba(16,24,40,.05); }
      div[data-testid="stMetricLabel"] p { font-size:.72rem; color:#64748b; font-weight:600;
             text-transform:uppercase; letter-spacing:.3px; }
      /* dataframe header */
      .stDataFrame thead tr th { background:#f1f5f9 !important; font-weight:700 !important; }
      section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e6eaf1; }
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label, value, sub="", tone="neu"):
    return (f'<div class="kpi"><div class="lbl">{label}</div>'
            f'<div class="val {tone}">{value}</div>'
            f'<div class="sub {tone}">{sub}</div></div>')


def load_symbols():
    df = pd.read_csv(SYMBOLS_CSV)
    names = {}
    if NAME_COLUMN in df.columns:
        names = dict(zip(df[SYMBOL_COLUMN], df[NAME_COLUMN]))
    return df[SYMBOL_COLUMN].dropna().astype(str).str.strip().tolist(), names


@st.cache_data(ttl=900)
def fetch(symbol: str) -> pd.DataFrame | None:
    df = yf.download(symbol, period=PERIOD, interval=INTERVAL,
                     auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) < WEEK_52 + 5:
        return None
    return df


@st.cache_data(ttl=1800)
def excel_history():
    """Load the single nifty500.xlsx Price History once → {symbol: OHLCV df}."""
    if not os.path.exists(markets.DATA_XLSX):
        return {}
    raw = pd.read_excel(markets.DATA_XLSX, sheet_name=markets.HISTORY_SHEET, parse_dates=["Date"])
    out = {}
    for sym, g in raw.groupby("Symbol"):
        d = g.drop(columns=["Symbol"]).sort_values("Date").set_index("Date")
        if len(d) >= WEEK_52 + 5:
            out[sym] = d
    return out


def get_hist(symbol, fresh):
    """History for one symbol: live from Yahoo if fresh, else from the workbook."""
    if fresh:
        return fetch(symbol)
    return excel_history().get(symbol)


SECTORS_CSV = "sectors.csv"
SECTOR_LOOKBACK = 126   # ~6 trading months


@st.cache_data(ttl=86400)
def load_sectors():
    """symbol -> sector name (from sectors.csv)."""
    if not os.path.exists(SECTORS_CSV):
        return {}
    df = pd.read_csv(SECTORS_CSV)
    return dict(zip(df["Symbol"].astype(str), df["Sector"].astype(str)))


@st.cache_data(ttl=1800)
def sector_strength():
    """Per-sector 6-month strength from the workbook → {sector: (label, median_ret%)}.
    Bullish ≥ +10%, Bearish < 0%, Neutral in between (median of member 6-mo returns)."""
    hist = excel_history()
    sec_map = load_sectors()
    rets = {}
    for sym, d in hist.items():
        sec = sec_map.get(sym)
        if not sec or len(d) < SECTOR_LOOKBACK + 1:
            continue
        c = d["Close"]
        r = float(c.iloc[-1]) / float(c.iloc[-(SECTOR_LOOKBACK + 1)]) - 1
        rets.setdefault(sec, []).append(r)
    out = {}
    for sec, vals in rets.items():
        med = float(np.median(vals)) * 100
        label = "🟢 Bullish" if med >= 10 else ("🔴 Bearish" if med < 0 else "🟡 Neutral")
        out[sec] = (label, med)
    return out


def _sector_trend_txt(strength, sector):
    lab, ret = strength.get(sector, ("—", None))
    return f"{lab} {ret:+.0f}%" if ret is not None else lab


def _signed_color(v):
    try:
        return "color:#059669;font-weight:600" if float(v) > 0 else (
               "color:#dc2626;font-weight:600" if float(v) < 0 else "")
    except (TypeError, ValueError):
        return ""


def _rsi_color(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if x <= 35:
        return "color:#dc2626;font-weight:600"      # oversold
    if x >= 70:
        return "color:#059669;font-weight:600"      # strong
    return "color:#475569"


def style_screen(df):
    """Pretty, color-coded screener table (pandas Styler)."""
    sty = df.style
    if "ENTRY SIGNAL" in df.columns:
        def _row(row):
            hot = row.get("ENTRY SIGNAL") is True
            return ["background-color:#ecfdf5" if hot else ""] * len(row)
        sty = sty.apply(_row, axis=1)
    if "Sector 6M" in df.columns:
        def _sec(v):
            s = str(v)
            if "Bullish" in s:
                return "color:#059669;font-weight:700"
            if "Bearish" in s:
                return "color:#dc2626;font-weight:700"
            if "Neutral" in s:
                return "color:#b45309;font-weight:600"
            return ""
        sty = sty.applymap(_sec, subset=["Sector 6M"])
    for c in df.columns:
        lc = str(c).lower()
        if df[c].dtype == bool:
            sty = sty.applymap(
                lambda v: "color:#059669;font-weight:700" if v is True else "color:#cbd5e1",
                subset=[c])
        elif "rsi" in lc:
            sty = sty.applymap(_rsi_color, subset=[c])
        elif ("%" in str(c) or "return" in lc) and "above" not in lc:
            sty = sty.applymap(_signed_color, subset=[c])
    return sty.format(precision=2)


@st.cache_data(ttl=900)
def market_regime():
    """Universal regime gate: Nifty 100 vs its 200 DMA + India VIX."""
    out = {}
    for idx_sym in ("^CNX100", "^NSEI"):  # Nifty 100, fall back to Nifty 50
        try:
            d = yf.download(idx_sym, period="2y", interval="1d",
                            progress=False, auto_adjust=False)
            if d is None or d.empty:
                continue
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            c = d["Close"].dropna()
            sma200 = c.rolling(200).mean()
            out["idx_name"] = "Nifty 100" if idx_sym == "^CNX100" else "Nifty 50"
            out["idx_above_200"] = float(c.iloc[-1]) > float(sma200.iloc[-1])
            break
        except Exception:
            continue
    try:
        v = yf.download("^INDIAVIX", period="1mo", interval="1d",
                        progress=False, auto_adjust=False)
        if v is not None and not v.empty:
            if isinstance(v.columns, pd.MultiIndex):
                v.columns = v.columns.get_level_values(0)
            out["vix"] = float(v["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return out


def render_regime():
    reg = market_regime()
    if not reg:
        return
    cards = []
    if "idx_above_200" in reg:
        above = reg["idx_above_200"]
        cards.append(kpi_card(
            f"{reg.get('idx_name','Index')} vs 200 DMA",
            "ABOVE ▲" if above else "BELOW ▼",
            "longs allowed" if above else "reduce / stand aside",
            "pos" if above else "neg"))
    if "vix" in reg:
        v = reg["vix"]
        tone = "pos" if v < 15 else ("neu" if v < 20 else "neg")
        tag = "calm" if v < 15 else ("elevated — widen stops" if v < 20 else "high — halve size")
        cards.append(kpi_card("India VIX", f"{v:.1f}", tag, tone))
    cards.append(kpi_card("Risk per trade", "1–2% cap", "R:R ≥ 1:2", "neu"))
    st.markdown('<div class="kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def check_password():
    """Password gate. Active ONLY when an 'app_password' secret is set (i.e. online).
    Locally, with no secret, the dashboard opens normally."""
    try:
        expected = st.secrets.get("app_password")
    except Exception:
        expected = None
    if not expected:
        return True
    if st.session_state.get("auth_ok"):
        return True
    st.markdown('<div class="hero"><h1>🔒 NSE Strategy Screener</h1>'
                '<p>Enter the password to continue.</p></div>', unsafe_allow_html=True)
    pw = st.text_input("Password", type="password", label_visibility="collapsed",
                       placeholder="Password")
    if pw:
        if pw == expected:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def main():
    inject_css()
    if not check_password():
        st.stop()
    st.markdown(
        '<div class="hero"><h1>📈 NSE Strategy Screener</h1>'
        '<p>Multi-strategy screener &amp; backtester across NSE markets · '
        'Yahoo Finance data (~15-min delayed) · research/education only.</p></div>',
        unsafe_allow_html=True)

    # ---- controls in the sidebar ----
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        mchs = markets.choices()
        mlabels = [l for _, l in mchs]
        mkeys = [k for k, _ in mchs]
        mpick = st.selectbox("Market", mlabels, index=mkeys.index(markets.DEFAULT))
        market = mkeys[mlabels.index(mpick)]

        chs = strategies.choices()
        labels = [lbl for _, lbl in chs]
        keys = [k for k, _ in chs]
        picked = st.selectbox("Strategy", labels, index=keys.index(strategies.DEFAULT_KEY))
        strat = strategies.STRATEGIES[keys[labels.index(picked)]]

        symbols = markets.market_symbols(market)
        names = markets.market_names(market)
        st.caption(f"**{len(symbols)}** stocks in {markets.label(market)}")
        with st.expander(f"📜 Rules — {strat['name']} (by {strat['author']})", expanded=False):
            st.markdown(strat["description"])

    render_regime()

    tab_live, tab_chart, tab_bt, tab_pos, tab_cmp, tab_z = st.tabs(
        ["🔴 Live Screen", "📈 Chart", "📊 Backtest Results", "📋 My Positions",
         "🆚 Compare All", "💼 Zerodha"])
    with tab_live:
        render_live_screen(symbols, names, strat, market)
    with tab_chart:
        render_chart(symbols, names, strat)
    with tab_bt:
        render_backtest(strat, market)
    with tab_pos:
        render_positions(strat, symbols, names)
    with tab_cmp:
        render_compare(market)
    with tab_z:
        render_zerodha(strat, symbols, names)


# =====================================================
# LIVE SCREEN TAB
# =====================================================
@st.cache_data(ttl=900)
def load_snapshot_from_excel():
    return pd.read_excel(EXCEL_FILE, sheet_name=SNAPSHOT_SHEET)


@st.cache_data(ttl=900)
def fetch_session_time():
    """Timestamp of the current cached live pull. Resets when the cache is cleared
    (i.e. when the user clicks Refresh), so it reflects the last live fetch."""
    return datetime.now()


def render_live_screen(symbols, names, strat, market):
    have_xl = os.path.exists(markets.DATA_XLSX)

    colA, colB = st.columns([1, 3])
    with colA:
        view = st.radio("Show", ["Entry signals only", "Passes selection", "All stocks"])
        fresh = st.checkbox("Fetch fresh from Yahoo (slow)", value=not have_xl,
                            help="Off = compute from the saved workbook (fast). "
                                 "On = pull live daily data per stock from Yahoo (slow for big markets).")
        use_live = st.checkbox("Use Zerodha live prices (LTP)", value=False,
                               help="Overlay Kite real-time LTP onto the latest bar before evaluating.")
        run = st.button("🔄 Refresh / recompute", type="primary")
    if run:
        st.cache_data.clear()

    evaluate_fn = strat["evaluate"]
    rows, errors, last_bar = [], [], None

    live = {}
    if use_live and kite_helper is not None:
        kobj = kite_client()
        ok_live, _ = kite_helper.connection_status(kobj)
        if ok_live:
            live = kite_helper.live_ltp(kobj, symbols)
            st.caption(f"Overlaying Zerodha live LTP on {len(live)} symbols.")
        else:
            st.caption("Zerodha not connected — see the 💼 Zerodha tab.")

    if fresh and len(symbols) > 150:
        st.warning(f"Fetching {len(symbols)} symbols live from Yahoo — this can take a few minutes.")

    progress = st.progress(0.0, text="Computing indicators...")
    for i, sym in enumerate(symbols, 1):
        try:
            df = get_hist(sym, fresh)
            if df is None:
                errors.append(sym)
            else:
                if sym in live and live[sym] > 0:
                    df = df.copy()
                    df.iloc[-1, df.columns.get_loc("Close")] = live[sym]
                bar = pd.to_datetime(df.index[-1])
                last_bar = bar if last_bar is None else max(last_bar, bar)
                rv = evaluate_fn(df)
                if rv is None:
                    errors.append(sym)
                else:
                    row = {"Symbol": sym, "Name": names.get(sym, "")}
                    row.update(rv)
                    rows.append(row)
        except Exception:
            errors.append(sym)
        if i % 10 == 0 or i == len(symbols):
            progress.progress(i / len(symbols), text=f"{strat['name']}: {i}/{len(symbols)}")
    progress.empty()
    if not rows:
        st.error("No data. Build the workbook with build_market_excel.py, or tick "
                 "'Fetch fresh from Yahoo'.")
        return
    res = pd.DataFrame(rows)
    if fresh:
        ts = fetch_session_time()
        bar_txt = f" · latest bar {last_bar.date()}" if last_bar is not None else ""
        st.success(f"🕒 Live data fetched at **{ts:%d %b %Y, %I:%M:%S %p}**{bar_txt} "
                   "(Yahoo, ~15-min delayed).")
    else:
        bt = last_bar.date() if last_bar is not None else "?"
        st.caption(f"🗂 From workbook `{markets.DATA_XLSX}` ({markets.label(market)}, "
                   f"{len(res)} stocks) · data to {bt}. Tick 'Fetch fresh from Yahoo' for live.")

    # Sector + 6-month sector strength, shown just before Close.
    sec_map = load_sectors()
    strength = sector_strength()
    res.insert(2, "Sector", res["Symbol"].map(sec_map).fillna("Other"))
    res.insert(3, "Sector 6M", res["Sector"].map(lambda s: _sector_trend_txt(strength, s)))

    # Momentum rotation is cross-sectional: flag this month's top-N as ENTRY.
    if strat.get("type") == "rotation" and "6M Return %" in res.columns:
        topn = strat.get("topn", 10)
        elig = res[res["Selected"] == True].sort_values("6M Return %", ascending=False)
        top_syms = set(elig["Symbol"].head(topn))
        res["ENTRY SIGNAL"] = res["Symbol"].isin(top_syms)

    entries = res[res["ENTRY SIGNAL"] == True].copy()
    selected = res[res["Selected"] == True].copy()

    m1, m2, m3 = st.columns(3)
    m1.metric("Universe scanned", len(res))
    m2.metric("Pass selection", len(selected))
    m3.metric("Live ENTRY signals", len(entries))

    sort_col = strat.get("sort_col")
    sort_asc = strat.get("sort_asc", False)

    def _sort(d):
        if sort_col and sort_col in d.columns:
            return d.sort_values(sort_col, ascending=sort_asc)
        return d

    if view == "Entry signals only":
        out = _sort(entries)
    elif view == "Passes selection":
        out = _sort(selected)
    else:
        out = res.sort_values(["ENTRY SIGNAL", "Selected"], ascending=False)

    if out.empty:
        st.info("No stocks match this view right now.")
    else:
        try:
            st.dataframe(style_screen(out), use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download as CSV",
            out.to_csv(index=False).encode(),
            file_name=f"screen_{strat['key']}.csv",
            mime="text/csv",
        )

    if errors:
        st.caption(f"Could not fetch/evaluate {len(errors)} symbols "
                   f"(insufficient data or delisted).")


# =====================================================
# CHART TAB (native candlestick + TradingView link)
# =====================================================
def tv_symbol(sym: str) -> str:
    """Yahoo NSE ticker (e.g. 'M&M.NS') -> TradingView symbol ('NSE:M_M')."""
    base = sym.replace(".NS", "").replace(".BO", "")
    base = base.replace("&", "_").replace("-", "_")
    return f"NSE:{base}"


def _ma_line(d, col, clr):
    return alt.Chart(d).mark_line(color=clr, strokeWidth=1.5).encode(
        x="Date:T", y=alt.Y(f"{col}:Q"))


def render_chart(symbols, names, strat):
    st.caption("Candlestick chart from Yahoo Finance with this strategy's indicators "
               "overlaid. (TradingView's free embed blocks NSE data, so charts render "
               "natively here — use the link below for the full TradingView chart.)")

    labels = [f"{s.replace('.NS', '')} — {names.get(s, '')}".strip(" —") for s in symbols]
    default_sym = st.session_state.get("chart_symbol")
    idx = symbols.index(default_sym) if default_sym in symbols else 0

    c1, c2 = st.columns([3, 1])
    with c1:
        pick = st.selectbox("Stock", labels, index=idx)
    sym = symbols[labels.index(pick)]
    with c2:
        months = st.selectbox("Lookback", [6, 12, 24], index=1,
                              format_func=lambda m: f"{m} months")

    df = fetch(sym)
    if df is None or df.empty:
        st.warning("No daily data available for this symbol.")
        st.markdown(f"[Open on TradingView ↗](https://www.tradingview.com/chart/?symbol={tv_symbol(sym)})")
        return

    d = df.copy().reset_index()
    dcol = "Date" if "Date" in d.columns else d.columns[0]
    d = d.rename(columns={dcol: "Date"})
    d["Date"] = pd.to_datetime(d["Date"])

    c = d["Close"]
    d["EMA20"], d["EMA50"], d["EMA200"] = S._ema(c, 20), S._ema(c, 50), S._ema(c, 200)
    d["SMA50"], d["SMA150"], d["EMA220"] = S._sma(c, 50), S._sma(c, 150), S._ema(c, 220)
    d["RSI"] = S._rsi(c, 14)
    d["ADX"] = S._adx(d, 14)
    basis, upper, lower = S._bbands(c, 20, 2.0)
    d["BBu"], d["BBm"], d["BBl"] = upper, basis, lower
    st_line, _ = S._supertrend(d, 10, 3.0)
    d["ST"] = st_line
    d = d.tail(int(months * 21)).copy()

    color = alt.condition("datum.Open <= datum.Close",
                          alt.value("#26a69a"), alt.value("#ef5350"))
    base = alt.Chart(d).encode(x=alt.X("Date:T", axis=alt.Axis(title=None)))
    layers = [
        base.mark_rule().encode(y="Low:Q", y2="High:Q", color=color),
        base.mark_bar().encode(
            y=alt.Y("Open:Q", title="Price", scale=alt.Scale(zero=False)),
            y2="Close:Q", color=color),
    ]

    key = strat["key"]
    lower_panel = None
    if key == "onepct_breakout":
        layers += [_ma_line(d, "SMA50", "#1f77b4"), _ma_line(d, "SMA150", "#9467bd"),
                   _ma_line(d, "EMA220", "#d62728")]
    elif key == "s1_rsi_pullback":
        layers += [_ma_line(d, "EMA20", "#ff7f0e"), _ma_line(d, "EMA50", "#1f77b4"),
                   _ma_line(d, "EMA200", "#d62728")]
        lower_panel = "RSI"
    elif key == "s2_supertrend":
        layers += [_ma_line(d, "EMA50", "#1f77b4"), _ma_line(d, "EMA200", "#d62728"),
                   _ma_line(d, "ST", "#2ca02c")]
        lower_panel = "ADX"
    elif key == "s3_squeeze":
        layers += [_ma_line(d, "BBu", "#1f77b4"), _ma_line(d, "BBm", "#999999"),
                   _ma_line(d, "BBl", "#1f77b4")]
        lower_panel = "BBW"
    elif key == "vm_rsi40_support":
        layers += [_ma_line(d, "SMA50", "#1f77b4"), _ma_line(d, "EMA200", "#d62728")]
        lower_panel = "RSI"

    price = alt.layer(*layers).properties(height=380)
    vol = alt.Chart(d).mark_bar().encode(
        x=alt.X("Date:T", axis=alt.Axis(title=None)),
        y=alt.Y("Volume:Q", title="Vol"), color=color).properties(height=80)
    panels = [price, vol]

    if lower_panel == "RSI":
        panels.append(alt.layer(
            alt.Chart(d).mark_line(color="#7e57c2").encode(
                x="Date:T", y=alt.Y("RSI:Q", scale=alt.Scale(domain=[0, 100]), title="RSI")),
            alt.Chart(pd.DataFrame({"y": [30, 70]})).mark_rule(
                strokeDash=[4, 4], color="#bbbbbb").encode(y="y:Q"),
        ).properties(height=110))
    elif lower_panel == "ADX":
        panels.append(alt.layer(
            alt.Chart(d).mark_line(color="#e67e22").encode(
                x="Date:T", y=alt.Y("ADX:Q", title="ADX")),
            alt.Chart(pd.DataFrame({"y": [25]})).mark_rule(
                strokeDash=[4, 4], color="#bbbbbb").encode(y="y:Q"),
        ).properties(height=110))
    elif lower_panel == "BBW":
        d2 = d.assign(BBW=(d["BBu"] - d["BBl"]) / d["BBm"] * 100)
        panels.append(alt.Chart(d2).mark_area(opacity=0.5, color="#26c6da").encode(
            x=alt.X("Date:T", axis=alt.Axis(title=None)),
            y=alt.Y("BBW:Q", title="BB width %")).properties(height=90))

    st.altair_chart(alt.vconcat(*panels).resolve_scale(x="shared"),
                    use_container_width=True)
    st.markdown(
        f"[Open **{tv_symbol(sym)}** on TradingView ↗]"
        f"(https://www.tradingview.com/chart/?symbol={tv_symbol(sym)})"
    )


# =====================================================
# BACKTEST RESULTS TAB
# =====================================================
@st.cache_data(ttl=300)
def load_backtest(market, key):
    folder = os.path.join(RESULTS_FOLDER, market, key)
    eqp = os.path.join(folder, "equity_curve.csv")
    trp = os.path.join(folder, "trades.csv")
    if not os.path.exists(eqp):          # legacy fallback (pre-market layout)
        eqp = os.path.join(RESULTS_FOLDER, key, "equity_curve.csv")
        trp = os.path.join(RESULTS_FOLDER, key, "trades.csv")
    eq = pd.read_csv(eqp, parse_dates=["Date"]) if os.path.exists(eqp) else None
    tr = pd.read_csv(trp, parse_dates=["EntryDate", "ExitDate"]) if os.path.exists(trp) else None
    return eq, tr


CAPITAL = 100_000.0  # ₹1,00,000 starting capital (matches backtest.py)


def render_backtest(strat, market):
    eq, tr = load_backtest(market, strat["key"])
    if eq is None or eq.empty:
        st.info(
            f"No backtest results for **{strat['name']}** on **{markets.label(market)}** yet. "
            f"Run:\n\n`python backtest.py --market {market} --strategy {strat['key']}`   "
            "(or `--market all --strategy all`)."
        )
        return

    eq = eq.sort_values("Date").set_index("Date")
    equity_full = eq["Equity"]
    full_start, full_end = equity_full.index[0], equity_full.index[-1]
    avail_years = (full_end - full_start).days / 365.25

    # ---- period selector (1Y / 3Y / 5Y / 10Y / Full) ----
    opts = [o for o in ["1Y", "3Y", "5Y", "10Y"] if int(o[:-1]) <= avail_years + 0.5]
    opts.append("Full")
    sel = st.radio("Backtest window", opts, index=len(opts) - 1, horizontal=True)
    if sel == "Full":
        equity = equity_full
    else:
        yrs = int(sel[:-1])
        cutoff = full_end - pd.Timedelta(days=int(round(yrs * 365.25)))
        equity = equity_full[equity_full.index >= cutoff]
        if len(equity) < 2:
            equity = equity_full

    # rebase the chosen window so it starts at ₹1,00,000 ("growth of ₹1L over this window")
    eq_rb = equity / float(equity.iloc[0]) * CAPITAL
    end_val = float(eq_rb.iloc[-1])
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total_ret = end_val / CAPITAL - 1
    cagr = (end_val / CAPITAL) ** (1 / years) - 1
    daily = eq_rb.pct_change().dropna()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    drawdown = eq_rb / eq_rb.cummax() - 1
    max_dd = drawdown.min()

    st.caption(
        f"Backtest of **{strategies.label(strat['key'])}** on **{markets.label(market)}** · "
        f"₹1,00,000 start · max 10%/stock · 0.1%/side · window "
        f"{equity.index[0].date()} → {equity.index[-1].date()} ({years:.1f} yrs)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Equity", f"₹{end_val:,.0f}", f"{total_ret*100:+.1f}%")
    c2.metric("CAGR", f"{cagr*100:.2f}%")
    c3.metric("Max Drawdown", f"{max_dd*100:.2f}%")
    c4.metric("Sharpe (ann.)", f"{sharpe:.2f}")

    # trades that were entered inside the selected window
    trw = None
    if tr is not None and not tr.empty:
        trw = tr[pd.to_datetime(tr["EntryDate"]) >= equity.index[0]].copy()
        wins = trw[trw["PnL"] > 0]
        gl = abs(trw[trw["PnL"] <= 0]["PnL"].sum())
        gw = wins["PnL"].sum()
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Trades", len(trw))
        c6.metric("Win Rate", f"{(len(wins)/len(trw)*100) if len(trw) else 0:.1f}%")
        c7.metric("Profit Factor", f"{(gw/gl) if gl > 0 else float('inf'):.2f}")
        c8.metric("Avg Hold (days)", f"{trw['HoldingDays'].mean() if len(trw) else 0:.0f}")

    eqdf = pd.DataFrame({"Date": eq_rb.index, "Equity": eq_rb.values})
    ddf = pd.DataFrame({"Date": drawdown.index, "Drawdown": (drawdown * 100).values})

    st.markdown("#### 📈 Equity curve — growth of ₹1,00,000")
    area = alt.Chart(eqdf).mark_area(
        line={"color": "#1d4ed8", "strokeWidth": 2},
        color=alt.Gradient(gradient="linear",
            stops=[alt.GradientStop(color="#eff4ff", offset=0),
                   alt.GradientStop(color="#93b4fb", offset=1)],
            x1=1, x2=1, y1=1, y2=0)
    ).encode(
        x=alt.X("Date:T", axis=alt.Axis(title=None)),
        y=alt.Y("Equity:Q", scale=alt.Scale(zero=False), title="₹ (from 1,00,000)"),
        tooltip=["Date:T", alt.Tooltip("Equity:Q", format=",.0f", title="Equity ₹")],
    ).properties(height=300)
    baseln = alt.Chart(pd.DataFrame({"y": [CAPITAL]})).mark_rule(
        strokeDash=[4, 4], color="#94a3b8").encode(y="y:Q")
    st.altair_chart(area + baseln, use_container_width=True)

    st.markdown("#### 📉 Drawdown")
    dd_area = alt.Chart(ddf).mark_area(
        color="#fecaca", line={"color": "#dc2626"}).encode(
        x=alt.X("Date:T", axis=alt.Axis(title=None)),
        y=alt.Y("Drawdown:Q", title="Drawdown %"),
        tooltip=["Date:T", alt.Tooltip("Drawdown:Q", format=".1f", title="DD %")],
    ).properties(height=170)
    st.altair_chart(dd_area, use_container_width=True)

    if "OpenPositions" in eq.columns:
        with st.expander("Open positions over time"):
            st.bar_chart(eq["OpenPositions"].loc[equity.index], height=200,
                         use_container_width=True)

    st.subheader("Trade history")
    show = trw if trw is not None else tr
    if show is None or show.empty:
        st.info("No trades in this window.")
    else:
        show = show.sort_values("ExitDate", ascending=False)
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download trade log",
            show.to_csv(index=False).encode(),
            file_name=f"trades_{strat['key']}.csv", mime="text/csv",
        )


# =====================================================
# COMPARE-ALL TAB (equity overlay + walk-forward)
# =====================================================
@st.cache_data(ttl=300)
def load_all_equities(market):
    out = {}
    for key in strategies.ORDER:
        f = os.path.join(RESULTS_FOLDER, market, key, "equity_curve.csv")
        if not os.path.exists(f):
            f = os.path.join(RESULTS_FOLDER, key, "equity_curve.csv")   # legacy
        if os.path.exists(f):
            out[key] = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")["Equity"]
    return out


def _cmp_metrics(eq):
    eq = eq.dropna()
    if len(eq) < 10:
        return None
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    dl = eq.pct_change().dropna()
    sh = dl.mean() / dl.std() * np.sqrt(252) if dl.std() > 0 else 0.0
    return cagr * 100, dd * 100, sh


def render_compare(market):
    eqs = load_all_equities(market)
    if not eqs:
        st.info(f"No backtests for **{markets.label(market)}** yet. Run "
                f"`python backtest.py --market {market} --strategy all`.")
        return
    st.caption(f"All strategies on **{markets.label(market)}**.")

    # ---- equity overlay, each normalised to a starting value of 100 ----
    frames = []
    for key, e in eqs.items():
        n = (e / e.iloc[0] * 100).rename("Growth of 100").reset_index()
        n["Strategy"] = strategies.STRATEGIES[key]["name"]
        frames.append(n)
    big = pd.concat(frames, ignore_index=True)
    chart = alt.Chart(big).mark_line().encode(
        x=alt.X("Date:T", axis=alt.Axis(title=None)),
        y=alt.Y("Growth of 100:Q", title="Growth of ₹100"),
        color=alt.Color("Strategy:N", legend=alt.Legend(orient="bottom", title=None)),
    ).properties(height=420)
    st.altair_chart(chart, use_container_width=True)

    # ---- summary + walk-forward (IS first 60% vs OOS last 40%) ----
    rows = []
    for key, e in eqs.items():
        full = _cmp_metrics(e)
        split = e.index[int(len(e) * 0.6)]
        ins = _cmp_metrics(e.loc[:split])
        oos = _cmp_metrics(e.loc[split:])
        if not (full and ins and oos):
            continue
        rows.append({
            "Strategy": strategies.STRATEGIES[key]["name"],
            "Full CAGR %": round(full[0], 1), "Full Sharpe": round(full[2], 2),
            "Max DD %": round(full[1], 1),
            "IS CAGR %": round(ins[0], 1), "IS Sharpe": round(ins[2], 2),
            "OOS CAGR %": round(oos[0], 1), "OOS Sharpe": round(oos[2], 2),
        })
    st.subheader("Performance & walk-forward")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "IS = in-sample (first 60%), OOS = out-of-sample (last 40%, unseen). "
        "If OOS CAGR/Sharpe collapse vs IS, the edge was likely regime/luck, not robust. "
        "All figures are in-sample on today's constituents (survivorship bias)."
    )


# =====================================================
# MY POSITIONS TAB — HOLD / SELL + stop level per strategy
# =====================================================
def _load_positions():
    if os.path.exists(POSITIONS_FILE):
        try:
            df = pd.read_csv(POSITIONS_FILE)
            for col, default in (("Symbol", ""), ("Buy Price", 0.0), ("Qty", 0)):
                if col not in df.columns:
                    df[col] = default
            return df[["Symbol", "Buy Price", "Qty"]]
        except Exception:
            pass
    return pd.DataFrame({"Symbol": ["RELIANCE", "TCS"],
                         "Buy Price": [1400.0, 3800.0], "Qty": [5, 2]})


def _save_positions(df):
    try:
        keep = df.copy()
        keep["Symbol"] = keep["Symbol"].astype(str).str.strip()
        keep = keep[(keep["Symbol"] != "") & (keep["Symbol"].str.upper() != "NAN")]
        keep.to_csv(POSITIONS_FILE, index=False)
    except Exception:
        pass


def render_positions(strat, symbols, names):
    st.subheader("📋 My Positions vs strategy")
    st.caption(f"Enter each holding (Symbol · Buy Price · Qty). Advice uses the strategy "
               f"selected above — **{strat['name']}** — telling you HOLD or SELL and the "
               "stop-loss level to keep. Add/remove rows freely.")

    if "pos_df" not in st.session_state:
        st.session_state["pos_df"] = _load_positions()   # restore from disk
    if "pos_ver" not in st.session_state:
        st.session_state["pos_ver"] = 0

    # ---- import from a portfolio screenshot (OCR) ----
    with st.expander("📷 Import from a portfolio screenshot (auto-fill)"):
        parsed = None
        up = st.file_uploader("Drop a screenshot (PNG/JPG) of your holdings", type=["png", "jpg", "jpeg"], key="pf_img")
        if up is not None and portfolio_ocr is not None:
            ok, err = portfolio_ocr.ocr_available()
            if not ok:
                st.warning("OCR engine not set up. Install once: `brew install tesseract` "
                           f"and `pip install pytesseract pillow`. ({err}) "
                           "You can still paste your holdings as text below.")
            else:
                st.image(up.getvalue(), caption="Uploaded screenshot", width=380)
                with st.spinner("Reading the screenshot..."):
                    text = portfolio_ocr.image_to_text(up.getvalue())
                parsed = portfolio_ocr.parse_holdings(text)
                with st.expander("Raw text the OCR read (debug)"):
                    st.text(text or "(empty)")
        st.caption("…or paste holdings text (one row each, e.g. `RELIANCE 5 1400.00`):")
        pasted = st.text_area("Paste here", height=90, key="pf_txt", label_visibility="collapsed")
        if pasted.strip() and portfolio_ocr is not None:
            parsed = portfolio_ocr.parse_holdings(pasted)

        if parsed is not None and not parsed.empty:
            st.write("**Detected — review/fix, then add:**")
            prev = st.data_editor(parsed, num_rows="dynamic", use_container_width=True, key="pf_prev")
            if st.button("➕ Add these to my positions", type="primary"):
                base = _load_positions()
                merged = pd.concat([base, prev], ignore_index=True)
                merged["Symbol"] = (merged["Symbol"].astype(str).str.upper()
                                    .str.replace(".NS", "", regex=False).str.strip())
                merged = merged[merged["Symbol"] != ""].drop_duplicates(subset="Symbol", keep="last")
                _save_positions(merged)
                st.session_state["pos_df"] = merged
                st.session_state["pos_ver"] += 1
                st.success(f"Added {len(prev)} row(s) to your positions.")
                st.rerun()
        elif parsed is not None:
            st.info("Couldn't auto-detect rows — try a clearer/cropped screenshot, "
                    "or paste the text. Then review before adding.")

    if kite_helper is not None:
        k = kite_client()
        ok, _ = kite_helper.connection_status(k)
        if ok and st.button("⬇️ Load my Zerodha holdings"):
            h = kite_helper.get_holdings(k)
            if h is not None and not h.empty:
                st.session_state["pos_df"] = pd.DataFrame({
                    "Symbol": h["tradingsymbol"].astype(str),
                    "Buy Price": h["average_price"].astype(float),
                    "Qty": h["quantity"].astype(float).astype(int)})
                st.session_state["pos_ver"] += 1
                st.success(f"Loaded {len(h)} holdings from Zerodha.")

    edited = st.data_editor(
        st.session_state["pos_df"], num_rows="dynamic", use_container_width=True,
        key=f"pos_editor_{st.session_state['pos_ver']}",
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", help="NSE symbol, e.g. RELIANCE"),
            "Buy Price": st.column_config.NumberColumn("Buy Price", format="%.2f"),
            "Qty": st.column_config.NumberColumn("Qty", format="%d")})

    _save_positions(edited)   # autosave to disk so it survives refresh / restart
    st.caption(f"💾 Saved automatically to `{POSITIONS_FILE}` — your entries persist "
               "across refresh and restarts.")

    # optional real-time LTP from Zerodha
    ltp_map = {}
    if kite_helper is not None:
        k = kite_client()
        ok, _ = kite_helper.connection_status(k)
        if ok:
            ys = []
            for s in edited["Symbol"]:
                s = str(s).strip().upper()
                if s:
                    ys.append(s if s.endswith(".NS") else s + ".NS")
            ltp_map = kite_helper.live_ltp(k, ys)

    rows = []
    for _, r in edited.iterrows():
        raw = str(r.get("Symbol", "")).strip().upper()
        if not raw or raw == "NAN":
            continue
        ysym = raw if raw.endswith(".NS") else raw + ".NS"
        df = fetch(ysym)
        if df is None:
            rows.append({"Symbol": raw, "Action": "❓ no data (check symbol)"})
            continue
        buy = float(r.get("Buy Price") or 0)
        qty = float(r.get("Qty") or 0)
        adv = strategies.position_advice(strat["key"], df, buy)
        if adv is None:
            rows.append({"Symbol": raw, "Action": "insufficient history"})
            continue
        ltp = ltp_map.get(ysym, adv["Close"])
        pnl = (ltp / buy - 1) * 100 if buy > 0 else 0.0
        stop = adv["Stop level"]
        rows.append({
            "Symbol": raw.replace(".NS", ""), "Qty": int(qty), "Buy": round(buy, 2),
            "LTP": round(ltp, 2), "P&L %": round(pnl, 1),
            "Action": "🔴 SELL" if adv["Action"] == "SELL" else "✅ HOLD",
            "Stop-loss": stop,
            "Stop dist %": round((stop / ltp - 1) * 100, 1) if stop else None,
            "Why": adv["Reason"],
        })

    if rows:
        rdf = pd.DataFrame(rows)
        st.dataframe(rdf, use_container_width=True, hide_index=True)
        sells = [x for x in rows if x.get("Action") == "🔴 SELL"]
        if sells:
            st.warning(f"**{len(sells)}** position(s) flagged to SELL by {strat['name']}: "
                       + ", ".join(x["Symbol"] for x in sells)
                       + ". Review and place the exit yourself in Kite.")
        else:
            st.success(f"All positions are **HOLD** per {strat['name']} — keep your stops at "
                       "the levels shown. Exit if price closes below the stop.")
        st.download_button("⬇️ Download advice", rdf.to_csv(index=False).encode(),
                           file_name="my_positions_advice.csv", mime="text/csv")
    st.caption("Stop-loss = the strategy's protective level (you're advised to exit on a "
               "close below it). Educational only — not investment advice; place orders yourself.")


# =====================================================
# ZERODHA TAB (Kite Connect — read-only)
# =====================================================
def render_zerodha(strat, symbols, names):
    st.subheader("💼 Zerodha · Kite Connect")
    st.caption("Read-only: live prices, holdings & P&L, and exit flags. "
               "I never place orders — any order you place yourself in Kite.")

    if kite_helper is None or not kite_helper.have_kite_lib():
        st.info("Integration not installed. Run `pip install kiteconnect`, set up "
                "`.streamlit/secrets.toml`, then `python kite_login.py`. "
                "See README → **Zerodha setup**.")
        return

    kite = kite_client()
    ok, msg = kite_helper.connection_status(kite)
    (st.success if ok else st.warning)(msg)
    if not ok:
        with st.expander("Setup steps", expanded=True):
            st.markdown(
                "1. Create a Kite Connect app at **developers.kite.trade** (₹500/mo); "
                "set the redirect URL to `http://127.0.0.1`.\n"
                "2. Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` "
                "and fill `api_key` + `api_secret`.\n"
                "3. Each trading day run `python kite_login.py` and paste the request_token.\n"
                "4. Click **Refresh / recompute** or reload this tab.")
        return

    funds = kite_helper.get_funds(kite)
    if funds is not None:
        st.metric("Available funds (equity)", f"₹{funds:,.0f}")

    st.markdown("#### Holdings")
    hold = kite_helper.get_holdings(kite)
    if hold is None or hold.empty:
        st.caption("No holdings found in this account.")
    else:
        st.dataframe(hold, use_container_width=True, hide_index=True)

        st.markdown(f"#### Holdings vs **{strat['name']}** exit signal")
        sig_fn = strat.get("signals")
        rows = []
        for _, h in hold.iterrows():
            ysym = kite_helper.to_yahoo(str(h.get("tradingsymbol", "")))
            if ysym not in symbols:
                continue
            df = fetch(ysym)
            if df is None:
                continue
            sell, reason = False, ""
            try:
                if sig_fn is not None:
                    sell = bool(sig_fn(df.reset_index())["exit"].iloc[-1])
                    reason = "exit rule"
                else:
                    sma200 = S._sma(df["Close"], 200)
                    sell = float(df["Close"].iloc[-1]) < float(sma200.iloc[-1])
                    reason = "below 200DMA"
            except Exception:
                pass
            rows.append({"Symbol": h.get("tradingsymbol"), "Qty": h.get("quantity"),
                         "Avg": round(float(h.get("average_price", 0)), 1),
                         "LTP": round(float(h.get("last_price", 0)), 1),
                         "P&L": round(float(h.get("pnl", 0)), 0),
                         "Signal": "🔴 SELL" if sell else "✅ hold",
                         "Why": reason if sell else ""})
        if rows:
            sdf = pd.DataFrame(rows)
            st.dataframe(sdf, use_container_width=True, hide_index=True)
            n_sell = int((sdf["Signal"] == "🔴 SELL").sum())
            if n_sell:
                st.warning(f"{n_sell} holding(s) flagged to EXIT by {strat['name']}. "
                           "Review and place the sell yourself in Kite.")

    st.markdown("#### Manual order ticket")
    st.caption("Pre-fills the order — review and enter it yourself in Kite. No auto-execution.")
    cc1, cc2 = st.columns(2)
    with cc1:
        labels = [s.replace(".NS", "") for s in symbols]
        pick = st.selectbox("Stock", labels)
        ysym = symbols[labels.index(pick)]
    with cc2:
        side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
    ltp = kite_helper.live_ltp(kite, [ysym]).get(ysym)
    if ltp:
        ticket = kite_helper.order_ticket(ysym, side, ltp, capital=CAPITAL,
                                          weight=0.10, stop_pct=strat.get("stop_pct"))
        ticket["LTP"] = round(ltp, 2)
        st.table(pd.DataFrame([ticket]).T.rename(columns={0: "value"}))
    else:
        st.caption("Could not fetch a live price for this symbol.")


if __name__ == "__main__":
    main()
