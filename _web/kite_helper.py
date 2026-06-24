"""
Zerodha Kite Connect helper for the dashboard (READ-ONLY).

Provides: connection from saved credentials, live LTP quotes, holdings,
positions, and funds. It deliberately does NOT place orders — order tickets are
pre-filled in the dashboard for you to review and place yourself in Kite.

Setup (one-time):
  1. Create a Kite Connect app at https://developers.kite.trade (₹500/mo).
     Note the api_key and api_secret. Set the redirect URL to http://127.0.0.1
  2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill
     api_key + api_secret.
  3. Each trading day, run:  python kite_login.py   (logs in, saves access token)

Requires:  pip install kiteconnect
"""

import os
import json

TOKEN_FILE = "kite_token.json"   # stores the daily access_token (gitignore this)

try:
    from kiteconnect import KiteConnect
    _HAVE_KITE = True
except Exception:
    _HAVE_KITE = False


def have_kite_lib() -> bool:
    return _HAVE_KITE


# ---------- credentials ----------
def _read_secrets():
    """Return (api_key, api_secret) from Streamlit secrets or env vars."""
    api_key = api_secret = None
    try:
        import streamlit as st
        if "kite" in st.secrets:
            api_key = st.secrets["kite"].get("api_key")
            api_secret = st.secrets["kite"].get("api_secret")
    except Exception:
        pass
    api_key = api_key or os.environ.get("KITE_API_KEY")
    api_secret = api_secret or os.environ.get("KITE_API_SECRET")
    return api_key, api_secret


def _read_access_token():
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE) as f:
                return json.load(f).get("access_token")
        except Exception:
            return None
    return os.environ.get("KITE_ACCESS_TOKEN")


def save_access_token(token: str):
    with open(TOKEN_FILE, "w") as f:
        json.dump({"access_token": token}, f)


# ---------- symbol mapping ----------
def to_kite(yahoo_symbol: str) -> str:
    """'RELIANCE.NS' -> 'NSE:RELIANCE'  (Kite keeps & and - in tradingsymbols)."""
    base = yahoo_symbol.replace(".NS", "").replace(".BO", "")
    return f"NSE:{base}"


def to_yahoo(tradingsymbol: str) -> str:
    return f"{tradingsymbol}.NS"


# ---------- connection ----------
def get_kite():
    """Return an authenticated KiteConnect client, or None if not set up."""
    if not _HAVE_KITE:
        return None
    api_key, _ = _read_secrets()
    token = _read_access_token()
    if not api_key or not token:
        return None
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(token)
        return kite
    except Exception:
        return None


def connection_status(kite):
    """Return (ok: bool, message: str). Calls profile() to validate the token."""
    if not _HAVE_KITE:
        return False, "kiteconnect not installed (pip install kiteconnect)."
    if kite is None:
        return False, "Not configured — add api_key/secret and run kite_login.py."
    try:
        p = kite.profile()
        return True, f"Connected as {p.get('user_name', p.get('user_id', 'user'))}."
    except Exception as e:
        return False, f"Token invalid/expired — re-run kite_login.py. ({e})"


# ---------- read-only data ----------
def live_ltp(kite, yahoo_symbols):
    """dict: yahoo_symbol -> last_price, via batched kite.ltp()."""
    out = {}
    if kite is None:
        return out
    ksyms = [to_kite(s) for s in yahoo_symbols]
    for i in range(0, len(ksyms), 200):              # batch to be safe
        chunk = ksyms[i:i + 200]
        try:
            data = kite.ltp(chunk)
            for k, v in data.items():                # k like 'NSE:RELIANCE'
                out[to_yahoo(k.split(":", 1)[1])] = float(v["last_price"])
        except Exception:
            continue
    return out


def get_holdings(kite):
    import pandas as pd
    if kite is None:
        return pd.DataFrame()
    try:
        h = pd.DataFrame(kite.holdings())
    except Exception:
        return pd.DataFrame()
    if h.empty:
        return h
    keep = ["tradingsymbol", "quantity", "average_price", "last_price",
            "pnl", "day_change_percentage"]
    h = h[[c for c in keep if c in h.columns]].copy()
    if {"quantity", "average_price"}.issubset(h.columns):
        h["invested"] = (h["quantity"] * h["average_price"]).round(0)
    if {"quantity", "last_price"}.issubset(h.columns):
        h["current"] = (h["quantity"] * h["last_price"]).round(0)
    return h


def get_positions(kite):
    import pandas as pd
    if kite is None:
        return pd.DataFrame()
    try:
        net = kite.positions().get("net", [])
        return pd.DataFrame(net)
    except Exception:
        return pd.DataFrame()


def get_funds(kite):
    if kite is None:
        return None
    try:
        m = kite.margins()
        return float(m.get("equity", {}).get("available", {}).get("live_balance", 0.0))
    except Exception:
        return None


def order_ticket(symbol_yahoo, side, ltp, capital=100000.0, weight=0.10, stop_pct=None):
    """Build a PRE-FILLED order summary (does NOT place anything)."""
    base = symbol_yahoo.replace(".NS", "")
    qty = int((capital * weight) // ltp) if ltp and ltp > 0 else 0
    ticket = {
        "Exchange": "NSE", "Symbol": base, "Side": side,
        "Qty": qty, "Order type": "MARKET / LIMIT (your choice)",
        "Approx value": round(qty * ltp, 0) if ltp else 0,
    }
    if stop_pct and ltp:
        ticket["Suggested stop"] = round(ltp * (1 - stop_pct), 2)
    return ticket
