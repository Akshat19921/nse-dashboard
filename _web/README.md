# Nifty 100 — Strong Stock Breakout (Algo Strategy)

Python port of the Pro Trader Bootcamp Nasdaq-100 strategy, adapted for the
**NSE Nifty 100** universe. All market data lives in **one Excel workbook**.

## Files

- `nifty100_symbols.csv` — the universe (editable). Symbol, Name.
- `build_nifty_excel.py` — downloads all 100 stocks from Yahoo into **one** `.xlsx`.
- `backtest.py` — backtests ANY strategy (`--strategy <key>` or `all`) → `backtest_results/<key>/`.
- `run_backtests.command` — one-click: backtest all strategies on the Nifty 100.
- `walk_forward.py` — in-sample vs out-of-sample robustness check across strategies.
- `verify.py` — audit script: cross-checks indicators, engine accounting (cash identity,
  no-lookahead, both-side costs) and performance metrics. Run `python verify.py` (24 checks).
- `kite_helper.py` / `kite_login.py` — optional Zerodha Kite Connect integration (read-only).
- `portfolio_ocr.py` — reads a broker portfolio **screenshot** → Symbol/Qty/Buy-price rows
  (Tesseract OCR; needs `brew install tesseract`). Used by the 📋 My Positions tab.

## 📋 My Positions tab

Enter your holdings (Symbol · Buy Price · Qty) — or **drop a screenshot** of your broker
portfolio to auto-fill them (review before adding), or click **Load my Zerodha holdings**.
For the selected strategy it tells you **HOLD / SELL** and the **stop-loss level** to keep,
with P&L. Entries **autosave** to `my_positions.csv` and survive refresh/restart.

## Zerodha (Kite Connect) setup — optional

The dashboard's **💼 Zerodha** tab shows your live funds, holdings & P&L, flags which
holdings hit the selected strategy's exit rule, and builds a pre-filled order ticket.
The 🔴 Live Screen can also overlay Zerodha real-time LTP. **It is strictly read-only —
no orders are ever placed for you; you place any order yourself in Kite.**

1. `pip install kiteconnect` (already in requirements.txt).
2. Create a **Kite Connect** app at https://developers.kite.trade (≈₹500/month). Set the
   app's redirect URL to `http://127.0.0.1`. Note the **api_key** and **api_secret**.
3. `cp .streamlit/secrets.toml.example .streamlit/secrets.toml` and fill in api_key/secret.
4. Each trading day run **`python kite_login.py`** — open the printed URL, log in to
   Zerodha (you enter your password only on Zerodha's site), and paste back the
   `request_token`. It saves a daily access token (valid until ~6am next day).
5. Reload the dashboard → the 💼 Zerodha tab connects.

> Kite Connect is a paid personal API. Historical intraday data is a separate add-on, so
> indicators still use Yahoo daily history; Kite is used for **real-time LTP, holdings and
> funds**. Never commit `.streamlit/secrets.toml` or `kite_token.json` (see `.gitignore`).
- `strategies.py` — pluggable strategy registry (the dashboard dropdown reads this).
- `dashboard.py` — Streamlit app: strategy dropdown + live screener + backtest charts.
- `setup_and_run.command` — double-click to do everything (build Excel → backtest → dashboard).
- `run_dashboard.command` — double-click to just open the dashboard.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python build_nifty_excel.py     # 1. all 100 stocks -> nifty100.xlsx
python backtest.py --strategy all   # 2. backtest every strategy -> backtest_results/<key>/
streamlit run dashboard.py      # 3. live screener + backtest charts
```

…or just double-click `setup_and_run.command`.

## The single Excel workbook — `nifty100.xlsx`

One file, three sheets (no more one-CSV-per-stock):

1. **Live Snapshot** — one row per stock: live/last price, day change %, the
   50/150-day SMAs, 220-day EMA, 52-week high/low, % above the 52w low, and a
   pass/fail for each of the 5 selection rules + the breakout. Sorted so live
   entry signals sit on top.
2. **Price History** — full daily OHLCV for every stock stacked in one sheet
   (`Symbol, Date, Open, High, Low, Close, AdjClose, Volume`). This is what the
   backtest reads.
3. **Symbols** — the master universe list.

## Multiple strategies (dropdown)

The dashboard has a **Strategy** dropdown at the top. Each strategy is defined in
`strategies.py`, screens live from Yahoo, charts its own indicators, **and is
backtested on the Nifty 100**. Shipped strategies:

- **Strong Stock Breakout — 1% Club** (SMA/EMA stack + new 52w-high breakout)
- **Momentum Rotation (Top-10, 6-month) — Tuned** ⭐ best returns in testing: monthly
  rebalance into the strongest names above their 200-DMA
- **Strong Breakout (ATR Trailing) — Tuned** 1% Club entries + a 3×ATR chandelier exit
  (≈half the drawdown, smoother)
- **RSI Pullback in Uptrend — Playbook** (Close>EMA50>EMA200; RSI crosses back >40 + volume)
- **Supertrend (10,3) + ADX Trend Rider — Playbook** (ST flip up + ADX>25 + EMA50>EMA200)
- **Bollinger Squeeze Breakout — Playbook** (close above upper band + RVOL>2, above EMA200)

> **Long-term MA convention:** the 1% Club family (Strong Breakout + ATR Trailing) uses the
> original **220-day** EMA. Everything else — Momentum Rotation's filter, the regime banner,
> and the three Playbook strategies (RSI / Supertrend / Bollinger) — uses **200-day** MAs.

Backtest comparison on the Nifty 100 — **full 10 years** (2016-06 → 2026-06, ₹1L start,
0.1%/side, in-sample). The dashboard's Backtest tab lets you re-slice to **1Y / 3Y / 5Y / 10Y**.

| Strategy | Return | CAGR | Max DD | Sharpe | Trades |
|---|---|---|---|---|---|
| Momentum Rotation Top-10 (200-DMA) | +611% | **21.7%** | −32% | **1.46** | 381 |
| Strong Breakout (ATR Trailing) | +123% | 8.4% | **−16.6%** | 0.98 | 313 |
| Bollinger Squeeze | +116% | 8.0% | −20.1% | 0.80 | 1008 |
| 1% Club Strong Breakout | +110% | 7.7% | −27.8% | 0.65 | 121 |
| Supertrend + ADX | +73% | 5.6% | −14.0% | 0.72 | 278 |
| RSI Pullback | +61% | 4.9% | −8.1% | 0.81 | 1286 |

> ⚠️ In-sample, today's-constituents (survivorship-biased) results. Treat as a relative
> ranking, not a future guarantee.

### Walk-forward / out-of-sample (the reality check)

`python walk_forward.py` splits each equity curve into in-sample (first 60%, ~2016-2022)
and out-of-sample (last 40%, ~2022-2026, unseen). Over a **full 10-year cycle, all six
strategies hold up out-of-sample** — a much healthier result than the 5-year window
(which had put the entire weak 2024-26 patch in the OOS slice):

| Strategy | IS CAGR / Sharpe | OOS CAGR / Sharpe | Holds up? |
|---|---|---|---|
| Momentum Top-10 (200-DMA) | 32.4% / 1.67 | 7.2% / **1.56** | yes |
| Strong Breakout (ATR Trailing) | 8.7% / 0.91 | 7.9% / 1.21 | yes |
| 1% Club Breakout | 4.1% / 0.36 | 13.3% / 1.19 | yes |
| Bollinger Squeeze | 8.6% / 0.73 | 7.2% / 1.18 | yes |
| RSI Pullback | 5.4% / 0.82 | 4.0% / 0.81 | yes |
| Supertrend + ADX | 7.9% / 0.85 | 2.4% / 0.45 | yes |

**Takeaway:** over a full cycle, Momentum Rotation is the standout (and stays strong
out-of-sample), while the risk-managed ATR-trailing breakout is the steadiest. The 5-year
read was misleadingly pessimistic — always check multiple windows. Still, don't deploy on
backtest returns alone; the
honest next steps are a point-in-time (no-survivorship) constituent list and live
paper-trading. The dashboard's **Compare All** tab shows this visually.

**Backtest any/all of them** (₹1,00,000 capital, max 10%/stock, 0.1% cost/side,
each strategy's own exit + stop):

```
python backtest.py --strategy s2_supertrend   # one strategy
python backtest.py --strategy all              # all of them
```

Results go to `backtest_results/<strategy_key>/`; the dashboard's **Backtest
Results** tab shows whichever strategy is selected in the dropdown. There's also
a one-click **`run_backtests.command`** that backtests all strategies.

Each strategy also defines a `signals(df)` (full-series entry/exit for the
backtest) and an `evaluate(df)` (latest-bar screen). To add your own, copy one
set of functions, append an entry to `STRATEGIES` + `ORDER`, and it appears in
the dropdown and the backtester automatically.

A market-regime banner (Nifty 100 vs its 200 DMA + India VIX) sits at the top as
a universal go/no-go gate for longs.

## The 1% Club strategy (backtested)

**Selection (closing basis, all must be true):**
1. `SMA(150) > EMA(220)` — long-term trend bullish
2. `Close > SMA(50)` — short-term strength
3. `SMA(50) > SMA(150)` — trend stacking
4. `Close > 1.25 × 52-week low` — more than 25% above the 52w low
5. Low dipped below `EMA(220)` at least once in the last 90 days

**Entry:** new 52-week high on a *closing* basis while all 5 rules hold.
Signal on the close → executed at the **next day's open**.

**Exit:** close below `EMA(220)`, OR 15% below the entry price — whichever first.

**Portfolio:** ₹1,00,000 capital, equal-weighted, max 10% (₹10,000) per stock
(→ max 10 positions), 0.10% cost per side.

## Notes & caveats

- **Universe:** `nifty100_symbols.csv` is an editable Nifty 100 list. To refresh
  it against the official index, download `ind_nifty100list.csv` from
  niftyindices.com and paste its Symbol column (add the `.NS` suffix Yahoo uses).
- **Survivorship bias:** the list is *today's* constituents.
- **52-week rules need ~1 year of data** before they produce valid signals.
- The dashboard's Live Screen reads the workbook's snapshot instantly; tick
  "Fetch fresh from Yahoo" to pull live prices for all 100 (slower).
- Research/education only — not investment advice.
```
