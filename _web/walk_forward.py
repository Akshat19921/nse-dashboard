"""
Walk-forward / out-of-sample robustness check.

Reads each strategy's saved equity curve (backtest_results/<key>/equity_curve.csv)
and splits the timeline into an in-sample (IS) front portion and an out-of-sample
(OOS) tail. A strategy that "works" should still perform in the OOS window it was
never reasoned about — a big IS→OOS drop is a red flag for overfitting / luck.

Run `python backtest.py --strategy all` first, then:
    python walk_forward.py
Saves backtest_results/walk_forward.csv and prints a table.
"""

import os
import numpy as np
import pandas as pd
import strategies

RESULTS = "backtest_results"
SPLIT = 0.60   # first 60% = in-sample, last 40% = out-of-sample


def metrics(eq):
    eq = eq.dropna()
    if len(eq) < 10:
        return None
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    r = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    dl = eq.pct_change().dropna()
    sh = dl.mean() / dl.std() * np.sqrt(252) if dl.std() > 0 else 0.0
    return {"ret": r * 100, "cagr": cagr * 100, "dd": dd * 100, "sharpe": sh}


def main():
    rows = []
    for key in strategies.ORDER:
        f = os.path.join(RESULTS, key, "equity_curve.csv")
        if not os.path.exists(f):
            continue
        eq = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")["Equity"]
        split_dt = eq.index[int(len(eq) * SPLIT)]
        is_m = metrics(eq.loc[:split_dt])
        oos_m = metrics(eq.loc[split_dt:])
        full = metrics(eq)
        if not (is_m and oos_m and full):
            continue
        rows.append({
            "Strategy": strategies.label(key),
            "Split": str(split_dt.date()),
            "Full CAGR %": round(full["cagr"], 1),
            "Full Sharpe": round(full["sharpe"], 2),
            "IS CAGR %": round(is_m["cagr"], 1),
            "IS Sharpe": round(is_m["sharpe"], 2),
            "OOS CAGR %": round(oos_m["cagr"], 1),
            "OOS Sharpe": round(oos_m["sharpe"], 2),
            "OOS Max DD %": round(oos_m["dd"], 1),
            "Holds up?": "yes" if oos_m["sharpe"] >= 0.5 * is_m["sharpe"] and oos_m["cagr"] > 0 else "weak",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No backtest results found. Run: python backtest.py --strategy all")
        return
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "walk_forward.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\nWalk-forward (IS = first {int(SPLIT*100)}%, OOS = last {int((1-SPLIT)*100)}%):\n")
    print(df.to_string(index=False))
    print(f"\nSaved: {os.path.join(RESULTS, 'walk_forward.csv')}")


if __name__ == "__main__":
    main()
