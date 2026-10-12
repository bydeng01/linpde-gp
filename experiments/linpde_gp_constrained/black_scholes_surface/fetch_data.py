"""
Snapshot the ^SPX option chain to data/spx_chain_<date>.csv (real-data input).

Run OUTSIDE the container (needs network + yfinance):

    pip install yfinance
    python experiments/linpde_gp_constrained/black_scholes_surface/fetch_data.py

Output columns (the contract expected by run.load_chain_csv):
    quote_date, expiry, strike, bid, ask, mid, call_put

Both calls and puts are kept: run.py needs the pairs for the per-expiry
put-call-parity regression (forward + rate). All quote filtering (bid = 0,
relative spread > 25%, maturity window) happens in run.py, not here --- the
snapshot is raw.
"""

from __future__ import annotations

import csv
from datetime import date
import os
import sys


def main(ticker: str = "^SPX") -> None:
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("fetch_data.py needs yfinance: pip install yfinance")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    today = date.today().isoformat()
    out_csv = os.path.join(out_dir, f"spx_chain_{today}.csv")

    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        raise SystemExit(f"no option expiries returned for {ticker}.")

    rows = []
    for exp in expiries:
        chain = tk.option_chain(exp)
        for frame, cp in ((chain.calls, "C"), (chain.puts, "P")):
            for _, rec in frame.iterrows():
                bid = float(rec.get("bid") or 0.0)
                ask = float(rec.get("ask") or 0.0)
                if ask <= 0.0:
                    continue
                rows.append(
                    {
                        "quote_date": today,
                        "expiry": exp,
                        "strike": float(rec["strike"]),
                        "bid": bid,
                        "ask": ask,
                        "mid": 0.5 * (bid + ask),
                        "call_put": cp,
                    }
                )
        print(f"  {exp}: cumulative {len(rows)} quotes", flush=True)

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "quote_date",
                "expiry",
                "strike",
                "bid",
                "ask",
                "mid",
                "call_put",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} quotes -> {out_csv}")
    print(f"Run: python run.py --data data/{os.path.basename(out_csv)}")


if __name__ == "__main__":
    main(ticker=sys.argv[1] if len(sys.argv) > 1 else "^SPX")
