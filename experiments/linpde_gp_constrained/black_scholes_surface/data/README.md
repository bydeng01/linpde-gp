# SPX option-chain data

`../run.py --data` and `../sweep_spx.py` read a snapshot of the S&P 500 index
(^SPX) option chain from this directory. The results in the paper use the chain
captured on 2026-06-13 (`spx_chain_2026-06-13.csv`). That file is not included
in this repository: it was downloaded from Yahoo Finance with
[yfinance](https://github.com/ranaroussi/yfinance), which notes that the Yahoo
Finance API is intended for personal use only, so the quotes are not
redistributed here.

To create a snapshot (needs network access; run outside the container):

    pip install yfinance
    python ../fetch_data.py

This writes `spx_chain_<today>.csv` with columns
`quote_date, expiry, strike, bid, ask, mid, call_put`. Pass it to the scripts as
`--data data/spx_chain_<date>.csv`, or run `../reproduce_spx.sh` with that path.
Option chains change every trading day, so a snapshot from another date gives
numbers that differ from those reported in the paper. The results derived from
the 2026-06-13 chain are committed next to the scripts
(`../black_scholes_spx_sweep.csv`, `../black_scholes_spx_dense.json`,
`../bs_spx_numbers.tex` and the `figure_black_scholes_spx*.pdf` figures).
