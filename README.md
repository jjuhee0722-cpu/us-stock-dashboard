# US Stock Dashboard

Streamlit dashboard for screening US stocks by sector, profitability, momentum, analyst target price, and valuation metrics.

It also includes a personal watchlist dashboard. Add tickers in the watchlist box and save them; the list is stored in the URL as a `fav` query parameter so the same link can reopen the same watchlist.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Render

Render can deploy this repository using `render.yaml`.

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`

The dashboard uses Yahoo Finance data through `yfinance`; values may be delayed or missing for some tickers.
