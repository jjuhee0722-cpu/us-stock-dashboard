# US Stock Dashboard

Static US stock dashboard for S&P 500 screening, major ETF tracking, additional tickers, and a personal watchlist.

The public site is just static files:

- `public/index.html`
- `public/data.json`

Python is only used to refresh Yahoo Finance data and write `public/data.json`.

## Local Build

```bash
pip install -r requirements.txt
python generate_data.py
python -m http.server 8000 --directory public
```

Then open `http://localhost:8000`.

## Deploy

### GitHub Pages

The workflow in `.github/workflows/update-static-dashboard.yml` refreshes data every 3 hours and publishes the `public` directory to the `gh-pages` branch.

Public URL:

```text
https://jjuhee0722-cpu.github.io/us-stock-dashboard/
```

### Cloudflare Pages

Use these settings:

- Build command: `pip install -r requirements.txt && python generate_data.py`
- Output directory: `public`

The dashboard uses Yahoo Finance data through `yfinance`; values may be delayed or missing for some tickers. The static search universe is S&P 500 constituents plus major ETFs and the extra tickers configured in `generate_data.py`.
