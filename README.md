# US Stock Dashboard

Static US stock dashboard for sector screening and a personal watchlist.

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

The workflow in `.github/workflows/update-static-dashboard.yml` refreshes data once per hour and deploys the `public` directory to GitHub Pages.

In GitHub, set Pages source to **GitHub Actions**.

### Cloudflare Pages

Use these settings:

- Build command: `pip install -r requirements.txt && python generate_data.py`
- Output directory: `public`

The dashboard uses Yahoo Finance data through `yfinance`; values may be delayed or missing for some tickers.
