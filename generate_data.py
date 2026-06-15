import json
import math
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.error import URLError

import pandas as pd
import requests
import yfinance as yf


CURATED_SECTOR_TICKERS = {
    "Technology": ["NVDA", "AVGO", "MSFT", "AAPL", "GOOGL", "META", "CRM", "ORCL"],
    "Semiconductors": ["TSM", "AMD", "LRCX", "ASML", "AMAT", "QCOM", "MU", "KLAC"],
    "Infrastructure/Data": ["AMT", "EQIX", "ANET", "CSCO", "NET", "DDOG", "SNOW"],
    "Cybersecurity": ["PANW", "CRWD", "ZS", "FTNT", "OKTA"],
    "Cloud/Software": ["NOW", "ADBE", "INTU", "SHOP", "MDB", "TEAM"],
    "Consumer/Internet": ["AMZN", "TSLA", "NFLX", "COST", "WMT", "HD", "MCD", "NKE", "UBER"],
    "Healthcare": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ISRG"],
    "Financials": ["JPM", "BAC", "V", "MA", "BRK-B", "GS", "MS"],
    "Energy/Industrial": ["XOM", "CVX", "CAT", "GE", "RTX", "BA", "LIN", "NEE"],
    "ETFs": ["SPY", "QQQ", "VOO", "VTI", "SCHD"],
}

S_AND_P_500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MAJOR_ETFS = ["SPY", "QQQ", "VOO", "VTI", "SCHD", "DIA", "IWM", "SOXX", "SMH", "XLK", "XLF", "XLV", "XLE"]
EXTRA_TICKERS = [
    "TSM", "ASML", "SHOP", "BABA", "NVO", "ARM", "PLTR", "COIN", "MSTR", "ARKK",
    "RIVN", "HOOD", "NET", "CRWD", "DDOG", "SNOW", "OKTA", "MDB", "SE", "MELI",
    "RKLB", "MRVL", "BMNR", "CRDO", "ARKX", "BITO", "BOTZ", "DRAM", "IREN", "IONQ",
]
DEFAULT_FAVORITES = [
    "RKLB", "LITE", "MRVL", "MSFT", "MU", "META", "VRT", "AVGO", "BMNR", "SNDK",
    "COHR", "CRDO", "TSLA", "PLTR", "AMD", "ARKX", "ARM", "BITO", "BOTZ", "DRAM",
    "STX", "AMZN", "IREN", "IONQ", "GOOGL", "AAPL", "NVDA", "WDC", "ETN", "INTC",
]
FAVORITE_GROUPS = {
    "반도체/하드웨어": [
        "LITE", "MRVL", "MU", "AVGO", "SNDK", "COHR", "CRDO", "AMD", "ARM", "DRAM",
        "STX", "WDC", "NVDA", "INTC",
    ],
    "AI/플랫폼": ["MSFT", "META", "PLTR", "AMZN", "GOOGL", "AAPL"],
    "전력/인프라": ["VRT", "ETN"],
    "우주/로보틱스/퀀텀": ["RKLB", "ARKX", "BOTZ", "IONQ"],
    "크립토/비트코인": ["BMNR", "BITO", "IREN"],
    "전기차": ["TSLA"],
}
PUBLIC_DIR = Path(__file__).resolve().parent / "public"
DATA_PATH = PUBLIC_DIR / "data.json"


def flatten_tickers(sector_map: dict[str, list[str]]) -> list[str]:
    tickers: list[str] = []
    for sector_tickers in sector_map.values():
        tickers.extend(sector_tickers)
    return sorted(set(tickers))


def yahoo_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def load_sp500_sectors() -> dict[str, list[str]]:
    try:
        response = requests.get(
            S_AND_P_500_URL,
            headers={"User-Agent": "Mozilla/5.0 us-stock-dashboard data refresh"},
            timeout=30,
        )
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[0]
    except (ImportError, ValueError, URLError, OSError, requests.RequestException) as error:
        print(f"Could not load S&P 500 constituents: {error}")
        return {}

    sectors: dict[str, list[str]] = {}
    for _, row in table.iterrows():
        ticker = yahoo_symbol(row.get("Symbol", ""))
        sector = str(row.get("GICS Sector") or "S&P 500").strip()
        if not ticker:
            continue
        sectors.setdefault(sector, []).append(ticker)

    return {sector: sorted(set(tickers)) for sector, tickers in sorted(sectors.items())}


def build_sector_universe() -> tuple[dict[str, list[str]], str]:
    sectors = load_sp500_sectors()
    if not sectors:
        sectors = {sector: list(tickers) for sector, tickers in CURATED_SECTOR_TICKERS.items()}
        source = "Curated fallback list"
    else:
        sectors["Major ETFs"] = sorted(set(MAJOR_ETFS))
        sectors["Additional Tickers"] = sorted(set(EXTRA_TICKERS))
        source = "S&P 500 구성종목 + 주요 ETF + 추가 티커"

    for ticker in DEFAULT_FAVORITES:
        sectors.setdefault("Favorites", [])
        if ticker not in sectors["Favorites"] and ticker not in flatten_tickers(sectors):
            sectors["Favorites"].append(ticker)
    if not sectors.get("Favorites"):
        sectors.pop("Favorites", None)

    return sectors, source


def parse_ticker_text(text: str) -> list[str]:
    raw_tickers = re.split(r"[\s,;/]+", text.upper().strip())
    tickers = []
    for ticker in raw_tickers:
        ticker = re.sub(r"[^A-Z0-9.\-]", "", ticker)
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers[:80]


def clean_number(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def pct(value: Optional[float]) -> Optional[float]:
    value = clean_number(value)
    return None if value is None else value * 100


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def normalize_info(info: object) -> dict:
    return info if isinstance(info, dict) else {}


def opinion_from_info(info: object) -> tuple[Optional[int], str]:
    info = normalize_info(info)
    key = str(info.get("recommendationKey") or "").lower()
    key_map = {
        "strong_buy": 5,
        "buy": 4,
        "hold": 3,
        "underperform": 2,
        "sell": 1,
        "strong_sell": 1,
    }
    score = key_map.get(key)

    if score is None:
        mean = clean_number(info.get("recommendationMean"))
        if mean is not None:
            score = int(round(6 - mean))
            score = max(1, min(5, score))

    label_map = {
        1: "강력매도",
        2: "매도추천",
        3: "중립",
        4: "매수추천",
        5: "강력매수",
    }
    return score, label_map.get(score, "의견 없음")


def load_price_history(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    return yf.download(
        tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )


def load_fundamentals(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        yf_ticker = yf.Ticker(ticker)
        try:
            info = yf_ticker.get_info()
        except Exception:
            try:
                info = yf_ticker.info
            except Exception:
                info = {}
        info = normalize_info(info)
        opinion_score, opinion_label = opinion_from_info(info)

        rows.append(
            {
                "Ticker": ticker,
                "Name": info.get("longName") or info.get("shortName") or ticker,
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "Business Summary": info.get("longBusinessSummary"),
                "ROE(%)": pct(info.get("returnOnEquity")),
                "Operating Margin(%)": pct(info.get("operatingMargins")),
                "Gross Margin(%)": pct(info.get("grossMargins")),
                "Profit Margin(%)": pct(info.get("profitMargins")),
                "Info 52W High": clean_number(info.get("fiftyTwoWeekHigh")),
                "Target Price": clean_number(info.get("targetMeanPrice")),
                "PER": clean_number(info.get("trailingPE")),
                "Forward PER": clean_number(info.get("forwardPE")),
                "PBR": clean_number(info.get("priceToBook")),
                "PSR": clean_number(info.get("priceToSalesTrailing12Months")),
                "Dividend Yield(%)": pct(info.get("dividendYield")),
                "Revenue Growth(%)": pct(info.get("revenueGrowth")),
                "Earnings Growth(%)": pct(info.get("earningsGrowth")),
                "Debt/Equity": clean_number(info.get("debtToEquity")),
                "Market Cap": clean_number(info.get("marketCap")),
                "Total Revenue": clean_number(info.get("totalRevenue")),
                "Net Income": clean_number(info.get("netIncomeToCommon")),
                "Free Cashflow": clean_number(info.get("freeCashflow")),
                "Total Cash": clean_number(info.get("totalCash")),
                "Total Debt": clean_number(info.get("totalDebt")),
                "Current Ratio": clean_number(info.get("currentRatio")),
                "Opinion Score": opinion_score,
                "Opinion": opinion_label,
            }
        )
    return pd.DataFrame(rows)


def get_close_frame(history: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=tickers)
    if isinstance(history.columns, pd.MultiIndex):
        closes = {}
        for ticker in tickers:
            if (ticker, "Close") in history.columns:
                closes[ticker] = history[(ticker, "Close")]
            elif ("Close", ticker) in history.columns:
                closes[ticker] = history[("Close", ticker)]
        return pd.DataFrame(closes)
    if "Close" in history.columns and len(tickers) == 1:
        return history[["Close"]].rename(columns={"Close": tickers[0]})
    return pd.DataFrame(columns=tickers)


def build_stock_rows(tickers: list[str]) -> tuple[list[dict], dict[str, list[dict]]]:
    history = load_price_history(tickers)
    fundamentals = load_fundamentals(tickers)
    close_frame = get_close_frame(history, tickers)

    rows = []
    charts: dict[str, list[dict]] = {}
    for ticker in tickers:
        close = close_frame.get(ticker, pd.Series(dtype="float64")).dropna()
        price = clean_number(close.iloc[-1]) if not close.empty else None
        ma50 = clean_number(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma200 = clean_number(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        history_high = clean_number(close.max()) if not close.empty else None

        f_row = fundamentals[fundamentals["Ticker"] == ticker]
        if f_row.empty:
            continue
        f_row = f_row.iloc[0]

        high_52w = clean_number(f_row["Info 52W High"]) or history_high
        high_gap = ((price / high_52w) - 1) * 100 if price and high_52w else None
        target_price = clean_number(f_row["Target Price"])
        target_upside = ((target_price / price) - 1) * 100 if price and target_price else None
        one_year_return = ((price / clean_number(close.iloc[0])) - 1) * 100 if len(close) > 1 and price else None
        cash_to_debt = safe_ratio(clean_number(f_row["Total Cash"]), clean_number(f_row["Total Debt"]))

        valuation_score = 0
        valuation_score += 1 if clean_number(f_row["PER"]) is not None and clean_number(f_row["PER"]) <= 25 else 0
        valuation_score += 1 if clean_number(f_row["PBR"]) is not None and clean_number(f_row["PBR"]) <= 8 else 0
        valuation_score += 1 if target_upside is not None and target_upside > 10 else 0
        valuation = "저평가 여지" if valuation_score >= 2 else "중립" if valuation_score == 1 else "비싼 편"

        row = {
            "Ticker": ticker,
            "Name": f_row["Name"],
            "Stock": f"{ticker} - {f_row['Name']}",
            "Sector": f_row["Sector"],
            "Industry": f_row["Industry"],
            "Business Summary": f_row["Business Summary"],
            "Price": price,
            "Target Price": target_price,
            "Target Upside(%)": target_upside,
            "Opinion Score": clean_number(f_row["Opinion Score"]),
            "Opinion": f_row["Opinion"],
            "ROE(%)": clean_number(f_row["ROE(%)"]),
            "Operating Margin(%)": clean_number(f_row["Operating Margin(%)"]),
            "Gross Margin(%)": clean_number(f_row["Gross Margin(%)"]),
            "Profit Margin(%)": clean_number(f_row["Profit Margin(%)"]),
            "PER": clean_number(f_row["PER"]),
            "Forward PER": clean_number(f_row["Forward PER"]),
            "PBR": clean_number(f_row["PBR"]),
            "PSR": clean_number(f_row["PSR"]),
            "Dividend Yield(%)": clean_number(f_row["Dividend Yield(%)"]),
            "Revenue Growth(%)": clean_number(f_row["Revenue Growth(%)"]),
            "Earnings Growth(%)": clean_number(f_row["Earnings Growth(%)"]),
            "Debt/Equity": clean_number(f_row["Debt/Equity"]),
            "Current Ratio": clean_number(f_row["Current Ratio"]),
            "Cash/Debt": cash_to_debt,
            "Market Cap": clean_number(f_row["Market Cap"]),
            "Total Revenue": clean_number(f_row["Total Revenue"]),
            "Net Income": clean_number(f_row["Net Income"]),
            "Free Cashflow": clean_number(f_row["Free Cashflow"]),
            "52W High": high_52w,
            "52W High Gap(%)": high_gap,
            "Above MA50": bool(price and ma50 and price > ma50),
            "Above MA200": bool(price and ma200 and price > ma200),
            "MA50": ma50,
            "MA200": ma200,
            "One Year Return(%)": one_year_return,
            "Valuation": valuation,
        }
        rows.append(row)

        if not close.empty:
            chart_df = pd.DataFrame({"Close": close})
            chart_df["MA50"] = chart_df["Close"].rolling(50).mean()
            chart_df["MA200"] = chart_df["Close"].rolling(200).mean()
            charts[ticker] = [
                {
                    "date": index.strftime("%Y-%m-%d"),
                    "close": clean_number(chart_row["Close"]),
                    "ma50": clean_number(chart_row["MA50"]),
                    "ma200": clean_number(chart_row["MA200"]),
                }
                for index, chart_row in chart_df.tail(260).iterrows()
            ]

    return rows, charts


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def main() -> None:
    sectors, universe_source = build_sector_universe()
    tickers = flatten_tickers(sectors)
    rows, charts = build_stock_rows(tickers)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = {
        "meta": {
            "generatedAt": generated_at,
            "source": "Yahoo Finance via yfinance",
            "refreshCadence": "GitHub Actions schedule: every 3 hours",
            "universe": universe_source,
            "tickerCount": len(tickers),
            "stockCount": len(rows),
        },
        "sectors": sectors,
        "defaultFavorites": DEFAULT_FAVORITES,
        "favoriteGroups": FAVORITE_GROUPS,
        "stocks": rows,
        "charts": charts,
    }

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(
        json.dumps(sanitize_for_json(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {DATA_PATH} with {len(rows)} stocks at {generated_at}")


if __name__ == "__main__":
    main()
