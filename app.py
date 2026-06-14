import math
import re
from datetime import datetime
from html import escape
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="US Equity Sector Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


SECTOR_TICKERS = {
    "Technology": ["NVDA", "AVGO", "MSFT", "AAPL", "GOOGL", "META", "CRM", "ORCL"],
    "Semiconductors": ["TSM", "AMD", "LRCX", "ASML", "AMAT", "QCOM", "MU", "KLAC"],
    "Infrastructure/Data": ["AMT", "EQIX", "ANET", "CSCO", "NET", "DDOG", "SNOW"],
    "Cybersecurity": ["PANW", "CRWD", "ZS", "FTNT", "OKTA"],
    "Cloud/Software": ["NOW", "ADBE", "INTU", "SHOP", "MDB", "TEAM"],
}

DEFAULT_FAVORITES = ["NVDA", "MSFT", "AAPL", "GOOGL", "AVGO"]


def flatten_tickers(sector_map: dict[str, list[str]]) -> list[str]:
    tickers: list[str] = []
    for sector_tickers in sector_map.values():
        tickers.extend(sector_tickers)
    return sorted(set(tickers))


def parse_ticker_text(text: str) -> list[str]:
    raw_tickers = re.split(r"[\s,;/]+", text.upper().strip())
    tickers = []
    for ticker in raw_tickers:
        ticker = re.sub(r"[^A-Z0-9.\-]", "", ticker)
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers[:20]


def pct(value: Optional[float]) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value) * 100


def clean_number(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


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


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_price_history(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame:
    data = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    return data


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        yf_ticker = yf.Ticker(ticker)
        info = {}
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
                "ROE(%)": pct(clean_number(info.get("returnOnEquity"))),
                "Operating Margin(%)": pct(clean_number(info.get("operatingMargins"))),
                "Gross Margin(%)": pct(clean_number(info.get("grossMargins"))),
                "Profit Margin(%)": pct(clean_number(info.get("profitMargins"))),
                "Info 52W High": clean_number(info.get("fiftyTwoWeekHigh")),
                "Target Price": clean_number(info.get("targetMeanPrice")),
                "PER": clean_number(info.get("trailingPE")),
                "Forward PER": clean_number(info.get("forwardPE")),
                "PBR": clean_number(info.get("priceToBook")),
                "PSR": clean_number(info.get("priceToSalesTrailing12Months")),
                "Dividend Yield(%)": pct(clean_number(info.get("dividendYield"))),
                "Revenue Growth(%)": pct(clean_number(info.get("revenueGrowth"))),
                "Earnings Growth(%)": pct(clean_number(info.get("earningsGrowth"))),
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


def build_screening_table(
    tickers: list[str],
    min_roe: int,
    min_operating_margin: int,
    require_above_ma50: bool,
    max_high_gap: int,
) -> pd.DataFrame:
    ticker_tuple = tuple(tickers)
    history = load_price_history(ticker_tuple)
    fundamentals = load_fundamentals(ticker_tuple)
    close_frame = get_close_frame(history, tickers)

    rows = []
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

        rows.append(
            {
                "Ticker": ticker,
                "Name": f_row["Name"],
                "Price": price,
                "Target Price": target_price,
                "Target Upside(%)": target_upside,
                "Opinion Score": clean_number(f_row["Opinion Score"]),
                "Opinion": f_row["Opinion"],
                "ROE(%)": clean_number(f_row["ROE(%)"]),
                "Operating Margin(%)": clean_number(f_row["Operating Margin(%)"]),
                "PER": clean_number(f_row["PER"]),
                "Forward PER": clean_number(f_row["Forward PER"]),
                "PBR": clean_number(f_row["PBR"]),
                "Revenue Growth(%)": clean_number(f_row["Revenue Growth(%)"]),
                "Debt/Equity": clean_number(f_row["Debt/Equity"]),
                "52W High": high_52w,
                "52W High Gap(%)": high_gap,
                "Above MA50": bool(price and ma50 and price > ma50),
                "MA50": ma50,
                "MA200": ma200,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    mask &= df["ROE(%)"].fillna(-999) >= min_roe
    mask &= df["Operating Margin(%)"].fillna(-999) >= min_operating_margin
    mask &= df["52W High Gap(%)"].fillna(-999) >= -max_high_gap
    if require_above_ma50:
        mask &= df["Above MA50"]

    return df.loc[mask].sort_values(["ROE(%)", "Operating Margin(%)"], ascending=False)


def build_favorites_table(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    ticker_tuple = tuple(tickers)
    history = load_price_history(ticker_tuple)
    fundamentals = load_fundamentals(ticker_tuple)
    close_frame = get_close_frame(history, tickers)

    rows = []
    for ticker in tickers:
        close = close_frame.get(ticker, pd.Series(dtype="float64")).dropna()
        price = clean_number(close.iloc[-1]) if not close.empty else None
        one_year_return = ((price / clean_number(close.iloc[0])) - 1) * 100 if len(close) > 1 and price else None
        ma50 = clean_number(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma200 = clean_number(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        f_row = fundamentals[fundamentals["Ticker"] == ticker]
        if f_row.empty:
            continue
        f_row = f_row.iloc[0]

        target_price = clean_number(f_row["Target Price"])
        target_upside = ((target_price / price) - 1) * 100 if price and target_price else None
        cash_to_debt = safe_ratio(clean_number(f_row["Total Cash"]), clean_number(f_row["Total Debt"]))
        valuation_score = 0
        valuation_score += 1 if clean_number(f_row["PER"]) is not None and clean_number(f_row["PER"]) <= 25 else 0
        valuation_score += 1 if clean_number(f_row["PBR"]) is not None and clean_number(f_row["PBR"]) <= 8 else 0
        valuation_score += 1 if target_upside is not None and target_upside > 10 else 0
        valuation_note = "저평가 여지" if valuation_score >= 2 else "중립" if valuation_score == 1 else "비싼 편"

        rows.append(
            {
                "Ticker": ticker,
                "Name": f_row["Name"],
                "Stock": f"{ticker} - {f_row['Name']}",
                "Price": price,
                "Target Price": target_price,
                "Target Upside(%)": target_upside,
                "Opinion Score": clean_number(f_row["Opinion Score"]),
                "Opinion": f_row["Opinion"],
                "Market Cap": clean_number(f_row["Market Cap"]),
                "Dividend Yield(%)": clean_number(f_row["Dividend Yield(%)"]),
                "PER": clean_number(f_row["PER"]),
                "Forward PER": clean_number(f_row["Forward PER"]),
                "PBR": clean_number(f_row["PBR"]),
                "PSR": clean_number(f_row["PSR"]),
                "ROE(%)": clean_number(f_row["ROE(%)"]),
                "Gross Margin(%)": clean_number(f_row["Gross Margin(%)"]),
                "Operating Margin(%)": clean_number(f_row["Operating Margin(%)"]),
                "Profit Margin(%)": clean_number(f_row["Profit Margin(%)"]),
                "Revenue Growth(%)": clean_number(f_row["Revenue Growth(%)"]),
                "Earnings Growth(%)": clean_number(f_row["Earnings Growth(%)"]),
                "Debt/Equity": clean_number(f_row["Debt/Equity"]),
                "Current Ratio": clean_number(f_row["Current Ratio"]),
                "Cash/Debt": cash_to_debt,
                "Total Revenue": clean_number(f_row["Total Revenue"]),
                "Net Income": clean_number(f_row["Net Income"]),
                "Free Cashflow": clean_number(f_row["Free Cashflow"]),
                "One Year Return(%)": one_year_return,
                "Above MA50": bool(price and ma50 and price > ma50),
                "Above MA200": bool(price and ma200 and price > ma200),
                "Valuation": valuation_note,
            }
        )

    return pd.DataFrame(rows)


def get_ticker_history(ticker: str) -> pd.DataFrame:
    history = load_price_history((ticker,), period="1y")
    close_frame = get_close_frame(history, [ticker])
    if ticker not in close_frame:
        return pd.DataFrame()

    chart_df = pd.DataFrame({"Close": close_frame[ticker]}).dropna()
    chart_df["MA50"] = chart_df["Close"].rolling(50).mean()
    chart_df["MA200"] = chart_df["Close"].rolling(200).mean()
    return chart_df


def make_price_chart(ticker: str, company_name: str, chart_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["Close"],
            mode="lines",
            name="Close",
            line=dict(width=2.4, color="#2563eb"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MA50"],
            mode="lines",
            name="MA50",
            line=dict(width=1.8, color="#f59e0b"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MA200"],
            mode="lines",
            name="MA200",
            line=dict(width=1.8, color="#16a34a"),
        )
    )
    fig.update_layout(
        title=f"{ticker} - {company_name} 최근 1년 주가 및 이동평균선",
        height=500,
        margin=dict(l=16, r=16, t=60, b=30),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="Price (USD)",
    )
    return fig


def format_percent(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:.2f}%"


def format_price(value: float) -> str:
    return "-" if pd.isna(value) else f"${value:,.2f}"


def format_multiple(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:.2f}배"


def format_large_money(value: float) -> str:
    if pd.isna(value):
        return "-"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"


def format_plain(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:.2f}"


def render_mobile_cards(df: pd.DataFrame) -> None:
    for row in df.to_dict("records"):
        stock = escape(str(row["Stock"]))
        price = escape(format_price(row["Price"]))
        target = escape(format_price(row["Target Price"]))
        upside = escape(format_percent(row["Target Upside(%)"]))
        opinion = escape(f"{int(row['Opinion Score'])} · {row['Opinion']}" if pd.notna(row["Opinion Score"]) else str(row["Opinion"]))
        roe = escape(format_percent(row["ROE(%)"]))
        margin = escape(format_percent(row["Operating Margin(%)"]))
        per = escape(f"{row['PER']:.2f}" if pd.notna(row["PER"]) else "-")
        pbr = escape(f"{row['PBR']:.2f}" if pd.notna(row["PBR"]) else "-")
        high = escape(format_price(row["52W High"]))
        gap = escape(format_percent(row["52W High Gap(%)"]))
        st.markdown(
            f"""
            <div class="stock-card">
                <div class="stock-card-title">{stock}</div>
                <div class="opinion-bar">
                    <strong>{opinion}</strong>
                    <span>목표주가 {target} · 상승여력 {upside}</span>
                </div>
                <div class="stock-card-grid">
                    <div><span>현재가</span><strong>{price}</strong></div>
                    <div><span>목표주가</span><strong>{target}</strong></div>
                    <div><span>ROE</span><strong>{roe}</strong></div>
                    <div><span>영업이익률</span><strong>{margin}</strong></div>
                    <div><span>PER</span><strong>{per}</strong></div>
                    <div><span>PBR</span><strong>{pbr}</strong></div>
                    <div><span>52주 고가</span><strong>{high}</strong></div>
                    <div><span>고가 대비</span><strong>{gap}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_favorite_cards(df: pd.DataFrame) -> None:
    for row in df.to_dict("records"):
        stock = escape(str(row["Stock"]))
        valuation = escape(str(row["Valuation"]))
        price = escape(format_price(row["Price"]))
        target = escape(format_price(row["Target Price"]))
        upside = escape(format_percent(row["Target Upside(%)"]))
        per = escape(format_multiple(row["PER"]))
        pbr = escape(format_multiple(row["PBR"]))
        psr = escape(format_multiple(row["PSR"]))
        roe = escape(format_percent(row["ROE(%)"]))
        dividend = escape(format_percent(row["Dividend Yield(%)"]))
        revenue_growth = escape(format_percent(row["Revenue Growth(%)"]))
        profit_margin = escape(format_percent(row["Profit Margin(%)"]))
        debt_equity = escape(format_plain(row["Debt/Equity"]))
        trend = "50일/200일 평균선 위" if row["Above MA50"] and row["Above MA200"] else "추세 확인 필요"
        st.markdown(
            f"""
            <div class="stock-card favorite-card">
                <div class="stock-card-title">{stock}</div>
                <div class="opinion-bar">
                    <strong>{valuation}</strong>
                    <span>현재가 {price} · 목표주가 {target} · 상승여력 {upside}</span>
                </div>
                <div class="stock-card-grid">
                    <div><span>PER</span><strong>{per}</strong></div>
                    <div><span>PBR</span><strong>{pbr}</strong></div>
                    <div><span>PSR</span><strong>{psr}</strong></div>
                    <div><span>ROE</span><strong>{roe}</strong></div>
                    <div><span>배당수익률</span><strong>{dividend}</strong></div>
                    <div><span>매출성장률</span><strong>{revenue_growth}</strong></div>
                    <div><span>순이익률</span><strong>{profit_margin}</strong></div>
                    <div><span>부채/자본</span><strong>{debt_equity}</strong></div>
                    <div><span>추세</span><strong>{escape(trend)}</strong></div>
                    <div><span>시가총액</span><strong>{escape(format_large_money(row["Market Cap"]))}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def make_metric_bar_chart(df: pd.DataFrame, metric: str, label: str) -> go.Figure:
    plot_df = df[["Ticker", metric]].dropna().sort_values(metric)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df[metric],
            y=plot_df["Ticker"],
            orientation="h",
            marker_color="#2563eb",
            text=[f"{value:.2f}" for value in plot_df[metric]],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=f"{label} 비교",
        height=max(280, 58 * len(plot_df)),
        margin=dict(l=12, r=12, t=48, b=24),
        template="plotly_white",
        xaxis_title=label,
        yaxis_title=None,
        showlegend=False,
    )
    return fig


def render_overview_cards(
    passed_count: int,
    total_count: int,
    selected_sector: str,
    min_roe: int,
    min_operating_margin: int,
    max_high_gap: int,
) -> None:
    card_items = [
        ("후보 종목", f"{passed_count}개", f"전체 {total_count}개 중 조건 통과"),
        ("선택 섹터", selected_sector, "전체 보기면 모든 섹터를 함께 비교"),
        ("수익성 기준", f"ROE {min_roe}% 이상", f"영업이익률 {min_operating_margin}% 이상"),
        ("가격 위치", f"고점 대비 {max_high_gap}% 이내", "최근 강한 흐름을 유지한 종목 우선"),
    ]
    cards = []
    for label, value, helper in card_items:
        cards.append(
            "<div class='summary-card'>"
            f"<span>{escape(label)}</span>"
            f"<strong>{escape(value)}</strong>"
            f"<small>{escape(helper)}</small>"
            "</div>"
        )
    st.markdown(f"<div class='summary-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_beginner_guide() -> None:
    st.markdown(
        """
        <div class="guide-panel">
            <div>
                <p class="eyebrow">처음 볼 때는 이렇게 보세요</p>
                <h3>이 화면은 '바로 사라'가 아니라, 더 공부할 후보를 줄여주는 필터입니다.</h3>
            </div>
            <ol>
                <li><strong>후보 종목 수</strong>가 너무 적으면 조건이 빡빡한 것입니다.</li>
                <li><strong>ROE와 영업이익률</strong>은 회사가 돈을 잘 버는지 보는 기준입니다.</li>
                <li><strong>MA50 위</strong>와 <strong>52주 고가 근처</strong>는 주가 흐름이 아직 살아있는지 보는 기준입니다.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_picks(df: pd.DataFrame) -> None:
    if df.empty:
        return

    top_rows = (
        df.sort_values(
            ["Opinion Score", "Target Upside(%)", "ROE(%)"],
            ascending=[False, False, False],
            na_position="last",
        )
        .head(3)
        .to_dict("records")
    )
    cards = []
    for row in top_rows:
        gap = row["52W High Gap(%)"]
        gap_text = "고점 근처" if pd.notna(gap) and gap >= -8 else "고점 대비 조정"
        opinion = f"{int(row['Opinion Score'])} {row['Opinion']}" if pd.notna(row["Opinion Score"]) else str(row["Opinion"])
        cards.append(
            "<div class='pick-card'>"
            "<div>"
            f"<strong>{escape(str(row['Stock']))}</strong>"
            f"<p>현재 {escape(format_price(row['Price']))} · 목표 {escape(format_price(row['Target Price']))}</p>"
            "</div>"
            "<div class='pick-tags'>"
            f"<span>{escape(opinion)}</span>"
            f"<span>상승여력 {escape(format_percent(row['Target Upside(%)']))}</span>"
            f"<span>{escape(gap_text)}</span>"
            "</div>"
            "</div>"
        )
    st.markdown(
        "<section class='quick-section'>"
        "<p class='eyebrow'>먼저 볼 후보</p>"
        f"<div class='pick-grid'>{''.join(cards)}</div>"
        "</section>",
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8fb;
        color: #0f172a;
    }
    .block-container {
        padding-top: 1.4rem;
        max-width: 1180px;
    }
    h1, h2, h3, p, label, span, div {
        color: inherit;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #164e63 55%, #0f766e 100%);
        border-radius: 10px;
        padding: 22px 22px 20px;
        color: #ffffff;
        margin-bottom: 16px;
    }
    .hero h1 {
        color: #ffffff;
        margin: 0 0 8px;
        font-size: 2rem;
        line-height: 1.2;
        letter-spacing: 0;
    }
    .hero p {
        color: #dbeafe;
        margin: 0;
        line-height: 1.55;
        font-size: 0.96rem;
    }
    .eyebrow {
        margin: 0 0 6px;
        color: #0f766e;
        font-size: 0.82rem;
        font-weight: 750;
    }
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 14px 0 18px;
    }
    .summary-card {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 8px;
        padding: 15px 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    .summary-card span {
        display: block;
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .summary-card strong {
        display: block;
        color: #0f172a;
        font-size: 1.35rem;
        line-height: 1.2;
        font-weight: 800;
        margin-bottom: 6px;
        word-break: keep-all;
    }
    .summary-card small {
        display: block;
        color: #475569;
        font-size: 0.78rem;
        line-height: 1.35;
    }
    .guide-panel {
        background: #ecfeff;
        border: 1px solid #99f6e4;
        border-radius: 8px;
        padding: 16px 18px;
        margin: 8px 0 18px;
    }
    .guide-panel h3 {
        margin: 0 0 12px;
        color: #134e4a;
        font-size: 1.08rem;
        line-height: 1.42;
    }
    .guide-panel ol {
        margin: 0;
        padding-left: 20px;
        color: #164e63;
    }
    .guide-panel li {
        margin: 6px 0;
        line-height: 1.45;
    }
    .quick-section {
        margin: 6px 0 18px;
    }
    .pick-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
    }
    .pick-card {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 8px;
        padding: 13px;
    }
    .pick-card strong {
        display: block;
        color: #0f172a;
        line-height: 1.35;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }
    .pick-card p {
        margin: 0 0 10px;
        color: #475569;
        font-size: 0.82rem;
    }
    .pick-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .pick-tags span {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        color: #334155;
        font-size: 0.72rem;
        padding: 3px 8px;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        color: #0f172a;
    }
    [data-testid="stMetric"] * { color: #0f172a !important; }
    [data-testid="stSidebar"] {
        background: #f8fafc;
        color: #0f172a;
    }
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 8px;
        color: #0f172a;
    }
    [data-testid="stExpander"] *,
    [data-testid="stWidgetLabel"] *,
    [data-testid="stSelectbox"] *,
    [data-testid="stSlider"] *,
    [data-testid="stCheckbox"] *,
    [data-testid="stButton"] button {
        color: #0f172a !important;
    }
    [data-testid="stSelectbox"] div,
    [data-testid="stSelectbox"] input {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    [data-testid="stButton"] button {
        background: #0f172a !important;
        border: 1px solid #0f172a !important;
        color: #ffffff !important;
    }
    [data-testid="stButton"] button * {
        color: #ffffff !important;
    }
    .stock-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 14px 12px;
        margin-bottom: 10px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    .stock-card-title {
        font-weight: 700;
        font-size: 0.98rem;
        color: #111827;
        margin-bottom: 10px;
        line-height: 1.35;
    }
    .opinion-bar {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-radius: 8px;
        padding: 10px 11px;
        margin-bottom: 11px;
    }
    .opinion-bar strong {
        display: block;
        color: #115e59;
        font-size: 1rem;
        margin-bottom: 3px;
    }
    .opinion-bar span {
        display: block;
        color: #0f766e;
        font-size: 0.82rem;
        line-height: 1.35;
    }
    .stock-card-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px 12px;
    }
    .stock-card-grid span {
        display: block;
        color: #64748b;
        font-size: 0.78rem;
        margin-bottom: 2px;
    }
    .stock-card-grid strong {
        color: #0f172a;
        font-size: 0.92rem;
        font-weight: 650;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.85rem;
            padding-right: 0.85rem;
            padding-top: 0.85rem;
        }
        .hero {
            padding: 18px 16px;
            margin-bottom: 12px;
        }
        .hero h1 {
            font-size: 1.38rem !important;
            line-height: 1.28 !important;
        }
        .hero p {
            font-size: 0.88rem;
        }
        h2, h3 {
            font-size: 1.1rem !important;
        }
        .summary-grid,
        .pick-grid {
            grid-template-columns: 1fr;
        }
        .summary-card {
            padding: 13px 14px;
        }
        .summary-card strong {
            font-size: 1.18rem;
        }
        .guide-panel {
            padding: 14px 14px;
        }
        [data-testid="stMetric"] {
            padding: 12px;
        }
    }
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    .stDeployButton {
        display: none;
    }
    .filter-help {
        color: #475569;
        font-size: 0.86rem;
        line-height: 1.45;
        margin: -4px 0 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <section class="hero">
        <h1>미국 주식 후보 찾기</h1>
        <p>
            어려운 지표를 한 번에 다 보려 하지 말고, 수익성 좋은 회사 중
            주가 흐름이 아직 살아있는 후보만 먼저 좁혀보세요.
            <br>Yahoo Finance 데이터 기준 · 마지막 갱신 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

sector_options = ["전체 보기"] + list(SECTOR_TICKERS.keys())

with st.expander("조건 조정하기", expanded=True):
    st.markdown(
        "<p class='filter-help'>처음에는 기본값 그대로 보고, 후보가 너무 적으면 ROE·영업이익률·52주 고가 조건을 조금 낮춰보세요.</p>",
        unsafe_allow_html=True,
    )

    selected_sector = st.selectbox(
        "어떤 산업군을 볼까요?",
        sector_options,
        help="전체 보기는 기술주, 반도체, 데이터 인프라, 보안, 클라우드 종목을 한 번에 비교합니다.",
    )

    fundamental_cols = st.columns(2)
    with fundamental_cols[0]:
        min_roe = st.slider(
            "최소 ROE: 자본 대비 이익률",
            min_value=0,
            max_value=60,
            value=15,
            step=1,
            help="높을수록 주주 자본으로 이익을 잘 내는 회사만 남깁니다.",
        )
    with fundamental_cols[1]:
        min_operating_margin = st.slider(
            "최소 영업이익률: 본업 수익성",
            min_value=0,
            max_value=60,
            value=15,
            step=1,
            help="높을수록 본업에서 마진이 좋은 회사만 남깁니다.",
        )

    momentum_cols = st.columns(2)
    with momentum_cols[0]:
        require_above_ma50 = st.checkbox(
            "현재가가 50일 평균선 위인 종목만 보기",
            value=True,
            help="최근 주가 흐름이 평균보다 강한 종목을 우선 보려는 조건입니다.",
        )
    with momentum_cols[1]:
        max_high_gap = st.slider(
            "52주 고가에서 너무 멀어진 종목 제외",
            min_value=0,
            max_value=30,
            value=15,
            step=1,
            help="15%라면 최근 1년 고점보다 15% 이상 내려간 종목은 제외합니다.",
        )

    if st.button("데이터 다시 불러오기", width="stretch"):
        st.cache_data.clear()
        st.rerun()

selected_tickers = (
    flatten_tickers(SECTOR_TICKERS)
    if selected_sector == "전체 보기"
    else SECTOR_TICKERS[selected_sector]
)

try:
    with st.spinner("주가와 펀더멘털 데이터를 불러오는 중입니다..."):
        screened_df = build_screening_table(
            selected_tickers,
            min_roe,
            min_operating_margin,
            require_above_ma50,
            max_high_gap,
        )
except Exception as exc:
    screened_df = pd.DataFrame()
    st.error(
        "일부 금융 데이터를 불러오는 중 문제가 생겼습니다. "
        "잠시 후 '데이터 다시 불러오기'를 눌러 주세요."
    )
    st.caption(f"오류 요약: {type(exc).__name__}")

render_overview_cards(
    len(screened_df),
    len(selected_tickers),
    selected_sector,
    min_roe,
    min_operating_margin,
    max_high_gap,
)
render_beginner_guide()

st.divider()
st.subheader("스크리닝 결과")

display_columns = [
    "Ticker",
    "Name",
    "Price",
    "Target Price",
    "Target Upside(%)",
    "Opinion Score",
    "Opinion",
    "ROE(%)",
    "Operating Margin(%)",
    "PER",
    "Forward PER",
    "PBR",
    "Revenue Growth(%)",
    "Debt/Equity",
    "52W High",
    "52W High Gap(%)",
]

if screened_df.empty:
    st.warning("현재 필터 조건을 통과한 종목이 없습니다. 슬라이더 조건을 조금 완화해 보세요.")
else:
    display_df = screened_df[display_columns].copy()
    numeric_cols = [
        "Price",
        "Target Price",
        "Target Upside(%)",
        "Opinion Score",
        "ROE(%)",
        "Operating Margin(%)",
        "PER",
        "Forward PER",
        "PBR",
        "Revenue Growth(%)",
        "Debt/Equity",
        "52W High",
        "52W High Gap(%)",
    ]
    display_df[numeric_cols] = display_df[numeric_cols].round(2)
    display_df.insert(0, "Stock", display_df["Ticker"] + " - " + display_df["Name"])

    table_df = display_df[
        [
            "Stock",
            "Price",
            "Target Price",
            "Target Upside(%)",
            "Opinion Score",
            "Opinion",
            "ROE(%)",
            "Operating Margin(%)",
            "PER",
            "PBR",
            "Revenue Growth(%)",
            "Debt/Equity",
            "52W High",
            "52W High Gap(%)",
        ]
    ]

    render_top_picks(display_df)

    card_tab, table_tab = st.tabs(["쉬운 카드 보기", "상세 표 보기"])
    with card_tab:
        render_mobile_cards(display_df)
    with table_tab:
        table_event = st.dataframe(
            table_df,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Stock": st.column_config.TextColumn("종목 / 회사명", width="large"),
                "Price": st.column_config.NumberColumn("현재가", format="$%.2f"),
                "Target Price": st.column_config.NumberColumn("목표주가", format="$%.2f"),
                "Target Upside(%)": st.column_config.NumberColumn("목표 대비 상승여력(%)", format="%.2f"),
                "Opinion Score": st.column_config.NumberColumn("투자의견(1-5)", format="%.0f"),
                "Opinion": st.column_config.TextColumn("의견"),
                "ROE(%)": st.column_config.NumberColumn("ROE(%)", format="%.2f"),
                "Operating Margin(%)": st.column_config.NumberColumn("영업이익률(%)", format="%.2f"),
                "PER": st.column_config.NumberColumn("PER", format="%.2f"),
                "PBR": st.column_config.NumberColumn("PBR", format="%.2f"),
                "Revenue Growth(%)": st.column_config.NumberColumn("매출성장률(%)", format="%.2f"),
                "Debt/Equity": st.column_config.NumberColumn("부채/자본", format="%.2f"),
                "52W High": st.column_config.NumberColumn("52주 고가", format="$%.2f"),
                "52W High Gap(%)": st.column_config.NumberColumn(
                    "52주 고가 대비 마이너스 괴리율(%)", format="%.2f"
                ),
            },
        )

    selected_rows = table_event.selection.rows if table_event and table_event.selection else []
    default_ticker = (
        display_df.iloc[selected_rows[0]]["Ticker"] if selected_rows else display_df.iloc[0]["Ticker"]
    )

    st.divider()
    chart_header_cols = st.columns([2, 1])
    chart_header_cols[0].subheader("상세 차트")
    ticker_labels = dict(zip(display_df["Stock"], display_df["Ticker"]))
    ticker_names = dict(zip(display_df["Ticker"], display_df["Name"]))
    default_label = display_df.loc[display_df["Ticker"] == default_ticker, "Stock"].iloc[0]
    selected_label = chart_header_cols[1].selectbox(
        "차트 종목",
        list(ticker_labels.keys()),
        index=list(ticker_labels.keys()).index(default_label),
        label_visibility="collapsed",
    )
    selected_ticker = ticker_labels[selected_label]
    selected_name = ticker_names[selected_ticker]

    chart_df = get_ticker_history(selected_ticker)
    if chart_df.empty:
        st.info(f"{selected_ticker} - {selected_name}의 차트 데이터를 불러오지 못했습니다.")
    else:
        st.plotly_chart(
            make_price_chart(selected_ticker, selected_name, chart_df),
            config={"displayModeBar": False, "responsive": True},
        )

st.divider()
st.subheader("관심종목 대시보드")
st.caption("티커를 쉼표나 공백으로 입력하고 저장하면 URL에 관심종목이 반영됩니다. 같은 링크를 열면 같은 목록을 볼 수 있습니다.")

saved_fav_text = st.query_params.get("fav", ",".join(DEFAULT_FAVORITES))
if isinstance(saved_fav_text, list):
    saved_fav_text = saved_fav_text[0] if saved_fav_text else ",".join(DEFAULT_FAVORITES)

favorite_input = st.text_input(
    "관심종목 티커",
    value=saved_fav_text,
    placeholder="예: NVDA, MSFT, AAPL, GOOGL",
    help="최대 20개까지 입력할 수 있습니다. 쉼표, 공백, 줄바꿈으로 구분해도 됩니다.",
)
favorite_tickers = parse_ticker_text(favorite_input)

fav_action_cols = st.columns([1, 1])
with fav_action_cols[0]:
    if st.button("관심종목 저장", width="stretch"):
        st.query_params["fav"] = ",".join(favorite_tickers)
        st.rerun()
with fav_action_cols[1]:
    if st.button("기본 관심종목으로 변경", width="stretch"):
        st.query_params["fav"] = ",".join(DEFAULT_FAVORITES)
        st.rerun()

if not favorite_tickers:
    st.warning("관심종목 티커를 하나 이상 입력해 주세요.")
else:
    try:
        with st.spinner("관심종목 투자지표를 불러오는 중입니다..."):
            favorite_df = build_favorites_table(favorite_tickers)
    except Exception as exc:
        favorite_df = pd.DataFrame()
        st.error("관심종목 데이터를 불러오는 중 문제가 생겼습니다. 잠시 후 다시 시도해 주세요.")
        st.caption(f"오류 요약: {type(exc).__name__}")

    if favorite_df.empty:
        st.warning("관심종목 데이터를 불러오지 못했습니다. 티커를 확인하거나 잠시 후 다시 시도해 주세요.")
    else:
        favorite_df = favorite_df.round(
            {
                "Price": 2,
                "Target Price": 2,
                "Target Upside(%)": 2,
                "Dividend Yield(%)": 2,
                "PER": 2,
                "Forward PER": 2,
                "PBR": 2,
                "PSR": 2,
                "ROE(%)": 2,
                "Gross Margin(%)": 2,
                "Operating Margin(%)": 2,
                "Profit Margin(%)": 2,
                "Revenue Growth(%)": 2,
                "Earnings Growth(%)": 2,
                "Debt/Equity": 2,
                "Current Ratio": 2,
                "Cash/Debt": 2,
                "One Year Return(%)": 2,
            }
        )

        st.markdown("#### 투자 지표")
        render_favorite_cards(favorite_df)

        st.markdown("#### 가치평가 비교")
        metric_options = {
            "PER": "PER",
            "PBR": "PBR",
            "PSR": "PSR",
            "ROE(%)": "ROE",
            "Dividend Yield(%)": "배당수익률",
        }
        selected_metric = st.segmented_control(
            "비교 지표",
            options=list(metric_options.keys()),
            format_func=lambda value: metric_options[value],
            default="PER",
        )
        if selected_metric and favorite_df[selected_metric].notna().any():
            st.plotly_chart(
                make_metric_bar_chart(favorite_df, selected_metric, metric_options[selected_metric]),
                config={"displayModeBar": False, "responsive": True},
            )
        else:
            st.info("선택한 지표의 데이터가 없습니다.")

        st.markdown("#### 재무 / 실적 요약")
        finance_columns = [
            "Stock",
            "Price",
            "Market Cap",
            "PER",
            "PBR",
            "PSR",
            "Dividend Yield(%)",
            "ROE(%)",
            "Operating Margin(%)",
            "Profit Margin(%)",
            "Revenue Growth(%)",
            "Earnings Growth(%)",
            "Debt/Equity",
            "Current Ratio",
            "Cash/Debt",
            "Total Revenue",
            "Net Income",
            "Free Cashflow",
            "Target Price",
            "Target Upside(%)",
            "Valuation",
        ]
        finance_df = favorite_df[finance_columns].copy()
        st.dataframe(
            finance_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Stock": st.column_config.TextColumn("종목 / 회사명", width="large"),
                "Price": st.column_config.NumberColumn("현재가", format="$%.2f"),
                "Market Cap": st.column_config.NumberColumn("시가총액", format="$%.0f"),
                "PER": st.column_config.NumberColumn("PER", format="%.2f"),
                "PBR": st.column_config.NumberColumn("PBR", format="%.2f"),
                "PSR": st.column_config.NumberColumn("PSR", format="%.2f"),
                "Dividend Yield(%)": st.column_config.NumberColumn("배당수익률(%)", format="%.2f"),
                "ROE(%)": st.column_config.NumberColumn("ROE(%)", format="%.2f"),
                "Operating Margin(%)": st.column_config.NumberColumn("영업이익률(%)", format="%.2f"),
                "Profit Margin(%)": st.column_config.NumberColumn("순이익률(%)", format="%.2f"),
                "Revenue Growth(%)": st.column_config.NumberColumn("매출성장률(%)", format="%.2f"),
                "Earnings Growth(%)": st.column_config.NumberColumn("이익성장률(%)", format="%.2f"),
                "Debt/Equity": st.column_config.NumberColumn("부채/자본", format="%.2f"),
                "Current Ratio": st.column_config.NumberColumn("유동비율", format="%.2f"),
                "Cash/Debt": st.column_config.NumberColumn("현금/부채", format="%.2f"),
                "Total Revenue": st.column_config.NumberColumn("매출", format="$%.0f"),
                "Net Income": st.column_config.NumberColumn("순이익", format="$%.0f"),
                "Free Cashflow": st.column_config.NumberColumn("잉여현금흐름", format="$%.0f"),
                "Target Price": st.column_config.NumberColumn("목표주가", format="$%.2f"),
                "Target Upside(%)": st.column_config.NumberColumn("상승여력(%)", format="%.2f"),
                "Valuation": st.column_config.TextColumn("가치평가"),
            },
        )

st.divider()
st.subheader("용어 설명")
term_cols = st.columns(2)
with term_cols[0]:
    st.markdown(
        """
        **ROE(%)**: 자기자본이익률입니다. 회사가 주주 자본을 활용해 얼마나 효율적으로 이익을 내는지 보여줍니다.

        **영업이익률(%)**: 매출에서 영업이익이 차지하는 비율입니다. 본업의 수익성을 보는 지표입니다.

        **52주 고가**: 최근 1년 동안 기록한 가장 높은 주가입니다.

        **목표주가**: 애널리스트들이 제시한 평균 목표 가격입니다. 예측치라서 반드시 도달한다는 뜻은 아닙니다.

        **투자의견(1-5)**: 1은 강력매도, 2는 매도추천, 3은 중립, 4는 매수추천, 5는 강력매수입니다.
        """
    )
with term_cols[1]:
    st.markdown(
        """
        **52주 고가 대비 괴리율(%)**: 현재가가 52주 고가보다 얼마나 낮은지 나타냅니다. `-10%`라면 고점 대비 10% 아래라는 뜻입니다.

        **MA50**: 최근 50거래일 종가 평균선입니다. 단기에서 중기 모멘텀을 확인할 때 씁니다.

        **MA200**: 최근 200거래일 종가 평균선입니다. 장기 추세를 확인할 때 자주 쓰입니다.

        **PER**: 주가가 이익 대비 비싼지 보는 지표입니다. 낮다고 무조건 싸고, 높다고 무조건 나쁜 것은 아닙니다.

        **PBR**: 주가가 장부가치 대비 어느 정도 평가받는지 보는 지표입니다.
        """
    )
