import math
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


def flatten_tickers(sector_map: dict[str, list[str]]) -> list[str]:
    tickers: list[str] = []
    for sector_tickers in sector_map.values():
        tickers.extend(sector_tickers)
    return sorted(set(tickers))


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


def opinion_from_info(info: dict) -> tuple[Optional[int], str]:
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

        rows.append(
            {
                "Ticker": ticker,
                "Name": info.get("longName") or info.get("shortName") or ticker,
                "ROE(%)": pct(clean_number(info.get("returnOnEquity"))),
                "Operating Margin(%)": pct(clean_number(info.get("operatingMargins"))),
                "Info 52W High": clean_number(info.get("fiftyTwoWeekHigh")),
                "Target Price": clean_number(info.get("targetMeanPrice")),
                "PER": clean_number(info.get("trailingPE")),
                "Forward PER": clean_number(info.get("forwardPE")),
                "PBR": clean_number(info.get("priceToBook")),
                "Revenue Growth(%)": pct(clean_number(info.get("revenueGrowth"))),
                "Debt/Equity": clean_number(info.get("debtToEquity")),
                "Opinion Score": opinion_from_info(info)[0],
                "Opinion": opinion_from_info(info)[1],
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

    top_rows = df.head(3).to_dict("records")
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
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("필터 컨트롤")

    sector_options = ["전체 보기"] + list(SECTOR_TICKERS.keys())
    selected_sector = st.selectbox("섹터 선택", sector_options)

    st.divider()
    st.subheader("펀더멘털")
    min_roe = st.slider("최소 ROE (%)", min_value=0, max_value=60, value=15, step=1)
    min_operating_margin = st.slider(
        "최소 영업이익률 (%)", min_value=0, max_value=60, value=15, step=1
    )

    st.divider()
    st.subheader("모멘텀")
    require_above_ma50 = st.checkbox("현재가가 50일 이동평균선 위", value=True)
    max_high_gap = st.slider(
        "52주 고가 대비 최대 괴리율 (%)", min_value=0, max_value=30, value=15, step=1
    )

    st.divider()
    if st.button("캐시 새로고침", width="stretch"):
        st.cache_data.clear()
        st.rerun()


selected_tickers = (
    flatten_tickers(SECTOR_TICKERS)
    if selected_sector == "전체 보기"
    else SECTOR_TICKERS[selected_sector]
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

with st.spinner("주가와 펀더멘털 데이터를 불러오는 중입니다..."):
    screened_df = build_screening_table(
        selected_tickers,
        min_roe,
        min_operating_margin,
        require_above_ma50,
        max_high_gap,
    )

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
