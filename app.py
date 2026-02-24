"""
Grant's Notebook — Live Portfolio Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, time as dtime
from pathlib import Path
import pytz

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grant's Notebook",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Password Gate ──────────────────────────────────────────────────────────────
def check_password():
    """Simple password authentication using Streamlit secrets."""
    def password_entered():
        if st.session_state.get("password") == st.secrets["passwords"]["grant"]:
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        '<div style="display:flex; justify-content:center; align-items:center; min-height:60vh;">'
        '<div style="text-align:center;">'
        '<p style="font-size:2.5rem; margin-bottom:0.25rem;">📓</p>'
        '<h2 style="margin-bottom:1.5rem; color:#f1f5f9;">Grant\'s Notebook</h2>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.text_input("Enter password", type="password", key="password", on_change=password_entered)
    if "authenticated" in st.session_state and not st.session_state["authenticated"]:
        st.error("Incorrect password.")
    st.stop()

check_password()

# ─── Auto-refresh: every 15 seconds ────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=15_000, limit=None, key="live_refresh")
except ImportError:
    pass  # graceful fallback if package missing

# ─── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
HOLDINGS_PATH = DATA_DIR / "holdings.csv"
BALANCE_PATH = DATA_DIR / "portfolio_balance.csv"
CLOSES_PATH = DATA_DIR / "daily_closes.csv"

ET = pytz.timezone("US/Eastern")

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.2rem;
    }
    .dashboard-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .dashboard-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.25rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* KPI cards */
    .kpi-row {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    .kpi-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        flex: 1;
        min-width: 160px;
    }
    .kpi-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
    }
    .kpi-sub {
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.15rem;
    }
    .green { color: #22c55e; }
    .red   { color: #ef4444; }
    .neutral { color: #e5e7eb; }

    /* Market status badge */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
    }
    .badge-live {
        background: rgba(34,197,94,0.15);
        color: #22c55e;
        border: 1px solid rgba(34,197,94,0.3);
    }
    .badge-closed {
        background: rgba(107,114,128,0.15);
        color: #6b7280;
        border: 1px solid rgba(107,114,128,0.3);
    }

    /* Table styling */
    .dataframe-container {
        border-radius: 10px;
        overflow: hidden;
    }
    div[data-testid="stDataFrame"] > div {
        border-radius: 10px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ───────────────────────────────────────────────────────────

def is_market_open() -> bool:
    """Check if US market is currently open (9:30 AM - 4:00 PM ET, weekdays)."""
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    market_open = dtime(9, 30)
    market_close = dtime(16, 0)
    return market_open <= now_et.time() <= market_close


def color_class(val: float) -> str:
    if val > 0:
        return "green"
    elif val < 0:
        return "red"
    return "neutral"


def fmt_currency(val: float, decimals: int = 2) -> str:
    if pd.isna(val):
        return "—"
    sign = "-" if val < 0 else ""
    return f"{sign}${abs(val):,.{decimals}f}"


def fmt_pct(val: float) -> str:
    if pd.isna(val):
        return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:,.2f}%"


def render_kpi(col, label: str, value_str: str, sub_str: str = "", delta_val: float = 0):
    with col:
        color = "#22c55e" if delta_val > 0 else "#ef4444" if delta_val < 0 else "#e5e7eb"
        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1f2937; border-radius:10px; padding:1rem 1.25rem;">
            <div style="font-size:0.7rem; font-weight:500; color:#6b7280; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.3rem;">{label}</div>
            <div style="font-size:1.35rem; font-weight:700; font-family:'JetBrains Mono',monospace; color:{color};">{value_str}</div>
            <div style="font-size:0.75rem; font-family:'JetBrains Mono',monospace; color:{color}; margin-top:0.15rem;">{sub_str}</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Load Holdings ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_holdings() -> pd.DataFrame:
    df = pd.read_csv(HOLDINGS_PATH)
    df["date_bought"] = pd.to_datetime(df["date_bought"], format="%m/%d/%Y")
    return df


# ─── Fetch Live Prices ─────────────────────────────────────────────────────────

@st.cache_data(ttl=14)
def fetch_live_prices(tickers: tuple) -> dict:
    """Batch-fetch current prices via yfinance."""
    stock_tickers = [t for t in tickers if t != "Foreign Stock"]
    prices = {}
    if stock_tickers:
        data = yf.download(
            stock_tickers,
            period="1d",
            interval="1m",
            progress=False,
            threads=True,
        )
        if not data.empty:
            close_col = data["Close"] if "Close" in data.columns else data[("Close",)]
            if isinstance(close_col, pd.Series):
                prices[stock_tickers[0]] = float(close_col.dropna().iloc[-1])
            else:
                for t in stock_tickers:
                    if t in close_col.columns:
                        series = close_col[t].dropna()
                        if not series.empty:
                            prices[t] = float(series.iloc[-1])
    return prices


# ─── Load Historical Data ──────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_balance_history() -> pd.DataFrame:
    df = pd.read_csv(BALANCE_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=300)
def load_daily_closes() -> pd.DataFrame:
    df = pd.read_csv(CLOSES_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─── Build Portfolio DataFrame ──────────────────────────────────────────────────

def build_portfolio(holdings: pd.DataFrame, live_prices: dict, closes_df: pd.DataFrame) -> pd.DataFrame:
    df = holdings.copy()

    # Current prices: live for stocks, static for Foreign Stock
    df["current_price"] = df["ticker"].map(live_prices)
    foreign_mask = df["ticker"] == "Foreign Stock"
    df.loc[foreign_mask, "current_price"] = df.loc[foreign_mask, "buy_price"]

    # Core metrics
    df["total_buy"] = df["shares"] * df["buy_price"]
    df["total_current"] = df["shares"] * df["current_price"]
    df["pnl_total"] = df["total_current"] - df["total_buy"]
    df["return_pct"] = (df["pnl_total"] / df["total_buy"]) * 100

    # Historical reference prices
    if not closes_df.empty:
        last_row = closes_df.iloc[-1]
        today = pd.Timestamp.today().normalize()
        current_month_start = today.replace(day=1)
        current_year_start = today.replace(month=1, day=1)

        # Yesterday (last recorded close)
        for t in df["ticker"]:
            if t in closes_df.columns and t != "Foreign Stock":
                prev_val = last_row.get(t)
                if pd.notna(prev_val):
                    df.loc[df["ticker"] == t, "prev_close"] = float(prev_val)

        # MTD start: first close of current month
        mtd_rows = closes_df[closes_df["date"] >= current_month_start]
        if not mtd_rows.empty:
            mtd_row = mtd_rows.iloc[0]
            for t in df["ticker"]:
                if t in closes_df.columns and t != "Foreign Stock":
                    val = mtd_row.get(t)
                    if pd.notna(val):
                        df.loc[df["ticker"] == t, "mtd_start"] = float(val)

        # YTD start: first close of current year
        ytd_rows = closes_df[closes_df["date"] >= current_year_start]
        if not ytd_rows.empty:
            ytd_row = ytd_rows.iloc[0]
            for t in df["ticker"]:
                if t in closes_df.columns and t != "Foreign Stock":
                    val = ytd_row.get(t)
                    if pd.notna(val):
                        df.loc[df["ticker"] == t, "ytd_start"] = float(val)

    # Fill Foreign Stock reference prices
    df.loc[foreign_mask, "prev_close"] = df.loc[foreign_mask, "buy_price"]
    df.loc[foreign_mask, "mtd_start"] = df.loc[foreign_mask, "buy_price"]
    df.loc[foreign_mask, "ytd_start"] = df.loc[foreign_mask, "buy_price"]

    # Period PnL calculations
    for ref_col, pnl_col, ret_col in [
        ("prev_close", "pnl_1d", "return_1d_pct"),
        ("mtd_start", "pnl_mtd", "return_mtd_pct"),
        ("ytd_start", "pnl_ytd", "return_ytd_pct"),
    ]:
        if ref_col in df.columns:
            df[f"price_delta_{ref_col}"] = df["current_price"] - df[ref_col]
            df[pnl_col] = df["shares"] * df[f"price_delta_{ref_col}"]
            df[ret_col] = (df[f"price_delta_{ref_col}"] / df[ref_col]) * 100
        else:
            df[pnl_col] = np.nan
            df[ret_col] = np.nan

    # Holding period
    today_norm = pd.Timestamp.today().normalize()
    df["hold_days"] = (today_norm - df["date_bought"]).dt.days
    df["hold_years"] = df["hold_days"] / 365.25

    return df


# ─── MAIN APP ───────────────────────────────────────────────────────────────────

holdings = load_holdings()
closes_df = load_daily_closes()
balance_hist = load_balance_history()

# Fetch live prices
stock_tickers = tuple(holdings[holdings["ticker"] != "Foreign Stock"]["ticker"].tolist())
live_prices = fetch_live_prices(stock_tickers)

# Build portfolio
df = build_portfolio(holdings, live_prices, closes_df)

# ─── Header ─────────────────────────────────────────────────────────────────────
now_et = datetime.now(ET)
market_open = is_market_open()
badge_class = "badge-live" if market_open else "badge-closed"
badge_text = "● LIVE" if market_open else "● MARKET CLOSED"

st.markdown(f"""
<div class="dashboard-header">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <p class="dashboard-title">📓 Grant's Notebook</p>
            <p style="font-size:0.75rem; color:#94a3b8; margin-top:0.1rem; font-style:italic;">Only Big G allowed</p>
            <p class="dashboard-subtitle">Last refreshed: {now_et.strftime("%b %d, %Y  %I:%M:%S %p")} ET</p>
        </div>
        <div>
            <span class="badge {badge_class}">{badge_text}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI Cards ──────────────────────────────────────────────────────────────────
total_value = df["total_current"].sum()
total_pnl = df["pnl_total"].sum()
pnl_1d = df["pnl_1d"].sum() if "pnl_1d" in df.columns else 0
pnl_mtd = df["pnl_mtd"].sum() if "pnl_mtd" in df.columns else 0
pnl_ytd = df["pnl_ytd"].sum() if "pnl_ytd" in df.columns else 0

start_of_day_val = np.nansum(df["shares"] * df["prev_close"]) if "prev_close" in df.columns else total_value
ret_1d = (pnl_1d / start_of_day_val * 100) if start_of_day_val else 0
mtd_start_val = np.nansum(df["shares"] * df["mtd_start"]) if "mtd_start" in df.columns else total_value
ret_mtd = (pnl_mtd / mtd_start_val * 100) if mtd_start_val else 0
ytd_start_val = np.nansum(df["shares"] * df["ytd_start"]) if "ytd_start" in df.columns else total_value
ret_ytd = (pnl_ytd / ytd_start_val * 100) if ytd_start_val else 0

agg_return = (total_pnl / df["total_buy"].sum()) * 100

kpi_cols = st.columns(5)
render_kpi(kpi_cols[0], "Portfolio Value", fmt_currency(total_value, 0), f"Cost basis: {fmt_currency(df['total_buy'].sum(), 0)}", 1)
render_kpi(kpi_cols[1], "Total PnL", fmt_currency(total_pnl, 0), fmt_pct(agg_return), total_pnl)
render_kpi(kpi_cols[2], "PnL (1D)", fmt_currency(pnl_1d, 0), fmt_pct(ret_1d), pnl_1d)
render_kpi(kpi_cols[3], "PnL (MTD)", fmt_currency(pnl_mtd, 0), fmt_pct(ret_mtd), pnl_mtd)
render_kpi(kpi_cols[4], "PnL (YTD)", fmt_currency(pnl_ytd, 0), fmt_pct(ret_ytd), pnl_ytd)

# ─── Holdings Table ─────────────────────────────────────────────────────────────
st.markdown("### Holdings")

display_df = df[[
    "ticker", "name", "shares", "buy_price", "current_price",
    "total_buy", "total_current", "pnl_total", "return_pct",
    "pnl_1d", "return_1d_pct", "pnl_mtd", "return_mtd_pct",
    "pnl_ytd", "return_ytd_pct",
]].copy()

display_df.columns = [
    "Ticker", "Name", "Shares", "Buy Price", "Current Price",
    "Total Cost", "Current Value", "$ PnL", "% Return",
    "$ PnL (1D)", "Return (1D) %", "$ PnL (MTD)", "Return (MTD) %",
    "$ PnL (YTD)", "Return (YTD) %",
]

# Sort by current value descending
display_df = display_df.sort_values("Current Value", ascending=False).reset_index(drop=True)

st.dataframe(
    display_df.style.format({
        "Shares": "{:,.0f}",
        "Buy Price": "${:,.2f}",
        "Current Price": "${:,.2f}",
        "Total Cost": "${:,.0f}",
        "Current Value": "${:,.0f}",
        "$ PnL": "${:,.0f}",
        "% Return": "{:,.2f}%",
        "$ PnL (1D)": "${:,.0f}",
        "Return (1D) %": "{:,.2f}%",
        "$ PnL (MTD)": "${:,.0f}",
        "Return (MTD) %": "{:,.2f}%",
        "$ PnL (YTD)": "${:,.0f}",
        "Return (YTD) %": "{:,.2f}%",
    }).map(
        lambda v: f"color: {'#22c55e' if v > 0 else '#ef4444' if v < 0 else '#e5e7eb'}; font-weight: 600"
        if isinstance(v, (int, float)) and not pd.isna(v) else "",
        subset=["$ PnL", "% Return", "$ PnL (1D)", "Return (1D) %",
                "$ PnL (MTD)", "Return (MTD) %", "$ PnL (YTD)", "Return (YTD) %"],
    ),
    use_container_width=True,
    height=600,
    hide_index=True,
)

# ─── Charts ─────────────────────────────────────────────────────────────────────
st.markdown("### Analytics")
tab_balance, tab_pnl, tab_alloc, tab_returns, tab_age = st.tabs([
    "📈 Balance History", "💰 Top PnL", "🍩 Allocation", "📊 Returns", "⏳ Holding Age"
])

CHART_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(0,0,0,0)"
CHART_COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
                "#06b6d4", "#ec4899", "#f97316", "#14b8a6", "#a855f7"]

# --- Balance History ---
with tab_balance:
    if not balance_hist.empty:
        fig_bal = go.Figure()
        fig_bal.add_trace(go.Scatter(
            x=balance_hist["date"],
            y=balance_hist["balance"],
            mode="lines+markers",
            marker=dict(size=5, color="#22c55e"),
            line=dict(color="#22c55e", width=2),
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.08)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Balance: $%{y:,.0f}<extra></extra>",
        ))
        fig_bal.update_layout(
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            title=dict(text="Portfolio Balance Over Time", font=dict(size=20)),
            xaxis_title="Date",
            yaxis_title="Balance ($)",
            yaxis_tickprefix="$",
            yaxis_tickformat=",",
            height=500,
            margin=dict(l=60, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_bal, use_container_width=True)
    else:
        st.info("No balance history recorded yet.")

# --- Top PnL Contributors ---
with tab_pnl:
    pnl_period = st.selectbox("Period", ["1D", "MTD", "YTD"], key="pnl_period")
    col_map = {"1D": "pnl_1d", "MTD": "pnl_mtd", "YTD": "pnl_ytd"}
    pnl_col = col_map[pnl_period]

    if pnl_col in df.columns:
        pnl_data = df[["ticker", pnl_col]].dropna().copy()
        pnl_data = pnl_data[pnl_data["ticker"] != "Foreign Stock"]

        # Top 10 winners and losers
        top_winners = pnl_data.nlargest(10, pnl_col)
        top_losers = pnl_data.nsmallest(5, pnl_col)
        pnl_show = pd.concat([top_winners, top_losers]).drop_duplicates().sort_values(pnl_col, ascending=True)

        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in pnl_show[pnl_col]]

        fig_pnl = go.Figure(go.Bar(
            x=pnl_show[pnl_col],
            y=pnl_show["ticker"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>PnL: $%{x:,.0f}<extra></extra>",
        ))
        fig_pnl.update_layout(
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            title=dict(text=f"Top PnL Contributors ({pnl_period})", font=dict(size=20)),
            xaxis_title="$ PnL",
            xaxis_tickprefix="$",
            xaxis_tickformat=",",
            height=500,
            margin=dict(l=80, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

# --- Allocation Pie ---
with tab_alloc:
    alloc = df[["ticker", "total_current"]].copy()
    alloc = alloc[alloc["ticker"] != "Foreign Stock"]
    alloc = alloc.sort_values("total_current", ascending=False)

    top_n = 10
    top_alloc = alloc.head(top_n).copy()
    other_val = alloc.iloc[top_n:]["total_current"].sum()
    if other_val > 0:
        top_alloc = pd.concat([
            top_alloc,
            pd.DataFrame([{"ticker": "Other", "total_current": other_val}])
        ], ignore_index=True)

    fig_pie = px.pie(
        top_alloc,
        names="ticker",
        values="total_current",
        hole=0.45,
        color_discrete_sequence=CHART_COLORS,
    )
    fig_pie.update_traces(
        textinfo="label+percent",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<br>Value: $%{value:,.0f}<extra></extra>",
        marker=dict(line=dict(color="#0a0f1a", width=2)),
    )

    total_display = top_alloc["total_current"].sum()
    fig_pie.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        title=dict(text="Portfolio Allocation", font=dict(size=20)),
        height=550,
        margin=dict(l=20, r=160, t=60, b=20),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        annotations=[dict(
            text=f"Total<br>${total_display:,.0f}",
            x=0.5, y=0.5, font_size=14, showarrow=False, font_color="#e5e7eb",
        )],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Returns Bar ---
with tab_returns:
    ret_data = df[["ticker", "return_pct"]].dropna().copy()
    ret_data = ret_data[ret_data["ticker"] != "Foreign Stock"]
    ret_data = ret_data.sort_values("return_pct", ascending=True)

    colors_ret = ["#22c55e" if v >= 0 else "#ef4444" for v in ret_data["return_pct"]]

    fig_ret = go.Figure(go.Bar(
        x=ret_data["return_pct"],
        y=ret_data["ticker"],
        orientation="h",
        marker_color=colors_ret,
        hovertemplate="<b>%{y}</b><br>Return: %{x:,.1f}%<extra></extra>",
    ))
    fig_ret.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        title=dict(text="Total % Return by Ticker", font=dict(size=20)),
        xaxis_title="% Return",
        xaxis_ticksuffix="%",
        height=max(500, len(ret_data) * 22),
        margin=dict(l=80, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_ret, use_container_width=True)

# --- Holding Age ---
with tab_age:
    age_df = df[["ticker", "name", "date_bought", "hold_years", "shares",
                 "buy_price", "current_price", "pnl_total", "return_pct"]].copy()
    age_df = age_df.sort_values("hold_years", ascending=False).reset_index(drop=True)

    col_oldest, col_newest = st.columns(2)

    with col_oldest:
        st.markdown("**🏛️ Top 10 Oldest Positions**")
        oldest = age_df.head(10).copy()
        oldest["date_bought"] = oldest["date_bought"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            oldest[["ticker", "name", "date_bought", "hold_years", "shares", "return_pct"]].rename(columns={
                "ticker": "Ticker", "name": "Name", "date_bought": "Date Bought",
                "hold_years": "Years Held", "shares": "Shares", "return_pct": "% Return",
            }).style.format({"Years Held": "{:.1f}", "Shares": "{:,.0f}", "% Return": "{:,.1f}%"}),
            use_container_width=True,
            hide_index=True,
        )

    with col_newest:
        st.markdown("**🆕 Top 10 Newest Positions**")
        newest = age_df.tail(10).sort_values("hold_years").copy()
        newest["date_bought"] = newest["date_bought"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            newest[["ticker", "name", "date_bought", "hold_years", "shares", "return_pct"]].rename(columns={
                "ticker": "Ticker", "name": "Name", "date_bought": "Date Bought",
                "hold_years": "Years Held", "shares": "Shares", "return_pct": "% Return",
            }).style.format({"Years Held": "{:.1f}", "Shares": "{:,.0f}", "% Return": "{:,.1f}%"}),
            use_container_width=True,
            hide_index=True,
        )

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="text-align:center; color:#4b5563; font-size:0.75rem; font-family: JetBrains Mono, monospace;">'
    f'Grant\'s Notebook · Auto-refreshing every 15s · Data via Yahoo Finance'
    f'</p>',
    unsafe_allow_html=True,
)
