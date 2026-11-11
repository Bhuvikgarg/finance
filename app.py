"""
Market Predictor & Quantitative Intelligence Dashboard
Tri-Class Probabilities + Unsupervised Regimes + Live News Sentiment NLP +
Target Projections + Behavior Attribution + Fakeout/Anomaly Auditing.
Supports ETERNAL.NS, all NIFTY 50 stocks, and custom tickers.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from src.data_loader import load_market_dataset
from src.features import build_feature_dataset
from src.regimes import add_regime_features
from src.model import train_time_series_model
from src.evaluate import evaluate_predictions, backtest_probability_strategy
from src.audit_recent import audit_last_n_days
from src.news_sentiment import fetch_live_news_sentiment
from src.anomaly_detector import (
    compute_target_projections,
    analyze_trend_structure,
    explain_market_behavior,
    detect_fakeouts_and_anomalies,
)

# Streamlit Page Setup
st.set_page_config(
    page_title="Quant Intelligence & Regime Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data" / "raw"

# Comprehensive Stock Presets (NIFTY 50 + Top Growth Stocks)
NIFTY_STOCKS = {
    "⭐ FEATURED: Eternal Limited (Zomato)": "ETERNAL.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "State Bank of India (SBIN)": "SBIN.NS",
    "ITC Limited": "ITC.NS",
    "Larsen & Toubro (LT)": "LT.NS",
    "Hindustan Unilever (HINDUNILVR)": "HINDUNILVR.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Titan Company": "TITAN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "NTPC": "NTPC.NS",
    "Oil & Natural Gas Corp (ONGC)": "ONGC.NS",
    "Power Grid Corp": "POWERGRID.NS",
    "Wipro": "WIPRO.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Coal India": "COALINDIA.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Mahindra & Mahindra (M&M)": "M&M.NS",
    "Bharat Electronics (BEL)": "BEL.NS",
    "Nestle India": "NESTLEIND.NS",
    "Hindustan Aeronautics (HAL)": "HAL.NS",
    "DLF Limited": "DLF.NS",
    "Grasim Industries": "GRASIM.NS",
    "Varun Beverages (VBL)": "VBL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Cipla": "CIPLA.NS",
    "Trent Limited": "TRENT.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Apple Inc. (US)": "AAPL",
    "NVIDIA Corp (US)": "NVDA",
    "Tesla Inc (US)": "TSLA",
    "Custom Ticker (Enter Symbol)": "CUSTOM",
}

# ----------------- SIDEBAR -----------------
st.sidebar.title("🎛️ Control Panel")

# Stock Selection
st.sidebar.subheader("1. Equity Selection")
selected_label = st.sidebar.selectbox("Select Asset / NIFTY Stock", list(NIFTY_STOCKS.keys()), index=0)

if NIFTY_STOCKS[selected_label] == "CUSTOM":
    raw_sym = st.sidebar.text_input("Enter NSE/Global Ticker (e.g. ETERNAL, TATAPOWER)", value="ETERNAL.NS").strip().upper()
    # Smart normalizer: if user enters ETERNAL or ZOMATO or typical Indian stock without suffix
    if raw_sym in ["ETERNAL", "ZOMATO"]:
        stock_ticker = "ETERNAL.NS"
    elif "." not in raw_sym and "^" not in raw_sym and raw_sym not in ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "SPY", "QQQ"]:
        stock_ticker = f"{raw_sym}.NS"
    else:
        stock_ticker = raw_sym
else:
    stock_ticker = NIFTY_STOCKS[selected_label]

bench_presets = ["^NSEI", "^BSESN", "SPY", "QQQ"]
benchmark_ticker = st.sidebar.selectbox("Market Context Benchmark", bench_presets, index=0)

# Refresh News & Re-Analyze Trigger
refresh_news_btn = st.sidebar.button("🔄 Refresh Live News & Re-Analyze", use_container_width=True)

# Triple Barrier Parameters
st.sidebar.subheader("2. Triple Barrier Risk Settings")
horizon = st.sidebar.slider("Max Holding Window (Days)", min_value=2, max_value=10, value=5, step=1)
pt_mult = st.sidebar.slider("Profit Target (× ATR)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
sl_mult = st.sidebar.slider("Stop Loss Barrier (× ATR)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

# Advanced ML Settings
st.sidebar.subheader("3. Model & Regimes")
use_regimes = st.sidebar.checkbox("Include Unsupervised Regimes (GMM)", value=True)
train_split = st.sidebar.slider("Chronological Train Split", min_value=0.60, max_value=0.90, value=0.80, step=0.05)
force_data_refresh = st.sidebar.checkbox("Force Redownload Fresh Price Bars", value=False)


# Caching Data & Info
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_market_data(stock_sym: str, bench_sym: str, force: bool):
    return load_market_dataset(
        stock_ticker=stock_sym,
        benchmark_ticker=bench_sym,
        start_date="2020-01-01",
        cache_dir=CACHE_DIR,
        force_download=force,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_company_info(ticker: str):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


# News Sentiment Caching (TTL 5 minutes, invalidated when button clicked)
@st.cache_data(ttl=300, show_spinner=False)
def get_cached_news_sentiment(ticker: str, comp_name: str, cache_buster: int):
    return fetch_live_news_sentiment(ticker, company_name=comp_name)


# Manage cache buster for news
if "news_cache_counter" not in st.session_state:
    st.session_state["news_cache_counter"] = 0

if refresh_news_btn:
    st.session_state["news_cache_counter"] += 1
    st.toast("Refreshed live headlines & sentiment analysis!", icon="📰")

# ----------------- PIPELINE EXECUTION -----------------
with st.spinner(f"Ingesting live market data and analyzing {stock_ticker}..."):
    try:
        stock_df, bench_df = get_cached_market_data(stock_ticker, benchmark_ticker, force_data_refresh)
    except Exception as e:
        st.error(f"Error fetching data for {stock_ticker}: {e}")
        st.stop()

company_info = get_cached_company_info(stock_ticker)
company_name = company_info.get("shortName") or company_info.get("longName") or stock_ticker

# Fetch Live News & Sentiment
news_data = get_cached_news_sentiment(
    stock_ticker, company_name, st.session_state["news_cache_counter"]
)

# Feature Engineering
df, feature_cols = build_feature_dataset(
    stock_df, bench_df, forecast_horizon=horizon, pt_mult=pt_mult, sl_mult=sl_mult
)

# Unsupervised Regimes
detector = None
if use_regimes:
    df, detector, regime_cols = add_regime_features(df, train_ratio=train_split)
    feature_cols = feature_cols + regime_cols

# Train Calibrated Model
model, train_df, test_df, importance_df = train_time_series_model(
    df, feature_cols, target_col="target_class", train_ratio=train_split
)

# Evaluations & Backtest
metrics = evaluate_predictions(test_df)
bt_df = backtest_probability_strategy(test_df)
audit_table = audit_last_n_days(df, feature_cols, model, n_days=3)

# Executive Analytics
projections = compute_target_projections(df, company_info)
trend_info = analyze_trend_structure(df)
bench_ret_1d = float(bench_df["close"].pct_change().iloc[-1]) if len(bench_df) > 1 else 0.0
behavior_info = explain_market_behavior(df, news_sentiment=news_data, bench_ret_1d=bench_ret_1d)
anomaly_info = detect_fakeouts_and_anomalies(df, news_sentiment=news_data)

# Latest Probability Forecast
latest_feats = df[feature_cols].iloc[[-1]]
latest_probs = model.predict_proba(latest_feats)[0]
p_chop, p_bull, p_bear = latest_probs[0], latest_probs[1], latest_probs[2]
net_edge = p_bull - p_bear
latest_date = df.index[-1].strftime("%Y-%m-%d")
latest_close = float(df["close"].iloc[-1])

# Signal Badge
if p_bull >= 0.38 and net_edge >= 0.08:
    signal_badge = "🟢 BUY / LONG"
elif p_bear >= 0.40 or p_bear > p_bull:
    signal_badge = "🔴 AVOID / BEARISH"
else:
    signal_badge = "⚪ CHOP / CASH"

# ----------------- APP HEADER -----------------
st.title(f"⚡ {company_name} (`{stock_ticker}`)")
st.markdown(
    f"**Context Benchmark:** `{benchmark_ticker}` | **Data Points:** {len(df):,} sessions | **Status:** Real-time Analysis Active"
)

# ----------------- TOP KPI BANNER -----------------
k1, k2, k3, k4, k5 = st.columns([1.2, 1.2, 1.2, 1.4, 1.5])
k1.metric(f"Price ({latest_date})", f"₹{latest_close:,.2f}" if ".NS" in stock_ticker else f"${latest_close:,.2f}", f"Trend: {trend_info['trend_status']}")
k2.metric("P(Bull: Target Hit)", f"{p_bull * 100:.1f}%", f"Edge: {net_edge * 100:+.1f}%")
k3.metric("P(Bear: Stop Hit)", f"{p_bear * 100:.1f}%")
k4.metric("Live Signal Status", signal_badge)
k5.metric("News Sentiment", news_data["sentiment_label"].split()[0] + f" ({news_data['sentiment_index']:+d}/100)")

# Prominent Anomaly Banner Alert
if anomaly_info["severity"] == "DANGER":
    st.error(f"🚨 **ANOMALY ALERT:** {anomaly_info['anomalies'][0]['description']}")
elif anomaly_info["severity"] == "SUSPICIOUS":
    st.warning(f"⚠️ **FAKE / TRAP ALERT:** {anomaly_info['anomalies'][0]['description']}")
else:
    st.success(f"🛡️ **PRICE INTEGRITY VERIFIED:** {anomaly_info['anomalies'][0]['description']}")

st.divider()

# ----------------- TABS -----------------
tabs = st.tabs([
    "🧭 Executive Intelligence & Why is it Behaving?",
    "📰 Live News & Sentiment NLP",
    "🏢 Company Fundamentals",
    "🔍 Last 3 Days Walk-Forward Audit",
    "🔮 Tri-Class Probabilities",
    "📈 Strategy Backtest",
    "📊 Market & Flow Charts",
    "🌊 Unsupervised Regimes",
])

# ----------------- TAB 1: EXECUTIVE INTELLIGENCE -----------------
with tabs[0]:
    st.subheader("🎯 Target Projections: How Far Would It Go?")
    c_tgt1, c_tgt2, c_tgt3, c_tgt4 = st.columns(4)
    c_tgt1.metric(
        "Upside Target 1 (1.5σ ATR)",
        f"₹{projections['upside_target_1']:,.2f}",
        f"{projections['upside_target_1_pct']:+.1f}%",
    )
    c_tgt2.metric(
        "Upside Target 2 (2.5σ Breakout)",
        f"₹{projections['upside_target_2']:,.2f}",
        f"{projections['upside_target_2_pct']:+.1f}%",
    )
    c_tgt3.metric(
        "Downside Stop / Support 1",
        f"₹{projections['downside_support_1']:,.2f}",
        f"{projections['downside_support_1_pct']:+.1f}%",
        delta_color="inverse",
    )
    if projections["analyst_mean"]:
        c_tgt4.metric(
            f"Analyst Consensus ({projections['analyst_recommendation']})",
            f"₹{projections['analyst_mean']:,.2f}",
            f"{projections['analyst_mean_pct']:+.1f}%",
        )
    else:
        c_tgt4.metric("Pivot R1 Resistance", f"₹{projections['pivot_r1']:,.2f}")

    st.markdown("---")

    col_why, col_fake = st.columns([1.2, 1.2])

    with col_why:
        st.subheader("🔬 Why is it Behaving Now?")
        st.markdown(f"**Macro Trend:** `{trend_info['macro_trend']}`")
        st.markdown(f"**RSI (14):** `{trend_info['rsi_14']:.1f}` | **50-day SMA Distance:** `{trend_info['dist_sma_50_pct']:+.1f}%`")
        st.markdown("#### Driver Decomposition:")
        for r in behavior_info["reasons"]:
            st.markdown(f"- {r}")

    with col_fake:
        st.subheader("🛡️ Is There Any Fake Thing Going On?")
        st.markdown(f"**Integrity Verdict:** `{anomaly_info['status_badge']}`")
        for an in anomaly_info["anomalies"]:
            if an["level"] == "DANGER":
                st.error(f"**{an['type']}**\n\n{an['description']}")
            elif an["level"] in ["WARNING", "ALERT"]:
                st.warning(f"**{an['type']}**\n\n{an['description']}")
            else:
                st.success(f"**{an['type']}**\n\n{an['description']}")

# ----------------- TAB 2: LIVE NEWS & SENTIMENT -----------------
with tabs[1]:
    st.subheader(f"📰 Live Financial News & Sentiment NLP: {company_name}")
    col_n1, col_n2, col_n3 = st.columns([1, 1, 1])
    col_n1.metric("Live Sentiment Score", f"{news_data['sentiment_index']:+d} / 100")
    col_n2.metric("Headlines Scanned", f"{len(news_data['articles'])} articles")
    col_n3.metric("Bullish vs Bearish Headlines", f"{news_data['bullish_count']} Bull / {news_data['bearish_count']} Bear")

    st.markdown("#### Real-time Headlines (Click to Read):")
    if not news_data["articles"]:
        st.info("No recent news headlines returned for this ticker.")
    else:
        for art in news_data["articles"]:
            col_b, col_t = st.columns([1, 4])
            with col_b:
                if art["sentiment"] == "Bullish":
                    st.markdown(f"🟢 **Bullish** (`{art['compound']:+.2f}`)")
                elif art["sentiment"] == "Bearish":
                    st.markdown(f"🔴 **Bearish** (`{art['compound']:+.2f}`)")
                else:
                    st.markdown(f"⚪ **Neutral** (`{art['compound']:+.2f}`)")
                st.caption(art["pub_date"][:16] if art["pub_date"] else "")
            with col_t:
                st.markdown(f"[{art['title']}]({art['link']})")
                st.caption(f"Source: {art['source']}")

# ----------------- TAB 3: FUNDAMENTALS -----------------
with tabs[2]:
    st.subheader(f"🏢 Company Fundamentals: {company_name}")
    if not company_info:
        st.info("No fundamental valuation metrics available for this ticker.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        pe = company_info.get("trailingPE")
        fwd_pe = company_info.get("forwardPE")
        pb = company_info.get("priceToBook")
        ev_ebitda = company_info.get("enterpriseToEbitda")
        
        f1.metric("Trailing P/E", f"{pe:.1f}" if pe else "N/A")
        f2.metric("Forward P/E", f"{fwd_pe:.1f}" if fwd_pe else "N/A")
        f3.metric("Price / Book", f"{pb:.2f}" if pb else "N/A")
        f4.metric("EV / EBITDA", f"{ev_ebitda:.1f}" if ev_ebitda else "N/A")

        f5, f6, f7, f8 = st.columns(4)
        mcap = company_info.get("marketCap")
        roe = company_info.get("returnOnEquity")
        op_margin = company_info.get("operatingMargins")
        rec = company_info.get("recommendationKey")

        f5.metric("Market Cap", f"₹{mcap/1e7:,.0f} Cr" if mcap and ".NS" in stock_ticker else (f"${mcap/1e9:.1f}B" if mcap else "N/A"))
        f6.metric("Return on Equity (ROE)", f"{roe*100:.1f}%" if roe else "N/A")
        f7.metric("Operating Margin", f"{op_margin*100:.1f}%" if op_margin else "N/A")
        f8.metric("Consensus Rating", rec.upper() if rec else "N/A")

        st.markdown("#### Business Summary")
        st.write(company_info.get("longBusinessSummary", "No company profile text available."))

# ----------------- TAB 4: 3-DAY AUDIT -----------------
with tabs[3]:
    st.subheader("🔍 Day-by-Day Walk-Forward Audit: Model Impact Over Last 3 Sessions")
    for idx, row in audit_table.iterrows():
        with st.expander(f"📅 Session {row['Date']} — {row['Model Signal']} — Verdict: {row['Model Audit Verdict']}", expanded=True):
            ca, cb, cc = st.columns([1.2, 1.5, 2.0])
            with ca:
                st.markdown(f"**Closing Price:** {row['Close Price']}")
                st.markdown(f"**Market Regime:** `{row['Market Regime']}`")
                st.markdown(f"**Action Signal:** {row['Model Signal']}")
            with cb:
                st.markdown("**Calibrated Probability Triplet:**")
                st.write(f"- 🟢 **P(Bullish Win):** {row['P(Bull)']}")
                st.write(f"- ⚪ **P(Chop/Noise):** {row['P(Chop)']}")
                st.write(f"- 🔴 **P(Bearish Stop):** {row['P(Bear)']}")
                st.write(f"- **Directional Edge:** `{row['Edge (P_Bull - P_Bear)']}`")
            with cc:
                st.markdown("**Next Day Market Reality:**")
                st.info(f"📊 {row['Next Day Reality']}")
                st.markdown(f"**Audit Impact Assessment:**\n#### {row['Model Audit Verdict']}")

# ----------------- TAB 5: PROBABILITIES -----------------
with tabs[4]:
    st.subheader("🔮 Tri-Class Probability Trajectory (Out-of-Sample Window)")
    prob_fig = go.Figure()
    prob_fig.add_trace(go.Scatter(x=test_df.index, y=test_df['prob_bull'], name='P(Bullish Win)', line=dict(color='#00e676', width=1.8)))
    prob_fig.add_trace(go.Scatter(x=test_df.index, y=test_df['prob_chop'], name='P(Chop / Timeout)', line=dict(color='#ffd600', width=1.2)))
    prob_fig.add_trace(go.Scatter(x=test_df.index, y=test_df['prob_bear'], name='P(Bearish Loss)', line=dict(color='#ff1744', width=1.8)))
    prob_fig.update_layout(height=420, template="plotly_dark", yaxis_title="Probability", yaxis=dict(range=[0.0, 1.0]))
    st.plotly_chart(prob_fig, use_container_width=True)

# ----------------- TAB 6: BACKTEST -----------------
with tabs[5]:
    st.subheader("Strategy Backtest vs Buy & Hold Benchmark")
    strat_tot = (bt_df['cum_strategy'].iloc[-1] - 1.0) * 100
    bench_tot = (bt_df['cum_benchmark'].iloc[-1] - 1.0) * 100
    alpha = strat_tot - bench_tot

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Model Strategy Return", f"{strat_tot:+.2f}%")
    col_b.metric(f"Buy & Hold Return ({stock_ticker})", f"{bench_tot:+.2f}%")
    col_c.metric("Alpha (Excess Return)", f"{alpha:+.2f}%")

    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['cum_benchmark'], mode='lines', name=f'Buy & Hold ({stock_ticker})', line=dict(color='#90a4ae', width=1.5, dash='dash')))
    equity_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['cum_strategy'], mode='lines', name='Tri-Class Risk Strategy', line=dict(color='#00e676', width=2.2)))
    equity_fig.update_layout(height=420, template="plotly_dark", yaxis_title="Portfolio Wealth ($1 base)")
    st.plotly_chart(equity_fig, use_container_width=True)

# ----------------- TAB 7: CHARTS -----------------
with tabs[6]:
    st.subheader(f"Price Action, Market Flow & Context: {stock_ticker}")
    cfig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.25, 0.20],
        subplot_titles=("Price & Moving Averages", "Volume & Flow Layer", "RSI Momentum (14)")
    )
    cfig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close Price', line=dict(color='#29b6f6', width=1.5)), row=1, col=1)
    cfig.add_trace(go.Scatter(x=df.index, y=df['close'].rolling(10).mean(), name='SMA 10', line=dict(color='#ffa726', width=1)), row=1, col=1)
    cfig.add_trace(go.Scatter(x=df.index, y=df['close'].rolling(50).mean(), name='SMA 50', line=dict(color='#ab47bc', width=1)), row=1, col=1)
    cfig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='#37474f'), row=2, col=1)
    cfig.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], name='RSI 14', line=dict(color='#26a69a', width=1.2)), row=3, col=1)
    cfig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=3, col=1)
    cfig.add_hline(y=30, line_dash="dash", line_color="#66bb6a", row=3, col=1)
    cfig.update_layout(height=650, template="plotly_dark")
    st.plotly_chart(cfig, use_container_width=True)

# ----------------- TAB 8: REGIMES -----------------
with tabs[7]:
    st.subheader("🌊 Unsupervised Market Regimes (GMM)")
    if not use_regimes:
        st.warning("Unsupervised Regimes are disabled in the sidebar.")
    else:
        regime_colors = {"Bull Regime": "#00e676", "Sideways / Choppy": "#ffd600", "Bear Regime": "#ff1744"}
        r_fig = go.Figure()
        for r_name, r_color in regime_colors.items():
            mask = df["regime_name"] == r_name
            r_fig.add_trace(go.Scatter(x=df.index[mask], y=df["close"][mask], mode="markers", marker=dict(color=r_color, size=4), name=r_name))
        r_fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", line=dict(color="#546e7a", width=1), name="Price Path"))
        r_fig.update_layout(height=450, template="plotly_dark", yaxis_title="Price")
        st.plotly_chart(r_fig, use_container_width=True)
