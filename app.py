"""
Market Predictor Interactive Dashboard (Streamlit + Plotly)
Tri-Class Triple-Barrier Probability Engine + Unsupervised Regimes + 3-Day Walk-Forward Audit
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_loader import load_market_dataset
from src.features import build_feature_dataset
from src.regimes import add_regime_features
from src.model import train_time_series_model
from src.evaluate import evaluate_predictions, backtest_probability_strategy
from src.audit_recent import audit_last_n_days

# Streamlit Page Setup
st.set_page_config(
    page_title="Tri-Class Quant Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data" / "raw"

# Sidebar Configuration
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("Configure Triple-Barrier risk targets, regimes, and models.")

# 1. Asset & Context
st.sidebar.subheader("1. Asset & Context")
stock_presets = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "NVDA", "TSLA", "Custom"]
selected_preset = st.sidebar.selectbox("Target Equity", stock_presets, index=0)

if selected_preset == "Custom":
    stock_ticker = st.sidebar.text_input("Enter Ticker", value="RELIANCE.NS")
else:
    stock_ticker = selected_preset

bench_presets = ["^NSEI", "^BSESN", "SPY", "QQQ", "Custom"]
selected_bench = st.sidebar.selectbox("Market Context Benchmark", bench_presets, index=0)
if selected_bench == "Custom":
    benchmark_ticker = st.sidebar.text_input("Enter Benchmark Ticker", value="^NSEI")
else:
    benchmark_ticker = selected_bench

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))

# 2. Triple-Barrier Settings
st.sidebar.subheader("2. Triple Barrier Settings")
horizon = st.sidebar.slider("Max Holding Window (Days)", min_value=2, max_value=10, value=5, step=1)
pt_mult = st.sidebar.slider("Profit Target (× ATR)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
sl_mult = st.sidebar.slider("Stop Loss Barrier (× ATR)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

# 3. Model & Regimes Configuration
st.sidebar.subheader("3. Model & Regimes")
use_regimes = st.sidebar.checkbox("Include Unsupervised Regimes (GMM)", value=True)
train_split = st.sidebar.slider("Train/Test Chronological Split", min_value=0.60, max_value=0.90, value=0.80, step=0.05)
n_trees = st.sidebar.slider("Number of Trees (LightGBM)", min_value=50, max_value=300, value=100, step=10)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.03, 0.05, 0.10], value=0.03)
force_download = st.sidebar.checkbox("Force Redownload Fresh Quotes", value=False)
if st.sidebar.button("🔄 Sync Live Market Data Now", use_container_width=True):
    st.cache_data.clear()
    force_download = True
    st.rerun()


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_market_data(stock_sym: str, bench_sym: str, start: str, force: bool):
    return load_market_dataset(
        stock_ticker=stock_sym,
        benchmark_ticker=bench_sym,
        start_date=start,
        cache_dir=CACHE_DIR,
        force_download=force,
    )


# App Header
st.title("⚡ Quantitative Market Predictor")
st.markdown(
    f"**Architecture:** Triple-Barrier Labeling + Calibrated Tri-Class Probabilities + GMM Regimes on **`{stock_ticker}`** vs **`{benchmark_ticker}`**"
)

with st.spinner("Fetching market data and building feature pipeline..."):
    try:
        stock_df, bench_df = get_cached_market_data(
            stock_ticker, benchmark_ticker, str(start_date), force_download
        )
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# Feature Engineering
df, feature_cols = build_feature_dataset(
    stock_df, bench_df, forecast_horizon=horizon, pt_mult=pt_mult, sl_mult=sl_mult
)

detector = None
if use_regimes:
    df, detector, regime_cols = add_regime_features(df, train_ratio=train_split)
    feature_cols = feature_cols + regime_cols

# Model Training
model_params = {
    "n_estimators": n_trees,
    "learning_rate": learning_rate,
    "max_depth": 3,
    "num_leaves": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "verbose": -1,
}

model, train_df, test_df, importance_df = train_time_series_model(
    df, feature_cols, target_col="target_class", train_ratio=train_split, params=model_params
)

# Run Evaluation & Backtest
metrics = evaluate_predictions(test_df)
bt_df = backtest_probability_strategy(test_df)

# Run Last 3 Days Walk-Forward Audit
audit_table = audit_last_n_days(df, feature_cols, model, n_days=3)

# Latest Bar Metrics
latest_bar = df.iloc[-1]
latest_feats = df[feature_cols].iloc[[-1]]
latest_probs = model.predict_proba(latest_feats)[0]
p_chop, p_bull, p_bear = latest_probs[0], latest_probs[1], latest_probs[2]
latest_edge = p_bull - p_bear
latest_date = df.index[-1].strftime("%Y-%m-%d")
latest_close = float(latest_bar["close"])
latest_regime = latest_bar.get("regime_name", "N/A")

if p_bull >= 0.38 and latest_edge >= 0.08:
    live_badge = "🟢 BUY / LONG"
elif p_bear >= 0.40 or p_bear > p_bull:
    live_badge = "🔴 AVOID / DOWNSIDE RISK"
else:
    live_badge = "⚪ CHOP / CASH"

# Top Live KPI Banner
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric(f"Close ({latest_date})", f"₹{latest_close:,.2f}" if ".NS" in stock_ticker else f"${latest_close:,.2f}", f"Regime: {latest_regime}")
kpi2.metric("P(Bull: Hit Target)", f"{p_bull * 100:.1f}%")
kpi3.metric("P(Chop: Range-bound)", f"{p_chop * 100:.1f}%")
kpi4.metric("P(Bear: Hit Stop Loss)", f"{p_bear * 100:.1f}%")
kpi5.metric("Model Action Signal", live_badge, f"Edge: {latest_edge * 100:+.1f}%")

st.divider()

# Navigation Tabs
tab_titles = [
    "🔍 Last 3 Days Walk-Forward Audit",
    "🔮 Tri-Class Probabilities",
    "📈 Strategy Backtest",
    "📊 Price & Flow Charts",
    "🌊 Unsupervised Regimes",
    "🧠 Feature Diagnostics",
]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

# ----------------- TAB 1: 3-DAY AUDIT -----------------
with tab1:
    st.subheader("🔍 Day-by-Day Walk-Forward Audit: Testing Model Impact on Recent Trading Sessions")
    st.markdown(
        """
        **Direct verification of model decisions over the last 3 trading days:**
        For each session, we inspect what the model predicted at close based *only* on information up to that moment, 
        and compare it directly with the actual market price movement that unfolded next.
        """
    )

    for idx, row in audit_table.iterrows():
        with st.expander(f"📅 Session {row['Date']} — {row['Model Signal']} — Verdict: {row['Model Audit Verdict']}", expanded=True):
            col_a, col_b, col_c = st.columns([1.2, 1.5, 2.0])
            with col_a:
                st.markdown(f"**Closing Price:** {row['Close Price']}")
                st.markdown(f"**Market Regime:** `{row['Market Regime']}`")
                st.markdown(f"**Action Signal:** {row['Model Signal']}")
            with col_b:
                st.markdown("**Calibrated Probability Triplet:**")
                st.write(f"- 🟢 **P(Bullish Win):** {row['P(Bull)']}")
                st.write(f"- ⚪ **P(Chop/Noise):** {row['P(Chop)']}")
                st.write(f"- 🔴 **P(Bearish Stop):** {row['P(Bear)']}")
                st.write(f"- **Directional Edge:** `{row['Edge (P_Bull - P_Bear)']}`")
            with col_c:
                st.markdown("**Next Day Market Reality:**")
                st.info(f"📊 {row['Next Day Reality']}")
                st.markdown(f"**Audit Impact Assessment:**\n#### {row['Model Audit Verdict']}")

    st.markdown("#### Full Audit Summary Table")
    st.dataframe(audit_table, use_container_width=True)

# ----------------- TAB 2: PROBABILITIES -----------------
with tab2:
    st.subheader("🔮 Full Tri-Class Probability Trajectory (Out-of-Sample Window)")
    st.markdown(
        "Unlike naive binary models, our engine tracks **three concurrent probabilities**: "
        "Profit Target reach, Sideways Chop, and Stop Loss risk."
    )

    prob_fig = go.Figure()
    prob_fig.add_trace(go.Scatter(x=test_df.index, y=test_df['prob_bull'], name='P(Bullish Win)', line=dict(color='#00e676', width=1.8)))
    prob_fig.add_trace(go.Scatter(x=test_df.index, y=test_df['prob_chop'], name='P(Chop / Timeout)', line=dict(color='#ffd600', width=1.2)))
    prob_fig.add_trace(go.Scatter(x=test_df.index, y=test_df['prob_bear'], name='P(Bearish Loss)', line=dict(color='#ff1744', width=1.8)))
    prob_fig.update_layout(
        height=450,
        template="plotly_dark",
        yaxis_title="Calibrated Probability",
        yaxis=dict(range=[0.0, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(prob_fig, use_container_width=True)

    # Net Edge
    st.markdown("#### Directional Edge Score ($P_{\\text{Bull}} - P_{\\text{Bear}}$)")
    edge_fig = go.Figure()
    edge_fig.add_trace(go.Bar(
        x=test_df.index,
        y=test_df['edge'],
        marker_color=np.where(test_df['edge'] > 0, '#00e676', '#ff1744'),
        name='Edge'
    ))
    edge_fig.update_layout(height=250, template="plotly_dark", yaxis_title="Net Edge", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(edge_fig, use_container_width=True)

# ----------------- TAB 3: BACKTEST -----------------
with tab3:
    st.subheader("Strategy Backtest: Capital Preservation & Risk-Adjusted Returns")
    
    strat_tot = (bt_df['cum_strategy'].iloc[-1] - 1.0) * 100
    bench_tot = (bt_df['cum_benchmark'].iloc[-1] - 1.0) * 100
    alpha = strat_tot - bench_tot

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Model Strategy Return", f"{strat_tot:+.2f}%")
    col_b.metric(f"Buy & Hold Return ({stock_ticker})", f"{bench_tot:+.2f}%")
    col_c.metric("Alpha (Capital Protected)", f"{alpha:+.2f}%")

    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df['cum_benchmark'],
        mode='lines', name=f'Buy & Hold ({stock_ticker})',
        line=dict(color='#90a4ae', width=1.5, dash='dash')
    ))
    equity_fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df['cum_strategy'],
        mode='lines', name='Tri-Class Risk Strategy',
        line=dict(color='#00e676', width=2.2)
    ))
    equity_fig.update_layout(
        height=450,
        template="plotly_dark",
        title="Out-of-Sample Portfolio Wealth ($1.00 starting capital)",
        yaxis_title="Portfolio Wealth",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(equity_fig, use_container_width=True)

# ----------------- TAB 4: CHARTS -----------------
with tab4:
    st.subheader(f"Price Action, Market Flow & Technicals: {stock_ticker}")
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.25, 0.20],
        subplot_titles=("Price & Moving Averages", "Volume & Flow Layer", "RSI Momentum (14)")
    )
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close Price', line=dict(color='#29b6f6', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['close'].rolling(10).mean(), name='SMA 10', line=dict(color='#ffa726', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['close'].rolling(30).mean(), name='SMA 30', line=dict(color='#ab47bc', width=1)), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='#37474f'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['volume'].rolling(20).mean(), name='Vol 20 MA', line=dict(color='#ff7043', width=1.2)), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], name='RSI 14', line=dict(color='#26a69a', width=1.2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#66bb6a", row=3, col=1)
    fig.update_layout(height=700, template="plotly_dark", showlegend=True, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ----------------- TAB 5: REGIMES -----------------
with tab5:
    st.subheader("🌊 Unsupervised Market Regimes (GMM)")
    if not use_regimes:
        st.warning("Unsupervised Regimes are disabled in the sidebar.")
    else:
        regime_colors = {
            "Bull Regime": "#00e676",
            "Sideways / Choppy": "#ffd600",
            "Bear Regime": "#ff1744",
        }
        r_fig = go.Figure()
        for r_name, r_color in regime_colors.items():
            mask = df["regime_name"] == r_name
            r_fig.add_trace(go.Scatter(
                x=df.index[mask], y=df["close"][mask], mode="markers",
                marker=dict(color=r_color, size=4, opacity=0.8), name=r_name
            ))
        r_fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", line=dict(color="#546e7a", width=1), name="Price Path"))
        r_fig.update_layout(height=450, template="plotly_dark", yaxis_title="Price", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(r_fig, use_container_width=True)

# ----------------- TAB 6: DIAGNOSTICS -----------------
with tab6:
    st.subheader("Model Diagnostics & Feature Importance")
    c_m1, c_m2 = st.columns([1, 1.5])
    with c_m1:
        st.markdown("#### Test Metrics")
        st.table(pd.DataFrame([
            {"Metric": "Accuracy", "Value": f"{metrics['Accuracy'] * 100:.2f}%"},
            {"Metric": "Log Loss", "Value": f"{metrics['Log_Loss']:.4f}"},
            {"Metric": "Bull Rate in Test", "Value": f"{metrics['Bull_Win_Rate'] * 100:.1f}%"},
            {"Metric": "Bear Rate in Test", "Value": f"{metrics['Bear_Loss_Rate'] * 100:.1f}%"},
            {"Metric": "Chop Rate in Test", "Value": f"{metrics['Chop_Timeout_Rate'] * 100:.1f}%"},
        ]))
    with c_m2:
        st.markdown("#### Top Predictive Features")
        top_imp = importance_df.head(10).sort_values(by="importance", ascending=True)
        imp_fig = go.Figure(go.Bar(
            x=top_imp['importance'], y=top_imp['feature'], orientation='h', marker_color='#42a5f5'
        ))
        imp_fig.update_layout(height=350, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(imp_fig, use_container_width=True)
