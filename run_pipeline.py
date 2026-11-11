"""
Run Pipeline: End-to-end execution of the Tri-Class Market Predictor
With Triple-Barrier Labeling, Unsupervised Regimes, Live News Sentiment NLP,
Target Projections, Behavior Attribution, and Fakeout/Anomaly Auditing.
"""

import argparse
from pathlib import Path
import yfinance as yf
from src.data_loader import load_market_dataset
from src.features import build_feature_dataset
from src.regimes import add_regime_features
from src.model import train_time_series_model
from src.evaluate import evaluate_predictions, backtest_probability_strategy, plot_summary_report
from src.audit_recent import audit_last_n_days
from src.news_sentiment import fetch_live_news_sentiment
from src.anomaly_detector import (
    compute_target_projections,
    analyze_trend_structure,
    explain_market_behavior,
    detect_fakeouts_and_anomalies,
)


def run(
    stock_ticker: str = "ETERNAL.NS",
    benchmark_ticker: str = "^NSEI",
    use_regimes: bool = True,
    refresh_data: bool = True,
):
    base_dir = Path(__file__).resolve().parent
    cache_dir = base_dir / "data" / "raw"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"🚀 RUNNING QUANT INTELLIGENCE ENGINE: {stock_ticker} vs {benchmark_ticker}")
    print(f"   Architecture: Triple Barrier + Calibrated Probabilities + Regimes + NLP")
    print("=" * 75)

    # 1. Ingest Data (fetch latest live quotes)
    stock_df, bench_df = load_market_dataset(
        stock_ticker=stock_ticker,
        benchmark_ticker=benchmark_ticker,
        start_date="2020-01-01",
        cache_dir=cache_dir,
        force_download=refresh_data,
    )
    print(f"✅ Data Loaded: {len(stock_df)} stock records, {len(bench_df)} benchmark records.")

    # Fetch Info & Live News Sentiment
    company_info = {}
    try:
        company_info = yf.Ticker(stock_ticker).info
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch company info: {e}")

    company_name = company_info.get("shortName") or company_info.get("longName") or stock_ticker
    print(f"🏢 Company: {company_name}")

    print("\n[NLP] Fetching real-time news headlines & computing sentiment...")
    news_res = fetch_live_news_sentiment(stock_ticker, company_name=company_name)
    print(f"📰 Live News Sentiment: {news_res['sentiment_label']} (Score: {news_res['sentiment_index']:+d}/100)")
    for a in news_res["articles"][:3]:
        print(f"   - [{a['sentiment']}] {a['title']}")

    # 2. Build Features & Triple Barrier Labels
    df, feature_cols = build_feature_dataset(
        stock_df, bench_df, forecast_horizon=5, pt_mult=1.5, sl_mult=1.0
    )
    print(f"\n✅ Triple-Barrier Features: {len(feature_cols)} features across {len(df)} bars.")

    # 3. Unsupervised Regime Discovery
    if use_regimes:
        df, detector, regime_cols = add_regime_features(df, train_ratio=0.8)
        feature_cols = feature_cols + regime_cols
        print(f"✅ Unsupervised Regimes Attached: {regime_cols}")

    # 4. Train Calibrated Tri-Class Model
    model, train_df, test_df, importance_df = train_time_series_model(
        df, feature_cols, target_col="target_class", train_ratio=0.8
    )
    print(f"✅ Calibrated Model Trained on {len(train_df)} bars | Testing on {len(test_df)} out-of-sample bars.")

    # 5. Out-of-Sample Metrics & Backtest
    metrics = evaluate_predictions(test_df)
    bt_df = backtest_probability_strategy(test_df)
    bench_ret = (bt_df['cum_benchmark'].iloc[-1] - 1.0) * 100
    strat_ret = (bt_df['cum_strategy'].iloc[-1] - 1.0) * 100

    print("\n--- OUT-OF-SAMPLE TEST METRICS ---")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")
    print(f"  Buy & Hold Return   : {bench_ret:+.2f}%")
    print(f"  Strategy Return     : {strat_ret:+.2f}%")

    # 6. HOW FAR WOULD IT GO & TREND ANALYSIS
    projections = compute_target_projections(df, company_info)
    trend = analyze_trend_structure(df)
    print("\n" + "=" * 75)
    print("🎯 HOW FAR WOULD IT GO? (Target Projections & Boundaries)")
    print("=" * 75)
    print(f"  Current Price          : ₹{projections['current_price']:,.2f}")
    print(f"  Upside Target 1 (1.5σ) : ₹{projections['upside_target_1']:,.2f} ({projections['upside_target_1_pct']:+.1f}%)")
    print(f"  Upside Target 2 (2.5σ) : ₹{projections['upside_target_2']:,.2f} ({projections['upside_target_2_pct']:+.1f}%)")
    print(f"  Support 1 (Stop Loss)  : ₹{projections['downside_support_1']:,.2f} ({projections['downside_support_1_pct']:+.1f}%)")
    if projections["analyst_mean"]:
        print(f"  Analyst Consensus Target: ₹{projections['analyst_mean']:,.2f} ({projections['analyst_mean_pct']:+.1f}%) | Consensus: {projections['analyst_recommendation']}")
    print(f"  Overall Trend Status   : {trend['macro_trend']}")

    # 7. WHY IS IT BEHAVING NOW? (Attribution)
    bench_ret_1d = float(bench_df["close"].pct_change().iloc[-1]) if len(bench_df) > 1 else 0.0
    behavior = explain_market_behavior(df, news_sentiment=news_res, bench_ret_1d=bench_ret_1d)
    print("\n" + "=" * 75)
    print("🔬 WHY IS IT BEHAVING NOW? (Behavior Attribution)")
    print("=" * 75)
    for reason in behavior["reasons"]:
        clean_text = reason.replace("**", "").replace("*", "")
        print(f"  • {clean_text}")

    # 8. IS THERE ANY FAKE THING GOING ON? (Anomaly / Fakeout Audit)
    fakeouts = detect_fakeouts_and_anomalies(df, news_sentiment=news_res)
    print("\n" + "=" * 75)
    print("🛡️ IS THERE ANY FAKE THING GOING ON? (Anomaly & Manipulation Audit)")
    print("=" * 75)
    print(f"  Overall Status: {fakeouts['status_badge']}")
    for anomaly in fakeouts["anomalies"]:
        print(f"\n  [{anomaly['level']}] {anomaly['type']}")
        print(f"  Details: {anomaly['description']}")

    # 9. WALK-FORWARD AUDIT OVER THE LAST 3 TRADING DAYS
    print("\n" + "=" * 75)
    print("🔍 LAST 3 TRADING DAYS WALK-FORWARD IMPACT AUDIT")
    print("=" * 75)
    audit_table = audit_last_n_days(df, feature_cols, model, n_days=3)
    for _, row in audit_table.iterrows():
        print(f"\n📅 Date: {row['Date']} | Close: {row['Close Price']} | Regime: {row['Market Regime']}")
        print(f"   Probability Vector : [Bull: {row['P(Bull)']} | Chop: {row['P(Chop)']} | Bear: {row['P(Bear)']}]")
        print(f"   Edge: {row['Edge (P_Bull - P_Bear)']} | Signal: {row['Model Signal']}")
        print(f"   Next Day Reality   : {row['Next Day Reality']}")
        print(f"   Audit Assessment   : {row['Model Audit Verdict']}")

    # 10. Generate Visualization
    plot_path = reports_dir / "performance_summary.png"
    plot_summary_report(bt_df, importance_df, plot_path, symbol_name=stock_ticker)
    print("\n" + "=" * 75)
    print(f"🎉 Analysis Complete! Visual report updated at:\n   {plot_path}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tri-Class Market Predictor")
    parser.add_argument("--stock", type=str, default="ETERNAL.NS", help="Target equity ticker (default: ETERNAL.NS)")
    parser.add_argument("--bench", type=str, default="^NSEI", help="Benchmark context ticker")
    parser.add_argument("--no-regimes", action="store_true", help="Disable unsupervised regime features")
    parser.add_argument("--no-refresh", action="store_true", help="Use local cache without downloading")
    args = parser.parse_args()

    run(
        stock_ticker=args.stock,
        benchmark_ticker=args.bench,
        use_regimes=not args.no_regimes,
        refresh_data=not args.no_refresh,
    )
