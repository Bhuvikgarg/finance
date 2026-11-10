"""
Run Pipeline: End-to-end execution of the Tri-Class Market Predictor MVP
With Triple-Barrier Labeling, Unsupervised Regimes, and Recent Walk-Forward Audit.
"""

import argparse
from pathlib import Path
from src.data_loader import load_market_dataset
from src.features import build_feature_dataset
from src.regimes import add_regime_features
from src.model import train_time_series_model
from src.evaluate import evaluate_predictions, backtest_probability_strategy, plot_summary_report
from src.audit_recent import audit_last_n_days


def run(
    stock_ticker: str = "RELIANCE.NS",
    benchmark_ticker: str = "^NSEI",
    use_regimes: bool = True,
    refresh_data: bool = True,
):
    base_dir = Path(__file__).resolve().parent
    cache_dir = base_dir / "data" / "raw"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"🚀 RUNNING TRI-CLASS QUANT ENGINE: {stock_ticker} vs {benchmark_ticker}")
    print(f"   Architecture: Triple Barrier (ATR) + Calibrated Probabilities + Regimes")
    print("=" * 70)

    # 1. Ingest Data (fetch latest live quotes)
    stock_df, bench_df = load_market_dataset(
        stock_ticker=stock_ticker,
        benchmark_ticker=benchmark_ticker,
        start_date="2020-01-01",
        cache_dir=cache_dir,
        force_download=refresh_data,
    )
    print(f"✅ Data Loaded: {len(stock_df)} stock records, {len(bench_df)} benchmark records.")

    # 2. Build Features & Triple Barrier Labels
    df, feature_cols = build_feature_dataset(
        stock_df, bench_df, forecast_horizon=5, pt_mult=1.5, sl_mult=1.0
    )
    print(f"✅ Triple-Barrier Features Constructed: {len(feature_cols)} features across {len(df)} bars.")

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

    # 6. WALK-FORWARD AUDIT OVER THE LAST 3 TRADING DAYS
    print("\n" + "=" * 70)
    print("🔍 LAST 3 TRADING DAYS WALK-FORWARD IMPACT AUDIT")
    print("=" * 70)
    audit_table = audit_last_n_days(df, feature_cols, model, n_days=3)
    
    for _, row in audit_table.iterrows():
        print(f"\n📅 Date: {row['Date']} | Close: {row['Close Price']} | Regime: {row['Market Regime']}")
        print(f"   Probability Vector : [Bull: {row['P(Bull)']} | Chop: {row['P(Chop)']} | Bear: {row['P(Bear)']}]")
        print(f"   Edge: {row['Edge (P_Bull - P_Bear)']} | Exp Value (EV): {row['Exp Return (EV)']} | Signal: {row['Model Signal']}")
        print(f"   Next Day Reality   : {row['Next Day Reality']}")
        print(f"   Audit Assessment   : {row['Model Audit Verdict']}")

    # 7. Generate Visualization
    plot_path = reports_dir / "performance_summary.png"
    plot_summary_report(bt_df, importance_df, plot_path, symbol_name=stock_ticker)
    print("\n" + "=" * 70)
    print(f"🎉 Pipeline Complete! Visual report updated at:\n   {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tri-Class Market Predictor")
    parser.add_argument("--stock", type=str, default="RELIANCE.NS", help="Target equity ticker")
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
