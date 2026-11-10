# Market Predictor & Regime Discovery Engine

An end-to-end quantitative machine learning pipeline for market forecasting, built on technical price action, volume flow, benchmark context, and unsupervised regime discovery.

## Target Architecture Mapping

This project implements the foundational layers of the multi-modal trading architecture:

1. **Market Context**: Benchmark index (e.g., NIFTY 50 `^NSEI`).
2. **Technical (Price Action)**: Normalized multi-horizon returns, moving average distances, RSI (7 & 14), normalized ATR, and MACD.
3. **Market Flow Layer**: Volume moving average ratios and On-Balance Volume (OBV) trend slope.
4. **Unsupervised Market Regimes**: Gaussian Mixture Model (GMM) discovering latent macro regimes (*Bull*, *Bear*, *Choppy/Sideways*) without manual annotations.
5. **Future Probability**: Probabilistic output ($P(\text{Up})$) using Gradient Boosted Trees (`LightGBM`).

## Project Layout

```
market_predictor_mvp/
├── data/
│   ├── raw/                 # Cached raw OHLCV market data
│   └── processed/           # Processed datasets
├── reports/
│   └── performance_summary.png  # Generated backtest charts & feature importances
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Historical market data ingestion via yfinance
│   ├── features.py          # Technical indicators & volume flow engineering
│   ├── regimes.py           # Unsupervised GMM regime detection
│   ├── model.py             # LightGBM classifier & probabilistic training
│   └── evaluate.py          # Temporal backtesting & equity curves
├── app.py                   # Interactive Streamlit & Plotly dashboard
├── run_pipeline.py          # Main CLI orchestrator
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Run via CLI:
```bash
cd /home/bhuvik/market_predictor_mvp

# Run with defaults (RELIANCE.NS vs NIFTY 50, with unsupervised regimes)
python3 run_pipeline.py

# Run on other tickers
python3 run_pipeline.py --stock TCS.NS --bench ^NSEI
python3 run_pipeline.py --stock AAPL --bench SPY
```

### 2. Launch the Interactive Dashboard:
```bash
cd /home/bhuvik/market_predictor_mvp
streamlit run app.py
```
Open your browser at `http://localhost:8501` to interactively adjust forecast horizons, probability triggers, inspect regime shifts, and view live trading signals.
