"""
Feature engineering module with Triple-Barrier Labeling, Technicals, Market Flow, and Context.
Implements Marcos López de Prado's Triple Barrier Method for volatility-adjusted classification.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (absolute points)."""
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def apply_triple_barrier_labels(
    df: pd.DataFrame,
    pt_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_holding_days: int = 5,
) -> pd.DataFrame:
    """
    Labels each bar using the Triple Barrier Method:
    - Upper Barrier: entry_close + (pt_mult * ATR) -> Label 1 (Bullish Win)
    - Lower Barrier: entry_close - (sl_mult * ATR) -> Label 2 (Bearish Loss)
    - Vertical Barrier: max_holding_days reached -> Label 0 (Chop / Neutral)
    
    Returns DataFrame with:
    - target_class: 0 (Chop), 1 (Bull), 2 (Bear)
    - target_barrier: 'ProfitTarget', 'StopLoss', 'Timeout'
    - target_return: Realized return upon barrier touch or timeout
    - target_pt_pct: Profit target percentage
    - target_sl_pct: Stop loss percentage
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    atr = df['atr_14'].values
    n = len(df)

    classes = np.full(n, np.nan)
    barriers = ['None'] * n
    realized_rets = np.full(n, np.nan)
    pt_pcts = np.full(n, np.nan)
    sl_pcts = np.full(n, np.nan)

    for i in range(n):
        if i + 1 >= n or np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry_price = close[i]
        pt_dist = pt_mult * atr[i]
        sl_dist = sl_mult * atr[i]
        
        pt_price = entry_price + pt_dist
        sl_price = entry_price - sl_dist
        
        pt_pcts[i] = pt_dist / entry_price
        sl_pcts[i] = sl_dist / entry_price

        end_idx = min(i + max_holding_days + 1, n)
        future_highs = high[i + 1 : end_idx]
        future_lows = low[i + 1 : end_idx]
        future_closes = close[i + 1 : end_idx]

        hit_class = 0  # Default: Timeout (Chop)
        hit_barrier = 'Timeout'
        final_ret = (future_closes[-1] - entry_price) / entry_price if len(future_closes) > 0 else 0.0

        for step in range(len(future_highs)):
            is_pt = future_highs[step] >= pt_price
            is_sl = future_lows[step] <= sl_price

            if is_pt and not is_sl:
                hit_class = 1  # Bull
                hit_barrier = 'ProfitTarget'
                final_ret = pt_pcts[i]
                break
            elif is_sl and not is_pt:
                hit_class = 2  # Bear
                hit_barrier = 'StopLoss'
                final_ret = -sl_pcts[i]
                break
            elif is_pt and is_sl:
                # Conservative quant rule: assume stop loss hit first
                hit_class = 2
                hit_barrier = 'StopLoss'
                final_ret = -sl_pcts[i]
                break

        classes[i] = hit_class
        barriers[i] = hit_barrier
        realized_rets[i] = final_ret

    df['target_class'] = classes
    df['target_barrier'] = barriers
    df['target_return'] = realized_rets
    df['target_pt_pct'] = pt_pcts
    df['target_sl_pct'] = sl_pcts
    return df


def build_feature_dataset(
    stock_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    forecast_horizon: int = 5,
    pt_mult: float = 1.5,
    sl_mult: float = 1.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct stationary features from Technicals, Volume Flow, Context,
    and apply Triple-Barrier labeling.
    """
    df = stock_df.copy()
    b_df = bench_df.copy()
    
    # Align dates
    common_idx = df.index.intersection(b_df.index)
    df = df.loc[common_idx].copy()
    b_df = b_df.loc[common_idx].copy()

    # --- 1. Technical (Price Action & Momentum) ---
    df['ret_1d'] = df['close'].pct_change()
    df['ret_3d'] = df['close'].pct_change(3)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)

    # Moving Average Deviations
    df['sma_10_dist'] = df['close'] / df['close'].rolling(10).mean() - 1.0
    df['sma_30_dist'] = df['close'] / df['close'].rolling(30).mean() - 1.0

    # Volatility & ATR
    df['vol_10d'] = df['ret_1d'].rolling(10).std()
    df['vol_20d'] = df['ret_1d'].rolling(20).std()
    df['atr_14'] = compute_atr(df, period=14)
    df['norm_atr_14'] = df['atr_14'] / df['close']

    # Momentum
    df['rsi_14'] = compute_rsi(df['close'], period=14)
    df['rsi_7'] = compute_rsi(df['close'], period=7)

    # MACD approximation
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = (macd_line - signal_line) / df['close']

    # --- 2. Market Flow Layer (Volume & Liquidity) ---
    vol_ma20 = df['volume'].rolling(20).mean()
    df['vol_ratio_20'] = df['volume'] / (vol_ma20 + 1e-9)
    df['vol_ratio_5'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-9)
    
    obv_direction = np.sign(df['ret_1d']).fillna(0)
    obv = (obv_direction * df['volume']).cumsum()
    df['obv_slope_10'] = (obv - obv.rolling(10).mean()) / (df['volume'].rolling(10).mean() * 10 + 1e-9)

    # --- 3. Market Context (NIFTY Benchmark) ---
    b_df['bench_ret_1d'] = b_df['close'].pct_change()
    b_df['bench_ret_5d'] = b_df['close'].pct_change(5)
    b_df['bench_vol_20d'] = b_df['bench_ret_1d'].rolling(20).std()

    df['rel_ret_1d'] = df['ret_1d'] - b_df['bench_ret_1d']
    df['rel_ret_5d'] = df['ret_5d'] - b_df['bench_ret_5d']
    df['bench_vol_20d'] = b_df['bench_vol_20d']
    df['corr_nifty_20'] = df['ret_1d'].rolling(20).corr(b_df['bench_ret_1d'])

    # --- 4. Apply Triple Barrier Labeling ---
    df = apply_triple_barrier_labels(
        df, pt_mult=pt_mult, sl_mult=sl_mult, max_holding_days=forecast_horizon
    )

    feature_cols = [
        'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d',
        'sma_10_dist', 'sma_30_dist',
        'vol_10d', 'vol_20d', 'norm_atr_14',
        'rsi_14', 'rsi_7', 'macd_hist',
        'vol_ratio_20', 'vol_ratio_5', 'obv_slope_10',
        'rel_ret_1d', 'rel_ret_5d', 'bench_vol_20d', 'corr_nifty_20'
    ]

    # Clean missing values resulting from rolling warmup
    # Note: keep uncompleted latest bars with valid features for live/walk-forward prediction
    valid_features_mask = df[feature_cols].notna().all(axis=1)
    df = df[valid_features_mask].copy()

    return df, feature_cols
