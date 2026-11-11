"""
Behavior Attribution, Target Projections, and Fakeout/Anomaly Detector Engine.
Answers:
1. How far would it go? (ATR Volatility Targets & Analyst Targets)
2. What is its trend? (Multi-timeframe Moving Average & Momentum structure)
3. Why is it behaving now? (Market Beta, Volume Flow, News Attribution)
4. Is there any fake thing going on? (Bull traps, Bear traps, RSI Divergences, Speculative pumps)
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_target_projections(
    df: pd.DataFrame,
    info: Optional[Dict] = None,
    atr_period: int = 14,
) -> Dict:
    """
    Computes mathematical price projections and upside/downside boundaries.
    """
    last_close = float(df["close"].iloc[-1])
    atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else float(df["close"].pct_change().std() * last_close * np.sqrt(14))

    # Volatility Cones (5-day & 20-day projections)
    upside_target_1 = last_close + (1.5 * atr)
    upside_target_2 = last_close + (2.5 * atr)
    downside_support_1 = max(0.0, last_close - (1.0 * atr))
    downside_support_2 = max(0.0, last_close - (2.0 * atr))

    # Classical Pivot Points from recent 20-day range
    recent_high = float(df["high"].tail(20).max())
    recent_low = float(df["low"].tail(20).min())
    pivot_point = (recent_high + recent_low + last_close) / 3.0
    r1 = (2.0 * pivot_point) - recent_low
    s1 = (2.0 * pivot_point) - recent_high

    # Institutional Analyst Targets from info
    analyst_mean = None
    analyst_high = None
    analyst_low = None
    recommendation = "N/A"

    if info:
        analyst_mean = info.get("targetMeanPrice")
        analyst_high = info.get("targetHighPrice")
        analyst_low = info.get("targetLowPrice")
        recommendation = info.get("recommendationKey", "N/A").upper()

    return {
        "current_price": last_close,
        "atr_value": atr,
        "upside_target_1": upside_target_1,
        "upside_target_1_pct": (upside_target_1 - last_close) / last_close * 100,
        "upside_target_2": upside_target_2,
        "upside_target_2_pct": (upside_target_2 - last_close) / last_close * 100,
        "downside_support_1": downside_support_1,
        "downside_support_1_pct": (downside_support_1 - last_close) / last_close * 100,
        "downside_support_2": downside_support_2,
        "downside_support_2_pct": (downside_support_2 - last_close) / last_close * 100,
        "pivot_r1": r1,
        "pivot_s1": s1,
        "analyst_mean": analyst_mean,
        "analyst_mean_pct": ((analyst_mean - last_close) / last_close * 100) if analyst_mean else None,
        "analyst_high": analyst_high,
        "analyst_low": analyst_low,
        "analyst_recommendation": recommendation,
    }


def analyze_trend_structure(df: pd.DataFrame) -> Dict:
    """
    Evaluates multi-timeframe trend alignment and momentum strength.
    """
    last_close = float(df["close"].iloc[-1])
    sma_20 = float(df["close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else last_close
    sma_50 = float(df["close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else last_close
    sma_200 = float(df["close"].rolling(200).mean().iloc[-1]) if len(df) >= 200 else last_close

    rsi_14 = float(df["rsi_14"].iloc[-1]) if "rsi_14" in df.columns else 50.0

    # Trend Categorization
    if last_close > sma_50 > sma_200:
        macro_trend = "🟢 Strong Uptrend (Golden Alignment)"
        trend_status = "BULLISH"
    elif last_close < sma_50 < sma_200:
        macro_trend = "🔴 Strong Downtrend (Death Alignment)"
        trend_status = "BEARISH"
    elif last_close > sma_50:
        macro_trend = "🟡 Moderate Uptrend / Consolidation"
        trend_status = "MODERATE_BULLISH"
    else:
        macro_trend = "🟡 Weak / Corrective Consolidation"
        trend_status = "MODERATE_BEARISH"

    return {
        "macro_trend": macro_trend,
        "trend_status": trend_status,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "dist_sma_50_pct": (last_close - sma_50) / sma_50 * 100,
        "dist_sma_200_pct": (last_close - sma_200) / sma_200 * 100 if len(df) >= 200 else 0.0,
        "rsi_14": rsi_14,
    }


def explain_market_behavior(
    df: pd.DataFrame,
    news_sentiment: Optional[Dict] = None,
    bench_ret_1d: float = 0.0,
) -> Dict:
    """
    Decomposes and attributes why the stock is behaving the way it is right now.
    """
    last_close = float(df["close"].iloc[-1])
    ret_1d = float(df["ret_1d"].iloc[-1]) * 100 if "ret_1d" in df.columns else 0.0
    ret_5d = float(df["ret_5d"].iloc[-1]) * 100 if "ret_5d" in df.columns else 0.0
    vol_ratio = float(df["vol_ratio_20"].iloc[-1]) if "vol_ratio_20" in df.columns else 1.0
    corr_nifty = float(df["corr_nifty_20"].iloc[-1]) if "corr_nifty_20" in df.columns else 0.5
    rsi = float(df["rsi_14"].iloc[-1]) if "rsi_14" in df.columns else 50.0

    reasons = []

    # 1. Market Context / NIFTY Drag or Boost
    market_effect = corr_nifty * (bench_ret_1d * 100)
    if abs(market_effect) >= 0.5:
        dir_word = "tailwind (boosting price)" if market_effect > 0 else "headwind (dragging price down)"
        reasons.append(f"**Broad Market (NIFTY 50):** Provided a {market_effect:+.2f}% {dir_word}.")
    else:
        reasons.append(f"**Broad Market (NIFTY 50):** Minimal index influence ({market_effect:+.2f}%). Stock is moving independently.")

    # 2. Volume & Liquidity Flow
    if vol_ratio > 1.8:
        flow_word = "Aggressive Institutional Inflow (Accumulation)" if ret_1d > 0 else "Heavy Institutional Outflow (Distribution / Selling)"
        reasons.append(f"**Market Flow Layer:** Volume is {vol_ratio:.1f}x higher than normal — indicates {flow_word}.")
    elif vol_ratio < 0.6:
        reasons.append(f"**Market Flow Layer:** Thin trading volume ({vol_ratio:.1f}x of 20d average) — move lacks strong institutional participation.")
    else:
        reasons.append(f"**Market Flow Layer:** Normal liquidity turnover ({vol_ratio:.1f}x average).")

    # 3. News Sentiment Catalyst
    if news_sentiment and news_sentiment.get("articles"):
        avg_s = news_sentiment.get("average_compound", 0.0)
        top_art = news_sentiment["articles"][0]["title"]
        if avg_s >= 0.15:
            reasons.append(f"**News Catalyst:** Bullish news narrative (Score: {avg_s:+.2f}). Lead headline: *\"{top_art}\"*.")
        elif avg_s <= -0.15:
            reasons.append(f"**News Catalyst:** Bearish headlines pressuring sentiment (Score: {avg_s:+.2f}). Lead headline: *\"{top_art}\"*.")
        else:
            reasons.append("**News Catalyst:** Mixed/Neutral coverage. Price is driven primarily by technical positioning rather than fresh news.")

    # 4. Momentum / Overbought / Oversold
    if rsi >= 70:
        reasons.append(f"**Technical State:** Overbought conditions (RSI = {rsi:.1f}). Profit taking or consolidation is mathematically probable.")
    elif rsi <= 30:
        reasons.append(f"**Technical State:** Oversold territory (RSI = {rsi:.1f}). Sellers may be reaching exhaustion, opening a mean-reversion bounce.")
    else:
        reasons.append(f"**Technical State:** Healthy momentum equilibrium (RSI = {rsi:.1f}).")

    return {
        "ret_1d_pct": ret_1d,
        "ret_5d_pct": ret_5d,
        "volume_ratio": vol_ratio,
        "correlation_with_nifty": corr_nifty,
        "reasons": reasons,
    }


def detect_fakeouts_and_anomalies(
    df: pd.DataFrame,
    news_sentiment: Optional[Dict] = None,
) -> Dict:
    """
    Checks for market manipulation patterns, fakeouts, and divergences:
    - Bull Trap (Volume Divergence)
    - Bear Trap (Selling Exhaustion)
    - RSI Divergence (Price high, RSI lower)
    - Speculative Pump Anomaly (Extreme volume + price spike with no news)
    - Institutional Distribution / Sell-The-News
    """
    anomalies = []
    severity = "CLEAN"  # CLEAN, SUSPICIOUS, DANGER

    ret_1d = float(df["ret_1d"].iloc[-1]) if "ret_1d" in df.columns else 0.0
    ret_5d = float(df["ret_5d"].iloc[-1]) if "ret_5d" in df.columns else 0.0
    vol_ratio = float(df["vol_ratio_20"].iloc[-1]) if "vol_ratio_20" in df.columns else 1.0
    obv_slope = float(df["obv_slope_10"].iloc[-1]) if "obv_slope_10" in df.columns else 0.0
    rsi = float(df["rsi_14"].iloc[-1]) if "rsi_14" in df.columns else 50.0

    recent_closes = df["close"].tail(10).values
    recent_rsis = df["rsi_14"].tail(10).values if "rsi_14" in df.columns else np.full(10, 50.0)

    # 1. Bull Trap: Price rising but volume declining or OBV dumping
    if ret_5d > 0.02 and (vol_ratio < 0.65 or obv_slope < -0.15):
        anomalies.append({
            "type": "⚠️ BULL TRAP WARNING (Volume Divergence)",
            "description": f"Price gained {ret_5d*100:+.1f}% over 5 days, but volume is drying up ({vol_ratio:.1f}x avg) and OBV slope is negative ({obv_slope:.2f}). Institutions are NOT supporting this upward move.",
            "level": "WARNING"
        })
        severity = "SUSPICIOUS"

    # 2. Bear Trap: Price dumped, but volume exhausted and RSI deeply oversold
    if ret_5d < -0.04 and vol_ratio < 0.70 and rsi < 32:
        anomalies.append({
            "type": "⚠️ BEAR TRAP / SHAKEOUT ALERT",
            "description": f"Sharp 5-day drop ({ret_5d*100:+.1f}%), but selling volume has dissipated ({vol_ratio:.1f}x) into deep oversold RSI ({rsi:.1f}). High probability of a bear squeeze / rebound.",
            "level": "ALERT"
        })
        if severity != "DANGER":
            severity = "SUSPICIOUS"

    # 3. Bearish RSI Divergence: Price higher high, RSI lower high
    if len(recent_closes) >= 10 and recent_closes[-1] > recent_closes[0] and recent_rsis[-1] < (recent_rsis[0] - 4.0) and rsi > 55:
        anomalies.append({
            "type": "⚠️ BEARISH MOMENTUM DIVERGENCE",
            "description": f"Price reached higher levels than 10 sessions ago, but RSI momentum declined from {recent_rsis[0]:.1f} to {recent_rsis[-1]:.1f}. Indicates upward momentum exhaustion.",
            "level": "WARNING"
        })
        severity = "SUSPICIOUS"

    # 4. Speculative Pump Anomaly: Massive volume + price spike with no news
    avg_sentiment = news_sentiment.get("average_compound", 0.0) if news_sentiment else 0.0
    if ret_1d > 0.045 and vol_ratio > 3.0 and avg_sentiment <= 0.05:
        anomalies.append({
            "type": "🚨 SPECULATIVE PUMP ANOMALY",
            "description": f"Surge of {ret_1d*100:+.1f}% with massive volume ({vol_ratio:.1f}x) occurred with ZERO major positive news catalyst. Beware of speculative social media or operator pump.",
            "level": "DANGER"
        })
        severity = "DANGER"

    # 5. "Sell The News" Distribution: Good news reported, but heavy selling
    if ret_1d < -0.015 and vol_ratio > 1.4 and avg_sentiment >= 0.25:
        anomalies.append({
            "type": "🚨 SELL-THE-NEWS DISTRIBUTION",
            "description": f"Strongly positive news flow (Sentiment: {avg_sentiment:+.2f}), but price fell {ret_1d*100:+.1f}% on above-average volume. Institutional smart money is selling into retail enthusiasm.",
            "level": "DANGER"
        })
        severity = "DANGER"

    # If no anomalies triggered
    if not anomalies:
        anomalies.append({
            "type": "🟢 CLEAN ORGANIC PRICE ACTION",
            "description": "No synthetic volume traps, divergences, or pump anomalies detected. Price action, momentum, and volume flow are in healthy structural alignment.",
            "level": "CLEAN"
        })

    # Summary Badge
    if severity == "CLEAN":
        status_badge = "🟢 VERIFIED CLEAN (No Fakeout Detected)"
    elif severity == "SUSPICIOUS":
        status_badge = "🟡 SUSPICIOUS TRAP DIVERGENCE (Caution Advised)"
    else:
        status_badge = "🔴 ANOMALY DETECTED (High Manipulation / Trap Risk)"

    return {
        "severity": severity,
        "status_badge": status_badge,
        "anomalies": anomalies,
    }
