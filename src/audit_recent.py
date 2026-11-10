"""
Walk-Forward Audit Engine: Evaluates model predictions over the last 3 trading sessions
and verifies day-by-day whether the model correctly anticipated market action.
"""

from typing import Dict, List
import numpy as np
import pandas as pd


def audit_last_n_days(
    df: pd.DataFrame,
    feature_cols: List[str],
    model,
    n_days: int = 3,
) -> pd.DataFrame:
    """
    Simulates a walk-forward check over the last n trading days.
    For each day:
    1. Reads features available at that day's close.
    2. Computes calibrated P(Bull), P(Chop), P(Bear), and Expected Return.
    3. Evaluates actual next-day price movement and classifies model decision impact.
    """
    # Filter out zero-volume holidays
    active_df = df[df["volume"] > 0].copy()
    
    if len(active_df) < n_days + 1:
        raise ValueError("Not enough trading sessions for walk-forward audit.")

    audit_records = []
    
    # We audit the last n_days
    # The last day (T) might not have next-day data yet if it's the live session
    test_indices = list(range(len(active_df) - n_days, len(active_df)))

    for idx in test_indices:
        current_bar = active_df.iloc[idx]
        date_str = active_df.index[idx].strftime("%Y-%m-%d")
        close_price = current_bar["close"]

        # 1. Model Prediction as of this day's close
        feat_vals = active_df[feature_cols].iloc[[idx]]
        probs = model.predict_proba(feat_vals)[0]
        p_chop, p_bull, p_bear = probs[0], probs[1], probs[2]
        edge = p_bull - p_bear
        
        atr = current_bar.get("atr_14", close_price * 0.015)
        pt_dist = 1.5 * atr / close_price
        sl_dist = 1.0 * atr / close_price
        ev = (p_bull * pt_dist) - (p_bear * sl_dist)

        # Signal
        if p_bull >= 0.38 and edge >= 0.08 and ev > 0:
            signal = "🟢 BUY / LONG"
        elif p_bear >= 0.40 or p_bear > p_bull:
            signal = "🔴 AVOID / BEARISH"
        else:
            signal = "⚪ CHOP / CASH"

        # 2. What actually happened next?
        has_next_bar = (idx + 1) < len(active_df)
        if has_next_bar:
            next_bar = active_df.iloc[idx + 1]
            next_date_str = active_df.index[idx + 1].strftime("%Y-%m-%d")
            next_close = next_bar["close"]
            next_high = next_bar["high"]
            next_low = next_bar["low"]
            next_ret = (next_close - close_price) / close_price * 100
            max_up = (next_high - close_price) / close_price * 100
            max_down = (next_low - close_price) / close_price * 100

            # 3. Model Verdict Assessment
            if "AVOID" in signal and next_ret < 0:
                verdict = "✅ SUCCESS: Protected Capital (Avoided Drop)"
            elif "BUY" in signal and next_ret > 0:
                verdict = "✅ SUCCESS: Captured Gain"
            elif "AVOID" in signal and next_ret > 0:
                verdict = "⚠️ MISSED: Bearish Bias during Rally"
            elif "BUY" in signal and next_ret < 0:
                verdict = "❌ LOSS: Incurred Drawdown"
            else:
                verdict = "⚪ NEUTRAL: Sidelined in Range"

            next_info = f"{next_date_str} ({next_ret:+.2f}%) [High: {max_up:+.2f}%, Low: {max_down:+.2f}%]"
        else:
            next_ret = np.nan
            verdict = "⏳ LIVE / PENDING: Currently Active Market Session"
            next_info = "Next Session Active"

        regime_label = current_bar.get("regime_name", "N/A")

        audit_records.append({
            "Date": date_str,
            "Close Price": f"₹{close_price:,.2f}",
            "Market Regime": regime_label,
            "P(Bull)": f"{p_bull * 100:.1f}%",
            "P(Chop)": f"{p_chop * 100:.1f}%",
            "P(Bear)": f"{p_bear * 100:.1f}%",
            "Edge (P_Bull - P_Bear)": f"{edge * 100:+.1f}%",
            "Exp Return (EV)": f"{ev * 100:+.2f}%",
            "Model Signal": signal,
            "Next Day Reality": next_info,
            "Model Audit Verdict": verdict,
        })

    return pd.DataFrame(audit_records)
