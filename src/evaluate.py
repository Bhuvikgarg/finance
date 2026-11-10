"""
Evaluation and backtesting module for calibrated Tri-Class Triple-Barrier market predictions.
"""

from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


def evaluate_predictions(test_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate ML multiclass classification and calibration metrics.
    """
    y_true = test_df["target_class"].astype(int)
    prob_matrix = test_df[["prob_chop", "prob_bull", "prob_bear"]].values
    y_pred = np.argmax(prob_matrix, axis=1)

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Log_Loss": float(log_loss(y_true, prob_matrix)),
        "Bull_Win_Rate": float((y_true == 1).mean()),
        "Bear_Loss_Rate": float((y_true == 2).mean()),
        "Chop_Timeout_Rate": float((y_true == 0).mean()),
    }
    return metrics


def backtest_probability_strategy(
    test_df: pd.DataFrame,
    return_col: str = "ret_1d",
) -> pd.DataFrame:
    """
    Simulate execution strategy:
    Go Long only when model issues '🟢 BUY / LONG' signal (edge >= +8% and positive EV).
    Otherwise hold Cash to protect capital against downside risk.
    """
    bt = test_df.copy()
    
    # Enter position on next session when model signals BUY at today's close
    bt["is_buy_signal"] = bt["pred_signal"].str.contains("BUY", na=False).astype(int)
    bt["position"] = bt["is_buy_signal"].shift(1).fillna(0)
    
    # Calculate strategy daily return
    bt["strat_ret"] = bt["position"] * bt[return_col]
    
    # Cumulative compound growth
    bt["cum_benchmark"] = (1 + bt[return_col]).cumprod()
    bt["cum_strategy"] = (1 + bt["strat_ret"]).cumprod()
    
    return bt


def plot_summary_report(
    bt_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    save_path: Path,
    symbol_name: str = "Target Equity",
):
    """
    Generate and save a visual summary plot comparing strategy vs buy & hold and top features.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Equity Curves
    ax1 = axes[0]
    ax1.plot(bt_df.index, bt_df["cum_benchmark"], label=f"Buy & Hold ({symbol_name})", color="gray", alpha=0.7)
    ax1.plot(bt_df.index, bt_df["cum_strategy"], label="Tri-Class Probability Strategy", color="forestgreen", lw=1.8)
    ax1.set_title("Out-of-Sample Performance Comparison")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Growth of $1.00")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Feature Importances
    ax2 = axes[1]
    top_features = importance_df.head(10).iloc[::-1]
    ax2.barh(top_features["feature"], top_features["importance"], color="steelblue")
    ax2.set_title("Top 10 Feature Importances (Calibrated Model)")
    ax2.set_xlabel("Importance Score")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Evaluation] Summary plot saved to: {save_path}")
