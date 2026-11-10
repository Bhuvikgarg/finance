"""
Tri-Class Calibrated Machine Learning Engine using LightGBM and CalibratedClassifierCV.
Outputs calibrated probabilities: P(Bullish Target Hit), P(Chop/Noise), P(Stop Loss Hit).
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV


def train_time_series_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_class",
    train_ratio: float = 0.8,
    params: Dict = None,
) -> Tuple[CalibratedClassifierCV, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train a calibrated multiclass model strictly preserving chronological temporal order.
    
    Classes:
        0: Timeout / Chop / Consolidation
        1: Bullish Win (Profit Target hit first)
        2: Bearish Loss (Stop Loss hit first)
    """
    # Only train on bars where target labeling has completed
    labeled_df = df[df[target_col].notna()].copy()
    
    n_samples = len(labeled_df)
    split_idx = int(n_samples * train_ratio)

    train_df = labeled_df.iloc[:split_idx].copy()
    test_df = labeled_df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    X_test = test_df[feature_cols]

    default_params = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": 100,
        "max_depth": 3,
        "num_leaves": 10,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    base_model = LGBMClassifier(**default_params)
    
    # Probability Calibration via Sigmoid
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
    calibrated_model.fit(X_train, y_train)

    # Predict Probabilities for Out-of-Sample Test Set
    test_probs = calibrated_model.predict_proba(X_test)
    test_df["prob_chop"] = test_probs[:, 0]
    test_df["prob_bull"] = test_probs[:, 1]
    test_df["prob_bear"] = test_probs[:, 2]

    # Calculate Edge and Expected Return (EV)
    test_df["edge"] = test_df["prob_bull"] - test_df["prob_bear"]
    pt_pct = test_df["target_pt_pct"].fillna(0.02)
    sl_pct = test_df["target_sl_pct"].fillna(0.015)
    test_df["expected_return"] = (test_df["prob_bull"] * pt_pct) - (test_df["prob_bear"] * sl_pct)

    # Actionable Signal Rule
    signals = []
    for _, row in test_df.iterrows():
        if row["prob_bull"] >= 0.38 and row["edge"] >= 0.08 and row["expected_return"] > 0:
            signals.append("🟢 BUY / LONG")
        elif row["prob_bear"] >= 0.40 or row["prob_bear"] > row["prob_bull"]:
            signals.append("🔴 AVOID / BEARISH")
        else:
            signals.append("⚪ CHOP / CASH")
    test_df["pred_signal"] = signals

    # Extract Base Model Feature Importance (average across calibrated folds)
    importances = np.mean([clf.estimator.feature_importances_ for clf in calibrated_model.calibrated_classifiers_], axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return calibrated_model, train_df, test_df, importance_df
