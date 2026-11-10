"""
Unsupervised Market Regime Detection using Gaussian Mixture Models (GMM).
Discovers Bull, Bear, and Volatile/Sideways market regimes without human labels.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class MarketRegimeDetector:
    """
    Detects macro and asset regimes in an unsupervised manner using returns,
    volatilities, and benchmark market context.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            n_init=3,
            random_state=random_state,
        )
        self.regime_mapping: Dict[int, str] = {}
        self.sorted_cluster_order: List[int] = []

    def extract_regime_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract low-frequency macro and volatility features suitable for regime clustering.
        """
        regime_df = pd.DataFrame(index=df.index)

        # Asset features
        if "ret_5d" in df.columns:
            regime_df["asset_ret_5d"] = df["ret_5d"]
        else:
            regime_df["asset_ret_5d"] = df["close"].pct_change(5)

        if "vol_20d" in df.columns:
            regime_df["asset_vol_20d"] = df["vol_20d"]
        else:
            regime_df["asset_vol_20d"] = df["close"].pct_change().rolling(20).std()

        # Benchmark / Market Context features
        if "bench_vol_20d" in df.columns:
            regime_df["bench_vol_20d"] = df["bench_vol_20d"]
        if "rel_ret_5d" in df.columns:
            regime_df["rel_ret_5d"] = df["rel_ret_5d"]

        feature_cols = [c for c in regime_df.columns if regime_df[c].notna().all()]
        return regime_df[feature_cols], feature_cols

    def fit(self, train_features: pd.DataFrame) -> "MarketRegimeDetector":
        """
        Fit GMM on training features and sort clusters semantically (Bull, Bear, Sideways).
        """
        scaled_X = self.scaler.fit_transform(train_features)
        self.gmm.fit(scaled_X)

        raw_clusters = self.gmm.predict(scaled_X)
        temp_df = train_features.copy()
        temp_df["cluster"] = raw_clusters

        # Sort clusters by average asset return: highest return = Bull (0), lowest return = Bear (2)
        mean_returns = temp_df.groupby("cluster")["asset_ret_5d"].mean()
        self.sorted_cluster_order = list(mean_returns.sort_values(ascending=False).index)

        # Map to friendly labels
        if self.n_regimes == 3:
            self.regime_mapping = {
                self.sorted_cluster_order[0]: "Bull Regime",
                self.sorted_cluster_order[1]: "Sideways / Choppy",
                self.sorted_cluster_order[2]: "Bear Regime",
            }
        else:
            for idx, c in enumerate(self.sorted_cluster_order):
                self.regime_mapping[c] = f"Regime {idx}"

        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime IDs, semantic names, and posterior probabilities for samples.
        """
        scaled_X = self.scaler.transform(features)
        raw_preds = self.gmm.predict(scaled_X)
        raw_probs = self.gmm.predict_proba(scaled_X)

        result_df = pd.DataFrame(index=features.index)
        
        # Remap raw cluster integers to standardized ordered IDs (0: Bull, 1: Sideways, 2: Bear)
        order_map = {raw: rank for rank, raw in enumerate(self.sorted_cluster_order)}
        result_df["regime_id"] = [order_map[r] for r in raw_preds]
        result_df["regime_name"] = [self.regime_mapping[r] for r in raw_preds]

        # Add soft probability columns
        for rank, raw_id in enumerate(self.sorted_cluster_order):
            label = self.regime_mapping[raw_id].lower().replace(" ", "_").replace("/", "")
            result_df[f"prob_{label}"] = raw_probs[:, raw_id]

        return result_df


def add_regime_features(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, MarketRegimeDetector, List[str]]:
    """
    Fits regime detector on chronological train split and appends regime signals to entire df.
    Strictly avoids lookahead bias by fitting only on train split.
    """
    detector = MarketRegimeDetector(n_regimes=3)
    regime_features, r_cols = detector.extract_regime_features(df)

    split_idx = int(len(df) * train_ratio)
    train_features = regime_features.iloc[:split_idx]

    detector.fit(train_features)
    regime_outputs = detector.transform(regime_features)

    combined_df = pd.concat([df, regime_outputs], axis=1)
    
    # Identify new feature columns to feed into supervised model
    new_cols = [c for c in regime_outputs.columns if c.startswith("prob_") or c == "regime_id"]
    return combined_df, detector, new_cols


if __name__ == "__main__":
    from pathlib import Path
    from data_loader import load_market_dataset
    from features import build_feature_dataset

    raw_cache = Path(__file__).resolve().parent.parent / "data" / "raw"
    stock_df, bench_df = load_market_dataset(cache_dir=raw_cache)
    df, feature_cols = build_feature_dataset(stock_df, bench_df)

    enriched_df, detector, regime_cols = add_regime_features(df)
    print("\nRegime Detection Summary:")
    print(enriched_df["regime_name"].value_counts())
    print("\nSample Output:")
    print(enriched_df[["close", "regime_name"] + regime_cols].tail(5))
