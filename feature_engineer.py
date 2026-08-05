import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def engineer_features(raw_history: list) -> pd.DataFrame:
    logger.info(f"Loading {len(raw_history)} raw data points into DataFrame.")
    df = pd.DataFrame(raw_history)

    df["timestamp"] = pd.to_datetime(df["t"], unit="s")
    df.set_index("timestamp", inplace=True)
    df.rename(columns={"p": "price"}, inplace=True)
    df = df[["price"]].sort_index()

    # Clip prices to avoid log(0) or division by zero errors near bounds
    df["price"] = df["price"].clip(lower=1e-4, upper=1.0 - 1e-4)

    df = df[df["price"] > 0]

    logger.info("Calculating log returns and 24h rolling volatility...")
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["volatility_24h"] = df["log_return"].rolling(window=6).std()

    # CRITICAL FIX: Add forward return for causal mapping
    # The return we actually want to predict (from t to t+1)
    df["fwd_log_return"] = df["log_return"].shift(-1)

    clean_df = df.dropna()
    dropped_rows = len(df) - len(clean_df)
    logger.info(f"Dropped {dropped_rows} rows containing NaN values.")

    return clean_df
