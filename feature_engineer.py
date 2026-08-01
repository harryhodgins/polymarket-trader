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

    initial_rows = len(df)
    df = df[df["price"] > 0]
    if len(df) < initial_rows:
        logger.warning(f"Dropped {initial_rows - len(df)} rows with zero prices.")

    logger.info("Calculating log returns and 24h rolling volatility...")
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["volatility_24h"] = df["log_return"].rolling(window=24).std()

    clean_df = df.dropna()
    dropped_rows = len(df) - len(clean_df)
    logger.info(
        f"Dropped {dropped_rows} rows containing NaN values from rolling calculations."
    )

    return clean_df
