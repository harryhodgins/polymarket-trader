import logging
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def train_regime_model(df, n_components=3):
    features = ["log_return", "volatility_24h"]
    X = df[features].values

    logger.info(f"Standardizing features: {features}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Initializing Gaussian HMM with {n_components} components...")
    model = GaussianHMM(
        n_components=n_components, covariance_type="diag", n_iter=100, random_state=42
    )

    logger.info("Fitting model to data (this may take a moment)...")
    model.fit(X_scaled)

    score = model.score(X_scaled)
    logger.info(f"Model training complete. Final log-likelihood score: {score:.4f}")

    hidden_states = model.predict(X_scaled)
    logger.info(f"Predicted hidden states for {len(hidden_states)} time steps.")

    return model, hidden_states
