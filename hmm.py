import logging
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def train_regime_model(df, n_components=3):
    features = ["log_return", "volatility_24h"]
    X = df[features].values

    logger.info(f"Standardizing features: {features}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use k-means to initialize means
    kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    logger.info(f"Initializing Gaussian HMM with {n_components} components...")
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",  # Changed from "diag" to "full"
        n_iter=200,  # More iterations to find true optimum
        tol=1e-4,  # Tighter convergence tolerance
        init_params="stmc",  # Let k-means set means, learn rest
        params="stmc",
        random_state=42,
    )
    model.means_ = kmeans.cluster_centers_

    logger.info("Fitting model to data...")
    model.fit(X_scaled)

    score = model.score(X_scaled)
    logger.info(f"Model training complete. Final log-likelihood: {score:.4f}")

    hidden_states = model.predict(X_scaled)
    logger.info(f"Predicted hidden states for {len(hidden_states)} time steps.")

    return model, hidden_states
