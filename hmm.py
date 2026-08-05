import logging
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def _fit_single_model(X_window, n_components, random_state=42):
    """Fit one GaussianHMM on a fixed window of already-scaled data."""
    kmeans = KMeans(n_clusters=n_components, random_state=random_state, n_init=10)
    kmeans.fit(X_window)

    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=200,
        tol=1e-4,
        init_params="stc",
        params="stmc",
        random_state=random_state,
    )
    model.means_ = kmeans.cluster_centers_
    model.fit(X_window)
    return model


def train_regime_model_walkforward(
    df,
    n_components=3,
    min_train_size=500,
    refit_every=24,
    features=None,
    random_state=42,
):
    if features is None:
        features = ["log_return", "volatility_24h"]

    X_raw = df[features].values
    n = len(df)

    states = np.full(n, -1, dtype=int)
    current_model = None
    current_scaler = None
    last_refit_at = -1

    # For state alignment across refits
    baseline_means_scaled = None
    state_mapping = {i: i for i in range(n_components)}

    logger.info(f"Starting walk-forward training...")

    for t in range(min_train_size, n):
        need_refit = current_model is None or (t - last_refit_at) >= refit_every

        if need_refit:
            X_window_raw = X_raw[: t + 1]
            scaler = StandardScaler()
            X_window_scaled = scaler.fit_transform(X_window_raw)

            try:
                model = _fit_single_model(
                    X_window_scaled, n_components, random_state=random_state
                )
            except Exception as e:
                logger.warning(f"Refit failed at t={t}: {e}. Reusing previous model.")
                if current_model is None:
                    continue
            else:
                # CRITICAL FIX: Align new model states to baseline states
                if baseline_means_scaled is None:
                    baseline_means_scaled = model.means_.copy()
                else:
                    cost_matrix = cdist(
                        baseline_means_scaled, model.means_, metric="euclidean"
                    )
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    state_mapping = {
                        new_s: old_s for old_s, new_s in zip(row_ind, col_ind)
                    }

                current_model = model
                current_scaler = scaler
                last_refit_at = t

            # CRITICAL FIX: Use predict_proba (filtering) instead of predict (Viterbi)
            # This gives the causal marginal posterior P(q_t | O_1..t)
            proba = current_model.predict_proba(X_window_scaled)
            window_states_raw = np.argmax(proba, axis=1)
            window_states_aligned = np.array(
                [state_mapping[s] for s in window_states_raw]
            )
            states[t] = window_states_aligned[-1]

        else:
            X_window_raw = X_raw[: t + 1]
            X_window_scaled = current_scaler.transform(X_window_raw)
            proba = current_model.predict_proba(X_window_scaled)
            window_states_raw = np.argmax(proba, axis=1)
            window_states_aligned = np.array(
                [state_mapping[s] for s in window_states_raw]
            )
            states[t] = window_states_aligned[-1]

    logger.info("Walk-forward training complete.")
    return current_model, states
