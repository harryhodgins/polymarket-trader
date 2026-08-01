import json
import logging
import matplotlib.pyplot as plt
from polymarket_client import PolymarketClient
from feature_engineer import engineer_features
from hmm import train_regime_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(module)-15s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    client = PolymarketClient()
    market_id = "1163699"

    logger.info(f"Fetching details for Market ID: {market_id}...")
    market = client.get_market_details(market_id)
    if not market:
        logger.error("Could not fetch market details. Exiting.")
        return

    logger.info(f"Selected Market: {market.get('question')}")

    token_ids = json.loads(market.get("clobTokenIds", "[]"))
    if not token_ids:
        logger.error("No token IDs found. Exiting.")
        return

    target_token_id = token_ids[0]

    logger.info("Starting chunked historical data download...")
    raw_response = client.get_token_history(
        token_id=target_token_id, fidelity=60, total_days=90, chunk_days=7
    )

    if not raw_response or "history" not in raw_response:
        logger.error("No historical data found. Exiting.")
        return

    logger.info("Starting feature engineering pipeline...")
    df = engineer_features(raw_response["history"])
    logger.info(f"Dataset prepared. Total clean rows ready for HMM: {len(df)}")

    n_components = 3
    logger.info(f"Initializing Regime Detection with {n_components} hidden states...")
    model, states = train_regime_model(df, n_components=n_components)

    df["state"] = states
    logger.info("Generating regime visualization...")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(f"HMM Regime Analysis: {market.get('question')}", fontsize=16)

    # Thin continuous line preserves the true price path
    ax1.plot(df.index, df["price"], color="gray", linewidth=0.8, alpha=0.5)

    # Scatter points colored by regime — no false connections across time gaps
    for i in range(n_components):
        mask = df["state"] == i
        ax1.scatter(
            df.index[mask], df["price"][mask], label=f"Regime {i}", s=12, alpha=0.85
        )

    ax1.set_ylabel("Price / Probability")
    ax1.set_title("Market Price colored by HMM Regimes")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(df.index, df["state"], drawstyle="steps-post", color="black")
    ax2.set_ylabel("Regime State")
    ax2.set_title("Detected Market Regimes over Time")
    ax2.set_yticks(range(n_components))

    plt.tight_layout()
    logger.info("Rendering plot window...")
    plt.show()


if __name__ == "__main__":
    main()
