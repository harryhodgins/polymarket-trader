import json
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

    colors = ["#66c2a5", "#fc8d62", "#8da0cb"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 14), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1]}
    )
    fig.suptitle(f"HMM Regime Analysis: {market.get('question')}", fontsize=16)

    # Plot 1: Background shading
    ax1.plot(df.index, df["price"], color="black", lw=1)
    for i in range(n_components):
        mask = df["state"] == i
        ax1.fill_between(
            df.index,
            df["price"].min(),
            df["price"].max(),
            where=mask,
            color=colors[i],
            alpha=0.15,
            step="post",
        )
    legend_handles = [
        Patch(color=colors[i], alpha=0.3, label=f"Regime {i}")
        for i in range(n_components)
    ]
    ax1.legend(handles=legend_handles, loc="upper left")
    ax1.set_ylabel("Price / Probability")
    ax1.set_title("Background Shading by Regime")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Price line colored by regime
    for i in range(len(df) - 1):
        s = df["state"].iloc[i]
        ax2.plot(
            df.index[i : i + 2],
            df["price"].iloc[i : i + 2],
            color=colors[s],
            lw=2,
        )
    ax2.set_ylabel("Price / Probability")
    ax2.set_title("Price Line Colored by Regime")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Plot 3: State timeline
    ax3.plot(df.index, df["state"], drawstyle="steps-post", color="black")
    ax3.set_ylabel("Regime State")
    ax3.set_title("Detected Market Regimes over Time")
    ax3.set_yticks(range(n_components))

    plt.tight_layout()
    logger.info("Rendering plot window...")
    plt.show()


if __name__ == "__main__":
    main()
