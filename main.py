import json
import time
import matplotlib.pyplot as plt
from datetime import datetime
from polymarket_client import PolymarketClient


def main():
    client = PolymarketClient()

    market_id = "1163699"
    print(f"Fetching details for Market ID: {market_id}...")

    # Step 1: Get the market details
    market = client.get_market_details(market_id)
    if not market:
        print("Could not fetch market details.")
        return

    print(f"Selected Market: {market.get('question')}")

    # Extract the clobTokenIds
    token_ids_str = market.get("clobTokenIds", "[]")
    try:
        token_ids = json.loads(token_ids_str)
        if not token_ids:
            print("No token IDs found for this market.")
            return
        target_token_id = token_ids[0]  # Grab the "Yes" token
    except json.JSONDecodeError:
        print("Invalid token ID format.")
        return

    print(f"Target Asset ID: {target_token_id}")

    # Fetch chunked history
    print("\nFetching historical price data (Hourly candles)...")
    raw_response = client.get_token_history(
        token_id=target_token_id, fidelity=60, total_days=30, chunk_days=7
    )

    if not raw_response or "history" not in raw_response:
        print("No historical data found.")
        return

    history = raw_response["history"]
    print(f"\nSuccessfully fetched and merged {len(history)} historical data points.")

    # Check for price movement
    unique_prices = set(point["p"] for point in history)
    print(f"Number of unique prices in this dataset: {len(unique_prices)}")

    # plot data

    # Extract timestamps and prices
    timestamps = [point["t"] for point in history]
    prices = [point["p"] for point in history]

    # Convert Unix timestamps to Python datetime objects
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, label="Market Price", color="blue", linewidth=1.5)
    plt.title(
        f"Polymarket Price History (Last 30 Days)\n{market.get('question')}",
        fontsize=14,
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price / Probability", fontsize=12)
    plt.gcf().autofmt_xdate()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
