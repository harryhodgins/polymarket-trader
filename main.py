from polymarket_client import PolymarketClient
import os

def main():
    client = PolymarketClient()

    print("Fetching open markets...")
    markets = client.get_markets()

    if markets:
        print(f"Successfully fetched {len(markets)} markets.")
        for market in markets[:5]:
            print(f"- {market.get('question')} (ID: {market.get('id')})")
    else:
        print("Failed to fetch markets or no markets found.")

if __name__ == "__main__":
    main()
