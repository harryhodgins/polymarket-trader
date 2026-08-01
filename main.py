import json
from polymarket_client import PolymarketClient

def main():
    client = PolymarketClient()

    print("Fetching open markets...")
    markets = client.get_markets()

    if not markets:
        print("No markets found")
        return

    print(f"Successfully fetched {len(markets)} markets.")

    market = markets[0]

    token_ids = json.loads(market.get("clobTokenIds", "[]"))

    print("Raw token_ids:", token_ids)

    if not token_ids:
        print("No token IDs found")
        return

    token_id = token_ids[0]

    print(f"Fetching price for token: {token_id}")

    price = client.get_token_price(token_id, side="BUY")

    print(price)

if __name__ == "__main__":
    main()