import requests
import time


class PolymarketClient:
    BASE_URL = "https://gamma-api.polymarket.com"

    def _make_request(self, method: str, endpoint: str, params=None, data=None):
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = requests.request(method, url, params=params, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} - {response.text}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"An unexpected error occurred: {req_err}")
        return None

    def get_markets(self, status: str = "open", limit: int = 10):
        endpoint = "markets"
        params = {"active": "true" if status == "open" else "false", "limit": limit}
        return self._make_request("GET", endpoint, params=params)

    def get_market_details(self, market_id: str):
        endpoint = f"markets/{market_id}"
        return self._make_request("GET", endpoint)

    def get_token_price(self, token_id, side="BUY"):
        if isinstance(token_id, list):
            raise ValueError(f"token_id must be a string, got list: {token_id}")

        token_id = str(token_id).strip()

        if token_id.startswith("["):
            raise ValueError(f"Invalid token_id format: {token_id}")

        url = "https://clob.polymarket.com/price"
        params = {"token_id": token_id, "side": side}

        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_token_history(
        self,
        token_id: str,
        fidelity: int = 60,
        total_days: int = 30,
        chunk_days: int = 7,
    ):
        """
        Fetches historical price data by chunking the time window to avoid API limits.
        """
        url = "https://clob.polymarket.com/prices-history"

        end_ts = int(time.time())
        start_ts = end_ts - (total_days * 24 * 60 * 60)

        all_history = []
        current_end = end_ts

        print(f"Fetching {total_days} days of data in {chunk_days}-day chunks...")

        while current_end > start_ts:
            current_start = current_end - (chunk_days * 24 * 60 * 60)
            if current_start < start_ts:
                current_start = start_ts

            params = {
                "market": token_id,
                "fidelity": fidelity,
                "startTs": current_start,
                "endTs": current_end,
            }

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                if "history" in data:
                    chunk_data = data["history"]
                    all_history.extend(chunk_data)
                    print(
                        f"  - Fetched {len(chunk_data)} points for chunk ending at {current_end}"
                    )
                else:
                    print(
                        f"  - No history key in response for chunk ending at {current_end}"
                    )
            except requests.exceptions.RequestException as req_err:
                print(
                    f"Error fetching chunk {current_start} to {current_end}: {req_err}"
                )
                if "response" in locals() and hasattr(response, "text"):
                    print(f"Response text: {response.text[:200]}...")

            current_end = current_start

            time.sleep(0.5)

        if not all_history:
            return None

        unique_history = {point["t"]: point for point in all_history}
        sorted_history = sorted(unique_history.values(), key=lambda x: x["t"])

        return {"history": sorted_history}
