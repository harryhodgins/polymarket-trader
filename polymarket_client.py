import requests

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
        params = {
            "active": "true" if status == "open" else "false",
            "limit": limit
        }
        return self._make_request("GET", endpoint, params=params)

    def get_token_price(self, token_id, side="BUY"):
        if isinstance(token_id, list):
            raise ValueError(f"token_id must be a string, got list: {token_id}")

        token_id = str(token_id).strip()

        if token_id.startswith("["):
            raise ValueError(f"Invalid token_id format: {token_id}")

        url = "https://clob.polymarket.com/price"
        params = {
            "token_id": token_id,
            "side": side
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()