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

    def get_markets(self, status: str = "open"):
        """
        Fetches markets from Polymarket.
        This is a placeholder, actual API endpoint and parameters might vary.
        """
        endpoint = "markets"
        params = params={"active": "true", "closed": "false", "limit": 1}

        return self._make_request("GET", endpoint, params=params)

    # Add more methods here for other API interactions (e.g., place_order, get_user_portfolio, etc.)
