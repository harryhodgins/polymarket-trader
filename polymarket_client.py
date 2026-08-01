import logging
import requests
import time

logger = logging.getLogger(__name__)


class PolymarketClient:
    BASE_URL = "https://gamma-api.polymarket.com"

    def _make_request(self, method: str, endpoint: str, params=None, data=None):
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = requests.request(method, url, params=params, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API Request failed: {e}")
            return None

    def get_markets(
        self,
        status: str = "open",
        limit: int = 10,
        order: str = "volume",
        ascending: bool = False,
    ):
        params = {
            "active": "true" if status == "open" else "false",
            "limit": limit,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        return self._make_request("GET", "markets", params=params)

    def get_market_details(self, market_id: str):
        return self._make_request("GET", f"markets/{market_id}")

    def get_token_price(self, token_id, side="BUY"):
        url = "https://clob.polymarket.com/price"
        params = {"token_id": str(token_id).strip(), "side": side}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_token_history(
        self,
        token_id: str,
        fidelity: int = 60,
        total_days: int = 90,
        chunk_days: int = 7,
    ):
        url = "https://clob.polymarket.com/prices-history"
        end_ts = int(time.time())
        start_ts = end_ts - (total_days * 24 * 60 * 60)

        all_history = []
        current_end = end_ts
        chunk_count = 0

        while current_end > start_ts:
            current_start = max(current_end - (chunk_days * 24 * 60 * 60), start_ts)
            params = {
                "market": token_id,
                "fidelity": fidelity,
                "startTs": current_start,
                "endTs": current_end,
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                chunk_data = response.json().get("history", [])
                all_history.extend(chunk_data)
                logger.info(
                    f"Downloaded chunk {chunk_count + 1}: {len(chunk_data)} data points."
                )
            else:
                logger.warning(
                    f"Chunk {chunk_count + 1} failed with status {response.status_code}."
                )

            current_end = current_start
            chunk_count += 1
            time.sleep(0.5)

        if not all_history:
            logger.error("No historical data was retrieved from any chunks.")
            return None

        unique_history = {point["t"]: point for point in all_history}
        logger.info(
            f"Data aggregation complete. Total unique points: {len(unique_history)}"
        )
        return {"history": sorted(unique_history.values(), key=lambda x: x["t"])}
