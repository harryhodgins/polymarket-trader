import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SimpleBacktester:
    """
    Bare-bones backtester for signal evaluation.

    Ignores: fees, slippage, position sizing, cash, order execution.
    Tracks: directional PnL assuming $1 notional exposure per bar.

    Signal convention:
      +1 = long YES  (profit when YES price rises)
      -1 = long NO   (profit when YES price falls)
       0 = flat      (no exposure)
    """

    def __init__(self):
        self.position = 0
        self.prev_price = None
        self.pnl_series = []  # (timestamp, bar_pnl, cumulative_pnl)
        self.cumulative_pnl = 0.0
        self.trade_count = 0
        self.last_signal = 0

    def update(self, timestamp: pd.Timestamp, price: float, signal: int):
        """Process one bar. signal must be +1, -1, or 0."""
        if signal != self.last_signal:
            self.trade_count += 1
            self.last_signal = signal

        bar_pnl = 0.0
        if self.prev_price is not None:
            price_change = price - self.prev_price

            # --- GAP RISK TEST ---
            # If the market moved more than 3% in a single 4h bar,
            # it likely gapped on news. Market makers widen spreads instantly.
            pct_move = abs(price_change / self.prev_price) if self.prev_price > 0 else 0

            if pct_move > 0.03:
                # Assume you lose 2% to slippage/spread trying to enter the gap
                slippage_penalty = 0.02 * abs(self.position)
                bar_pnl = (self.position * price_change) - slippage_penalty
            else:
                bar_pnl = self.position * price_change
            # -----------------------

        self.cumulative_pnl += bar_pnl
        self.pnl_series.append((timestamp, bar_pnl, self.cumulative_pnl))

        self.position = signal
        self.prev_price = price

    def update_with_gap_penalty(
        self,
        timestamp: pd.Timestamp,
        price: float,
        signal: int,
        gap_threshold: float = 0.03,
        gap_slippage: float = 0.02,
    ):
        """Process one bar with gap penalty applied to large moves."""
        if signal != self.last_signal:
            self.trade_count += 1
            self.last_signal = signal

        bar_pnl = 0.0
        if self.prev_price is not None:
            price_change = price - self.prev_price
            pct_move = abs(price_change / self.prev_price) if self.prev_price > 0 else 0

            if pct_move > gap_threshold:
                slippage_penalty = gap_slippage * abs(self.position)
                bar_pnl = (self.position * price_change) - slippage_penalty
            else:
                bar_pnl = self.position * price_change

        self.cumulative_pnl += bar_pnl
        self.pnl_series.append((timestamp, bar_pnl, self.cumulative_pnl))
        self.position = signal
        self.prev_price = price

    def get_equity_curve(self) -> pd.Series:
        if not self.pnl_series:
            return pd.Series(dtype=float)
        return pd.Series(
            [c for _, _, c in self.pnl_series],
            index=pd.to_datetime([t for t, _, _ in self.pnl_series]),
            name="equity",
        )

    def get_summary(self) -> dict:
        if not self.pnl_series:
            return {}

        bar_pnls = np.array([b for _, b, _ in self.pnl_series])
        cum_pnls = np.array([c for _, _, c in self.pnl_series])

        # Drawdown on cumulative curve (add initial 0 baseline)
        curve = np.concatenate([[0.0], cum_pnls])
        peak = np.maximum.accumulate(curve)
        drawdowns = peak - curve
        max_dd = float(drawdowns.max())

        winning_bars = bar_pnls[bar_pnls > 0]
        losing_bars = bar_pnls[bar_pnls < 0]

        return {
            "trade_count": self.trade_count,
            "total_bars": len(bar_pnls),
            "total_pnl": float(self.cumulative_pnl),
            "max_drawdown": max_dd,
            "win_rate_bars": (
                float((bar_pnls > 0).mean() * 100) if len(bar_pnls) else 0.0
            ),
            "avg_bar_pnl": float(bar_pnls.mean()) if len(bar_pnls) else 0.0,
            "avg_win_bar": float(winning_bars.mean()) if len(winning_bars) else 0.0,
            "avg_loss_bar": float(losing_bars.mean()) if len(losing_bars) else 0.0,
            "sharpe_per_bar": (
                float(bar_pnls.mean() / (bar_pnls.std() + 1e-9))
                if len(bar_pnls)
                else 0.0
            ),
            "final_equity": 1000.0 + self.cumulative_pnl,  # assuming $1000 start
        }
