"""
pipeline.py - Orchestrates the full HMM regime backtest pipeline.
Call this from Streamlit, CLI, or notebook.
"""

import json
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from polymarket_client import PolymarketClient
from feature_engineer import engineer_features
from hmm import train_regime_model_walkforward
from backtest_engine import SimpleBacktester

logging.basicConfig(level=logging.WARNING)


@dataclass
class PipelineResults:
    """Container for all pipeline outputs."""

    # Data
    df: pd.DataFrame = None
    market_details: dict = None

    # Model
    model: object = None
    n_components: int = 0

    # Backtest
    backtest_summary: dict = field(default_factory=dict)
    equity_curve: pd.Series = None

    # Diagnostics
    state_stats: pd.DataFrame = None
    fwd_return_stats: pd.DataFrame = None

    # Metadata
    error: Optional[str] = None


def run_pipeline(
    market_id: str,
    fidelity: int = 60,
    total_days: int = 90,
    n_components: int = 4,
    min_train_size: int = 150,
    refit_every: int = 12,
    min_samples_per_state: int = 30,
    mapping_refresh_every: int = 12,
    min_edge: float = 0.0,
    min_tstat: float = 0.0,
    gap_penalty: bool = False,
    gap_threshold: float = 0.03,
    gap_slippage: float = 0.02,
) -> PipelineResults:
    """
    Run the full HMM regime backtest pipeline.

    Parameters
    ----------
    market_id : str
        Polymarket market ID
    fidelity : int
        Bar size in minutes (60 = 1h, 240 = 4h, 1440 = daily)
    total_days : int
        How many days of history to fetch
    n_components : int
        Number of HMM hidden states
    min_train_size : int
        Warm-up bars before states are assigned
    refit_every : int
        Bars between HMM refits
    min_samples_per_state : int
        Min observations per state before mapping is computed
    mapping_refresh_every : int
        Bars between state->signal mapping updates
    min_edge : float
        Minimum mean forward return to trade a state (0 = disabled)
    min_tstat : float
        Minimum t-statistic to trade a state (0 = disabled)
    gap_penalty : bool
        Whether to apply slippage penalty on large moves
    gap_threshold : float
        Price move % that triggers gap penalty
    gap_slippage : float
        Slippage penalty applied on gap bars

    Returns
    -------
    PipelineResults with all data, model, backtest results, and diagnostics
    """
    results = PipelineResults()
    client = PolymarketClient()

    try:
        # =====================================================================
        # 1. FETCH DATA
        # =====================================================================
        market = client.get_market_details(market_id)
        results.market_details = market

        token_ids = json.loads(market.get("clobTokenIds", "[]"))
        if not token_ids:
            results.error = f"No token IDs found for market {market_id}"
            return results

        target_token_id = token_ids[0]

        raw_response = client.get_token_history(
            token_id=target_token_id,
            fidelity=fidelity,
            total_days=total_days,
            chunk_days=7,
        )

        # =====================================================================
        # 2. FEATURE ENGINEERING
        # =====================================================================
        df = engineer_features(raw_response["history"])

        if len(df) < min_train_size + 50:
            results.error = (
                f"Not enough data: {len(df)} bars fetched, "
                f"but min_train_size={min_train_size} requires at least {min_train_size + 50}"
            )
            return results

        # =====================================================================
        # 3. TRAIN HMM
        # =====================================================================
        model, states = train_regime_model_walkforward(
            df,
            n_components=n_components,
            min_train_size=min_train_size,
            refit_every=refit_every,
        )

        df["state"] = states

        # Drop warm-up period
        warmup_mask = df["state"] == -1
        df = df[~warmup_mask].copy()

        if len(df) == 0:
            results.error = "No bars remaining after warm-up. Reduce min_train_size."
            return results

        results.df = df
        results.model = model
        results.n_components = n_components

        # =====================================================================
        # 4. DIAGNOSTICS: Forward return stats
        # =====================================================================
        results.fwd_return_stats = df["fwd_log_return"].describe().to_frame().T

        # =====================================================================
        # 5. RUN BACKTEST
        # =====================================================================
        df_bt = df.reset_index(drop=False)
        time_col = df_bt.columns[0]

        signal = 0
        last_mapping_at = -1
        state_to_signal = {s: 0 for s in df_bt["state"].unique()}

        bt = SimpleBacktester()

        for i in range(len(df_bt)):
            row = df_bt.iloc[i]
            timestamp = row[time_col]
            price = row["price"]
            state = row["state"]

            # Recompute state -> signal mapping
            if (
                i - last_mapping_at >= mapping_refresh_every
                and i > min_samples_per_state
            ):

                past = df_bt.iloc[:i].dropna(subset=["fwd_log_return"])
                stats = past.groupby("state")["fwd_log_return"].agg(
                    ["mean", "std", "count"]
                )
                stats = stats[stats["count"] >= min_samples_per_state]

                new_mapping = {s: 0 for s in df_bt["state"].unique()}
                if not stats.empty:
                    stats["tstat"] = stats["mean"] / (
                        stats["std"] / np.sqrt(stats["count"])
                    )
                    best_long = stats["mean"].idxmax()
                    best_short = stats["mean"].idxmin()

                    long_ok = stats.loc[best_long, "mean"] > min_edge
                    long_sig = stats.loc[best_long, "tstat"] > min_tstat
                    short_ok = stats.loc[best_short, "mean"] < -min_edge
                    short_sig = stats.loc[best_short, "tstat"] < -min_tstat

                    if long_ok and long_sig:
                        new_mapping[best_long] = 1
                    if short_ok and short_sig:
                        new_mapping[best_short] = -1

                state_to_signal = new_mapping
                last_mapping_at = i

            signal = state_to_signal.get(state, 0)

            # Apply gap penalty if enabled
            if gap_penalty:
                bt.update_with_gap_penalty(
                    timestamp,
                    price,
                    signal,
                    gap_threshold=gap_threshold,
                    gap_slippage=gap_slippage,
                )
            else:
                bt.update(timestamp, price, signal)

        results.backtest_summary = bt.get_summary()
        results.equity_curve = bt.get_equity_curve()

        # =====================================================================
        # 6. STATE DIAGNOSTICS
        # =====================================================================
        state_stats = (
            df.dropna(subset=["fwd_log_return"])
            .groupby("state")["fwd_log_return"]
            .agg(
                count="size",
                mean="mean",
                std="std",
                median="median",
                pct_positive=lambda x: (x > 0).mean(),
            )
            .round(5)
        )
        state_stats["tstat"] = (
            state_stats["mean"] / (state_stats["std"] / np.sqrt(state_stats["count"]))
        ).round(3)
        results.state_stats = state_stats

    except Exception as e:
        results.error = f"Pipeline error: {str(e)}"
        logging.error(results.error, exc_info=True)

    return results
