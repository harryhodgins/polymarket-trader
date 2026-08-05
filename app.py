"""
app.py - Streamlit UI for Polymarket HMM Regime Backtester
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline import run_pipeline

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Polymarket HMM Trader",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Polymarket HMM Regime Backtester")
st.markdown(
    "Explore Hidden Markov Model regimes on Polymarket prediction markets. "
    "Adjust parameters in the sidebar and hit **Run Backtest**."
)

# =============================================================================
# SIDEBAR: PARAMETERS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Parameters")

    st.subheader("Market")
    market_id = st.text_input("Market ID", value="665374")

    fidelity_options = {
        "1 hour": 60,
        "4 hours": 240,
        "Daily": 1440,
    }
    fidelity_label = st.selectbox("Bar Size", list(fidelity_options.keys()), index=1)
    fidelity = fidelity_options[fidelity_label]

    total_days = st.slider("History (days)", 30, 365, 90)

    st.subheader("HMM Model")
    n_components = st.slider("Regimes (states)", 2, 6, 4)
    min_train_size = st.slider("Warm-up bars", 50, 500, 150)
    refit_every = st.slider("Refit every (bars)", 6, 48, 12)

    st.subheader("Signal Mapping")
    min_samples_per_state = st.slider("Min samples/state", 10, 100, 30)
    mapping_refresh_every = st.slider("Mapping refresh (bars)", 6, 48, 12)
    min_edge = st.slider("Min edge", 0.0, 0.02, 0.0, 0.001)
    min_tstat = st.slider("Min t-stat", 0.0, 3.0, 0.0, 0.1)

    st.subheader("Execution")
    gap_penalty = st.checkbox("Apply gap penalty", value=True)
    if gap_penalty:
        gap_threshold = st.slider("Gap threshold (%)", 1, 10, 3) / 100
        gap_slippage = st.slider("Gap slippage (%)", 1, 5, 2) / 100
    else:
        gap_threshold = 0.03
        gap_slippage = 0.02

    st.divider()
    run_button = st.button("🚀 Run Backtest", type="primary", width="stretch")

# =============================================================================
# MAIN: RUN PIPELINE
# =============================================================================
if run_button:
    with st.spinner("Running backtest..."):
        results = run_pipeline(
            market_id=market_id,
            fidelity=fidelity,
            total_days=total_days,
            n_components=n_components,
            min_train_size=min_train_size,
            refit_every=refit_every,
            min_samples_per_state=min_samples_per_state,
            mapping_refresh_every=mapping_refresh_every,
            min_edge=min_edge,
            min_tstat=min_tstat,
            gap_penalty=gap_penalty,
            gap_threshold=gap_threshold,
            gap_slippage=gap_slippage,
        )

    # Store results in session state so they persist across reruns
    st.session_state["results"] = results
    st.session_state["market_id"] = market_id

# Display results if available
if "results" in st.session_state:
    results = st.session_state["results"]

    if results.error:
        st.error(f"❌ {results.error}")
        st.stop()

    df = results.df

    # =========================================================================
    # MARKET INFO
    # =========================================================================
    st.subheader("📋 Market Info")
    market = results.market_details
    col1, col2, col3 = st.columns(3)
    col1.metric("Market", market.get("question", "Unknown")[:50])
    col2.metric("Bars", len(df))
    col3.metric("States", results.n_components)

    # =========================================================================
    # TABS FOR DIFFERENT VIEWS
    # =========================================================================
    tab_price, tab_equity, tab_stats, tab_diagnose = st.tabs(
        ["📈 Price & Regimes", "💰 Equity Curve", "📊 State Stats", "🔬 Diagnostics"]
    )

    # =========================================================================
    # TAB 1: PRICE CHART WITH REGIMES
    # =========================================================================
    with tab_price:
        colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#9b0a9b", "#a6d854", "#ffd92f"]

        fig = go.Figure()

        # Price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["price"],
                mode="lines",
                name="Price",
                line=dict(color="black", width=1),
            )
        )

        # Regime backgrounds
        for i in range(results.n_components):
            mask = df["state"] == i
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=df.index[mask],
                        y=df["price"][mask],
                        mode="markers",
                        name=f"Regime {i}",
                        marker=dict(color=colors[i % len(colors)], size=3, opacity=0.5),
                    )
                )

        fig.update_layout(
            title="Price Colored by HMM Regime",
            xaxis_title="Time",
            yaxis_title="Price (Probability)",
            height=500,
        )
        st.plotly_chart(fig, width="stretch")

    # =========================================================================
    # TAB 2: EQUITY CURVE
    # =========================================================================
    with tab_equity:
        equity = results.equity_curve
        summary = results.backtest_summary

        if equity is not None and not equity.empty:
            # Buy & hold benchmark
            bh_series = df["price"]
            bh_pnl = bh_series - bh_series.iloc[0]
            bh_aligned = bh_pnl.reindex(equity.index, method="ffill")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity.values,
                    mode="lines",
                    name="HMM Strategy",
                    line=dict(color="#2ecc71", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bh_aligned.index,
                    y=bh_aligned.values,
                    mode="lines",
                    name="Buy & Hold",
                    line=dict(color="#3498db", width=2, dash="dash"),
                )
            )
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.update_layout(
                title="Cumulative PnL ($1 Notional)",
                xaxis_title="Time",
                yaxis_title="PnL",
                height=400,
            )
            st.plotly_chart(fig, width="stretch")

            # Summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total PnL", f"${summary.get('total_pnl', 0):.4f}")
            col2.metric("Win Rate", f"{summary.get('win_rate_bars', 0):.1f}%")
            col3.metric("Sharpe/bar", f"{summary.get('sharpe_per_bar', 0):.3f}")
            col4.metric("Max DD", f"${summary.get('max_drawdown', 0):.4f}")
            col5.metric("Trades", summary.get("trade_count", 0))

    # =========================================================================
    # TAB 3: STATE STATISTICS
    # =========================================================================
    with tab_stats:
        st.subheader("Forward Return by State")
        if results.state_stats is not None:
            st.dataframe(results.state_stats, width="stretch")

            # Bar chart of mean forward returns
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[f"State {s}" for s in results.state_stats.index],
                    y=results.state_stats["mean"],
                    marker_color=results.state_stats["mean"].apply(
                        lambda x: "#2ecc71" if x > 0 else "#e74c3c"
                    ),
                )
            )
            fig.update_layout(
                title="Mean Forward Return by State",
                xaxis_title="State",
                yaxis_title="Mean Return",
                height=350,
            )
            st.plotly_chart(fig, width="stretch")

    # =========================================================================
    # TAB 4: DIAGNOSTICS
    # =========================================================================
    with tab_diagnose:
        st.subheader("Raw Forward Return Distribution")
        if results.fwd_return_stats is not None:
            st.dataframe(results.fwd_return_stats, width="stretch")

        # Histogram of forward returns
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=df["fwd_log_return"].dropna(),
                nbinsx=50,
                marker_color="#9b59b6",
            )
        )
        fig.update_layout(
            title="Distribution of Next-Bar Returns",
            xaxis_title="Log Return",
            yaxis_title="Count",
            height=350,
        )
        st.plotly_chart(fig, width="stretch")

        # Data quality warning
        zero_pct = (df["fwd_log_return"] == 0).mean() * 100
        if zero_pct > 40:
            st.warning(
                f"⚠️ {zero_pct:.0f}% of bars have zero forward return. "
                f"This market is very sparse - consider longer timeframes "
                f"or a more liquid market."
            )

else:
    st.info(
        "👈 Set your parameters in the sidebar and click **Run Backtest** to get started."
    )
