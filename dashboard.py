"""
FX Arbitrage Detection — Live Streamlit Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import os
from pathlib import Path
from datetime import datetime

from src.data import TickSimulator, build_price_matrix, FX_PAIRS
from src.agent import DQNArbitrageAgent, BellmanFordDetector

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FX Arbitrage Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #0e1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .arb-alert {
        background: linear-gradient(135deg, #1a3a1a, #0e1117);
        border: 1px solid #00ff88;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .stMetric > div { background: #0e1117; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("FX Arb Engine")
    st.caption("Bellman-Ford + DQN")
    st.divider()

    mode = st.radio("Mode", ["Live Simulation", "Load Training Results"])
    n_pairs = st.slider("Currency pairs", 6, 12, 10)
    min_profit = st.slider("Min profit (bps)", 1, 20, 5)
    arb_prob = st.slider("Arb injection prob", 0.01, 0.20, 0.07, step=0.01)
    speed = st.slider("Ticks/second", 1, 50, 10)
    st.divider()
    run = st.button("Start Engine", type="primary", use_container_width=True)
    stop = st.button("Stop", use_container_width=True)

    st.divider()
    st.caption("Stack")
    st.code("Bellman-Ford O(V·E)\nDueling DQN\nDouble DQN\nGARCH vol sim\nStreamlit + Plotly", language="text")


# ── Main dashboard ────────────────────────────────────────────────────────────
st.title("Real-Time FX Arbitrage Detection")
st.caption(f"Monitoring {n_pairs} currency pairs | Min profit threshold: {min_profit} bps")

# Metric row
m1, m2, m3, m4, m5, m6 = st.columns(6)
metric_tick    = m1.empty()
metric_arb     = m2.empty()
metric_pnl     = m3.empty()
metric_trades  = m4.empty()
metric_winrate = m5.empty()
metric_eps     = m6.empty()

st.divider()

# Layout: chart left, arb feed right
col_chart, col_feed = st.columns([2, 1])

with col_chart:
    tab1, tab2, tab3 = st.tabs(["PnL Curve", "Price Feed", "Loss Curve"])
    chart_pnl   = tab1.empty()
    chart_price = tab2.empty()
    chart_loss  = tab3.empty()

with col_feed:
    st.subheader("Live Arb Opportunities")
    feed_container = st.empty()

# Agent stats row
st.divider()
stats_row = st.empty()

# ── Session state ────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "pnl_history" not in st.session_state:
    st.session_state.pnl_history = []
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []
if "price_history" not in st.session_state:
    st.session_state.price_history = {p: [] for p in FX_PAIRS[:n_pairs]}
if "arb_log" not in st.session_state:
    st.session_state.arb_log = []
if "tick_count" not in st.session_state:
    st.session_state.tick_count = 0
if "total_pnl" not in st.session_state:
    st.session_state.total_pnl = 0.0
if "trades" not in st.session_state:
    st.session_state.trades = 0
if "wins" not in st.session_state:
    st.session_state.wins = 0

if run:
    st.session_state.running = True
if stop:
    st.session_state.running = False


def render_pnl_chart(pnl_history):
    if len(pnl_history) < 2:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=pnl_history,
        mode="lines",
        line=dict(color="#00ff88", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,255,136,0.08)",
        name="Cumulative PnL (bps)"
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#1e2130", color="#888"),
        xaxis=dict(gridcolor="#1e2130", color="#888"),
        showlegend=False
    )
    return fig


def render_price_chart(price_history, pairs):
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, pair in enumerate(pairs[:6]):
        prices = price_history.get(pair, [])
        if len(prices) > 1:
            prices_norm = (np.array(prices) / prices[0] - 1) * 100
            fig.add_trace(go.Scatter(
                y=prices_norm,
                mode="lines",
                line=dict(width=1, color=colors[i % len(colors)]),
                name=pair
            ))
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#1e2130", color="#888", title="% change"),
        xaxis=dict(gridcolor="#1e2130", color="#888"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10))
    )
    return fig


def render_arb_feed(arb_log):
    if not arb_log:
        return "<p style='color:#666;font-size:0.8rem'>No opportunities detected yet...</p>"
    lines = []
    for entry in reversed(arb_log[-15:]):
        color = "#00ff88" if entry["profit_bps"] > 10 else "#ffaa00"
        path_str = " → ".join(entry["path"])
        lines.append(
            f'<div class="arb-alert" style="border-color:{color}">'
            f'<span style="color:{color}">+{entry["profit_bps"]:.1f} bps</span> '
            f'&nbsp;|&nbsp; {path_str}'
            f'&nbsp;|&nbsp; <span style="color:#666">tick {entry["tick"]}</span>'
            f'</div>'
        )
    return "\n".join(lines)


# ── Load training results mode ────────────────────────────────────────────────
if mode == "Load Training Results":
    results_path = Path("logs/training_results.json")
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        res = data.get("results", {})
        st.success("Training results loaded")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total PnL", f"{res.get('total_pnl_bps', 0):.0f} bps")
        c2.metric("Win Rate", f"{res.get('win_rate_pct', 0):.1f}%")
        c3.metric("Sharpe Ratio", f"{res.get('sharpe_ratio', 0):.2f}")
        c4.metric("Total Trades", res.get("total_trades", 0))

        pnl_curve = res.get("pnl_curve", [])
        if pnl_curve:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=pnl_curve,
                mode="lines",
                line=dict(color="#00ff88", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,255,136,0.08)"
            ))
            fig.update_layout(
                title="Cumulative PnL (bps) — Training Run",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="#1e2130"),
                xaxis=dict(gridcolor="#1e2130")
            )
            st.plotly_chart(fig, use_container_width=True)

        execs = res.get("executions", [])
        if execs:
            df = pd.DataFrame(execs)
            st.subheader("Recent Executions")
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("No training results found. Run `python train.py` first.")
    st.stop()


# ── Live simulation loop ──────────────────────────────────────────────────────
if st.session_state.running:
    sim = TickSimulator(pairs=FX_PAIRS[:n_pairs], inject_arb_prob=arb_prob)
    detector = BellmanFordDetector(min_profit_bps=min_profit)
    agent = DQNArbitrageAgent(state_dim=20)

    model_path = Path("models/dqn_final.pt")
    if model_path.exists():
        agent.load(str(model_path))
        agent.epsilon = 0.05

    prev_state = None
    position = False
    delay = 1.0 / speed

    for _ in range(5000):
        if not st.session_state.running:
            break

        tick = sim.next_tick()
        st.session_state.tick_count += 1

        # Update price history
        for pair in FX_PAIRS[:n_pairs]:
            if pair in tick:
                st.session_state.price_history[pair].append(tick[pair]["mid"])
                if len(st.session_state.price_history[pair]) > 300:
                    st.session_state.price_history[pair].pop(0)

        log_matrix, currencies, _ = build_price_matrix(tick, FX_PAIRS[:n_pairs])
        opportunities = detector.detect(log_matrix, currencies)

        # Log arb opportunities
        for opp in opportunities[:3]:
            st.session_state.arb_log.append({
                "tick": st.session_state.tick_count,
                **opp
            })

        # Agent decision
        state = agent.encode_state(tick, opportunities)
        if prev_state is not None:
            action = agent.select_action(prev_state)
            reward, info = agent.compute_reward(action, opportunities, position)
            if action == 1 and opportunities:
                st.session_state.trades += 1
                net = info.get("profit_bps", 0)
                st.session_state.total_pnl += net
                if net > 0:
                    st.session_state.wins += 1
            agent.store(prev_state, action, reward, state, False)
            agent.train_step()
            st.session_state.loss_history.append(
                agent.loss_history[-1] if agent.loss_history else 0
            )
        prev_state = state
        st.session_state.pnl_history.append(st.session_state.total_pnl)

        # Update UI every 5 ticks
        if st.session_state.tick_count % 5 == 0:
            wr = (st.session_state.wins / max(st.session_state.trades, 1)) * 100
            metric_tick.metric("Ticks", f"{st.session_state.tick_count:,}")
            metric_arb.metric("Arb Detected", len(st.session_state.arb_log))
            metric_pnl.metric("Total PnL", f"{st.session_state.total_pnl:.1f} bps")
            metric_trades.metric("Trades", st.session_state.trades)
            metric_winrate.metric("Win Rate", f"{wr:.1f}%")
            metric_eps.metric("ε", f"{agent.epsilon:.3f}")

            fig_pnl = render_pnl_chart(st.session_state.pnl_history)
            if fig_pnl:
                chart_pnl.plotly_chart(fig_pnl, use_container_width=True)

            fig_price = render_price_chart(st.session_state.price_history, FX_PAIRS[:n_pairs])
            chart_price.plotly_chart(fig_price, use_container_width=True)

            feed_container.markdown(
                render_arb_feed(st.session_state.arb_log), unsafe_allow_html=True
            )

            stats = agent.get_stats()
            stats_row.json(stats)

        time.sleep(delay)
else:
    st.info("Configure settings in the sidebar and click **Start Engine** to begin.")
    st.markdown("""
    **How it works:**
    1. **Tick Simulator** generates 20 FX pairs with GARCH volatility clustering
    2. **Bellman-Ford** detects negative-weight cycles in the log-rate graph = arbitrage paths
    3. **Dueling DQN** learns *when* to execute (accounting for spread costs and latency)
    4. **Reward shaping** penalises phantom executions, rewards net positive trades

    Run `python train.py --ticks 50000` to pre-train the agent, then reload this dashboard.
    """)
