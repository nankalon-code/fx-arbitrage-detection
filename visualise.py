"""
FX Arbitrage — Visualisation Module
Generates: network graph, learning curves, Bellman-Ford steps, DQN architecture
Run standalone: python -m src.visualise
Or import: from src.visualise import plot_arb_graph, plot_learning_curves
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

OUT = Path("outputs/visuals")
OUT.mkdir(parents=True, exist_ok=True)

BG = "#0d1117"; CARD = "#1c2128"; BORDER = "#30363d"
GREEN = "#3fb950"; BLUE = "#58a6ff"; PURPLE = "#bc8cff"
AMBER = "#d29922"; CORAL = "#f78166"; TEXT = "#e6edf3"; MUTED = "#8b949e"


def plot_arb_graph(
    opportunities: List[Dict] = None,
    all_pairs: List[str] = None,
    save_path: str = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot currency network graph with arbitrage cycles highlighted.
    Can be called live from dashboard or standalone.

    Args:
        opportunities: list of dicts from BellmanFordDetector.detect()
        all_pairs: list of FX pair strings e.g. ['EURUSD', 'GBPUSD']
        save_path: if provided, saves figure there
    """
    currencies = ["EUR", "USD", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"]
    rates = {
        ("EUR","USD"):1.085, ("USD","EUR"):0.922,
        ("USD","JPY"):149.5, ("JPY","USD"):0.0067,
        ("GBP","USD"):1.270, ("USD","GBP"):0.787,
        ("EUR","GBP"):0.854, ("GBP","EUR"):1.170,
        ("USD","CHF"):0.890, ("CHF","USD"):1.123,
        ("AUD","USD"):0.655, ("USD","AUD"):1.527,
        ("USD","CAD"):1.360, ("CAD","USD"):0.735,
        ("NZD","USD"):0.605, ("USD","NZD"):1.653,
    }

    # Determine arb path currencies
    arb_currencies = set()
    arb_edges = set()
    if opportunities:
        best = opportunities[0]
        path = best.get("path", [])
        arb_currencies = set(path)
        for i in range(len(path)-1):
            arb_edges.add((path[i], path[i+1]))
        profit_bps = best.get("profit_bps", 0)
    else:
        # Demo: inject EUR→USD→JPY→EUR
        arb_currencies = {"EUR", "USD", "JPY"}
        arb_edges = {("EUR","USD"), ("USD","JPY"), ("JPY","EUR")}
        profit_bps = 12.4

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=BG)
    ax.set_facecolor(BG); ax.axis("off")

    G = nx.DiGraph()
    G.add_nodes_from(currencies)
    for (u,v), r in rates.items():
        G.add_edge(u, v, rate=r)

    pos = nx.circular_layout(G, scale=1.9)

    # Draw normal edges
    normal = [(u,v) for (u,v) in rates if (u,v) not in arb_edges]
    nx.draw_networkx_edges(G, pos, edgelist=normal,
        edge_color=BORDER, arrows=True, arrowsize=10,
        width=0.7, alpha=0.45, ax=ax,
        connectionstyle="arc3,rad=0.12",
        arrowstyle="-|>", min_source_margin=22, min_target_margin=22)

    # Draw arb edges with glow effect
    for u, v in arb_edges:
        x0,y0 = pos[u]; x1,y1 = pos[v]
        for lw, alpha in [(9,0.05),(6,0.12),(3,0.5),(1.5,1.0)]:
            ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle="-|>", color=GREEN,
                    lw=lw, alpha=alpha, connectionstyle="arc3,rad=0.18"),
                zorder=5)
        # Rate label
        mx = (x0+x1)/2 + (y1-y0)*0.15
        my = (y0+y1)/2 + (x0-x1)*0.15
        rate = rates.get((u,v), 0)
        ax.text(mx, my, f"{rate:.4f}", ha="center", fontsize=7.5,
                color=GREEN, alpha=0.85, fontweight="bold")

    # Nodes
    for node, (x,y) in pos.items():
        in_arb = node in arb_currencies
        color = GREEN if in_arb else CARD
        border= GREEN if in_arb else BORDER
        lw    = 2.5 if in_arb else 0.8
        size  = 1000 if in_arb else 700
        ax.scatter(x, y, s=size, c=color, zorder=10,
                   edgecolors=border, linewidths=lw)
        ax.text(x, y, node, ha="center", va="center",
                fontsize=11 if in_arb else 9,
                fontweight="bold",
                color=BG if in_arb else TEXT, zorder=11)

    # Profit label
    if arb_currencies:
        pts = np.array([pos[c] for c in arb_currencies if c in pos])
        cx, cy = pts.mean(axis=0)
        sign = "+" if profit_bps >= 0 else ""
        ax.text(cx, cy+0.1, f"{sign}{profit_bps:.1f} bps",
                ha="center", va="center", fontsize=14,
                fontweight="bold", color=GREEN,
                bbox=dict(boxstyle="round,pad=0.5", fc=CARD,
                          ec=GREEN, lw=1.5), zorder=12)

    # Legend
    legend = [
        mpatches.Patch(color=GREEN, label="Arbitrage cycle (detected)"),
        mpatches.Patch(color=BORDER, label="Normal exchange rate edges"),
    ]
    ax.legend(handles=legend, loc="lower right", facecolor=CARD,
              edgecolor=BORDER, labelcolor=TEXT, fontsize=9, framealpha=0.9)

    path_str = " → ".join(list(arb_currencies)[:3]) + " → " + list(arb_currencies)[0] if arb_currencies else ""
    ax.set_title(f"Live Arbitrage Detection  ·  {path_str}",
                 color=TEXT, fontsize=12, pad=10, fontweight="bold")

    plt.tight_layout()
    out = save_path or str(OUT / "arb_graph.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    if show: plt.show()
    plt.close()
    return fig


def plot_learning_curves(
    reward_history: List[float] = None,
    loss_history:   List[float] = None,
    pnl_history:    List[float] = None,
    epsilon_history: List[float]= None,
    save_path: str = None,
    show: bool = False
) -> plt.Figure:
    """Plot DQN training metrics. Pass actual histories or generates demo data."""
    np.random.seed(42)
    n = 500

    def demo_reward():
        r = -0.3 + 0.0006*np.arange(n) + 0.12*np.sin(np.arange(n)/35)
        return r + np.random.normal(0, 0.11, n)

    def demo_pnl(rw):
        pos = np.where(rw > 0, rw * 3.5, rw * 1.2)
        return np.cumsum(pos)

    rw  = np.array(reward_history[-n:]) if reward_history else demo_reward()
    pnl = np.array(pnl_history[-n:])    if pnl_history    else demo_pnl(rw)
    ls  = np.array(loss_history[-n:])   if loss_history   else (
          0.8 * np.exp(-np.arange(n)/120) + np.random.exponential(0.05, n))
    eps = np.array(epsilon_history[-n:]) if epsilon_history else (
          0.05 + 0.95 * np.exp(-np.arange(n)/150))

    def smooth(arr, w=30):
        return np.convolve(arr, np.ones(w)/w, mode="same")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor=BG)
    fig.patch.set_facecolor(BG)
    x = np.arange(len(rw))

    configs = [
        (axes[0,0], x, rw,  smooth(rw),  BLUE,   "Episode reward",       "Episode", "Reward"),
        (axes[0,1], x, pnl, smooth(pnl), GREEN,  "Cumulative PnL (bps)", "Episode", "PnL (bps)"),
        (axes[1,0], x, ls,  smooth(ls),  CORAL,  "Training loss",        "Step",    "Loss"),
        (axes[1,1], x, eps, eps,         AMBER,  "Epsilon (exploration)", "Step",    "ε"),
    ]

    for ax, xd, raw, sm, color, title, xl, yl in configs:
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel(xl, color=MUTED, fontsize=9)
        ax.set_ylabel(yl, color=MUTED, fontsize=9)
        ax.set_title(title, color=TEXT, fontsize=10, pad=6, fontweight="bold")
        ax.plot(xd, raw, color=color, alpha=0.18, linewidth=0.5)
        ax.plot(xd, sm,  color=color, linewidth=2.0)
        ax.fill_between(xd, sm, alpha=0.07, color=color)
        ax.axhline(0, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.6)
        ax.grid(axis="y", color=BORDER, linewidth=0.3, alpha=0.5)

    # Epsilon min line
    axes[1,1].axhline(0.05, color=CORAL, linewidth=0.9, linestyle="--", alpha=0.8)
    axes[1,1].text(len(rw)*0.7, 0.08, "min ε = 0.05", color=CORAL, fontsize=8)

    plt.suptitle("DQN Training Metrics — FX Arbitrage Agent",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = save_path or str(OUT / "learning_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    if show: plt.show()
    plt.close()
    return fig


def plot_volatility_heatmap(
    price_history: Dict[str, List[float]] = None,
    save_path: str = None,
    show: bool = False
) -> plt.Figure:
    """Volatility heatmap across all pairs — shows market microstructure"""
    from data import FX_PAIRS

    pairs = FX_PAIRS[:16]
    np.random.seed(7)

    if price_history:
        vols = []
        for p in pairs:
            prices = np.array(price_history.get(p, [1.0]*50))
            if len(prices) > 2:
                ret = np.diff(np.log(prices + 1e-10))
                vols.append(np.std(ret) * 10000)
            else:
                vols.append(0.0)
    else:
        base = {"EURUSD":1.2,"GBPUSD":1.8,"USDJPY":2.1,"USDCHF":0.9,
                "AUDUSD":1.4,"USDCAD":1.1,"NZDUSD":1.3,"EURGBP":0.8}
        vols = [base.get(p, 1.0) + np.random.uniform(-0.3,0.8) for p in pairs]

    vols = np.array(vols)
    mat  = vols.reshape(4, 4)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
    ax.set_facecolor(BG)

    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vols.min(), vmax=vols.max())

    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(pairs[0:4],  color=MUTED, fontsize=9)
    ax.set_yticklabels(pairs[::4],  color=MUTED, fontsize=9)

    for i in range(4):
        for j in range(4):
            idx = i*4+j
            if idx < len(pairs):
                v = mat[i,j]
                ax.text(j, i, f"{pairs[idx]}\n{v:.2f}", ha="center",
                        va="center", fontsize=8.5, color=BG if v > vols.mean() else TEXT,
                        fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Volatility (bps)", color=MUTED, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)

    ax.set_title("Real-Time Volatility Heatmap — 16 FX Pairs",
                 color=TEXT, fontsize=12, pad=10, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(BORDER)

    plt.tight_layout()
    out = save_path or str(OUT / "vol_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    if show: plt.show()
    plt.close()
    return fig


if __name__ == "__main__":
    print("Generating visualisations...")
    plot_arb_graph()
    print("  arb_graph.png")
    plot_learning_curves()
    print("  learning_curves.png")
    plot_volatility_heatmap()
    print("  vol_heatmap.png")
    print(f"\nAll saved to {OUT}/")
