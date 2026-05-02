"""
Training loop + backtester for FX Arbitrage DQN
Run: python train.py --ticks 50000 --log-interval 500
"""

import argparse
import json
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from src.data import TickSimulator, YFinanceLoader, build_price_matrix, FX_PAIRS
from src.agent import DQNArbitrageAgent, BellmanFordDetector


LOG_DIR = Path("logs")
MODEL_DIR = Path("models")
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def run_training(
    n_ticks: int = 50_000,
    log_interval: int = 500,
    save_interval: int = 5_000,
    inject_arb_prob: float = 0.08,
    min_profit_bps: float = 4.0
):
    print(f"\n{'='*60}")
    print(f"  FX Arbitrage DQN Training")
    print(f"  Ticks: {n_ticks:,} | Arb injection prob: {inject_arb_prob}")
    print(f"{'='*60}\n")

    sim = TickSimulator(pairs=FX_PAIRS[:12], inject_arb_prob=inject_arb_prob)
    detector = BellmanFordDetector(min_profit_bps=min_profit_bps)
    agent = DQNArbitrageAgent(state_dim=20)

    # Metrics
    episode_rewards = []
    episode_losses = []
    executions = []
    arb_detections = []
    pnl_curve = [0.0]
    running_pnl = 0.0
    position = False
    wins = 0
    total_trades = 0

    log_data = {
        "config": {
            "n_ticks": n_ticks,
            "inject_arb_prob": inject_arb_prob,
            "min_profit_bps": min_profit_bps,
            "timestamp": datetime.utcnow().isoformat()
        },
        "snapshots": []
    }

    prev_state = None
    start_time = time.time()

    for tick_i in range(n_ticks):
        # Get tick
        tick = sim.next_tick()

        # Build log-rate matrix for Bellman-Ford
        log_matrix, currencies, idx = build_price_matrix(tick, FX_PAIRS[:10])

        # Detect arbitrage
        opportunities = detector.detect(log_matrix, currencies)
        arb_detections.append(len(opportunities))

        # Encode state
        state = agent.encode_state(tick, opportunities)

        if prev_state is not None:
            # Agent decides action
            action = agent.select_action(prev_state)
            reward, info = agent.compute_reward(action, opportunities, position)

            # Update position
            if action == 1 and opportunities:
                position = True
                total_trades += 1
                if info.get("profit_bps", 0) > 0:
                    wins += 1
                running_pnl += info.get("profit_bps", 0)
                executions.append({
                    "tick": tick_i,
                    "profit_bps": info.get("profit_bps", 0),
                    "path": info.get("path", [])
                })
            elif action == 2:
                position = False

            pnl_curve.append(running_pnl)

            # Store transition
            agent.store(prev_state, action, reward, state, False)
            episode_rewards.append(reward)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

        prev_state = state

        # Log progress
        if (tick_i + 1) % log_interval == 0:
            stats = agent.get_stats()
            elapsed = time.time() - start_time
            tps = tick_i / elapsed
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            avg_arb = np.mean(arb_detections[-log_interval:])

            print(
                f"Tick {tick_i+1:>7,} | "
                f"PnL: {running_pnl:>8.1f}bps | "
                f"Trades: {total_trades:>5} | "
                f"WinRate: {win_rate:>5.1f}% | "
                f"ε: {stats.get('epsilon', 0):.3f} | "
                f"Loss: {stats.get('mean_loss_100', 0):.5f} | "
                f"Arb/tick: {avg_arb:.2f} | "
                f"{tps:.0f}t/s"
            )

            snapshot = {
                "tick": tick_i + 1,
                "pnl_bps": round(running_pnl, 2),
                "trades": total_trades,
                "win_rate": round(win_rate, 2),
                "avg_arb_per_tick": round(avg_arb, 3),
                **stats
            }
            log_data["snapshots"].append(snapshot)

        # Save checkpoint
        if (tick_i + 1) % save_interval == 0:
            agent.save(MODEL_DIR / f"dqn_step_{tick_i+1}.pt")

    # Final save
    agent.save(MODEL_DIR / "dqn_final.pt")

    # Compute final metrics
    monthly_alpha = running_pnl / (n_ticks / 50000) if n_ticks > 0 else 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    rewards_arr = np.array(episode_rewards)
    sharpe = (
        rewards_arr.mean() / (rewards_arr.std() + 1e-8) * np.sqrt(252 * 390)
        if len(rewards_arr) > 0 else 0
    )

    results = {
        "total_ticks": n_ticks,
        "total_trades": total_trades,
        "wins": wins,
        "win_rate_pct": round(win_rate, 2),
        "total_pnl_bps": round(running_pnl, 2),
        "monthly_alpha_bps": round(monthly_alpha, 2),
        "sharpe_ratio": round(float(sharpe), 3),
        "final_epsilon": round(agent.epsilon, 4),
        "pnl_curve": [round(p, 2) for p in pnl_curve[::max(1, len(pnl_curve)//500)]],
        "executions": executions[-100:],
        "training_time_s": round(time.time() - start_time, 1)
    }

    log_data["results"] = results

    with open(LOG_DIR / "training_results.json", "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"  Total PnL:      {running_pnl:.1f} bps")
    print(f"  Win Rate:       {win_rate:.1f}%")
    print(f"  Total Trades:   {total_trades}")
    print(f"  Sharpe Ratio:   {sharpe:.2f}")
    print(f"  Monthly Alpha:  ~{monthly_alpha:.0f} bps")
    print(f"{'='*60}\n")

    return results


def run_backtest(model_path: str = None, n_ticks: int = 10_000):
    """Quick backtest on held-out simulated data"""
    print("\nRunning backtest...")

    sim = TickSimulator(pairs=FX_PAIRS[:12], inject_arb_prob=0.06)
    detector = BellmanFordDetector(min_profit_bps=4.0)
    agent = DQNArbitrageAgent(state_dim=20)

    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0.01  # Exploit only
    else:
        print("No model found — running random baseline")

    pnl = 0.0
    trades = 0
    wins = 0
    prev_state = None

    for _ in range(n_ticks):
        tick = sim.next_tick()
        log_matrix, currencies, _ = build_price_matrix(tick, FX_PAIRS[:10])
        opportunities = detector.detect(log_matrix, currencies)
        state = agent.encode_state(tick, opportunities)

        if prev_state is not None:
            action = agent.select_action(prev_state)
            reward, info = agent.compute_reward(action, opportunities, False)
            if action == 1 and opportunities:
                trades += 1
                net = info.get("profit_bps", 0)
                pnl += net
                if net > 0:
                    wins += 1
        prev_state = state

    print(f"Backtest | Trades: {trades} | PnL: {pnl:.1f}bps | WinRate: {wins/max(trades,1)*100:.1f}%")
    return {"pnl_bps": pnl, "trades": trades, "win_rate": wins / max(trades, 1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=50_000)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--backtest-only", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.backtest_only:
        run_backtest(args.model)
    else:
        results = run_training(n_ticks=args.ticks, log_interval=args.log_interval)
        run_backtest(str(MODEL_DIR / "dqn_final.pt"))
