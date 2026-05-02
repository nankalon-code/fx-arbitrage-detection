"""
FX Arbitrage Agent
- Bellman-Ford: detects negative-weight cycles = triangular arbitrage paths
- DQN: learns WHEN to execute (accounts for latency, spread, slippage)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import List, Tuple, Optional, Dict
import json

# FX pairs hardcoded to avoid cross-module import issues on Streamlit Cloud
FX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
    "EURCHF", "AUDJPY", "GBPCHF", "EURAUD", "EURCAD",
    "GBPAUD", "GBPCAD", "AUDCAD", "AUDCHF", "CADJPY"
]

# ─────────────────────────────────────────────
# BELLMAN-FORD ARBITRAGE DETECTOR
# ─────────────────────────────────────────────

class BellmanFordDetector:
    """
    Detect triangular / multi-leg arbitrage using Bellman-Ford
    on the log-price graph.

    Graph edges: log(rate(A→B)) = -log(rate(B→A))
    A negative cycle means: starting with 1 unit of currency A,
    trading through the cycle returns > 1 unit of A → free profit.

    Time complexity: O(V * E) — fast enough for 20 currencies
    """

    def __init__(self, min_profit_bps: float = 5.0):
        """
        min_profit_bps: minimum profit in basis points to flag (filters noise)
        """
        self.min_profit_bps = min_profit_bps / 10000.0

    def detect(
        self,
        log_rate_matrix: np.ndarray,
        currencies: List[str]
    ) -> List[Dict]:
        """
        Returns list of detected arbitrage opportunities:
        [{"path": ["EUR","USD","JPY","EUR"], "profit_bps": 12.3, "profit_pct": 0.00123}]
        """
        n = len(currencies)
        opportunities = []

        # Run Bellman-Ford from each source currency
        for src in range(n):
            dist = [float('inf')] * n
            pred = [-1] * n
            dist[src] = 0.0

            # Relax all edges n-1 times
            for _ in range(n - 1):
                updated = False
                for u in range(n):
                    if dist[u] == float('inf'):
                        continue
                    for v in range(n):
                        w = -log_rate_matrix[u][v]  # negate for profit maximization
                        if w != 0 and dist[u] + w < dist[v]:
                            dist[v] = dist[u] + w
                            pred[v] = u
                            updated = True
                if not updated:
                    break

            # Check for negative cycles (n-th relaxation)
            for u in range(n):
                if dist[u] == float('inf'):
                    continue
                for v in range(n):
                    w = -log_rate_matrix[u][v]
                    if w != 0 and dist[u] + w < dist[v] - 1e-10:
                        # Negative cycle found — reconstruct path
                        path = self._reconstruct_cycle(pred, v, n)
                        if path:
                            profit = self._compute_profit(path, log_rate_matrix, currencies)
                            if profit > self.min_profit_bps:
                                path_names = [currencies[i] for i in path]
                                opportunities.append({
                                    "path": path_names,
                                    "path_idx": path,
                                    "profit_bps": round(profit * 10000, 2),
                                    "profit_pct": round(profit * 100, 6),
                                    "legs": len(path) - 1
                                })

        # Deduplicate by frozenset of path
        seen = set()
        unique = []
        for opp in opportunities:
            key = frozenset(opp["path"])
            if key not in seen:
                seen.add(key)
                unique.append(opp)

        return sorted(unique, key=lambda x: -x["profit_bps"])

    def _reconstruct_cycle(self, pred: List[int], start: int, n: int) -> Optional[List[int]]:
        """Walk predecessors to extract the cycle"""
        visited = [False] * n
        node = start
        for _ in range(n):
            if visited[node]:
                cycle_start = node
                break
            visited[node] = True
            node = pred[node]
            if node == -1:
                return None
        else:
            return None

        cycle = []
        cur = cycle_start
        while True:
            cycle.append(cur)
            cur = pred[cur]
            if cur == cycle_start:
                break
            if len(cycle) > n:
                return None
        cycle.append(cycle_start)
        cycle.reverse()
        return cycle

    def _compute_profit(
        self,
        path: List[int],
        log_rate_matrix: np.ndarray,
        currencies: List[str]
    ) -> float:
        """Compute actual profit (as fraction) from path"""
        total_log = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_log += log_rate_matrix[u][v]
        return np.exp(total_log) - 1.0


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# DQN NETWORK
# ─────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """
    Dueling DQN architecture:
    - Shared feature extractor
    - Value stream: V(s)
    - Advantage stream: A(s,a)
    - Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    Input state: [market features + arb features]
    Actions: 0=HOLD, 1=EXECUTE_ARB, 2=CLOSE_POSITION
    """

    def __init__(self, state_dim: int, action_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        val = self.value(feat)
        adv = self.advantage(feat)
        # Dueling combination
        return val + (adv - adv.mean(dim=1, keepdim=True))


# ─────────────────────────────────────────────
# DQN AGENT
# ─────────────────────────────────────────────

class DQNArbitrageAgent:
    """
    DQN agent that learns to execute FX arbitrage.

    State space (per tick):
    - Spread costs for each potential arb leg
    - Profit of best detected arb opportunity (bps)
    - Number of legs in best opportunity
    - Recent fill rate (EWMA)
    - Volatility of each pair in path
    - Time since last execution
    - Current position (0/1)

    Reward shaping:
    - +profit_bps if EXECUTE and arb is real
    - -spread_cost if EXECUTE and no arb
    - +0.01 per tick HOLD when no opportunity
    - -0.5 per tick HOLD when opportunity missed
    """

    ACTIONS = {0: "HOLD", 1: "EXECUTE", 2: "CLOSE"}

    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 3,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        device: str = "auto"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)
        self.buffer = ReplayBuffer()
        self.loss_history = []
        self.reward_history = []

        print(f"DQN Agent on {self.device} | State: {state_dim} | Actions: {action_dim}")
        print(f"Parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")

    def encode_state(self, tick_data: Dict, arb_opportunities: List[Dict]) -> np.ndarray:
        """
        Build a fixed-size state vector from tick + arb detector output
        """
        state = np.zeros(self.state_dim, dtype=np.float32)

        # Features 0-4: top arb opportunity stats
        if arb_opportunities:
            best = arb_opportunities[0]
            state[0] = min(best["profit_bps"] / 50.0, 1.0)   # normalized profit
            state[1] = best["legs"] / 5.0                      # normalized leg count
            state[2] = min(len(arb_opportunities) / 10.0, 1.0) # opportunity density
        else:
            state[0] = state[1] = state[2] = 0.0

        # Features 3-12: volatilities of first 10 pairs
        for i, pair in enumerate(FX_PAIRS[:10]):
            if pair in tick_data and isinstance(tick_data[pair], dict):
                state[3 + i] = min(tick_data[pair].get("vol", 0) * 10000, 1.0)

        # Features 13-14: time features (cyclical encoding)
        from datetime import datetime
        now = datetime.utcnow()
        state[13] = np.sin(2 * np.pi * now.hour / 24)
        state[14] = np.cos(2 * np.pi * now.hour / 24)

        # Feature 15: tick number (normalized)
        state[15] = min(tick_data.get("tick", 0) / 50000.0, 1.0)

        # Features 16-19: reserved (position, fill rate, etc.)
        state[16] = 0.0  # current position flag
        state[17] = 0.0  # recent pnl ewma
        state[18] = 0.0  # fill rate
        state[19] = 0.0  # spread cost

        return state

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection with decayed exploration"""
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * np.exp(
            -self.steps / self.epsilon_decay
        )
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()

    def compute_reward(
        self,
        action: int,
        arb_opportunities: List[Dict],
        position: bool
    ) -> Tuple[float, Dict]:
        """Reward shaping for arbitrage execution"""
        info = {}
        if action == 1:  # EXECUTE
            if arb_opportunities:
                best = arb_opportunities[0]
                raw_profit = best["profit_bps"]
                spread_cost = 2.0  # ~2bps typical round-trip
                net = raw_profit - spread_cost
                reward = net / 10.0  # scale to ~[-1, 1]
                info = {"executed": True, "profit_bps": net, "path": best["path"]}
            else:
                reward = -0.5  # penalty for phantom execution
                info = {"executed": False, "reason": "no_opportunity"}

        elif action == 0:  # HOLD
            reward = 0.01 if not arb_opportunities else -0.05
            info = {"held": True, "opportunity_missed": bool(arb_opportunities)}

        else:  # CLOSE
            reward = 0.0 if position else -0.1
            info = {"closed": True}

        return reward, info

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.reward_history.append(reward)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)

        # Double DQN: action selected by policy, value from target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def save(self, path: str):
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "steps": self.steps,
            "epsilon": self.epsilon,
            "loss_history": self.loss_history[-1000:],
            "reward_history": self.reward_history[-1000:],
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.steps = ckpt["steps"]
        self.epsilon = ckpt["epsilon"]
        self.loss_history = ckpt.get("loss_history", [])
        self.reward_history = ckpt.get("reward_history", [])
        print(f"Model loaded from {path} (step {self.steps})")

    def get_stats(self) -> Dict:
        if not self.reward_history:
            return {}
        rewards = np.array(self.reward_history[-500:])
        return {
            "steps": self.steps,
            "epsilon": round(self.epsilon, 4),
            "mean_reward_500": round(float(rewards.mean()), 4),
            "total_reward": round(float(sum(self.reward_history)), 2),
            "mean_loss_100": round(float(np.mean(self.loss_history[-100:])), 6) if self.loss_history else 0,
            "buffer_size": len(self.buffer),
            "sharpe_500": round(
                float(rewards.mean() / (rewards.std() + 1e-8)) * np.sqrt(252 * 6.5 * 3600), 2
            )
        }
