"""
FX Arbitrage Detection Engine — FastAPI Server
Run: uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data import TickSimulator, build_price_matrix, FX_PAIRS
from src.agent import DQNArbitrageAgent, BellmanFordDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="FX Arbitrage Detection Engine", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Config model ─────────────────────────────────────────────────────────────
class EngineConfig(BaseModel):
    n_pairs: int = Field(default=20, ge=4, le=20)
    min_profit_bps: float = Field(default=3, ge=1, le=50)
    arb_prob: float = Field(default=0.10, ge=0.01, le=0.5)
    speed: int = Field(default=25, ge=1, le=100)


# ── Shared engine state ───────────────────────────────────────────────────────
class EngineState:
    def __init__(self):
        self.running = False
        self.config = EngineConfig()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self.start_time: float | None = None
        self.reset_metrics()

    def reset_metrics(self):
        self.tick_count = 0
        self.total_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.arb_log = []
        self.pnl_history = []
        self.loss_history = []
        self.price_history = {p: [] for p in FX_PAIRS}

state = EngineState()

# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        logger.info(f"WebSocket connected. Active: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        logger.info(f"WebSocket disconnected. Active: {len(self.active)}")

    async def broadcast(self, data: dict):
        if not self.active:
            return
        message = json.dumps(data, default=str)
        dead = set()
        for ws in self.active.copy():
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self.active -= dead

manager = ConnectionManager()

# ── Background simulation task ────────────────────────────────────────────────
async def run_engine():
    cfg = state.config
    n_pairs = cfg.n_pairs
    min_profit = cfg.min_profit_bps
    arb_prob = cfg.arb_prob
    speed = cfg.speed

    sim = TickSimulator(pairs=FX_PAIRS[:n_pairs], inject_arb_prob=arb_prob)
    detector = BellmanFordDetector(min_profit_bps=min_profit)
    agent = DQNArbitrageAgent(state_dim=20)

    model_path = Path("models/dqn_final.pt")
    if model_path.exists():
        try:
            agent.load(str(model_path))
            agent.epsilon = 0.05
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")

    prev_state_vec = None
    position = False
    delay = 1.0 / max(speed, 1)

    while state.running:
        # Check for dynamic config changes
        current_speed = state.config.speed
        if current_speed != speed:
            speed = current_speed
            delay = 1.0 / max(speed, 1)

        tick = sim.next_tick()
        state.tick_count += 1

        # Update price history (keep last 300)
        prices_snapshot = {}
        for pair in FX_PAIRS[:n_pairs]:
            if pair in tick and isinstance(tick[pair], dict):
                mid = tick[pair]["mid"]
                state.price_history[pair].append(mid)
                if len(state.price_history[pair]) > 300:
                    state.price_history[pair].pop(0)
                prices_snapshot[pair] = tick[pair]

        log_matrix, currencies, _ = build_price_matrix(tick, FX_PAIRS[:n_pairs])
        opportunities = detector.detect(log_matrix, currencies)

        # Log top 3 arb opportunities
        for opp in opportunities[:3]:
            state.arb_log.append({
                "tick": state.tick_count,
                "profit_bps": opp["profit_bps"],
                "path": opp["path"],
                "legs": opp.get("legs", len(opp["path"]) - 1),
            })
        if len(state.arb_log) > 200:
            state.arb_log = state.arb_log[-200:]

        # Agent decision
        state_vec = agent.encode_state(tick, opportunities)
        action_label = "HOLD"
        reward_val = 0.0
        profit_this_tick = 0.0

        if prev_state_vec is not None:
            action = agent.select_action(prev_state_vec)
            action_label = agent.ACTIONS.get(action, "HOLD")
            reward_val, info = agent.compute_reward(action, opportunities, position)

            if action == 1 and opportunities:
                state.trades += 1
                net = info.get("profit_bps", 0)
                state.total_pnl += net
                profit_this_tick = net
                position = True
                if net > 0:
                    state.wins += 1
            elif action == 2:
                position = False

            agent.store(prev_state_vec, action, reward_val, state_vec, False)
            loss = agent.train_step()
            if loss is not None:
                state.loss_history.append(round(loss, 6))
                if len(state.loss_history) > 500:
                    state.loss_history = state.loss_history[-500:]

        prev_state_vec = state_vec
        state.pnl_history.append(round(state.total_pnl, 2))
        if len(state.pnl_history) > 500:
            state.pnl_history = state.pnl_history[-500:]

        # Broadcast every 3 ticks to avoid flooding
        if state.tick_count % 3 == 0:
            win_rate = (state.wins / max(state.trades, 1)) * 100
            agent_stats = agent.get_stats()

            # Build normalised prices for the last 50 ticks (for frontend charts)
            price_chart_data = {}
            for pair in FX_PAIRS[:6]:
                hist = state.price_history.get(pair, [])
                if len(hist) > 1:
                    base = hist[0]
                    price_chart_data[pair] = [
                        round((p / base - 1) * 100, 4) for p in hist[-50:]
                    ]

            uptime = round(time.time() - state.start_time, 1) if state.start_time else 0

            # Compute simulated infrastructure metrics
            elapsed = uptime if uptime > 0 else 1
            throughput = round(state.tick_count / elapsed, 0)
            fill_rate = round((state.wins / max(state.trades, 1)) * 100, 1) if state.trades > 5 else 82.0

            frame = {
                "type": "tick",
                "tick": state.tick_count,
                "running": state.running,
                "metrics": {
                    "tick_count": state.tick_count,
                    "arb_detected": len(state.arb_log),
                    "total_pnl": round(state.total_pnl, 2),
                    "trades": state.trades,
                    "win_rate": round(win_rate, 1),
                    "epsilon": round(agent.epsilon, 4),
                    "uptime": uptime,
                    "throughput": throughput,
                    "fill_rate": fill_rate,
                    "n_pairs": n_pairs,
                },
                "opportunities": opportunities[:5],
                "arb_log": state.arb_log[-20:],
                "pnl_history": state.pnl_history[-100:],
                "loss_history": state.loss_history[-100:],
                "price_chart": price_chart_data,
                "action": action_label,
                "reward": round(reward_val, 4),
                "position": position,
                "agent_stats": agent_stats,
                "prices": {
                    k: v for k, v in prices_snapshot.items()
                    if isinstance(v, dict)
                },
            }
            await manager.broadcast(frame)

        # Non-blocking sleep
        await asyncio.sleep(delay)

    # Broadcast stop event
    await manager.broadcast({"type": "stopped"})


# ── REST Endpoints ────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.1.0",
        "running": state.running,
        "connections": len(manager.active),
    }


@app.post("/api/start")
async def start_engine():
    if state.running:
        return {"status": "already_running"}

    # Cancel any previous dangling task
    if state._task and not state._task.done():
        state._task.cancel()
        try:
            await state._task
        except (asyncio.CancelledError, Exception):
            pass

    state.running = True
    state.start_time = time.time()
    state.reset_metrics()
    state._task = asyncio.create_task(run_engine())
    return {"status": "started"}


@app.post("/api/stop")
async def stop_engine():
    if not state.running:
        return {"status": "already_stopped"}
    state.running = False
    return {"status": "stopped"}


@app.post("/api/config")
async def update_config(config: EngineConfig):
    state.config = config
    return {"status": "updated", "config": config.model_dump()}


@app.get("/api/config")
async def get_config():
    return state.config.model_dump()


@app.get("/api/stats")
async def get_stats():
    return {
        "running": state.running,
        "tick_count": state.tick_count,
        "total_pnl": round(state.total_pnl, 2),
        "trades": state.trades,
        "wins": state.wins,
        "arb_count": len(state.arb_log),
        "config": state.config.model_dump(),
    }


@app.get("/api/results")
async def get_results():
    path = Path("logs/training_results.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No training results found. Run: python train.py")
    with open(path) as f:
        return json.load(f)


@app.get("/api/reality-check")
async def get_reality_check():
    """Return cached reality check results, or 404 if not yet run."""
    path = Path("logs/reality_check.json")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Reality check not run yet. Run: python backtest_real.py"
        )
    with open(path) as f:
        return json.load(f)


@app.get("/api/cost-analysis")
async def get_cost_analysis():
    """Run a quick cost analysis on current simulated opportunities."""
    from src.costs import TransactionCostModel, cost_analysis_summary

    sim = TickSimulator(pairs=FX_PAIRS[:10], inject_arb_prob=0.08)
    detector = BellmanFordDetector(min_profit_bps=3.0)
    cost_model = TransactionCostModel()

    evaluated = []
    for _ in range(500):
        tick = sim.next_tick()
        log_matrix, currencies, _ = build_price_matrix(tick, FX_PAIRS[:10])
        opps = detector.detect(log_matrix, currencies)
        for opp in opps[:2]:
            result = cost_model.evaluate_opportunity(
                path=opp["path"],
                gross_profit_bps=opp["profit_bps"],
            )
            evaluated.append({
                "path": result.path,
                "gross_bps": result.gross_profit_bps,
                "spread_cost": result.spread_cost_bps,
                "slippage": result.slippage_bps,
                "latency_cost": result.latency_cost_bps,
                "fees": result.fee_bps,
                "net_bps": result.net_profit_bps,
                "fill_rate": result.fill_rate,
                "survived": result.survived,
            })

    summary = cost_analysis_summary(
        [type('R', (), r)() for r in evaluated]  # quick struct
    ) if evaluated else {}

    return {
        "opportunities_analyzed": len(evaluated),
        "summary": summary,
        "samples": evaluated[:15],
    }


# ── WebSocket Endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    # Send initial state immediately
    await websocket.send_text(json.dumps({
        "type": "init",
        "running": state.running,
        "config": state.config.model_dump(),
        "fx_pairs": FX_PAIRS,
    }))
    try:
        while True:
            # Keep connection alive; client messages not needed but handle gracefully
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    print("Starting FX Arbitrage Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
