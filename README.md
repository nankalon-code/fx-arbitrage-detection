# ⚡ FX Arbitrage Detection Engine

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![D3.js](https://img.shields.io/badge/D3.js-7.x-f9a03c.svg)](https://d3js.org)

A high-performance, real-time algorithmic trading engine that detects and executes triangular (and multi-leg) arbitrage opportunities in Forex markets. 

Built with a **FastAPI + WebSockets** backend and a premium **React + D3.js** trading terminal frontend.

---

## 🧠 Architecture & Algorithms

### 1. Market Simulation (GARCH)
Real market data lacks enough arbitrage opportunities for rapid RL training. The `TickSimulator` generates realistic synthetic tick data across 12 major FX pairs, incorporating **GARCH(1,1) volatility clustering** and bid/ask spreads, while periodically injecting synthetic mispricings.

### 2. Detection (Bellman-Ford)
Arbitrage detection is modeled as finding **negative-weight cycles** in a directed graph:
* Nodes = Currencies (EUR, USD, JPY)
* Edges = Exchange Rates
* Weight = $-\log(\text{Rate})$

Running the **Bellman-Ford algorithm** $O(V \cdot E)$ on this graph detects cycles where trading 1 unit of currency returns $>1$ unit.

### 3. Execution (Dueling Double DQN)
Finding an arbitrage isn't enough—executing it incurs spread costs and latency risk. A Reinforcement Learning agent decides *when* to execute:
* **Double DQN**: Mitigates Q-value overestimation bias.
* **Dueling Architecture**: Separates the Value stream $V(s)$ from the Advantage stream $A(s,a)$, allowing the agent to learn the value of "holding" regardless of the action.
* **Reward Shaping**: Penalizes phantom executions and rewards net positive trades post-spread.

---

## 🚀 Quick Start

### Prerequisites
* Python 3.10+
* Node.js 18+

### Installation

1. **Clone the repository**
2. **Install Backend Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   ```

### Running the App

You can run both the backend and frontend simultaneously using the provided launcher:

**Windows:**
```cmd
start.bat
```

Or run them manually in separate terminals:

**Terminal 1 (Backend):**
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

The trading dashboard will be available at: **http://localhost:5173**

---

## 💻 Tech Stack

* **Backend**: Python, FastAPI, WebSockets, Uvicorn
* **Machine Learning**: PyTorch, NumPy, SciPy
* **Frontend**: React (Vite), TypeScript, D3.js (Force-directed graphs), Recharts, Vanilla CSS (Glassmorphism)
* **Data**: `yfinance` for historical backtesting, custom GARCH simulator for RL training.

---

## 📊 Dashboard Features

* **Real-time WebSockets**: True live data streaming at up to 50 ticks/second.
* **Live D3 Network Graph**: Visualizes the currency graph in real-time. Arbitrage cycles pulse neon green when detected.
* **Animated PnL & Loss Curves**: Tracks agent performance and training loss continuously.
* **Live Arb Feed**: A color-coded, scrolling terminal feed of detected opportunities.
* **Control Panel**: Adjust currency pairs, min profit threshold, injection probability, and simulation speed on the fly.
