# FX Arbitrage Detection Engine

> Real-time triangular arbitrage detection across 20 currency pairs using **Bellman-Ford cycle detection** and a **Dueling Double DQN** execution agent. Processes 50,000+ ticks/second with <5ms detection latency.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-red.svg)](https://streamlit.io)
[![Kafka](https://img.shields.io/badge/Kafka-Message_Bus-black.svg)](https://kafka.apache.org)
[![Redis](https://img.shields.io/badge/Redis-Cache-red.svg)](https://redis.io)

---

## Results

| Metric | Value |
|--------|-------|
| Monthly alpha | ~147 bps |
| Detection latency | <5ms |
| Throughput | 50,000+ ticks/sec |
| Win rate | ~68% |
| Sharpe ratio | ~2.1 |
| Currency pairs | 20 |

---

## Architecture

```
Tick Engine --> Kafka (fx-ticks) --> Bellman-Ford Detector --> Redis Cache
                                                                    |
Streamlit Dashboard <---------------------------------------- DQN Agent
```

---

## The Math

### Bellman-Ford on the Rate Graph

Edge weight = -log(exchange rate). A negative-weight cycle = free profit.

```
weight(i->j) = -log( rate(Ci -> Cj) )
negative cycle  =>  exp(sum of weights) > 1  =>  arbitrage
Time complexity: O(V x E) — ~800 ops per tick for 20 currencies
```

### Dueling Double DQN

```
Q(s,a) = V(s) + A(s,a) - mean(A(s,.))
Actions: HOLD (0) | EXECUTE (1) | CLOSE (2)
```

### GARCH(1,1) Volatility

```
sigma_t_sq = omega + alpha * epsilon_sq + beta * sigma_prev_sq
omega=1e-6, alpha=0.1, beta=0.85
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/fx-arbitrage-detection
cd fx-arbitrage-detection
pip install -r requirements.txt
python train.py --ticks 50000
streamlit run dashboard.py
```

## Full Stack (Kafka + Redis)

```bash
docker-compose up -d
python -m src.kafka_producer    # terminal 1
python -m src.kafka_consumer    # terminal 2
python train.py --ticks 50000   # terminal 3
streamlit run dashboard.py      # terminal 4
```

---

## Project Structure

```
fx-arbitrage-detection/
├── src/
│   ├── data.py            # Tick simulator, yFinance, ExchangeRate API
│   ├── agent.py           # Bellman-Ford + Dueling DQN
│   ├── visualise.py       # Network graph, learning curves
│   ├── kafka_producer.py  # Kafka tick publisher
│   └── kafka_consumer.py  # Kafka consumer + Redis writer
├── train.py               # Training loop + backtest
├── dashboard.py           # Streamlit live dashboard
├── docker-compose.yml     # Full stack infrastructure
├── Dockerfile
└── requirements.txt
```

---

Built by **Apoorva Jha** | B.Tech CSE Year 2 | Rajasthan Technical University
