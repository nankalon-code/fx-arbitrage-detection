"""
Kafka Consumer + Redis Writer
Reads ticks from Kafka, runs Bellman-Ford, writes results to Redis
Run: python -m src.kafka_consumer
"""
import json, os, time
import redis as rd
from kafka import KafkaConsumer
from src.agent import BellmanFordDetector
from src.data import build_price_matrix, FX_PAIRS

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
REDIS_HOST      = os.getenv("REDIS_HOST", "localhost")

r = rd.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
consumer = KafkaConsumer(
    "fx-ticks",
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_deserializer=lambda m: json.loads(m.decode()),
    auto_offset_reset="latest",
    group_id="arb-detector",
)
detector = BellmanFordDetector(min_profit_bps=5.0)
print(f"Consumer started ← Kafka:{KAFKA_BOOTSTRAP} → Redis:{REDIS_HOST}")

processed = 0
for message in consumer:
    tick = message.value
    log_matrix, currencies, _ = build_price_matrix(tick, FX_PAIRS[:10])
    opps = detector.detect(log_matrix, currencies)

    # Write to Redis
    r.set("latest_tick", json.dumps(tick))
    r.set("opportunities", json.dumps(opps))
    r.set("tick_count", tick.get("tick", 0))
    if opps:
        r.lpush("opp_history", json.dumps(opps[0]))
        r.ltrim("opp_history", 0, 999)

    processed += 1
    if processed % 1000 == 0:
        print(f"Processed {processed:,} ticks | Last arb count: {len(opps)}")
