"""
Kafka Tick Producer
Pumps FX ticks into the 'fx-ticks' Kafka topic
Run: python -m src.kafka_producer
"""
import json, time, os
from kafka import KafkaProducer
from src.data import TickSimulator, FX_PAIRS

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode(),
    compression_type="gzip",
    batch_size=16384,
    linger_ms=5,
)

sim = TickSimulator(pairs=FX_PAIRS[:12], inject_arb_prob=0.07)
print(f"Producer started → Kafka at {KAFKA_BOOTSTRAP}")

tick_count = 0
while True:
    tick = sim.next_tick()
    producer.send("fx-ticks", tick)
    tick_count += 1
    if tick_count % 1000 == 0:
        print(f"Published {tick_count:,} ticks")
    time.sleep(0.001)  # ~1000 ticks/sec
