import json
import time
import random
import logging
from typing import Dict, Any
from datetime import datetime
from kafka import KafkaProducer

# -------------------------------------------------------------
# Configuration & Logging 🛠️
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - ⚡ %(message)s')
KAFKA_BROKER = "localhost:9092"
TOPIC_NAME = "user_events"

# -------------------------------------------------------------
# Connect Producer 🔌
# -------------------------------------------------------------
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=3,  # Advanced: Added retries for robustness
        linger_ms=10 # Advanced: Micro-batching for higher throughput
    )
    logging.info(f"Connected to Kafka broker at {KAFKA_BROKER} 🚀")
except Exception as e:
    logging.error(f"Failed to connect to Kafka: {e} ❌")
    producer = None

# Realistic event distribution weights 📊
EVENT_TYPES = ["view", "like", "purchase", "scroll", "search", "comment", "bookmark"]
EVENT_WEIGHTS = [50.0, 15.0, 2.0, 20.0, 8.0, 3.0, 2.0]

def generate_event() -> Dict[str, Any]:
    """Generates a highly realistic mock user interaction event. 🎭"""
    user_id = random.randint(1, 100000) # Scaled up users
    item_id = random.randint(1, 500000) # Scaled up items
    
    event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]
    
    event = {
        "event_id": f"evt_{random.getrandbits(32):08x}", # Advanced: unique event ID
        "user_id": user_id,
        "item_id": item_id,
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session_id": f"sess_{random.randint(10000, 99999)}",
        "device_type": random.choices(["mobile", "desktop", "tablet"], weights=[70, 25, 5])[0]
    }
    return event

def main():
    if not producer:
        logging.critical("Kafka producer is not available. Exiting! 💀")
        return

    logging.info(f"Starting to produce events to topic '{TOPIC_NAME}'... 🎯")
    try:
        while True:
            event = generate_event()
            producer.send(TOPIC_NAME, event)
            logging.info(f"Produced event: {event['event_type']} by user {event['user_id']} 📤")
            # Simulate realistic bursty traffic using Pareto distribution
            time.sleep(random.paretovariate(1.5) * 0.1)
    except KeyboardInterrupt:
        logging.warning("Stopping event generation (KeyboardInterrupt) 🛑")
    finally:
        producer.close()
        logging.info("Producer closed cleanly. ✨")

if __name__ == "__main__":
    main()
