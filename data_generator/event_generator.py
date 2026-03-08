import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer

# Configuration
KAFKA_BROKER = "localhost:9092"
TOPIC_NAME = "user_events"

# Connect Producer
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    print(f"Connected to Kafka broker at {KAFKA_BROKER}")
except Exception as e:
    print(f"Failed to connect to Kafka: {e}")
    producer = None

EVENT_TYPES = ["view", "like", "purchase", "scroll", "search", "comment", "bookmark"]

def generate_event():
    user_id = random.randint(1, 10000)
    item_id = random.randint(1, 50000)
    event_type = random.choices(
        EVENT_TYPES, 
        weights=[50, 20, 5, 10, 5, 5, 5], 
        k=1
    )[0]
    
    event = {
        "user_id": user_id,
        "item_id": item_id,
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session_id": f"sess_{random.randint(1000, 9999)}",
        "device_type": random.choice(["mobile", "desktop", "tablet"])
    }
    return event

def main():
    if not producer:
        print("Kafka producer is not available. Exiting.")
        return

    print(f"Starting to produce events to topic '{TOPIC_NAME}'...")
    try:
        while True:
            event = generate_event()
            producer.send(TOPIC_NAME, event)
            print(f"Produced event: {event}")
            # Simulate real-time traffic
            time.sleep(random.uniform(0.1, 1.0))
    except KeyboardInterrupt:
        print("Stopping event generation.")
    finally:
        producer.close()

if __name__ == "__main__":
    main()
