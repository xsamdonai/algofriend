from datetime import timedelta
from feast import Entity, FeatureView, Field, PostgreSQLSource
from feast.types import Int32, Int64, Float32, String

# Define PostgreSQL Data Sources
user_stats_source = PostgreSQLSource(
    name="user_stats",
    query="SELECT * FROM user_features",
    timestamp_field="event_timestamp"
)

item_stats_source = PostgreSQLSource(
    name="item_stats",
    query="SELECT * FROM item_features",
    timestamp_field="event_timestamp"
)

# Define Entities
user = Entity(name="user_id", join_keys=["user_id"], value_type=Int32)
item = Entity(name="item_id", join_keys=["item_id"], value_type=Int32)

# Define Feature Views
user_stats_view = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="total_events_5m", dtype=Int64),
        Field(name="total_views_5m", dtype=Int64),
        Field(name="total_purchases_5m", dtype=Int64),
        Field(name="total_likes_5m", dtype=Int64),
    ],
    online=True,
    source=user_stats_source,
    tags={"team": "recommendation"}
)

item_stats_view = FeatureView(
    name="item_stats",
    entities=[item],
    ttl=timedelta(days=1),
    schema=[
        Field(name="item_total_events_5m", dtype=Int64),
        Field(name="item_total_views_5m", dtype=Int64),
        Field(name="item_total_purchases_5m", dtype=Int64),
        Field(name="item_total_likes_5m", dtype=Int64),
    ],
    online=True,
    source=item_stats_source,
    tags={"team": "recommendation"}
)
