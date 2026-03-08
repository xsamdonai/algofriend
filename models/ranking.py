import xgboost as xgb
import pandas as pd
import numpy as np
from feast import FeatureStore
import os

class RankingModel:
    def __init__(self, repo_path: str = "../feature_store"):
        self.repo_path = repo_path
        self.model = xgb.XGBRanker(
            tree_method="hist",
            objective="rank:pairwise",
            random_state=42,
            learning_rate=0.1,
            n_estimators=100
        )
        self.is_trained = False
        
        # In a real setup, Feast repo would be initialized here
        try:
            self.store = FeatureStore(repo_path=self.repo_path)
            self.has_feast = True
        except Exception as e:
            print(f"Feast store not found at {repo_path}. Using mock features. ({e})")
            self.has_feast = False

    def train_mock(self):
        """Simulate training the ranker with mock data."""
        print("Training Mock XGBoost Ranker...")
        # Create mock training data (User-Item interaction features)
        num_samples = 10000
        mock_X = pd.DataFrame({
            "user_total_events_5m": np.random.randint(1, 100, num_samples),
            "user_total_purchases_5m": np.random.randint(0, 10, num_samples),
            "item_total_events_5m": np.random.randint(10, 5000, num_samples),
            "item_total_purchases_5m": np.random.randint(0, 500, num_samples),
            "similarity_score": np.random.uniform(0.5, 1.0, num_samples) # From embedding
        })
        
        # Target: engagement score (e.g., click=1, purchase=5)
        mock_y = (
            mock_X["user_total_purchases_5m"] * 0.2 + 
            mock_X["item_total_purchases_5m"] * 0.1 + 
            mock_X["similarity_score"] * 50
        ) + np.random.normal(0, 2, num_samples)
        
        # Group info is required for learning to rank (e.g., sessions or queries)
        # We'll just group every 10 items as one "query"
        groups = [10] * (num_samples // 10)
        
        self.model.fit(mock_X, mock_y, group=groups)
        self.is_trained = True
        print("XGBoost Ranker trained successfully.")

    def score(self, user_id: int, candidate_item_ids: list[int]) -> list[tuple[int, float]]:
        if not self.is_trained:
            self.train_mock()
            
        num_candidates = len(candidate_item_ids)
        
        if self.has_feast:
            # Fetch from Feast Online Store (Redis)
            user_features = self.store.get_online_features(
                features=[
                    "user_stats:total_events_5m",
                    "user_stats:total_purchases_5m"
                ],
                entity_rows=[{"user_id": user_id}]
            ).to_dict()
            
            item_features = self.store.get_online_features(
                features=[
                    "item_stats:item_total_events_5m",
                    "item_stats:item_total_purchases_5m"
                ],
                entity_rows=[{"item_id": iid} for iid in candidate_item_ids]
            ).to_df()
            # Merge logic would go here
            pass
            
        # For demo purposes, we fallback to mock online features
        mock_features = pd.DataFrame({
            "user_total_events_5m": np.random.randint(10, 50, num_candidates),
            "user_total_purchases_5m": np.random.randint(0, 5, num_candidates),
            "item_total_events_5m": np.random.randint(100, 1000, num_candidates),
            "item_total_purchases_5m": np.random.randint(10, 100, num_candidates),
            "similarity_score": np.random.uniform(0.6, 0.95, num_candidates)
        })
        
        # Predict scores using the trained XGBoost model
        scores = self.model.predict(mock_features)
        
        # Combine items with their predicted score and sort descending
        ranked_items = list(zip(candidate_item_ids, scores))
        ranked_items.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_items

if __name__ == "__main__":
    ranker = RankingModel()
    candidates = [482, 109, 332, 981, 742]
    print(f"Input candidates for user 1938: {candidates}")
    ranked = ranker.score(user_id=1938, candidate_item_ids=candidates)
    print("Ranked results:")
    for item, score in ranked:
        print(f"Item {item}: Score {score:.4f}")
