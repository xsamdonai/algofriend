import numpy as np

# In a real scenario, this would be a PyTorch/TensorFlow Two-Tower model
# trained on User <-> Item interactions (e.g., clicks). 
# Here, we simulate generating embeddings.

EMBEDDING_DIM = 64

class MockTwoTowerModel:
    def __init__(self):
        # We simulate a "trained" embedding space
        np.random.seed(42)
        
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        # Simulate a deterministic embedding for a user
        np.random.seed(user_id)
        emb = np.random.randn(EMBEDDING_DIM).astype('float32')
        return emb / np.linalg.norm(emb)
        
    def get_item_embedding(self, item_id: int) -> np.ndarray:
        # Simulate a deterministic embedding for an item
        np.random.seed(item_id + 10000)
        emb = np.random.randn(EMBEDDING_DIM).astype('float32')
        return emb / np.linalg.norm(emb)
        
    def batch_item_embeddings(self, item_ids: list[int]) -> np.ndarray:
        embeddings = [self.get_item_embedding(uid) for uid in item_ids]
        return np.vstack(embeddings)

if __name__ == "__main__":
    model = MockTwoTowerModel()
    user_emb = model.get_user_embedding(1938)
    item_emb = model.get_item_embedding(482)
    similarity = np.dot(user_emb, item_emb)
    print(f"Similarity between User 1938 and Item 482: {similarity:.4f}")
