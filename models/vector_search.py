import faiss
import numpy as np
from models.two_tower import MockTwoTowerModel, EMBEDDING_DIM

class CandidateGenerator:
    def __init__(self, num_items: int = 50000):
        self.two_tower = MockTwoTowerModel()
        self.num_items = num_items
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM) # Inner Product -> Cosine Similarity for normalized vectors
        self.item_ids = list(range(1, num_items + 1))
        
        self._build_index()

    def _build_index(self):
        print(f"Building FAISS index with {self.num_items} items...")
        
        # Batch generation for efficiency
        batch_size = 10000
        for i in range(0, self.num_items, batch_size):
            batch_ids = self.item_ids[i:i + batch_size]
            embeddings = self.two_tower.batch_item_embeddings(batch_ids)
            self.index.add(embeddings)
            
        print("FAISS index built successfully.")

    def get_candidates(self, user_id: int, top_k: int = 100) -> list[int]:
        user_emb = self.two_tower.get_user_embedding(user_id)
        # Reshape to (1, D)
        query = np.expand_dims(user_emb, axis=0)
        
        # Search index
        distances, indices = self.index.search(query, top_k)
        
        # indices contains the row num in FAISS, which maps 1:1 to self.item_ids index
        # (Assuming item_ids 1 to N were added sequentially)
        candidate_item_ids = [self.item_ids[idx] for idx in indices[0]]
        return candidate_item_ids

if __name__ == "__main__":
    generator = CandidateGenerator(num_items=10000)
    user_id = 1938
    candidates = generator.get_candidates(user_id, top_k=10)
    print(f"Top 10 Candidate Items for User {user_id}: {candidates}")
