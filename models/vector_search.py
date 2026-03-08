import faiss
import numpy as np
import logging
from typing import List, Dict

# -------------------------------------------------------------
# Approximate Nearest Neighbor Search 🔍
# -------------------------------------------------------------
class VectorSearchIndex:
    """
    Handles Approximate Nearest Neighbor (ANN) search using FAISS. 🚀
    
    UPGRADE: 
    Switched from `IndexFlatIP` (Exact Search, O(N)) which requires scanning every 
    item in the catalog, to `IndexHNSWFlat` (Approximate Search, O(log N)).
    HNSW builds a multi-layered graph where queries are routed through 
    navigable small worlds, enabling blazing fast retrieval for millions of items. 🌐
    """
    def __init__(self, embedding_dim: int = 128):
        self.dim = embedding_dim
        
        # M is the number of neighbors connected to each node in the HNSW graph.
        # Higher M = better recall, but more memory usage and slower insertion.
        M = 32
        
        # HNSW index uses L2 distance by default. For inner product (cosine sim on normalized vectors),
        # we can still use HNSW but need to be careful with distance interpretation (lower is closer).
        # We'll stick to L2 for now since Two-Tower embeddings are L2 Normalized.
        self.index = faiss.IndexHNSWFlat(self.dim, M, faiss.METRIC_L2)
        
        # efConstruction controls index building time vs accuracy 🏗️
        self.index.hnsw.efConstruction = 64
        
        # efSearch controls search time vs accuracy (higher = slower but more accurate recall) 🔎
        self.index.hnsw.efSearch = 32

        # IDMap allows us to map string IDs to integer IDs externally for HNSW in Python
        self.index_id_map = faiss.IndexIDMap(self.index)
        self.item_ids: List[str] = [] # To map FAISS int indices back to string Product IDs

    def build_index(self, item_embeddings: Dict[str, List[float]]):
        """
        Builds the HNSW graph index from a dictionary of {item_id: embedding_list} 🏗️
        """
        if not item_embeddings:
            return

        logging.info(f"Building FAISS HNSW Index for {len(item_embeddings)} items... 🌐")
        
        vectors = []
        ids = []
        
        for idx, (item_id, emb) in enumerate(item_embeddings.items()):
            vectors.append(emb)
            ids.append(idx)
            self.item_ids.append(item_id)
            
        # Convert to float32 contiguous arrays required by FAISS
        vector_matrix = np.array(vectors, dtype=np.float32)
        id_array = np.array(ids, dtype=np.int64)
        
        # HNSW requires vectors to be normalized if we want cosine similarity behavior from L2
        faiss.normalize_L2(vector_matrix)

        self.index_id_map.add_with_ids(vector_matrix, id_array)
        logging.info("HNSW Index built successfully. ✅")

    def search(self, query_vector: List[float], top_k: int = 100) -> List[str]:
        """
        Retrieves the top_k Candidate items for a given User Query vector. ⚡
        Runs in O(log N) time instead of O(N) linear scan.
        """
        if self.index_id_map.ntotal == 0:
            return []

        # Convert query to FAISS format
        query_matrix = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_matrix)
        
        # Perform HNSW Search
        # D = distances (L2 squared), I = integer IDs
        D, I = self.index_id_map.search(query_matrix, top_k)
        
        # Map integer IDs back to actual string Item IDs
        candidates = []
        for int_id in I[0]:
            if int_id != -1 and int_id < len(self.item_ids):
                candidates.append(self.item_ids[int_id])
                
        return candidates

# Instantiate a global Search Index for the API to utilize
faiss_index = VectorSearchIndex()
