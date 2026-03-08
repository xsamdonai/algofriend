# Algofriend - Universal Deep Recommendation Engine

> **Algofriend** is a production-grade, two-stage AI recommendation system. It combines deep metric learning for candidate retrieval (Two-Tower architecture + FAISS HNSW) with a sophisticated Learning-to-Rank (LTR) scoring mechanism (XGBoost) to deliver highly personalized, scalable recommendations.

---

## 🏗️ Architecture: The Two-Stage Pipeline

A single model cannot efficiently score a user against 10 million items in real-time. Algofriend solves this via the industry-standard two-stage funnel:

### Stage 1: Fast Retrieval (Two-Tower DNN + FAISS HNSW)
*   **The Models:** We use deep neural networks (`models/two_tower.py`) — a **User Encoder** and an **Item Encoder**. These map disparate features (demographics, context, text descriptions) into a shared dense semantic vector space (e.g., 128 dimensions).
*   **The Training:** Trained using an **InfoNCE Contrastive Loss**, which learns to pull vectors of users and their interacted items closely together (dot product/cosine similarity), while pushing non-interacted items apart.
*   **The Search:** At inference, the User vector queries a **Hierarchical Navigable Small World (HNSW)** graph inside FAISS (`models/vector_search.py`). HNSW drops latency from $O(N)$ linear scans to $O(\log N)$, instantly retrieving the top ~100-500 candidate items out of millions.

### Stage 2: Precision Ranking (XGBoost LTR)
*   The retrieved candidates are sent to an **XGBoost Ranker** (`models/ranking.py`).
*   **Learning-to-Rank (LTR):** Unlike basic binary classification, this model is trained with the `rank:pairwise` (or `rank:ndcg`) objective using grouped queries (`DMatrix`). It directly optimizes the relative sorting order of items for a specific use context, evaluating heavy cross-features (historical interaction rates, real-time context) that the lightweight Two-Tower model cannot afford to process.
*   The output is sorted, and the final highly-curated Top-K items are returned to the user.

---

## 🚀 Features & Upgrades

*   ✅ **Deep Neural Encoders**: Replaced random embeddings with actual PyTorch `nn.Module` networks.
*   ✅ **Approximate Nearest Neighbors**: Upgraded FAISS from exact search (`IndexFlatIP`) to graph-based ANN (`IndexHNSWFlat`) for massive scaling.
*   ✅ **True LTR Objective**: XGBoost uses pairwise ranking objectives grouped by query context, not isolated CTR prediction.
*   ✅ **Async High-Performance API**: FastAPI with strict Pydantic validation and non-blocking background tasks for dynamic FAISS index updating.

---

## 💻 Tech Stack
*   **Deep Learning**: PyTorch
*   **Vector Search**: FAISS (Meta)
*   **Ranking**: XGBoost
*   **API Serving**: FastAPI, Uvicorn, Pydantic
*   **Data Manipulation**: NumPy, Pandas

---

## ⚡ Deployment & API Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. API Endpoints

#### `POST /recommend`
Retrieves personalized recommendations via the two-stage funnel.
```bash
curl -X POST http://localhost:8000/recommend \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user_42", "top_k": 5, "candidates_to_fetch": 150}'
```

#### `POST /index/items`
Dynamically embed and push new items into the HNSW search graph asynchronously.
```bash
curl -X POST http://localhost:8000/index/items \
     -H "Content-Type: application/json" \
     -d '[{"item_id": "new_sneaker_1", "text_content": "Red running shoes with high arch support."}]'
```

---

## 🔮 Roadmap
- [ ] Connect `feast` (Feature Store) to fetch online point-in-time features during the API call.
- [ ] Implement `Ray Serve` for managing multi-replica GPU batching on the PyTorch forward passes.
- [ ] Add `Kafka` consumers to update the FAISS index continually based on streaming item catalog changes.
