import asyncio
from functools import lru_cache
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List

# Import our ML components
from models.two_tower import two_tower_model
from models.vector_search import faiss_index
from models.ranking import xgb_ranker

app = FastAPI(
    title="Algofriend Universal Recommender API 🚀",
    description="A production-grade two-stage recommendation engine using Two-Tower DNNs, FAISS HNSW, and XGBoost LTR. 🧠⚡",
    version="2.1.0"
)

# -------------------------------------------------------------
# Pydantic Schemas for Validation
# -------------------------------------------------------------
class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user 👤")
    top_k: int = Field(10, ge=1, le=100, description="Number of final recommendations to return 🎯")
    candidates_to_fetch: int = Field(100, ge=10, le=1000, description="Number of candidates to retrieve from FAISS before ranking 🔍")

class RecommendationItem(BaseModel):
    item_id: str
    rank_score: float

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]

class IndexUpdateItem(BaseModel):
    item_id: str
    text_content: str = Field(..., description="Text description/metadata of the item to be embedded 📝")

# -------------------------------------------------------------
# Memory Cache for extreme performance on hot queries 🔥
# -------------------------------------------------------------
@lru_cache(maxsize=1000)
def get_cached_user_embedding(user_id: str):
    """Caches the heavy neural network forward pass for active users."""
    return two_tower_model.encode_user(user_id)

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------

@app.get("/health", tags=["Monitoring 📊"])
def health_check():
    return {"status": "ok ✅", "components": {"faiss_items": faiss_index.index_id_map.ntotal}}

@app.post("/recommend", response_model=RecommendResponse, tags=["Inference 🏃‍♂️"])
async def get_recommendations(request: RecommendRequest):
    """
    Two-Stage Recommendation Pipeline:
    1. Retrieval: Get O(100) candidates from FAISS vector database using user embedding.
    2. Ranking: Score and sort candidates using XGBoost LTR model.
    """
    if faiss_index.index_id_map.ntotal == 0:
        raise HTTPException(status_code=503, detail="Vector index is empty. Push items first! ⚠️")

    # STAGE 1: Retrieval (Deep Retrieval from Two-Tower model -> FAISS)
    user_embedding = get_cached_user_embedding(request.user_id)
    candidates = faiss_index.search(user_embedding, top_k=request.candidates_to_fetch)
    
    if not candidates:
        return RecommendResponse(user_id=request.user_id, recommendations=[])

    # STAGE 2: Ranking (Heavy cross-feature scoring via XGBoost)
    # In a real async environment, we might run this CPU-bound task in a threadpool
    ranked_results = await asyncio.to_thread(
        xgb_ranker.rank_candidates, request.user_id, candidates
    )
    
    # Trim to Final Top-K
    final_top_k = ranked_results[:request.top_k]
    
    # Format Response
    recs = [RecommendationItem(item_id=item, rank_score=float(score)) for item, score in final_top_k]
    
    return RecommendResponse(user_id=request.user_id, recommendations=recs)

def _background_index_update(items: List[IndexUpdateItem]):
    """Background task to encode items and update FAISS index"""
    embeddings_dict = {}
    for item in items:
        # Generate embedding via the Item Tower (mocking with random features)
        # In production: pass `item.text_content` through a text encoder / feature pipeline
        emb = two_tower_model.encode_item(item.item_id)
        embeddings_dict[item.item_id] = emb
    
    # Add to FAISS
    faiss_index.build_index(embeddings_dict)

@app.post("/index/items", status_code=202)
async def push_items_to_index(items: List[IndexUpdateItem], background_tasks: BackgroundTasks):
    """
    Ingest new items into the FAISS HNSW index.
    Processes the encodings and index updates in a background task to avoid blocking the API.
    """
    background_tasks.add_task(_background_index_update, items)
    return {"message": f"Accepted {len(items)} items for asynchronous indexing."}


@app.on_event("startup")
async def startup_event():
    """Build a mock initial FAISS index on startup so the API is usable immediately."""
    mock_items = {f"item_{i}": two_tower_model.encode_item(f"item_{i}") for i in range(1, 1001)}
    faiss_index.build_index(mock_items)
