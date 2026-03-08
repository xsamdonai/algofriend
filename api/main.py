from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import time

from models.vector_search import CandidateGenerator
from models.ranking import RankingModel

app = FastAPI(
    title="Universal Recommendation API",
    description="Multi-stage recommendation engine using Candidate Generation and XGBoost Ranking",
    version="1.0.0"
)

# Initialize models (in production, these might run as separate microservices)
print("Initializing Candidate Generator...")
candidate_generator = CandidateGenerator(num_items=50000)

print("Initializing Ranking Model...")
ranking_model = RankingModel(repo_path="../feature_store")
# Pretrain mock ranker just to have it ready
ranking_model.train_mock()

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[dict]
    latency_ms: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/v1/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    user_id: int = Query(..., description="The ID of the user requesting recommendations"),
    limit: int = Query(50, description="Number of final recommendations to return"),
    context_device: str = Query("mobile", description="Device context for personalization")
):
    try:
        start_time = time.time()
        
        # STAGE 1: Candidate Generation (Vector Search)
        # Retrieve top 1000 items that are semantically similar to user
        candidates = candidate_generator.get_candidates(user_id=user_id, top_k=1000)
        
        # STAGE 2: Ranking
        # Ranks the 1000 candidates using XGBoost and Feast features
        ranked_items = ranking_model.score(user_id=user_id, candidate_item_ids=candidates)
        
        # Take Top K
        top_k_ranked = ranked_items[:limit]
        
        # Format response
        recs = [
            {"item_id": int(item), "score": float(score)} 
            for item, score in top_k_ranked
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recs,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Typically run via `uvicorn api.main:app --reload`
    uvicorn.run(app, host="0.0.0.0", port=8000)
