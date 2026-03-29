from fastapi import APIRouter, HTTPException, Depends
from sentence_transformers import SentenceTransformer
from app.services.embeddings import compute_query_embedding
from app.services.vector_store import vector_store
from app.models.schemas import QueryRequest
from app.dependencies import get_model
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

@router.post("/")
def query_repo(request: QueryRequest, model : SentenceTransformer = Depends(get_model)):
    try : 
        logger.info(f"Processing query: {request.query}")

        request_embedding = compute_query_embedding(request.query, model)
        logger.info("Computed query embedding")

        search_result = vector_store.search(request_embedding, request.query, top_k=request.top_k)
        return {
            "query": request.query,
            "results": search_result,
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
