from fastapi import APIRouter, HTTPException, Depends
from sentence_transformers import SentenceTransformer
from app.services.embeddings import compute_query_embedding
from app.services.vector_store import vector_store
from app.services.llm import generate_answer
from app.models.schemas import QueryRequest
from app.dependencies import get_model, get_llm
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

@router.post("/")
def query_repo(request: QueryRequest, model : SentenceTransformer = Depends(get_model), llm_and_tokenizer = Depends(get_llm)):
    try : 
        logger.info(f"Processing query: {request.query}")

        request_embedding = compute_query_embedding(request.query, model)
        logger.info("Computed query embedding")

        search_result = vector_store.search(request_embedding, request.query, top_k=request.top_k)
        logger.info(f"Retrieved {len(search_result)} results from vector store")

        llm, tokenizer = llm_and_tokenizer
        answer = generate_answer(llm, tokenizer, request.query, search_result)

        return {
            "query": request.query,
            "results": search_result,
            "answer": answer,
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
