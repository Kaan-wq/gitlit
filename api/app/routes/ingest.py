from fastapi import APIRouter, HTTPException
from app.models.schemas import RepoIngestRequest
from app.services.scraper import scrape_repository
from app.services.chunker import chunk_files
from app.services.embeddings import compute_embeddings
from app.services.vector_store import vector_store
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.post("/repo")
def ingest_repo(request: RepoIngestRequest):
    try : 
        logger.info(f"Starting ingestion for {request.repo_url}")

        scraped_files = scrape_repository(request.repo_url)
        logger.info(f"Scraped {len(scraped_files)} files")

        chunks = chunk_files(scraped_files)
        logger.info(f"Created {len(chunks)} chunks")

        embedded_chunks = compute_embeddings(chunks)
        logger.info(f"Embedded {len(embedded_chunks)} chunks")

        insertion_result = vector_store.insert(embedded_chunks)
        return {
            "status": "success",
            "files_scraped": len(scraped_files),
            "chunks_indexed": len(embedded_chunks),
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
