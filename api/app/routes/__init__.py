from fastapi import APIRouter
from .health import router as health_router
from .ingest import router as ingest_router
from .query import router as query_router

router = APIRouter()
router.include_router(health_router)
router.include_router(ingest_router)
router.include_router(query_router)
