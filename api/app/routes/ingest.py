from fastapi import APIRouter

router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.post("/repo")
def ingest_repo():
    ...
