from pydantic import BaseModel, Field

class ScrapedFile(BaseModel):
    path: str
    content: str
    language: str

class Chunk(ScrapedFile):
    start_line: int
    end_line: int

class EmbeddedChunk(Chunk):
    embedding: list[float]

class RepoIngestRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=20)
