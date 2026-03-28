from pydantic import BaseModel

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
