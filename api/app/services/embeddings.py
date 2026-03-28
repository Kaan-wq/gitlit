from sentence_transformers import SentenceTransformer
from app.models.schemas import Chunk, EmbeddedChunk
import torch

device = torch.device("cpu")
model_id = "jinaai/jina-embeddings-v5-text-nano"
model = SentenceTransformer(
    model_id,
    trust_remote_code=True,
    device=device,
)

def compute_embeddings(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    contents = [chunk.content for chunk in chunks]
    embeddings = model.encode(
        contents,
        show_progress_bar=True,
        precision="float32",
        convert_to_numpy=True,
        device=device,
    )
    result = []
    for chunk, embedding in zip(chunks, embeddings):
        result.append(EmbeddedChunk(
            **chunk.model_dump(),
            embedding=embedding.tolist(),
        ))
    return result

def compute_query_embedding(query: str) -> list[float]:
    embedding = model.encode(
        query,
        show_progress_bar=False,
        precision="float32",
        convert_to_numpy=True,
        device=device,
    )
    return embedding.tolist()
