from sentence_transformers import SentenceTransformer
from app.models.schemas import Chunk, EmbeddedChunk
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "jinaai/jina-embeddings-v5-text-nano"

def load_model() -> SentenceTransformer:
    torch.backends.mps.is_available = lambda: False
    model = SentenceTransformer(
        model_id,
        trust_remote_code=True,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float32, 
            "default_task": "retrieval",
        },
    )
    return model

def compute_embeddings(chunks: list[Chunk], model: SentenceTransformer) -> list[EmbeddedChunk]:
    contents = [chunk.content for chunk in chunks]
    embeddings : torch.Tensor = model.encode(
        contents,
        show_progress_bar=True,
        precision="float32",
        convert_to_numpy=False,
        device=device,
    )
    return [
        EmbeddedChunk(**chunk.model_dump(), embedding=embedding.tolist())
        for chunk, embedding in zip(chunks, embeddings)
    ]

def compute_query_embedding(query: str, model: SentenceTransformer) -> list[float]:
    embedding : torch.Tensor = model.encode(
        query,
        show_progress_bar=False,
        precision="float32",
        convert_to_numpy=False,
        device=device,
    )
    return embedding.tolist()
