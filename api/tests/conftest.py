from app.models.schemas import ScrapedFile, Chunk, EmbeddedChunk
from app.services.vector_store import VectorStore
import pytest
import os

@pytest.fixture
def sample_scraped_file():
    return ScrapedFile(
        path="app/main.py",
        content="def hello():\n    return 'world'\n",
        language="python",
    )

@pytest.fixture
def sample_chunk():
    return Chunk(
        path="app/main.py",
        content="def hello():\n    return 'world'\n",
        language="python",
        start_line=1,
        end_line=2,
    )

@pytest.fixture
def sample_embedded_chunk():
    return EmbeddedChunk(
        path="app/main.py",
        content="def hello():\n    return 'world'\n",
        language="python",
        start_line=1,
        end_line=2,
        embedding=[0.1] * 768,
    )

@pytest.fixture
def sample_vector_store():
    vector_store = VectorStore(
        uri="http://localhost:19530",
        collection_name="gitlit_test",
    )
    yield vector_store

    # cleanup
    vector_store.client.drop_collection("gitlit_test")
    os.remove("milvus_test.db")
