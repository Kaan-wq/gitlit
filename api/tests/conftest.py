from app.models.schemas import ScrapedFile, Chunk, EmbeddedChunk
from app.services.vector_store import VectorStore
from fastapi.testclient import TestClient
from app.main import app
from app.services.embeddings import load_model
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import os


@pytest.fixture
def sample_client():
    yield TestClient(app)

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.encode.return_value = np.array([[0.1] * 768])
    return model

@pytest.fixture(scope="session")
def real_model():
    return load_model()

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
    with patch("app.services.vector_store.MilvusClient") as mock_class:
        mock_instance = MagicMock()
        mock_instance.has_collection.return_value = False
        mock_class.return_value = mock_instance
        store = VectorStore(uri="test.db", collection_name="gitlit_test")
        store._mock_client = mock_instance
        yield store
