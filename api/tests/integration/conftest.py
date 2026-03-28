import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def sample_client():
    yield TestClient(app)

@pytest.fixture
@pytest.mark.slow
def sample_model():
    from app.services.embeddings import model
    yield model
