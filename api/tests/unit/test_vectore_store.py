import pytest
from unittest.mock import MagicMock, patch
from app.services.vector_store import VectorStore
from app.models.schemas import EmbeddedChunk


# ── Additional fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def mock_client():
    with patch("app.services.vector_store.MilvusClient") as mock_class:
        mock_instance = MagicMock()
        mock_instance.has_collection.return_value = False
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def vector_store(mock_client):
    return VectorStore(uri="test.db", collection_name="gitlit_test")


@pytest.fixture
def another_embedded_chunk():
    return EmbeddedChunk(
        path="app/auth.py",
        content="def login(user, password):\n    return authenticate(user, password)\n",
        language="python",
        start_line=10,
        end_line=11,
        embedding=[0.2] * 768,
    )


@pytest.fixture
def mock_search_hit():
    return {
        "entity": {
            "path": "app/main.py",
            "content": "def hello():\n    return 'world'\n",
            "language": "python",
            "start_line": 1,
            "end_line": 2,
        },
        "distance": 0.95,
    }


# ── Init ──────────────────────────────────────────────────────────────────────

class TestVectorStoreInit:

    def test_creates_collection_when_not_exists(self, mock_client):
        mock_client.has_collection.return_value = False
        VectorStore(uri="test.db", collection_name="gitlit_test")
        mock_client.create_collection.assert_called_once()

    def test_skips_creation_when_collection_exists(self, mock_client):
        mock_client.has_collection.return_value = True
        VectorStore(uri="test.db", collection_name="gitlit_test")
        mock_client.create_collection.assert_not_called()

    def test_collection_name_is_set(self, vector_store):
        assert vector_store.collection_name == "gitlit_test"

    def test_checks_correct_collection_name(self, mock_client):
        mock_client.has_collection.return_value = False
        VectorStore(uri="test.db", collection_name="gitlit_test")
        mock_client.has_collection.assert_called_with("gitlit_test")

    def test_create_collection_called_with_correct_name(self, mock_client):
        mock_client.has_collection.return_value = False
        VectorStore(uri="test.db", collection_name="gitlit_test")
        assert mock_client.create_collection.call_args[0][0] == "gitlit_test"

    def test_uses_env_uri(self, monkeypatch):
        monkeypatch.setenv("MILVUS_URI", "env.db")
        monkeypatch.setenv("MILVUS_COLLECTION", "env_col")
        with patch("app.services.vector_store.MilvusClient") as mock_class:
            mock_class.return_value = MagicMock(has_collection=MagicMock(return_value=False))
            store = VectorStore()
            assert store.collection_name == "env_col"
            mock_class.assert_called_once_with("env.db")


# ── Insert ────────────────────────────────────────────────────────────────────

class TestInsert:

    def test_calls_client_insert(self, vector_store, mock_client, sample_embedded_chunk):
        vector_store.insert([sample_embedded_chunk])
        mock_client.insert.assert_called_once()

    def test_inserts_into_correct_collection(self, vector_store, mock_client, sample_embedded_chunk):
        vector_store.insert([sample_embedded_chunk])
        assert mock_client.insert.call_args[0][0] == "gitlit_test"

    def test_serializes_chunks_to_dicts(self, vector_store, mock_client, sample_embedded_chunk):
        vector_store.insert([sample_embedded_chunk])
        inserted_data = mock_client.insert.call_args[0][1]
        assert isinstance(inserted_data, list)
        assert isinstance(inserted_data[0], dict)

    def test_inserted_dict_contains_correct_fields(self, vector_store, mock_client, sample_embedded_chunk):
        vector_store.insert([sample_embedded_chunk])
        inserted_data = mock_client.insert.call_args[0][1]
        assert inserted_data[0]["path"] == sample_embedded_chunk.path
        assert inserted_data[0]["content"] == sample_embedded_chunk.content
        assert inserted_data[0]["language"] == sample_embedded_chunk.language
        assert inserted_data[0]["start_line"] == sample_embedded_chunk.start_line
        assert inserted_data[0]["end_line"] == sample_embedded_chunk.end_line
        assert inserted_data[0]["embedding"] == sample_embedded_chunk.embedding

    def test_inserts_multiple_chunks(self, vector_store, mock_client, sample_embedded_chunk, another_embedded_chunk):
        vector_store.insert([sample_embedded_chunk, another_embedded_chunk])
        inserted_data = mock_client.insert.call_args[0][1]
        assert len(inserted_data) == 2

    def test_returns_client_insert_result(self, vector_store, mock_client, sample_embedded_chunk):
        mock_client.insert.return_value = {"insert_count": 1}
        result = vector_store.insert([sample_embedded_chunk])
        assert result == {"insert_count": 1}


# ── Search ────────────────────────────────────────────────────────────────────

class TestSearch:

    def test_calls_hybrid_search(self, vector_store, mock_client, mock_search_hit):
        mock_client.hybrid_search.return_value = [[mock_search_hit]]
        vector_store.search([0.1] * 768, "how does auth work?")
        mock_client.hybrid_search.assert_called_once()

    def test_searches_correct_collection(self, vector_store, mock_client, mock_search_hit):
        mock_client.hybrid_search.return_value = [[mock_search_hit]]
        vector_store.search([0.1] * 768, "how does auth work?")
        assert mock_client.hybrid_search.call_args[0][0] == "gitlit_test"

    def test_returns_list_of_dicts(self, vector_store, mock_client, mock_search_hit):
        mock_client.hybrid_search.return_value = [[mock_search_hit]]
        result = vector_store.search([0.1] * 768, "how does auth work?")
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_result_contains_expected_fields(self, vector_store, mock_client, mock_search_hit):
        mock_client.hybrid_search.return_value = [[mock_search_hit]]
        result = vector_store.search([0.1] * 768, "how does auth work?")
        assert result[0]["path"] == mock_search_hit["entity"]["path"]
        assert result[0]["content"] == mock_search_hit["entity"]["content"]
        assert result[0]["language"] == mock_search_hit["entity"]["language"]
        assert result[0]["start_line"] == mock_search_hit["entity"]["start_line"]
        assert result[0]["end_line"] == mock_search_hit["entity"]["end_line"]
        assert result[0]["score"] == mock_search_hit["distance"]

    def test_returns_empty_list_when_no_results(self, vector_store, mock_client):
        mock_client.hybrid_search.return_value = [[]]
        result = vector_store.search([0.1] * 768, "how does auth work?")
        assert result == []

    def test_respects_top_k_parameter(self, vector_store, mock_client, mock_search_hit):
        mock_client.hybrid_search.return_value = [[mock_search_hit]]
        vector_store.search([0.1] * 768, "query", top_k=5)
        assert mock_client.hybrid_search.call_args[1]["limit"] == 5
