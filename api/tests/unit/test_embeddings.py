import numpy as np
from app.models.schemas import EmbeddedChunk
from app.services.embeddings import compute_embeddings, compute_query_embedding


# ── compute_embeddings ────────────────────────────────────────────────────────

class TestComputeEmbeddings:

    def test_returns_list_of_embedded_chunks(self, sample_chunk, mock_model):
        result = compute_embeddings([sample_chunk], mock_model)
        assert isinstance(result, list)
        assert all(isinstance(c, EmbeddedChunk) for c in result)

    def test_output_length_matches_input(self, sample_chunk, mock_model):
        mock_model.encode.return_value = np.array([[0.1] * 768] * 5)
        result = compute_embeddings([sample_chunk] * 5, mock_model)
        assert len(result) == 5

    def test_embedded_chunk_inherits_chunk_fields(self, sample_chunk, mock_model):
        result = compute_embeddings([sample_chunk], mock_model)
        assert result[0].path == sample_chunk.path
        assert result[0].content == sample_chunk.content
        assert result[0].language == sample_chunk.language
        assert result[0].start_line == sample_chunk.start_line
        assert result[0].end_line == sample_chunk.end_line

    def test_embedding_is_list_of_floats(self, sample_chunk, mock_model):
        result = compute_embeddings([sample_chunk], mock_model)
        assert isinstance(result[0].embedding, list)
        assert all(isinstance(v, float) for v in result[0].embedding)

    def test_embedding_dimension_is_correct(self, sample_chunk, mock_model):
        result = compute_embeddings([sample_chunk], mock_model)
        assert len(result[0].embedding) == 768

    def test_empty_list_returns_empty(self, mock_model):
        mock_model.encode.return_value = np.array([])
        result = compute_embeddings([], mock_model)
        assert result == []

    def test_encode_called_with_contents(self, sample_chunk, mock_model):
        compute_embeddings([sample_chunk], mock_model)
        assert mock_model.encode.call_args[0][0] == [sample_chunk.content]

    def test_real_embeddings_have_correct_dimension(self, sample_chunk, real_model):
        result = compute_embeddings([sample_chunk], real_model)
        assert len(result[0].embedding) == 768

    def test_real_embeddings_are_normalized(self, sample_chunk, real_model):
        result = compute_embeddings([sample_chunk], real_model)
        norm = sum(v ** 2 for v in result[0].embedding) ** 0.5
        assert 0.99 < norm < 1.01


# ── compute_query_embedding ───────────────────────────────────────────────────

class TestComputeQueryEmbedding:

    def test_returns_list_of_floats(self, mock_model):
        mock_model.encode.return_value = np.array([0.1] * 768)
        result = compute_query_embedding("how does authentication work?", mock_model)
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_returns_correct_dimension(self, mock_model):
        mock_model.encode.return_value = np.array([0.1] * 768)
        result = compute_query_embedding("how does authentication work?", mock_model)
        assert len(result) == 768

    def test_encode_called_with_query_text(self, mock_model):
        mock_model.encode.return_value = np.array([0.1] * 768)
        compute_query_embedding("find the login function", mock_model)
        assert mock_model.encode.call_args[0][0] == "find the login function"

    def test_empty_query_returns_embedding(self, mock_model):
        mock_model.encode.return_value = np.array([0.1] * 768)
        result = compute_query_embedding("", mock_model)
        assert isinstance(result, list)
        assert len(result) == 768

    def test_real_query_embedding_dimension(self, real_model):
        result = compute_query_embedding("how does authentication work?", real_model)
        assert len(result) == 768

    def test_different_queries_produce_different_embeddings(self, real_model):
        result1 = compute_query_embedding("authentication function", real_model)
        result2 = compute_query_embedding("database connection", real_model)
        assert result1 != result2

    def test_same_query_produces_same_embedding(self, real_model):
        result1 = compute_query_embedding("find the login function", real_model)
        result2 = compute_query_embedding("find the login function", real_model)
        assert result1 == result2
