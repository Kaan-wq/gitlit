from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
from app.models.schemas import EmbeddedChunk
import os

class VectorStore:
    def __init__(self):
        uri = os.environ.get("MILVUS_URI", "milvus_vector_store.db")
        self.client = MilvusClient(uri)
        self.collection_name = os.environ.get("MILVUS_COLLECTION", "gitlit")
        self._create_collection()

    def _create_collection(self):
        schema = MilvusClient.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("path", DataType.VARCHAR, max_length=150)
        schema.add_field("content", DataType.VARCHAR, max_length=1000)
        schema.add_field("language", DataType.VARCHAR, max_length=50)
        schema.add_field("start_line", DataType.INT32)
        schema.add_field("end_line", DataType.INT32)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=768)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

        bm25_function = Function(
            name="bm25_fn",
            input_field_names=["content"],
            output_field_names="sparse_vector",
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW_PQ",
            metric_type="COSINE",
            params={
                "M": 16,
                "efConstruction": 200,
                "m": 96,
                "nbits": 8,
                "refine": True,
                "refine_type": "SQ8",
            }
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                schema=schema,
                index_params=index_params,
            )

    def insert(self, data: list[EmbeddedChunk]) -> dict:
        return self.client.insert(
            self.collection_name,
            [chunk.model_dump() for chunk in data]
        )

    def search(self, query_embedding: list[float], query_text: str, top_k: int = 10) -> list[dict]:
        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="embedding",
            param={"ef": 100, "refine_k": top_k * 5},
            limit=top_k,
        )
        sparse_req = AnnSearchRequest(
            data=query_text,
            anns_field="sparse_vector",
            param={},
            limit=top_k,
        )
        results = self.client.hybrid_search(
            self.collection_name,
            [dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=top_k,
            output_fields=["path", "content", "language", "start_line", "end_line"],
        )
        return [
            {
                "path": hit["entity"]["path"],
                "content": hit["entity"]["content"],
                "language": hit["entity"]["language"],
                "start_line": hit["entity"]["start_line"],
                "end_line": hit["entity"]["end_line"],
                "score": hit["distance"],
            }
            for hit in results[0]
        ]


# Singleton instance
vector_store = VectorStore()
