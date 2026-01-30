from pymilvus import AnnSearchRequest, Function, FunctionType, MilvusClient

from core.vector.embedding_generator import EmbeddingGenerator

if __name__ == "__main__":
    embeddingGenerator = EmbeddingGenerator("http://172.18.10.61:8010")

    text = "新手怎样入门python？"
    dense_vec, lexical_weights = embeddingGenerator.embeddings(text)
    print(f"dense_vec: {dense_vec}")
    print(f"lexical_weights: {lexical_weights}")

    # text semantic search (dense)
    dense_search_param = {
        "data": [dense_vec],
        "anns_field": "dense_vector",
        "param": {"ef": 500},
        "limit": 3,
        "expr": "doc_id=={doc_id}",
        "expr_params": {"doc_id": 1}
    }
    dense_request = AnnSearchRequest(**dense_search_param)

    # full-text search (sparse)
    sparse_search_param = {
        "data": [lexical_weights],
        "anns_field": "sparse_vector",
        "param": {"metric_type": "IP"},
        "limit": 3,
        "expr": "doc_id=={doc_id}",
        "expr_params": {"doc_id": 1}
    }
    sparse_request = AnnSearchRequest(**sparse_search_param)

    ranker = Function(
        name="rrf",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={
            "reranker": "rrf",
            "k": 60
        }
    )

    client = MilvusClient(uri="http://172.18.10.65:19530", timeout=5.0)
    collection_name = "test_1"

    res = client.hybrid_search(
        collection_name=collection_name,
        reqs=[dense_request, sparse_request],
        ranker=ranker,
        limit=3,
        output_fields=["chunk_id", "raw_text"],
        timeout=5.0)

    print(res)

    res = client.query(
        collection_name=collection_name,
        filter="chunk_id==2",
        output_fields=["chunk_id", "raw_text"]
    )
    print(res)
