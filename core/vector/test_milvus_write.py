from concurrent.futures import wait
from typing import List, Callable

from pymilvus import MilvusClient, Collection

from core.tool.hash import text_to_sha256
from core.vector.milvus_conn_pool import MilvusConnPool
from core.tool.thread_pool import ThreadPool
from core.tool.time import get_current_timestamp_ms
from core.vector.embedding_generator import EmbeddingGenerator

def gene_data(emb_generator: EmbeddingGenerator, text: str, chunk_id: int) -> List[dict]:
    dense_vec, lexical_weights = emb_generator.embeddings(text)
    print(f"dense_vec: {dense_vec}")
    print(f"lexical_weights: {lexical_weights}")

    current_timestamp_ms = get_current_timestamp_ms()

    data = [{
        "id": text_to_sha256(text),
        "raw_text": text,
        "dense_vector": dense_vec,
        "sparse_vector": lexical_weights,
        "doc_id": 1,
        "file_name": "python教程",
        "chunk_id": chunk_id,
        "create_time": current_timestamp_ms,
        "update_time": current_timestamp_ms
    }]
    return data

def get_data1(emb_generator: EmbeddingGenerator):
    text = "怎样学习python？"
    chunk_id = 1
    return gene_data(emb_generator, text, chunk_id)

def get_data2(emb_generator: EmbeddingGenerator):
    text = "python是一门解释性语言"
    chunk_id = 2
    return gene_data(emb_generator, text, chunk_id)

def get_data3(emb_generator: EmbeddingGenerator):
    text = "python的入门门槛很低"
    chunk_id = 3
    return gene_data(emb_generator, text, chunk_id)

def get_data4(emb_generator: EmbeddingGenerator):
    text = "python学习要以教程为主"
    chunk_id = 4
    return gene_data(emb_generator, text, chunk_id)

def get_data5(emb_generator: EmbeddingGenerator):
    text = "python应用极为广泛"
    chunk_id = 5
    return gene_data(emb_generator, text, chunk_id)

def get_data6(emb_generator: EmbeddingGenerator):
    text = "python不适合高并发场景"
    chunk_id = 6
    return gene_data(emb_generator, text, chunk_id)

def write_data(conn_pool: MilvusConnPool,
               collection_name: str,
               embedding_func: Callable,
               emb_generator: EmbeddingGenerator):
    data = embedding_func(emb_generator)
    conn_alias = conn_pool.acquire()
    try:
        collection = Collection(collection_name, using=conn_alias)
        return collection.upsert(
            data=data,
            timeout=10.0,
            partial_update=False
        )
    finally:
        conn_pool.release(conn_alias)


if __name__ == "__main__":
    embedding_generator = EmbeddingGenerator("http://172.18.10.61:8010")
    data = get_data1(embedding_generator)

    client = MilvusClient(uri="http://172.18.10.65:19530", timeout=5.0)
    collection_name = "test_1"

    client.load_collection(
        collection_name=collection_name
    )
    res = client.get_load_state(
        collection_name=collection_name
    )
    for key, value in res.items():
        print(f"{key}: {value}")

    milvus_conn_pool = MilvusConnPool(uri="http://172.18.10.65:19530", pool_size=8)
    thread = ThreadPool(max_workers=4, queue_size=100)
    futures = [
        thread.submit(write_data, milvus_conn_pool,
                      collection_name, get_data1, embedding_generator),
        thread.submit(write_data, milvus_conn_pool,
                      collection_name, get_data2, embedding_generator),
        thread.submit(write_data, milvus_conn_pool,
                      collection_name, get_data3, embedding_generator),
        thread.submit(write_data, milvus_conn_pool,
                      collection_name, get_data4, embedding_generator),
        thread.submit(write_data, milvus_conn_pool,
                      collection_name, get_data5, embedding_generator),
        thread.submit(write_data, milvus_conn_pool,
                      collection_name, get_data6, embedding_generator)
    ]
    for future in futures:
        done, _ = wait([future], timeout=10.0)
        if not done:
            print("task timeout")
            continue
        else:
            print(f"done:{done}")
            print(future.result())