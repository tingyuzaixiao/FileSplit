import time
from typing import List

from pymilvus import Collection

from config.logging_config import logger
from core.tool.atomic_counter import AtomicCounter
from core.tool.hash import text_to_sha256
from core.tool.thread_pool import ThreadPool
from core.vector.milvus_conn_pool import MilvusConnPool
from core.tool.time import get_current_timestamp_ms
from core.vector.embedding_generator import EmbeddingGenerator


class MilvusWrite:
    MAX_TASK_NUM = 8
    MAX_RETRIES = 10

    def __init__(self, milvus_uri: str="http://172.18.10.65:19530",
                 embedding_uri: str="http://172.18.10.61:8010",
                 pool_size: int=4,
                 max_workers: int=4,
                 queue_size: int=100):
        self.conn_pool = MilvusConnPool(uri=milvus_uri, pool_size=pool_size)
        self.embedding_generator = EmbeddingGenerator(base_url=embedding_uri)
        self.task_thread = ThreadPool(max_workers=max_workers, queue_size=queue_size)
        self.task_counter = AtomicCounter(0)

    def write_batch(self, collection_name: str, data_batch: List[List[dict]]) -> None:
        for data in data_batch:
            while self.task_counter.increment() > self.MAX_TASK_NUM:
                self.task_counter.decrement()
                time.sleep(0.002)
            self.task_thread.submit(self._write, collection_name, data)

    def write(self, collection_name: str, data: List[dict]) -> None:
        while self.task_counter.increment() > self.MAX_TASK_NUM:
            self.task_counter.decrement()
            time.sleep(0.002)
        self.task_thread.submit(self._write, collection_name, data)

    def _write(self, collection_name:str, data: List[dict]) -> None:
        current_retry = 0
        while current_retry < self.MAX_RETRIES:
            conn_alias = None
            try:
                conn_alias = self.conn_pool.acquire()
                collection = Collection(collection_name, using=conn_alias)
                ret = collection.upsert(
                    data=data,
                    timeout=10.0,
                    partial_update=False
                )
                logger.info(f"upsert ret: {ret}")
                return
            except Exception as e:
                logger.error(f"write catch exception {e}")
                if conn_alias:
                    if not MilvusConnPool.test_connection(conn_alias):
                        self.conn_pool.create_connection(conn_alias)
                    time.sleep(0.005)
                    current_retry = current_retry + 1
                    continue
                raise
            finally:
                self.task_counter.decrement()
                if conn_alias:
                    self.conn_pool.release(conn_alias)

    def gene_data(self,
                  doc_id: int,
                  doc_name: str,
                  text: str,
                  chunk_id: int) -> List[dict]:
        dense_vec, lexical_weights = self.embedding_generator.embeddings(text)

        current_timestamp_ms = get_current_timestamp_ms()

        data = [{
            "id": text_to_sha256(text),
            "raw_text": text,
            "dense_vector": dense_vec,
            "sparse_vector": lexical_weights,
            "doc_id": doc_id,
            "file_name": doc_name,
            "chunk_id": chunk_id,
            "create_time": current_timestamp_ms,
            "update_time": current_timestamp_ms
        }]
        return data