import threading
from queue import Queue, Empty
from typing import Optional

from pymilvus import connections
from pymilvus.orm import utility

from core.tool.queue_full_error import QueueFullError


class MilvusConnPool:
    _instance: Optional['MilvusConnPool'] = None
    _lock = threading.Lock()

    def __new__(cls, uri: str, pool_size: int = 10):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_pool(uri, pool_size)
        return cls._instance

    def _init_pool(self, uri: str, pool_size: int = 10):
        self.uri = uri
        self.pool_size = pool_size
        self._pool = Queue(maxsize=pool_size)

        for i in range(pool_size):
            alias = f"conn_{i}"
            self.create_connection(alias=alias)

    def create_connection(self, alias: str):
        connections.connect(
            alias=alias,
            uri=self.uri
        )
        self._pool.put(alias)

    @staticmethod
    def test_connection(alias: str) -> bool:
        try:
            utility.get_server_version(using=alias, timeout=0.1)
            return True
        except Exception:
            return False

    def acquire(self, timeout: float = 5.0) -> str:
        try:
            alias = self._pool.get(timeout=timeout)
            # 健康检查：确认连接是否有效
            try:
                connections.get_connection_addr(alias)
                return alias
            except Exception:
                try:
                    connections.connect(alias=alias, uri=self.uri)
                    return alias
                except Exception as e:
                    self._pool.put(alias)  # 将别名放回池中
                    raise ConnectionError(f"Failed to reconnect alias {alias}") from e
        except Empty:
            raise QueueFullError("Milvus connection pool exhausted")

    def release(self, alias: str):
        self._pool.put(alias)

    def close(self):
        with self._lock:
            while not self._pool.empty():
                try:
                    alias = self._pool.get_nowait()
                    connections.disconnect(alias)
                except Empty:
                    break