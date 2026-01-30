import os
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    """服务配置"""
    # url配置
    embedding_url = "http://172.18.10.61:8010"
    milvus_url = "http://172.18.10.65:19530"
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "/home/zhangjiang/logs/file_split/file_split.log"

    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量加载配置
        self._load_from_env()

    def _load_from_env(self):
        """从环境变量加载配置"""
        if os.getenv("EMBEDDINGS_URL"):
            self.embedding_url = os.getenv("EMBEDDINGS_URL")
        if os.getenv("MILVUS_URL"):
            self.milvus_url = os.getenv("MILVUS_URL")
        if os.getenv("FILE_SPLIT_LOG_FILE"):
            self.log_file = os.getenv("FILE_SPLIT_LOG_FILE")
        if os.getenv("FILE_SPLIT_LOG_LEVEL"):
            self.log_level = os.getenv("FILE_SPLIT_LOG_LEVEL")

# 全局配置实例
config = ServiceConfig()