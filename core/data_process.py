import os

from config.logging_config import logger
from core.vector.collection_op import CollectionOp
from core.vector.milvus_write import MilvusWrite


class DataProcess:
    def __init__(self, collection_name: str,
                 milvus_uri: str,
                 embedding_uri: str,
                 doc_id: int,
                 input_file: str):
        self._milvus_write = MilvusWrite(milvus_uri=milvus_uri, embedding_uri=embedding_uri)
        self._collection_name = collection_name
        self._doc_id = doc_id
        self._doc_name = os.path.basename(input_file)
        self._collection = CollectionOp(milvus_uri=milvus_uri)

        if self._collection.has(self._collection_name):
            self._milvus_write.remove_doc(self._collection_name, self._doc_id)
        else:
            self._collection.create(collection_name=collection_name)

    def process(self, chunk_id: int, content: str, metadata: dict):
        logger.info(f"Processing doc_id:{self._doc_id}, chunk_id:{chunk_id}, content:{content}, metadata:{metadata}")
        data = self._milvus_write.gene_data(doc_id=self._doc_id,
                                            doc_name=self._doc_name,
                                            text=content,
                                            chunk_id=chunk_id)
        self._milvus_write.write(collection_name=self._collection_name,
                                 data=data)