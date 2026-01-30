from pymilvus import DataType, MilvusClient


def _create_schema(client: MilvusClient):
    schema = client.create_schema()
    schema.add_field(field_name="id",
                     datatype=DataType.VARCHAR,
                     max_length=64,
                     is_primary=True,
                     auto_id=False)
    schema.add_field(field_name="raw_text",
                     datatype=DataType.VARCHAR,
                     max_length=2048)
    schema.add_field(field_name="dense_vector",
                     datatype=DataType.FLOAT_VECTOR,
                     dim=1024)
    schema.add_field(field_name="sparse_vector",
                     datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="doc_id",
                     datatype=DataType.INT64,
                     is_partition_key=True)
    schema.add_field(field_name="file_name",
                     datatype=DataType.VARCHAR,
                     max_length=64)
    schema.add_field(field_name="chunk_id",
                     datatype=DataType.INT64)
    schema.add_field(field_name="metadata",
                     datatype=DataType.JSON,
                     nullable=True)
    schema.add_field(field_name="create_time",
                     datatype=DataType.INT64)
    schema.add_field(field_name="update_time",
                     datatype=DataType.INT64)
    return schema

def _create_index_params(client: MilvusClient):
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="raw_text",
                           index_type="INVERTED",
                           index_name="raw_text_index")
    index_params.add_index(field_name="dense_vector",
                           index_type="HNSW",
                           index_name="dense_vector_index",
                           metric_type="IP",
                           params={
                               "M": 64,
                               "efConstruction": 500
                           })
    index_params.add_index(field_name="sparse_vector",
                           index_type="SPARSE_INVERTED_INDEX",
                           index_name="sparse_vector_index",
                           metric_type="IP",
                           params={
                               "inverted_index_algo": "DAAT_MAXSCORE"
                           })
    index_params.add_index(field_name="doc_id",
                           index_type="BITMAP",
                           index_name="doc_id_index")
    index_params.add_index(field_name="file_name",
                           index_type="BITMAP",
                           index_name="file_name_index")
    index_params.add_index(field_name="chunk_id",
                           index_type="INVERTED",
                           index_name="chunk_id_index")
    index_params.add_index(field_name="create_time",
                           index_type="STL_SORT",
                           index_name="create_time_index")
    index_params.add_index(field_name="update_time",
                           index_type="STL_SORT",
                           index_name="update_time_index")
    return index_params

def create_collection(client: MilvusClient, collection_name: str):
    schema = _create_schema(client)
    index_params = _create_index_params(client)

    return client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        timeout=10.0,
        num_shards=1,
        # enable_mmap=False,
        consistency_level="Eventually",
        num_partitions=256,
        properties={
            "mmap.enabled": False,
            "partitionkey.isolation": True,
            "timezone": "Asia/Shanghai"
        }
    )
