from pymilvus import MilvusClient, DataType

from core.vector.collection_op import CollectionOp

if __name__ == "__main__":
    client = MilvusClient(uri="http://172.18.10.65:19530", timeout=5.0)
    collection = CollectionOp(milvus_uri="http://172.18.10.65:19530")

    collection_name = "test_1"

    if not collection.has(collection_name):
        collection = collection.create(collection_name = collection_name)
    else:
        collection.load(collection_name=collection_name)

    res = client.get_load_state(
        collection_name=collection_name
    )
    for key, value in res.items():
        print(f"{key}: {value}")

    res = client.list_collections()
    print(res)

    res = client.describe_collection(
        collection_name=collection_name
    )
    print(res)