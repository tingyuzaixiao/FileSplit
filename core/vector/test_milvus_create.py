from pymilvus import MilvusClient, DataType

from core.vector.collection import create_collection

if __name__ == "__main__":
    client = MilvusClient(uri="http://172.18.10.65:19530", timeout=5.0)

    collection_name = "test_1"
    need_create = True

    res = client.list_collections()
    print(res)
    for name in res:
        if name == collection_name:
            need_create = False
            break
    print(f"need_create: {need_create}")
    if need_create:
        collection = create_collection(client, collection_name)
    else:
        client.load_collection(
            collection_name=collection_name
        )

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