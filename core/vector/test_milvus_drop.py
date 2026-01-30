from pymilvus import MilvusClient

if __name__ == "__main__":
    client = MilvusClient(uri="http://172.18.10.65:19530", timeout=5.0)

    collection_name = "test_1"
    need_delete = True

    res = client.list_collections()
    print(res)
    for name in res:
        if name == collection_name:
            need_delete = True
            break
    print(f"need_delete: {need_delete}")
    if need_delete:
        collection = client.drop_collection(collection_name=collection_name)

    res = client.list_collections()
    print(res)