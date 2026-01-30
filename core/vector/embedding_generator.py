from urllib.parse import urljoin

from core.tool.http_req import send_request
from core.tool.thread_pool import logger


class EmbeddingGenerator:
    API_PATH = "api"
    EMBEDDINGS_API = API_PATH + "/embeddings"
    # EMBEDDINGS_URL = urljoin("http://172.18.10.61:8010", EMBEDDINGS_API)

    def __init__(self, base_url: str):
        self.embedding_url = urljoin(base_url, EmbeddingGenerator.EMBEDDINGS_API)

    def embeddings(self, query: str):
        payload = {
            "query": query
        }

        response = send_request(url=self.embedding_url,
                                method="POST",
                                json_data=payload,
                                timeout=5.0)
        if not response:
            raise Exception(f"请求失败, unknown error")

        res_data = response.json()
        if res_data.get("code") == 0:
            data = res_data.get("data")
            return data.get('dense_vec'), data.get('lexical_weights')
        else:
            logger.error(f"请求失败 (code:{res_data.get('code')}): {res_data.get('msg')}")
            raise Exception(f"请求失败 (code:{res_data.get('code')}): {res_data.get('msg')}")

if __name__ == "__main__":
    embeddingGenerator = EmbeddingGenerator("http://172.18.10.61:8010")
    dense_vec, lexical_weights = embeddingGenerator.embeddings("python怎样安装？")
    print(f"dense_vec: {dense_vec}")
    print(f"lexical_weights: {lexical_weights}")