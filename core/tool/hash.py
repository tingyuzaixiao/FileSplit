import hashlib


def text_to_sha256(text: str, encoding: str = 'utf-8') -> str:
    """
    将文本字符串转换为 SHA256 十六进制哈希值（小写）

    Args:
        text: 待哈希的文本
        encoding: 文本编码方式（默认 UTF-8）

    Returns:
        64 位小写十六进制字符串
    """
    if not isinstance(text, str):
        raise TypeError("输入必须是字符串类型")

    # 编码为字节 → 计算哈希 → 返回十六进制字符串
    return hashlib.sha256(text.encode(encoding)).hexdigest()

def compute_sha256(text):
    """
    计算文本的SHA-256哈希值

    Args:
        text (str): 要计算哈希的原始文本字符串

    Returns:
        str: 文本对应的SHA-256哈希值（十六进制字符串）
    """
    # 1. 创建SHA-256哈希对象
    sha256_hash = hashlib.sha256()

    # 2. 将文本字符串编码为字节（UTF-8），并更新到哈希对象中
    sha256_hash.update(text.encode('utf-8'))

    # 3. 获取十六进制格式的哈希摘要
    return sha256_hash.hexdigest()

if __name__ == "__main__":
    print(text_to_sha256("python的安装方法"))
    print(compute_sha256("python的安装方法"))
    print(len("3a55bb0d01b3984ba64c96d22aab0da484ab2a3bd1be49b3b5a25cb01794ec96"))