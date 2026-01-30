from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters.markdown import MarkdownTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer

from core.gilingual_text_splitter import BilingualTextSplitter

separators = [
    "\n\n",        # 段落分隔（优先级最高）
    "\n",          # 行分隔
    "。",          # 中文句号（句子结束）
    "？",          # 中文问号
    "！",          # 中文感叹号
    "……",         # 中文省略号（四个点，规范表示）
    "...",         # 英文省略号（三个点，常见于中英文混合文本）
    ".",           # 英文句号
    "?",           # 英文问号
    "!",           # 英文感叹号
    "；",          # 中文分号（用于长句分隔）
    "，",          # 中文逗号（谨慎使用，避免过度分割）
    " ",           # 空格（用于英文单词分隔）
    ""             # 最后按字符分割（兜底）
]

INPUT_FILE = "/Users/zhangjiang/Downloads/广东大湾区空天信息研究院2025年度职工考核评价办法.md"

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

def split_markdown_by_headers(input_file, headers_to_split_on, chunk_size, chunk_overlap):
    # 1. 读取markdown文件
    loader = TextLoader(input_file)
    documents = loader.load()

    # 2. 创建标题分割器(按标题层级分割)
    # 这是核心：保留标题结构，将标题信息作为元数据
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on = headers_to_split_on,
        return_each_line = False,
        strip_headers = True
    )

    # 3. 执行标题分割(按标题层级分割文档)
    # 例如：将"## 财务表现"作为边界，分割出财务表现部分
    headers_chunks = splitter.split_text(documents[0].page_content)

    # BGE-M3 等开源模型（HuggingFace）
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    # 4. 递归分割(处理过长的块)
    # 如果某个块内容过长(超过chunk_size)，按内容递归分割
    final_chunks = []
    length_function = lambda x: len(tokenizer.encode(x))
    for chunk in headers_chunks:
        # 如果块内容长度超过限制，则递归分割
        if len(chunk.page_content) > chunk_size:
            # 使用递归分割器处理长内容
            recursive_splitter = BilingualTextSplitter(
                chunk_size = CHUNK_SIZE,
                chunk_overlap = chunk_overlap,
                length_function = lambda x: len(tokenizer.encode(x))
            )
            # 递归分割内容(保留标题元数据)
            sub_chunks = recursive_splitter.split_text(chunk.page_content)

            # 为每个子块添加相同的元数据(保留标题结构)
            for sub_chunk in sub_chunks:
                # 保留父级标题信息(避免结构断裂)
                new_chunk = Document(
                    page_content = sub_chunk,
                    metadata=chunk.metadata.copy()  # 重要：复制metadata，避免引用问题
                )
                final_chunks.append(new_chunk)
        else:
            final_chunks.append(chunk)
    return final_chunks


if __name__ == "__main__":
    final_chunks = split_markdown_by_headers(INPUT_FILE, HEADERS_TO_SPLIT_ON, CHUNK_SIZE, CHUNK_OVERLAP)
    for i, chunk in enumerate(final_chunks):
        print(chunk.page_content)
        print("\n")