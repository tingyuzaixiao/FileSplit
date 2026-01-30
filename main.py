from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
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

HEADERS_TO_SPLIT_ON = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
    ("####", "Header4"),
]

MODEL_NAME = "/home/zhangjiang/bge-m3-model"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 0

HEAD_HEAD_SEG = " - "
HEAD_CONTENT_SEG = "："
LATER_HEAD_PREFIX = "续"

def merge_headers(metadata: dict) -> str:
    header_str = ""
    for header in HEADERS_TO_SPLIT_ON:
        key = header[1]
        if key in metadata.keys():
            if header_str:
                header_str += HEAD_HEAD_SEG + metadata[key]
            else:
                header_str += metadata[key]
    return header_str

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4. 递归分割(处理过长的块)
    # 如果某个块内容过长(超过chunk_size)，按内容递归分割
    final_chunks = []
    length_function = lambda x: len(tokenizer.encode(x))
    for chunk in headers_chunks:
        # 如果块内容长度超过限制，则递归分割
        header = merge_headers(chunk.metadata)
        header_len = length_function(header)

        if header_len + length_function(chunk.page_content) > chunk_size:
            # 使用递归分割器处理长内容
            recursive_splitter = BilingualTextSplitter(
                chunk_size = CHUNK_SIZE - header_len,
                chunk_overlap = chunk_overlap,
                length_function = length_function
            )
            # 递归分割内容(保留标题元数据)
            sub_chunks = recursive_splitter.split_text(chunk.page_content)

            # 为每个子块添加相同的元数据(保留标题结构)
            first_sub_chunk = True
            for sub_chunk in sub_chunks:
                if first_sub_chunk:
                    first_sub_chunk = False
                    sub_chunk_content = header + HEAD_CONTENT_SEG + sub_chunk
                else:
                    sub_chunk_content = LATER_HEAD_PREFIX + header + HEAD_CONTENT_SEG + sub_chunk
                # 保留父级标题信息(避免结构断裂)
                new_chunk = Document(
                    page_content = sub_chunk_content,
                    metadata = chunk.metadata.copy()  # 重要：复制metadata，避免引用问题
                )
                final_chunks.append(new_chunk)
        else:
            new_chunk = Document(
                page_content = header + HEAD_CONTENT_SEG + chunk.page_content,
                metadata = chunk.metadata
            )
            final_chunks.append(new_chunk)
    return final_chunks


if __name__ == "__main__":
    input_file = "/Users/zhangjiang/Downloads/广东大湾区空天信息研究院2025年度职工考核评价办法.md"
    final_chunks = split_markdown_by_headers(input_file, HEADERS_TO_SPLIT_ON, CHUNK_SIZE, CHUNK_OVERLAP)
    for i, chunk in enumerate(final_chunks):
        print(chunk.page_content)
        print("\n")