from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from transformers import AutoTokenizer

from core.gilingual_text_splitter import BilingualTextSplitter


class FileSplit:
    HEADER_SYMBOL = "#"
    HEADER_VALUE_PREFIX = "Header"
    HEADER_HEAD_SEG = " - "
    HEADER_CONTENT_SEG = "："
    LATER_HEADER_PREFIX = "续"

    def __init__(self, model_name: str="/home/zhangjiang/bge-m3-model",
                 chunk_size: int=512,
                 chunk_overlap: int=0,
                 header_level: int=4):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.headers_to_split_on = []
        for i in range(header_level):
            plus_one = i + 1
            symbol_str = self.HEADER_SYMBOL * plus_one
            header_name = self.HEADER_VALUE_PREFIX + str(plus_one)
            self.headers_to_split_on.append((symbol_str, header_name))

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._length_function = lambda x: len(tokenizer.encode(x))


    def _merge_headers(self, metadata: dict) -> str:
        header_str = ""
        for header in self.headers_to_split_on:
            key = header[1]
            if key in metadata.keys():
                if header_str:
                    header_str += self.HEADER_HEAD_SEG + metadata[key]
                else:
                    header_str += metadata[key]
        return header_str

    def _split_by_headers(self, input_file):
        # 1. 读取markdown文件
        loader = TextLoader(input_file)
        documents = loader.load()

        # 2. 创建标题分割器(按标题层级分割)
        # 这是核心：保留标题结构，将标题信息作为元数据
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            return_each_line=False,
            strip_headers=True
        )

        # 3. 执行标题分割(按标题层级分割文档)
        # 例如：将"## 财务表现"作为边界，分割出财务表现部分
        return splitter.split_text(documents[0].page_content)

    def split_markdown(self, input_file):
        # 1. 先按标题分割
        headers_chunks = self._split_by_headers(input_file)

        # 2. 递归分割(处理过长的块)
        # 如果某个块内容过长(超过chunk_size)，按内容递归分割
        final_chunks = []
        for chunk in headers_chunks:
            # 如果块内容长度超过限制，则递归分割
            header = self._merge_headers(chunk.metadata)
            header_len = self._length_function(header)

            if header_len + self._length_function(chunk.page_content) > self.chunk_size:
                # 使用递归分割器处理长内容
                recursive_splitter = BilingualTextSplitter(
                    chunk_size=self.chunk_size - header_len,
                    chunk_overlap=self.chunk_overlap,
                    length_function=self._length_function
                )
                # 递归分割内容(保留标题元数据)
                sub_chunks = recursive_splitter.split_text(chunk.page_content)

                # 为每个子块添加相同的元数据(保留标题结构)
                first_sub_chunk = True
                for sub_chunk in sub_chunks:
                    if first_sub_chunk:
                        first_sub_chunk = False
                        sub_chunk_content = header + self.HEADER_CONTENT_SEG + sub_chunk
                    else:
                        sub_chunk_content = (self.LATER_HEADER_PREFIX + header +
                                             self.HEADER_CONTENT_SEG + sub_chunk)
                    # 保留父级标题信息(避免结构断裂)
                    new_chunk = Document(
                        page_content=sub_chunk_content,
                        metadata=chunk.metadata.copy()  # 重要：复制metadata，避免引用问题
                    )
                    final_chunks.append(new_chunk)
            else:
                new_chunk = Document(
                    page_content=header + self.HEADER_CONTENT_SEG + chunk.page_content,
                    metadata=chunk.metadata
                )
                final_chunks.append(new_chunk)
        return final_chunks