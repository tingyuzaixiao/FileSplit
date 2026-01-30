import re
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BilingualTextSplitter(RecursiveCharacterTextSplitter):
    """中英文双语文本分割器"""

    def __init__(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
            separators: Optional[List[str]] = None,
            **kwargs
    ):
        if separators is None:
            # 使用优化的中英文分隔符
            separators = self.get_default_separators()

        # 如果是字符数，使用len；如果是token数，需使用tokenizer
        # length_function = kwargs.get('length_function', len)

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **kwargs
        )

    @staticmethod
    def get_default_separators() -> List[str]:
        """获取默认的中英文分隔符列表"""
        return [
            # 文档结构
            "\n\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",

            # 章节编号
            "\n\n一、", "\n\n二、", "\n\n三、", "\n\n四、", "\n\n五、",
            "\n\n六、", "\n\n七、", "\n\n八、", "\n\n九、", "\n\n十、",
            "\n\n1.", "\n\n2.", "\n\n3.", "\n\n4.", "\n\n5.",
            "\n\n6.", "\n\n7.", "\n\n8.", "\n\n9.", "\n\n10.",

            # 带括号编号
            "\n\n（一）", "\n\n（二）", "\n\n（三）", "\n\n（四）", "\n\n（五）",
            "\n\n(1)", "\n\n(2)", "\n\n(3)", "\n\n(4)", "\n\n(5)",

            # 段落和列表
            "\n\n", "\n* ", "\n- ", "\n+ ", "\n• ",

            # 句子结束符
            "。",  # 中文句号

            # 智能处理英文句号（避免缩写等问题）
            "(?<=[^A-Za-z0-9])[.](?=[^A-Za-z0-9])",  # 英文句号，但前后不是字母数字
            "(?<=[^A-Za-z])[.](?=\\s)",  # 英文句号，前面不是字母，后面是空白

            "！", "!",  # 感叹号
            "？", "?",  # 问号
            "……", "...",  # 省略号

            # 分号冒号
            "；", ";",
            "：", ": ",

            # 逗号顿号
            "，", ", ",
            "、",

            # 空格
            "  ", " ", "\t",

            # 最后
            ""
        ]

    def split_text(self, text: str) -> List[str]:
        """重写分割方法，处理中英文混合文本"""
        # 预处理：统一一些空白字符
        text = re.sub(r'\r\n', '\n', text)  # Windows换行转Unix
        text = re.sub(r'\r', '\n', text)  # Mac换行转Unix
        text = re.sub(r'\u3000', ' ', text)  # 全角空格转半角

        return super().split_text(text)