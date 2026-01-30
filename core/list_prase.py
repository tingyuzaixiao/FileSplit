import re


def extract_lists(text):
    """
    识别Markdown文本中的列表，包括嵌套列表。

    参数:
        text (str): Markdown文本输入。

    返回:
        list: 每个列表的信息列表，每个元素为元组 (start_pos, length, level)
            - start_pos: 列表起始位置的字符索引（从0开始）
            - length: 列表项的数量
            - level: 嵌套层级（根列表为0，嵌套一级为1，依此类推）
    """
    # 将文本按行分割（保留换行符），便于计算起始位置
    lines = text.splitlines(True)
    start_pos = 0  # 当前行的起始位置（字符索引）
    stack = []  # 栈：存储当前列表信息 (start_pos, current_length, indent)
    lists = []  # 输出列表：存储所有列表信息 (start_pos, length, level)

    for line in lines:
        # 计算当前行的前导空格数（缩进级别）
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        content = stripped

        # 检查是否为列表行（无序或有序）
        is_list = False
        if content.startswith('* ') or content.startswith('- ') or content.startswith('+ '):
            is_list = True
        else:
            # 匹配有序列表：数字 + 点 + 空格（如 "1. ", "10. "）
            if re.match(r'^\d+\.\s', content):
                is_list = True

        if is_list:
            # 结束所有嵌套级别更高的列表（当前缩进 < 栈顶缩进）
            while stack and indent < stack[-1][2]:
                start, length, _ = stack.pop()
                level = len(stack)  # 结束列表的嵌套层级
                lists.append((start, length, level))

            # 当前缩进与栈顶缩进相同：继续当前列表
            if stack and indent == stack[-1][2]:
                start, length, _ = stack.pop()
                stack.append((start, length + 1, indent))
            # 当前缩进 > 栈顶缩进：新嵌套列表
            else:
                stack.append((start_pos, 1, indent))
        else:
            # 非列表行：结束所有当前列表
            while stack:
                start, length, _ = stack.pop()
                level = len(stack)  # 结束列表的嵌套层级
                lists.append((start, length, level))

        # 更新下一行的起始位置
        start_pos += len(line)

    # 文本结束：结束所有剩余列表
    while stack:
        start, length, _ = stack.pop()
        level = len(stack)  # 结束列表的嵌套层级
        lists.append((start, length, level))

    return lists


# 示例用法
if __name__ == "__main__":
    markdown_text = """* Item 1
  1. Subitem 1
  2. Subitem 2
* Item 2
1. Ordered item
   - Nested unordered
2. Another item"""

    lists_info = extract_lists(markdown_text)
    for idx, (start, length, level) in enumerate(lists_info, 1):
        print(f"List {idx}: Start={start}, Length={length}, Level={level}")