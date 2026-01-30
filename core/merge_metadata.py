def merge_metadata_headers1(metadata):
    headers = []

    # 从metadata中提取多级标题
    if 'h1' in metadata:
        headers.append(metadata['h1'])
    if 'h2' in metadata:
        headers.append(metadata['h2'])
    if 'h3' in metadata:
        headers.append(metadata['h3'])
    # 可以根据需要继续添加h4, h5等

    return " > ".join(headers)  # 用箭头连接标题


def create_document_context(metadata):
    """创建完整的文档上下文路径"""
    context_parts = []

    # 添加文档基本信息
    if 'source' in metadata:
        context_parts.append(f"文档: {metadata['source']}")
    if 'page' in metadata:
        context_parts.append(f"第{metadata['page']}页")

    # 添加标题层级
    title_hierarchy = []
    for level in ['h1', 'h2', 'h3', 'h4']:
        if level in metadata and metadata[level]:
            title_hierarchy.append(metadata[level])

    if title_hierarchy:
        context_parts.append(" → ".join(title_hierarchy))

    return "\n".join(context_parts)


def prepare_for_embedding(chunk):
    """
    将metadata信息前置到chunk内容中
    适合大多数embedding模型
    """
    metadata_text = create_document_context(chunk.metadata)
    combined_text = f"{metadata_text}\n\n{chunk.page_content}"

    return {
        "text": combined_text,
        "metadata": chunk.metadata,  # 保留原始metadata
        "full_text": combined_text  # 用于embedding
    }


def prepare_with_separators(chunk):
    """
    使用特殊分隔符标记不同部分
    适合能处理结构化文本的embedding模型
    """
    # 为不同部分添加标记
    parts = []

    if 'source' in chunk.metadata:
        parts.append(f"[文档] {chunk.metadata['source']}")

    if 'page' in chunk.metadata:
        parts.append(f"[页码] {chunk.metadata['page']}")

    # 标题层级
    for level, title in sorted(chunk.metadata.items()):
        if level.startswith('h') and title:
            level_num = level[1:]
            parts.append(f"[标题{level_num}] {title}")

    parts.append(f"[内容]\n{chunk.page_content}")

    return {
        "text": "\n".join(parts),
        "metadata": chunk.metadata
    }


def prepare_with_template(chunk, template=None):
    """
    使用模板将metadata和内容组合
    """
    if template is None:
        template = """来源: {source} | 页码: {page}
章节: {section}
内容: {content}"""

    # 确保所有模板字段都有值
    template_data = {
        'source': chunk.metadata.get('source', '未知'),
        'page': chunk.metadata.get('page', '未知'),
        'section': " > ".join([
            chunk.metadata.get('h1', ''),
            chunk.metadata.get('h2', ''),
            chunk.metadata.get('h3', '')
        ]).strip(' >'),
        'content': chunk.page_content
    }

    combined_text = template.format(**template_data)

    return {
        "text": combined_text,
        "metadata": chunk.metadata
    }