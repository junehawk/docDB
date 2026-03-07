"""
메타데이터 기반 Context 접두사 생성기
청크 텍스트 앞에 붙여 임베딩 품질 향상
"""
from loguru import logger


def build_context_prefix(metadata: dict) -> str:
    """
    메타데이터에서 Context 접두사 생성

    내부 메타데이터가 풍부할 때:
        [2025-03 | 연구개발 사업계획서 | 홍길동 | 계획서.pdf]
    최소 정보만 있을 때:
        [연구자료/하위폴더 | 보고서.txt]

    Args:
        metadata: MetadataExtractor가 반환한 메타데이터 딕셔너리

    Returns:
        Context 접두사 문자열 (예: "[2025-03 | 제목 | 작성자 | file.pdf] ")
    """
    parts = []

    # 날짜 (doc_created 우선, fs_modified 폴백)
    date = metadata.get('doc_created') or metadata.get('fs_modified', '')
    if date:
        # YYYY-MM 형식으로 축약
        parts.append(date[:7] if len(date) >= 7 else date)

    # 제목 (doc_title 우선)
    title = metadata.get('doc_title', '')
    if title:
        parts.append(title[:50])  # 50자 제한

    # 작성자
    author = metadata.get('doc_author', '')
    if author:
        parts.append(author)

    file_name = metadata.get('file_name', '')

    if not parts and metadata.get('relative_path'):
        # 메타데이터가 전혀 없으면 상대경로 사용
        rel = metadata['relative_path']
        parent = '/'.join(rel.split('/')[:-1])  # 파일명 제외한 경로
        if parent:
            parts.append(parent)

    hierarchy = " | ".join(parts)
    if hierarchy and file_name:
        return f"[{hierarchy} | {file_name}] "
    elif hierarchy:
        return f"[{hierarchy}] "
    elif file_name:
        return f"[{file_name}] "
    return ""
