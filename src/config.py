"""
docDB 공통 설정 로더
main.py와 MCP 서버에서 공유하는 설정 로딩 로직
"""
import os
from typing import Optional, Dict
from loguru import logger

# 프로젝트 루트 디렉토리 (src/ 의 부모)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(path: str) -> str:
    """경로 확장: ~ → 홈, 상대경로 → 프로젝트 루트 기준 절대경로"""
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return path


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    설정 파일 로드 + 환경변수 오버라이드 + 경로 확장

    Args:
        config_path: YAML 설정 파일 경로 (없으면 기본 위치 탐색)

    Returns:
        설정 딕셔너리
    """
    import yaml

    config = {}

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    else:
        # 기본 위치 탐색 (프로젝트 루트 기준)
        for candidate in [
            os.path.join(PROJECT_ROOT, 'config', 'config.yaml'),
            os.path.join(PROJECT_ROOT, 'config.yaml'),
        ]:
            if os.path.exists(candidate):
                with open(candidate, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                break

        if not config:
            # 내장 기본 설정
            config = {
                'document_processing': {
                    'doc_root': '~/Documents',
                    'chunk_size': 800,
                    'chunk_overlap': 100,
                    'max_file_size_mb': 100,
                },
                'vectorstore': {
                    'chroma_path': 'data/chroma_db',
                },
                'indexing': {
                    'tracker_db': 'data/index_tracker.db',
                },
                'embedding': {
                    'model': 'BAAI/bge-m3',
                    'device': 'auto',
                },
            }

    # 환경변수 오버라이드
    if os.environ.get('DOCDB_ROOT'):
        config.setdefault('document_processing', {})['doc_root'] = os.environ['DOCDB_ROOT']

    # 경로 확장 (~ → 홈, 상대경로 → 프로젝트 루트 기준)
    for section_key, path_key in [
        ('vectorstore', 'chroma_path'),
        ('indexing', 'tracker_db'),
        ('document_processing', 'doc_root'),
        ('logging', 'log_dir'),
    ]:
        section = config.get(section_key, {})
        if isinstance(section, dict) and path_key in section:
            section[path_key] = _resolve_path(section[path_key])

    logger.info("설정 로드 완료")
    return config
