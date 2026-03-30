"""
공통 인덱싱 파이프라인
단일 파일의 인덱싱 로직을 통합하여 main.py와 server.py에서 재사용.

해결하는 이슈:
- C2: 인덱싱 로직 3곳 중복 제거
- C3: add_chunks 실패 시 tracker에 성공 기록되는 문제
- C4: full_index에서 stale 청크 미삭제
- C5: 변경 파일 처리 시 삭제-후-재처리 순서 문제
- M9: CLI 증분 인덱싱 실패 미기록
- M10: 경로 NFC 정규화 불일치
"""
import gc
import hashlib
from pathlib import Path
from typing import Dict, Any, Set
from loguru import logger
from src.compat import normalize_path as _platform_normalize


def normalize_path(file_path) -> str:
    """플랫폼 인식 경로 정규화 (macOS NFC, Windows 패스스루)"""
    return _platform_normalize(file_path)


def compute_mtime_hash(file_path: str, mtime: float) -> str:
    """경로+수정시간 기반 해시 계산"""
    return hashlib.sha256(f"{file_path}:{mtime}".encode()).hexdigest()


def index_single_file(
    file_path,
    processor,
    meta_extractor,
    emb_manager,
    chroma,
    tracker,
    config: Dict[str, Any],
    batch_size: int = 32,
    is_changed: bool = False,
    mtime: float = None,
) -> Dict[str, Any]:
    """
    단일 파일 인덱싱 파이프라인.

    1. 문서 추출 + 청킹
    2. 메타데이터 추출 + 병합
    3. Contextual Retrieval 접두사 추가
    4. 배치 임베딩
    5. ChromaDB 저장 (실패 시 에러 기록)
    6. Stale 청크 정리 (변경 파일)
    7. 추적 DB 업데이트

    Args:
        file_path: 파일 경로
        processor: DocumentProcessor 인스턴스
        meta_extractor: MetadataExtractor 인스턴스
        emb_manager: EmbeddingManager 인스턴스
        chroma: ChromaManager 인스턴스
        tracker: IndexTracker 인스턴스
        config: 전체 설정 딕셔너리
        batch_size: 임베딩 배치 크기
        is_changed: True면 기존 청크 정리 수행
        mtime: 파일 수정 시간 (None이면 내부에서 stat 호출)

    Returns:
        {'success': bool, 'chunks': int, 'error': str or None}
    """
    str_path = normalize_path(file_path)

    try:
        # mtime을 처리 시작 전에 확정 (TOCTOU 방지)
        if mtime is None:
            mtime = Path(str_path).stat().st_mtime
        # 1. 문서 추출 + 청킹
        chunks = processor.process_document(str_path)
        if not chunks:
            tracker.record_error(str_path, "추출 실패 또는 빈 결과")
            return {'success': False, 'chunks': 0, 'error': '추출 실패'}

        # 2. 메타데이터 추출 (첫 청크 텍스트만 사용 — meta_extractor는 최대 1000자만 참조)
        first_text = chunks[0]['text'] if chunks else ''
        doc_properties = chunks[0].get('metadata', {}).get('doc_properties', {})
        file_meta = meta_extractor.extract(
            str_path,
            extracted_text=first_text,
            doc_properties=doc_properties,
        )

        for chunk in chunks:
            chunk['metadata'].update(file_meta)
            chunk['metadata']['file_type'] = Path(str_path).suffix.lstrip('.')

        # 3. Contextual Retrieval: context 접두사 추가
        contextual_config = config.get('contextual', {})
        if contextual_config.get('enable', False):
            from src.search.context_builder import build_context_prefix
            for chunk in chunks:
                prefix = build_context_prefix({
                    **file_meta,
                    'file_name': Path(str_path).name,
                })
                chunk['text'] = prefix + chunk['text']

        # 4. 배치 임베딩 (빈 텍스트 청크 사전 필터링)
        valid_chunks = [c for c in chunks if c['text'].strip()]
        if not valid_chunks:
            tracker.record_error(str_path, "유효한 텍스트 청크 없음")
            return {'success': False, 'chunks': 0, 'error': '유효한 텍스트 없음'}

        # Stale 정리용 ID 사전 추출 후 chunks 해제
        new_ids = {c['chunk_id'] for c in valid_chunks} if is_changed else None
        del chunks  # valid_chunks만 유지하여 메모리 절감

        # 5. 스트리밍 서브배치 임베딩 + ChromaDB 저장 (C3: 실패 시 에러 기록)
        store_failed = False
        file_chunk_count = 0

        try:
            for batch_start in range(0, len(valid_chunks), batch_size):
                batch_chunks = valid_chunks[batch_start:batch_start + batch_size]

                batch_texts = [c['text'] for c in batch_chunks]
                batch_embeddings = emb_manager.embed_batch(batch_texts, batch_size=batch_size)

                batch_chroma = [
                    {'id': c['chunk_id'], 'text': c['text'], 'metadata': c['metadata']}
                    for c in batch_chunks
                ]
                if not chroma.add_chunks(batch_chroma, batch_embeddings):
                    store_failed = True
                    break

                file_chunk_count += len(batch_chunks)
                del batch_texts, batch_embeddings, batch_chroma
        finally:
            del valid_chunks
            gc.collect()

        if store_failed:
            tracker.record_error(str_path, "ChromaDB 저장 실패")
            return {'success': False, 'chunks': 0, 'error': 'ChromaDB 저장 실패'}

        # 6. Stale 청크 정리 (C4/C5: upsert 후 남은 기존 청크 삭제)
        if is_changed and new_ids:
            _cleanup_stale_chunks(chroma, str_path, new_ids)

        # 7. 추적 DB 업데이트 (mtime은 함수 시작 시 확정된 값 사용)
        mtime_hash = compute_mtime_hash(str_path, mtime)
        tracker.mark_indexed(str_path, mtime, mtime_hash)

        return {'success': True, 'chunks': file_chunk_count, 'error': None}

    except Exception as e:
        logger.warning(f"파일 인덱싱 실패: {str_path}: {e}")
        try:
            tracker.record_error(normalize_path(file_path), str(e))
        except Exception:
            pass
        return {'success': False, 'chunks': 0, 'error': str(e)}


def _cleanup_stale_chunks(chroma, file_path: str, new_chunk_ids: Set[str]):
    """
    파일의 기존 청크 중 새 청크 셋에 없는 stale 청크 삭제.

    upsert 후 호출하므로 새 데이터는 이미 저장된 상태.
    문서가 짧아져 청크 수가 줄었을 때 남는 잔여 청크를 정리.
    """
    try:
        # ChromaDB get()에 명시적 limit 설정 (기본 limit 적용 방지)
        existing_ids = []
        BATCH = 10000
        offset = 0
        while True:
            batch = chroma.collection.get(
                where={'file_path': {'$eq': file_path}},
                include=[],
                limit=BATCH,
                offset=offset,
            )
            if not batch or not batch['ids']:
                break
            existing_ids.extend(batch['ids'])
            if len(batch['ids']) < BATCH:
                break
            offset += BATCH

        if existing_ids:
            stale_ids = [cid for cid in existing_ids if cid not in new_chunk_ids]
            if stale_ids:
                chroma.collection.delete(ids=stale_ids)
                logger.info(f"Stale 청크 {len(stale_ids)}개 삭제: {Path(file_path).name}")
    except Exception as e:
        logger.warning(f"Stale 청크 정리 실패: {file_path}: {e}")
