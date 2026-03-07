"""
docDB - 메인 엔트리포인트
"""
import argparse
import asyncio
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict
from loguru import logger
from src.config import load_config as _load_config


def _setup_logger():
    """로거 설정"""
    from src.config import PROJECT_ROOT
    log_dir = os.path.join(PROJECT_ROOT, "data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        os.path.join(log_dir, "docdb.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="500 MB"
    )


def _create_chroma_manager(config: Dict, chroma_path: str = None):
    """
    설정에 따라 ChromaManager 생성 (로컬 Persistent)

    Args:
        config: 전체 설정
        chroma_path: ChromaDB 경로 (None이면 config에서 읽음)
    """
    from src.vectorstore.chroma_manager import ChromaManager

    vs_config = config.get('vectorstore', {})
    path = chroma_path or vs_config.get('chroma_path', '~/.docdb/chroma_db')
    return ChromaManager(persist_dir=path)


def run_mcp_server(config_path: Optional[str] = None):
    """MCP 서버 모드 실행 (stdio only)"""
    try:
        from src.mcp_server.server import main
        logger.info("MCP 서버 모드 시작 (stdio 전송)")
        asyncio.run(main(config_path))
    except ImportError as e:
        logger.error(f"MCP 라이브러리 임포트 실패: {e}")
        logger.error("설치 명령어: pip install 'mcp[cli]'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"MCP 서버 실행 실패: {e}")
        sys.exit(1)


def run_full_index(config_path: Optional[str] = None):
    """전체 인덱싱 모드 실행"""
    try:
        from src.document_processor.processor import DocumentProcessor
        from src.document_processor.metadata_extractor import MetadataExtractor
        from src.embedding.embedding_manager import EmbeddingManager
        from src.incremental.index_tracker import IndexTracker
        from tqdm import tqdm

        config = _load_config(config_path)
        dp_config = config.get('document_processing', {})
        emb_config = config.get('embedding', {})

        doc_root = dp_config.get('doc_root', '~/Documents')
        chroma_path = config.get('vectorstore', {}).get('chroma_path', '~/.docdb/chroma_db')
        tracker_path = config.get('indexing', {}).get('tracker_db', '~/.docdb/index_tracker.db')

        supported_ext = dp_config.get('supported_extensions', [
            'hwp', 'hwpx', 'pdf', 'docx', 'pptx', 'xlsx', 'txt', 'html', 'csv', 'rtf'
        ])
        max_size_mb = dp_config.get('max_file_size_mb', 100)
        excluded = config.get('excluded_patterns', ['.DS_Store', '__pycache__', '.git'])

        logger.info(f"전체 인덱싱 시작: {doc_root}")
        logger.info(f"지원 확장자: {supported_ext}")

        # 컴포넌트 초기화
        processor = DocumentProcessor({
            'chunk_size': dp_config.get('chunk_size', 800),
            'chunk_overlap': dp_config.get('chunk_overlap', 100),
            'ocr': config.get('ocr', {}),
        })

        chroma = _create_chroma_manager(config, chroma_path=chroma_path)
        tracker = IndexTracker(db_path=tracker_path)
        meta_extractor = MetadataExtractor(doc_root=doc_root)

        # 임베딩 매니저 초기화
        emb_manager = EmbeddingManager({
            'model': emb_config.get('model', 'BAAI/bge-m3'),
            'device': emb_config.get('device', 'auto'),
        })

        # 대상 파일 수집
        logger.info("파일 스캔 중...")
        root_path = Path(os.path.expanduser(doc_root))
        target_files = []

        import re as _re
        excluded_regex = [_re.compile(p) for p in excluded]

        for ext in supported_ext:
            for f in root_path.rglob(f"*.{ext}"):
                # 제외 패턴 체크 (정규식)
                str_f = str(f)
                if any(p.search(str_f) for p in excluded_regex):
                    continue

                # 파일 크기 체크
                try:
                    if f.stat().st_size > max_size_mb * 1024 * 1024:
                        logger.debug(f"파일 크기 초과 스킵: {f.name}")
                        continue
                except OSError:
                    continue

                target_files.append(f)

        logger.info(f"인덱싱 대상: {len(target_files)}개 파일")

        # 파일 처리
        success_count = 0
        fail_count = 0
        skip_count = 0
        total_chunks = 0
        start_time = time.time()

        for file_path in tqdm(target_files, desc="인덱싱"):
            try:
                str_path = str(file_path)

                # 이미 인덱싱된 파일 스킵
                mtime = file_path.stat().st_mtime
                # 경로+수정시간 기반 해시 (실제 파일 콘텐츠 해시가 아님)
                mtime_hash = hashlib.sha256(
                    f"{str_path}:{mtime}".encode()
                ).hexdigest()

                if tracker.is_indexed(str_path, mtime_hash):
                    skip_count += 1
                    continue

                # 문서 추출 + 청킹
                chunks = processor.process_document(str_path)

                if not chunks:
                    fail_count += 1
                    tracker.record_error(str_path, "추출 실패 또는 빈 결과")
                    continue

                # 메타데이터 추출
                extracted_text = "\n".join(chunk['text'] for chunk in chunks)
                doc_properties = chunks[0].get('metadata', {}).get('doc_properties', {})
                file_meta = meta_extractor.extract(
                    str_path,
                    extracted_text=extracted_text,
                    doc_properties=doc_properties,
                )

                for chunk in chunks:
                    chunk['metadata'].update(file_meta)
                    chunk['metadata']['file_type'] = file_path.suffix.lstrip('.')

                # Contextual Retrieval: context 접두사 추가
                contextual_config = config.get('contextual', {})
                if contextual_config.get('enable', False):
                    from src.search.context_builder import build_context_prefix
                    for chunk in chunks:
                        prefix = build_context_prefix({
                            **file_meta,
                            'file_name': file_path.name,
                        })
                        chunk['text'] = prefix + chunk['text']

                # 배치 임베딩
                texts = [chunk['text'] for chunk in chunks]
                try:
                    batch_size = emb_config.get('batch_size', 32)
                    embeddings = emb_manager.embed_batch(texts, batch_size=batch_size)

                    # 배치 ChromaDB 저장
                    chroma_chunks = [
                        {'id': chunk['chunk_id'], 'text': chunk['text'], 'metadata': chunk['metadata']}
                        for chunk in chunks
                    ]
                    chroma.add_chunks(chroma_chunks, embeddings)
                except Exception as e:
                    logger.warning(f"배치 임베딩/저장 실패: {file_path.name}: {e}")
                    fail_count += 1
                    continue

                total_chunks += len(chunks)

                # 추적 DB 업데이트
                tracker.mark_indexed(
                    file_path=str_path,
                    mtime=mtime,
                    content_hash=mtime_hash,
                )
                success_count += 1

            except Exception as e:
                fail_count += 1
                logger.warning(f"파일 처리 실패: {file_path.name}: {e}")
                try:
                    tracker.record_error(str(file_path), str(e))
                except Exception:
                    pass

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("전체 인덱싱 완료")
        logger.info(f"  대상 파일: {len(target_files)}개")
        logger.info(f"  성공: {success_count}개")
        logger.info(f"  실패: {fail_count}개")
        logger.info(f"  스킵(이미 인덱싱): {skip_count}개")
        logger.info(f"  총 청크: {total_chunks}개")
        logger.info(f"  소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
        if (success_count + fail_count) > 0:
            logger.info(f"  성공률: {success_count/(success_count+fail_count)*100:.1f}%")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"전체 인덱싱 실패: {e}", exc_info=True)
        sys.exit(1)


def run_incremental_index(config_path: Optional[str] = None):
    """증분 인덱싱 모드 실행"""
    try:
        from src.document_processor.processor import DocumentProcessor
        from src.document_processor.metadata_extractor import MetadataExtractor
        from src.embedding.embedding_manager import EmbeddingManager
        from src.incremental.index_tracker import IndexTracker
        from src.incremental.file_scanner import FileScanner

        config = _load_config(config_path)
        dp_config = config.get('document_processing', {})
        emb_config = config.get('embedding', {})

        doc_root = dp_config.get('doc_root', '~/Documents')
        chroma_path = config.get('vectorstore', {}).get('chroma_path', '~/.docdb/chroma_db')
        tracker_path = config.get('indexing', {}).get('tracker_db', '~/.docdb/index_tracker.db')

        logger.info("증분 인덱싱 시작")

        processor = DocumentProcessor({
            'chunk_size': dp_config.get('chunk_size', 800),
            'chunk_overlap': dp_config.get('chunk_overlap', 100),
            'ocr': config.get('ocr', {}),
        })

        chroma = _create_chroma_manager(config, chroma_path=chroma_path)
        tracker = IndexTracker(db_path=tracker_path)
        meta_extractor = MetadataExtractor(doc_root=doc_root)

        excluded = config.get('excluded_patterns', [])

        scanner = FileScanner(
            doc_root=doc_root,
            tracker=tracker,
            excluded_patterns=excluded,
            max_file_size_mb=dp_config.get('max_file_size_mb', 100),
        )

        emb_manager = EmbeddingManager({
            'model': emb_config.get('model', 'BAAI/bge-m3'),
            'device': emb_config.get('device', 'auto'),
        })

        # 차이 스캔
        new_files, changed_files, deleted_files = scanner.scan_and_diff()

        logger.info(
            f"변경 감지: 신규 {len(new_files)}, "
            f"변경 {len(changed_files)}, 삭제 {len(deleted_files)}"
        )

        # 삭제 처리
        for file_path in deleted_files:
            try:
                chroma.delete_by_file(file_path)
                tracker.mark_deleted(file_path)
                logger.debug(f"삭제 처리: {file_path}")
            except Exception as e:
                logger.warning(f"삭제 처리 실패: {file_path}: {e}")

        # 신규 + 변경 처리
        process_files = new_files + changed_files
        changed_set = set(changed_files)
        success = 0
        fail = 0

        for file_path in process_files:
            try:
                if file_path in changed_set:
                    chroma.delete_by_file(str(file_path))

                chunks = processor.process_document(str(file_path))
                if not chunks:
                    fail += 1
                    continue

                extracted_text = "\n".join(chunk['text'] for chunk in chunks)
                doc_properties = chunks[0].get('metadata', {}).get('doc_properties', {})
                file_meta = meta_extractor.extract(
                    str(file_path),
                    extracted_text=extracted_text,
                    doc_properties=doc_properties,
                )

                for chunk in chunks:
                    chunk['metadata'].update(file_meta)
                    chunk['metadata']['file_type'] = Path(file_path).suffix.lstrip('.')

                # Contextual Retrieval: context 접두사 추가
                contextual_config = config.get('contextual', {})
                if contextual_config.get('enable', False):
                    from src.search.context_builder import build_context_prefix
                    for chunk in chunks:
                        prefix = build_context_prefix({
                            **file_meta,
                            'file_name': Path(file_path).name,
                        })
                        chunk['text'] = prefix + chunk['text']

                # 배치 임베딩
                texts = [chunk['text'] for chunk in chunks]
                batch_size = emb_config.get('batch_size', 32)
                embeddings = emb_manager.embed_batch(texts, batch_size=batch_size)

                # 배치 ChromaDB 저장
                chroma_chunks = [
                    {'id': chunk['chunk_id'], 'text': chunk['text'], 'metadata': chunk['metadata']}
                    for chunk in chunks
                ]
                chroma.add_chunks(chroma_chunks, embeddings)

                mtime = Path(file_path).stat().st_mtime
                # 경로+수정시간 기반 해시 (실제 파일 콘텐츠 해시가 아님)
                mtime_hash = hashlib.sha256(
                    f"{file_path}:{mtime}".encode()
                ).hexdigest()
                tracker.mark_indexed(str(file_path), mtime, mtime_hash)
                success += 1

            except Exception as e:
                fail += 1
                logger.warning(f"증분 처리 실패: {file_path}: {e}")

        logger.info(
            f"증분 인덱싱 완료: 성공 {success}, "
            f"실패 {fail}, 삭제 {len(deleted_files)}"
        )

    except Exception as e:
        logger.error(f"증분 인덱싱 실패: {e}", exc_info=True)
        sys.exit(1)


def run_search(query: str, config_path: Optional[str] = None, n_results: int = 10):
    """검색 모드 실행"""
    try:
        if not query:
            logger.error("--query 파라미터는 필수입니다")
            sys.exit(1)

        from src.embedding.embedding_manager import EmbeddingManager
        from src.vectorstore.retriever import Retriever, ResultMerger

        config = _load_config(config_path)
        emb_config = config.get('embedding', {})

        chroma = _create_chroma_manager(config)

        emb_manager = EmbeddingManager({
            'model': emb_config.get('model', 'BAAI/bge-m3'),
            'device': emb_config.get('device', 'auto'),
        })

        # Contextual Retrieval: BM25 + Reranker 설정
        search_config = config.get('search', {})
        reranker_config = config.get('reranker', {})
        rrf_k = search_config.get('rrf_k', config.get('hybrid', {}).get('rrf_k', 60))

        bm25_index = None
        if search_config.get('mode', 'vector_only') == 'hybrid':
            from src.search.bm25_index import BM25Index
            bm25_index = BM25Index()
            bm25_index.build_from_chroma(chroma)

        reranker = None
        if reranker_config.get('enable', False):
            from src.search.reranker import Reranker as ChunkReranker
            reranker = ChunkReranker(
                model_name=reranker_config.get('model', 'BAAI/bge-reranker-v2-m3'),
                device=reranker_config.get('device', 'auto'),
                batch_size=reranker_config.get('batch_size', 16),
            )

        retriever = Retriever(
            chroma, emb_manager, ResultMerger(k=rrf_k),
            bm25_index=bm25_index,
            reranker=reranker,
        )

        logger.info(f"검색: '{query}'")
        results = retriever.search(query=query, n_results=n_results)

        if not results:
            print("\n검색 결과가 없습니다.")
            return

        print(f"\n검색 결과: {len(results)}건\n{'='*60}")
        for i, r in enumerate(results, 1):
            meta = r.get('metadata', {})
            source = r.get('source_collection', '?')
            score = r.get('rerank_score', r.get('score', 0))
            print(f"\n[{i}] (score: {score:.4f}, source: {source})")
            print(f"    파일: {meta.get('file_name', '?')}")
            print(f"    경로: {meta.get('file_path', '?')}")
            print(f"    유형: {meta.get('file_type', '?')} | 저자: {meta.get('doc_author', '?')}")
            text_preview = r.get('text', '')[:200]
            print(f"    내용: {text_preview}...")
        print(f"\n{'='*60}")

    except Exception as e:
        logger.error(f"검색 실패: {e}", exc_info=True)
        sys.exit(1)


def show_stats(config_path: Optional[str] = None):
    """통계 조회 모드"""
    try:
        from src.incremental.index_tracker import IndexTracker

        config = _load_config(config_path)
        tracker_path = config.get('indexing', {}).get('tracker_db', '~/.docdb/index_tracker.db')

        chroma = _create_chroma_manager(config)
        tracker = IndexTracker(db_path=tracker_path)

        chroma_stats = chroma.get_stats()

        print(f"\n{'='*50}")
        print("docDB 통계")
        print(f"{'='*50}")

        print("\n[벡터 DB (ChromaDB)]")
        for key, value in chroma_stats.items():
            print(f"  {key}: {value}")

        tracker_stats = tracker.get_stats()
        print("\n[인덱싱 추적 DB]")
        for key, value in tracker_stats.items():
            print(f"  {key}: {value}")

        print(f"\n{'='*50}")

    except Exception as e:
        logger.error(f"통계 조회 실패: {e}", exc_info=True)
        sys.exit(1)


def main():
    """메인 함수"""
    _setup_logger()

    parser = argparse.ArgumentParser(
        description='docDB - 로컬 문서 벡터 검색 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # MCP 서버 실행
  python -m src.main --mode mcp_server

  # 전체 인덱싱
  python -m src.main --mode full_index --config config/config.yaml

  # 증분 인덱싱
  python -m src.main --mode incremental_index

  # 검색
  python -m src.main --mode search --query "2025년 예산"

  # 통계 조회
  python -m src.main --mode stats
        """
    )

    parser.add_argument(
        '--mode',
        choices=['mcp_server', 'full_index', 'incremental_index', 'search', 'stats'],
        required=True,
        help='실행 모드'
    )

    parser.add_argument(
        '--config',
        default=None,
        help='설정 파일 경로 (YAML 형식)'
    )

    parser.add_argument(
        '--query',
        default=None,
        help='검색 쿼리 (--mode search인 경우만 필요)'
    )

    parser.add_argument(
        '--n-results',
        type=int,
        default=10,
        help='검색 결과 수 (기본값: 10)'
    )

    args = parser.parse_args()

    try:
        logger.info(f"docDB 시작 - 모드: {args.mode}")

        if args.mode == 'mcp_server':
            run_mcp_server(args.config)
        elif args.mode == 'full_index':
            run_full_index(args.config)
        elif args.mode == 'incremental_index':
            run_incremental_index(args.config)
        elif args.mode == 'search':
            run_search(args.query, args.config, n_results=args.n_results)
        elif args.mode == 'stats':
            show_stats(args.config)

    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(0)
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
