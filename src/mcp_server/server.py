"""
docDB MCP 서버
"""
import asyncio
import json
from typing import Optional, Dict, Any
from loguru import logger

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, CallToolResult
except ImportError:
    raise ImportError("MCP 라이브러리가 필요합니다. 설치: pip install 'mcp[cli]'")


class DocDBServer:
    """docDB MCP 서버"""

    def __init__(self, config_path: Optional[str] = None):
        """
        MCP 서버 초기화

        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self.config = {}
        self.chroma_manager = None
        self.embedding_manager = None
        self.retriever = None
        self.file_scanner = None
        self.tracker = None
        self.processor = None
        self.meta_extractor = None
        self._pending_bm25_build = False
        self._bm25_lock = None

        self.server = Server("docdb")
        self._register_tools()

    def _error_response(self, message: str):
        """에러 응답 생성 헬퍼 — isError=True 포함"""
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({"error": message}, ensure_ascii=False)
            )],
            isError=True
        )

    def _load_config(self):
        """설정 파일 로드"""
        try:
            from src.config import load_config
            self.config = load_config(self.config_path)
        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
            raise

    def _get_config_value(self, *keys, default=None):
        """중첩 config 키를 지원하는 헬퍼 (flat key 폴백 포함)"""
        _MISSING = object()
        # 중첩 키 시도 (예: 'document_processing', 'doc_root')
        if len(keys) >= 2:
            val = self.config.get(keys[0], {})
            if isinstance(val, dict):
                result = val.get(keys[1], _MISSING)
                if result is not _MISSING:
                    return result
        # flat key 폴백 (예: 'doc_root')
        for key in keys:
            val = self.config.get(key, _MISSING)
            if val is not _MISSING:
                return val
        return default

    def _initialize_components(self):
        """필요한 컴포넌트 초기화"""
        try:
            from src.vectorstore.chroma_manager import ChromaManager
            from src.vectorstore.retriever import Retriever, ResultMerger
            from src.embedding.embedding_manager import EmbeddingManager
            from src.document_processor.processor import DocumentProcessor
            from src.document_processor.metadata_extractor import MetadataExtractor
            from src.incremental.index_tracker import IndexTracker
            from src.incremental.file_scanner import FileScanner

            # 설정값 추출 (nested YAML + flat key 호환)
            chroma_path = self._get_config_value('vectorstore', 'chroma_path', default='~/.docdb/chroma_db')
            tracker_db = self._get_config_value('indexing', 'tracker_db', default='~/.docdb/index_tracker.db')
            doc_root = self._get_config_value('document_processing', 'doc_root', default='~/Documents')
            dp_config = self.config.get('document_processing', {})
            emb_config = self.config.get('embedding', {})

            # ChromaDB 매니저 (로컬)
            self.chroma_manager = ChromaManager(persist_dir=chroma_path)

            # 인덱싱 추적 + 파일 스캐너
            self.tracker = IndexTracker(db_path=tracker_db)

            excluded = self.config.get('excluded_patterns', [])
            self.file_scanner = FileScanner(
                doc_root=doc_root,
                tracker=self.tracker,
                excluded_patterns=excluded,
                max_file_size_mb=self._get_config_value('document_processing', 'max_file_size_mb', default=500)
            )

            # 임베딩 매니저 + 검색기 초기화
            self.embedding_manager = EmbeddingManager({
                'model': emb_config.get('model', 'BAAI/bge-m3'),
                'device': emb_config.get('device', 'auto'),
            })

            # Contextual Retrieval: BM25 + Reranker 설정
            search_config = self.config.get('search', {})
            reranker_config = self.config.get('reranker', {})

            # RRF k 값 (search.rrf_k 우선, hybrid.rrf_k 폴백)
            rrf_k = search_config.get('rrf_k', self.config.get('hybrid', {}).get('rrf_k', 60))

            # BM25 인덱스 (hybrid 모드일 때만, lazy build)
            bm25_index = None
            if search_config.get('mode', 'vector_only') == 'hybrid':
                from src.search.bm25_index import BM25Index
                bm25_index = BM25Index()
                # 서버 시작 시간 단축을 위해 lazy build (첫 검색 시 구축)
                self._pending_bm25_build = True

            # Reranker (활성화 시)
            reranker = None
            if reranker_config.get('enable', False):
                from src.search.reranker import Reranker as ChunkReranker
                reranker = ChunkReranker(
                    model_name=reranker_config.get('model', 'BAAI/bge-reranker-v2-m3'),
                    device=reranker_config.get('device', 'auto'),
                    batch_size=reranker_config.get('batch_size', 16),
                )

            self.retriever = Retriever(
                self.chroma_manager, self.embedding_manager,
                ResultMerger(k=rrf_k),
                bm25_index=bm25_index,
                reranker=reranker,
                vector_top_n=search_config.get('vector_top_n', 50),
                bm25_top_n=search_config.get('bm25_top_n', 50),
                rerank_top_n=search_config.get('rerank_top_n', 20),
            )

            # 문서 처리기
            self.processor = DocumentProcessor({
                'chunk_size': dp_config.get('chunk_size', 800),
                'chunk_overlap': dp_config.get('chunk_overlap', 100),
                'ocr': self.config.get('ocr', {}),
            })
            self.meta_extractor = MetadataExtractor(doc_root=doc_root)

            logger.info("서버 컴포넌트 초기화 완료 (로컬 모드)")

        except ImportError as e:
            logger.error(f"필수 모듈 임포트 실패: {e}")
            raise
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            raise

    def _register_tools(self):
        """MCP 도구 등록"""
        try:
            tools = [
                Tool(
                    name="search_documents",
                    description="문서 벡터 DB에서 시맨틱 검색. 파일 유형, 저자, 문서 유형으로 필터링 가능.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "검색 쿼리 (자연어)"
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "결과 수 (기본: 10)",
                                "default": 10
                            },
                            "file_type": {
                                "type": "string",
                                "description": "파일 유형 필터 (예: hwp, pdf)"
                            },
                            "author": {
                                "type": "string",
                                "description": "저자 필터"
                            },
                            "doc_type": {
                                "type": "string",
                                "description": "문서 유형 필터"
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="get_document",
                    description="특정 문서의 추출 텍스트를 반환. 큰 문서는 chunk_offset/chunk_limit으로 페이지네이션.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "문서 파일 경로"
                            },
                            "chunk_offset": {
                                "type": "integer",
                                "description": "시작 청크 인덱스 (기본: 0)",
                                "default": 0
                            },
                            "chunk_limit": {
                                "type": "integer",
                                "description": "반환할 최대 청크 수 (기본: 50, 최대: 50)",
                                "default": 50
                            },
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(
                    name="list_documents",
                    description="조건에 맞는 문서 목록 반환.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_type": {"type": "string"},
                            "author": {"type": "string"},
                            "doc_type": {"type": "string"},
                            "limit": {
                                "type": "integer",
                                "default": 20
                            },
                        },
                    },
                ),
                Tool(
                    name="reindex",
                    description="증분 인덱싱 수동 실행. 새로 추가/변경/삭제된 문서를 감지하여 벡터 DB 업데이트.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="get_stats",
                    description="벡터 DB 및 인덱싱 통계 조회.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
            ]

            @self.server.list_tools()
            async def list_tools():
                return tools

            @self.server.call_tool()
            async def call_tool(name, arguments):
                return await self._handle_tool_call(name, arguments)

        except Exception as e:
            logger.error(f"도구 등록 실패: {e}")
            raise

    async def _handle_tool_call(self, name: str, arguments: Dict[str, Any]):
        """도구 호출 처리 - MCP는 TextContent 리스트를 기대"""
        try:
            if name == "search_documents":
                result = await self._search_documents(arguments)
            elif name == "get_document":
                result = await self._get_document(arguments)
            elif name == "list_documents":
                result = await self._list_documents(arguments)
            elif name == "reindex":
                result = await self._reindex(arguments)
            elif name == "get_stats":
                result = await self._get_stats(arguments)
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps({
                            "error": f"알 수 없는 도구: {name}"
                        }, ensure_ascii=False, indent=2)
                    )],
                    isError=True
                )

            if isinstance(result, CallToolResult):
                return result
            # TextContent → CallToolResult로 정규화
            if isinstance(result, TextContent):
                return CallToolResult(content=[result])
            if isinstance(result, list):
                return CallToolResult(content=result)
            return CallToolResult(content=[result])

        except Exception as e:
            logger.error(f"도구 호출 실패 ({name}): {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e)
                    }, ensure_ascii=False, indent=2)
                )],
                isError=True
            )

    async def _search_documents(self, arguments: Dict[str, Any]):
        """문서 검색"""
        try:
            query = arguments.get('query')
            if not query:
                return self._error_response("query 파라미터는 필수입니다")

            n_results = arguments.get('n_results', 10)

            # BM25 인덱스 lazy build (첫 검색 시, 동시 build 방지 Lock 사용)
            if self._pending_bm25_build and self.retriever and self.retriever.bm25_index:
                if self._bm25_lock is None:
                    self._bm25_lock = asyncio.Lock()
                async with self._bm25_lock:
                    if self._pending_bm25_build:  # double-check after acquiring lock
                        logger.info("BM25 인덱스 구축 시작 (첫 검색, lazy build)...")
                        await asyncio.get_running_loop().run_in_executor(
                            None, self.retriever.bm25_index.build_from_chroma, self.chroma_manager
                        )
                        self._pending_bm25_build = False

            # 필터 구성
            filters = {}
            if 'file_type' in arguments:
                filters['file_type'] = arguments['file_type']
            if 'author' in arguments:
                filters['author'] = arguments['author']
            if 'doc_type' in arguments:
                filters['doc_type'] = arguments['doc_type']

            # 검색 실행
            if not self.retriever:
                return self._error_response("검색기가 초기화되지 않았습니다. 먼저 인덱싱을 완료하세요.")

            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.retriever.search(
                    query=query,
                    n_results=n_results,
                    filters=filters if filters else None
                )
            )

            # 검색 결과 포맷팅
            formatted_results = []
            for r in results:
                meta = dict(r.get('metadata', {}))
                formatted_results.append({
                    "chunk_id": r['chunk_id'],
                    "text": r['text'][:500] + "..." if len(r['text']) > 500 else r['text'],
                    "score": r.get('score', 0),
                    "source_collection": r.get('source_collection'),
                    "metadata": meta,
                })

            return TextContent(
                type="text",
                text=json.dumps({
                    "query": query,
                    "n_results": len(results),
                    "results": formatted_results,
                }, ensure_ascii=False, indent=2)
            )

        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return self._error_response(str(e))

    async def _get_document(self, arguments: Dict[str, Any]):
        """특정 문서 조회 - DocumentProcessor로 텍스트 추출"""
        try:
            file_path = arguments.get('file_path')
            if not file_path:
                return self._error_response("file_path 파라미터는 필수입니다")

            import os
            if not os.path.isfile(file_path):
                return self._error_response(f"파일을 찾을 수 없습니다: {file_path}")

            # path traversal 방지: doc_root 하위 경로만 허용
            doc_root = self._get_config_value('document_processing', 'doc_root', default='')
            if doc_root:
                resolved = os.path.realpath(file_path)
                root_resolved = os.path.realpath(os.path.expanduser(doc_root))
                if not resolved.startswith(root_resolved + os.sep) and resolved != root_resolved:
                    logger.warning(f"Path traversal 차단: {file_path} (doc_root: {doc_root})")
                    return self._error_response("허용되지 않은 경로입니다")

            if not self.processor:
                return self._error_response("DocumentProcessor가 초기화되지 않았습니다.")

            chunks = await asyncio.get_running_loop().run_in_executor(
                None, self.processor.process_document, file_path
            )

            if not chunks:
                return self._error_response(f"문서에서 텍스트를 추출할 수 없습니다: {file_path}")

            # 페이지네이션 파라미터
            chunk_offset = arguments.get('chunk_offset', 0)
            chunk_limit = min(arguments.get('chunk_limit', 50), 50)

            total_chunks = len(chunks)
            page_chunks = chunks[chunk_offset:chunk_offset + chunk_limit]
            full_text = "\n".join(chunk['text'] for chunk in page_chunks)
            has_more = (chunk_offset + chunk_limit) < total_chunks

            return TextContent(
                type="text",
                text=json.dumps({
                    "file_path": file_path,
                    "file_name": chunks[0].get('metadata', {}).get('file_name', ''),
                    "total_chunks": total_chunks,
                    "total_characters": sum(len(c['text']) for c in chunks),
                    "returned_chunks": len(page_chunks),
                    "chunk_offset": chunk_offset,
                    "chunk_limit": chunk_limit,
                    "has_more": has_more,
                    "next_offset": chunk_offset + chunk_limit if has_more else None,
                    "text": full_text
                }, ensure_ascii=False, indent=2)
            )

        except Exception as e:
            logger.error(f"문서 조회 실패: {e}")
            return self._error_response(str(e))

    async def _list_documents(self, arguments: Dict[str, Any]):
        """문서 목록 조회 - ChromaDB 메타데이터 기반 필터링"""
        try:
            import os

            file_type_filter = arguments.get('file_type')
            author_filter = arguments.get('author')
            doc_type_filter = arguments.get('doc_type')
            limit = arguments.get('limit', 20)

            if not self.chroma_manager:
                return self._error_response("ChromaDB가 초기화되지 않았습니다.")

            def _sync_work():
                # ChromaDB where 필터 구성
                conditions = []
                if file_type_filter:
                    conditions.append({'file_type': {'$eq': file_type_filter.lower().lstrip('.')}})
                if author_filter:
                    conditions.append({'doc_author': {'$eq': author_filter}})
                if doc_type_filter:
                    conditions.append({'doc_type': {'$eq': doc_type_filter}})

                where_filter = None
                if len(conditions) == 1:
                    where_filter = conditions[0]
                elif len(conditions) > 1:
                    where_filter = {'$and': conditions}

                # ChromaDB에서 메타데이터 조회
                try:
                    collection = self.chroma_manager.collection
                    result = collection.get(
                        where=where_filter,
                        include=['metadatas'],
                    )
                except Exception as e:
                    logger.warning(f"ChromaDB 조회 실패: {e}")
                    return []

                if not result or not result['ids']:
                    return []

                # file_path 기준으로 중복 제거 (청크 → 문서 단위)
                seen = {}
                for meta in result['metadatas']:
                    fp = meta.get('file_path', '')
                    if not fp or fp in seen:
                        continue
                    seen[fp] = {
                        "file_path": fp,
                        "file_name": os.path.basename(fp),
                        "file_type": meta.get('file_type', ''),
                        "doc_title": meta.get('doc_title', ''),
                        "doc_author": meta.get('doc_author', ''),
                        "doc_type": meta.get('doc_type', ''),
                    }

                return list(seen.values())

            documents = await asyncio.get_running_loop().run_in_executor(None, _sync_work)

            total_matched = len(documents)
            documents = documents[:limit]

            return TextContent(
                type="text",
                text=json.dumps({
                    "total_matched": total_matched,
                    "filters": {
                        "file_type": file_type_filter,
                        "author": author_filter,
                        "doc_type": doc_type_filter,
                        "limit": limit,
                    },
                    "documents": documents
                }, ensure_ascii=False, indent=2)
            )

        except Exception as e:
            logger.error(f"문서 목록 조회 실패: {e}")
            return self._error_response(str(e))

    async def _reindex(self, arguments: Dict[str, Any]):
        """증분 인덱싱 실행 - scan_and_diff 후 신규/변경/삭제 처리"""
        try:
            if not self.file_scanner or not self.tracker or not self.chroma_manager:
                return self._error_response("서버 컴포넌트가 초기화되지 않았습니다.")

            def _do_reindex():
                import hashlib
                import time
                from pathlib import Path

                processor = self.processor
                meta_extractor = self.meta_extractor
                emb_manager = self.embedding_manager

                start_time = time.time()

                # 변경 감지
                new_files, changed_files, deleted_files = self.file_scanner.scan_and_diff()
                changed_set = set(changed_files)

                # 삭제 처리
                deleted_count = 0
                for file_path in deleted_files:
                    try:
                        self.chroma_manager.delete_by_file(file_path)
                        self.tracker.mark_deleted(file_path)
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"삭제 처리 실패: {file_path}: {e}")

                # 신규 + 변경 처리
                process_files = new_files + changed_files
                success = 0
                fail = 0
                total_chunks = 0

                for file_path in process_files:
                    try:
                        # 변경된 파일은 기존 청크 삭제 후 재인덱싱
                        if file_path in changed_set:
                            self.chroma_manager.delete_by_file(str(file_path))

                        chunks = processor.process_document(str(file_path))
                        if not chunks:
                            fail += 1
                            self.tracker.record_error(str(file_path), "추출 실패 또는 빈 결과")
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
                        contextual_config = self.config.get('contextual', {})
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
                        embeddings = emb_manager.embed_batch(texts, batch_size=64)

                        # 배치 ChromaDB 저장
                        chroma_chunks = [
                            {'id': chunk['chunk_id'], 'text': chunk['text'], 'metadata': chunk['metadata']}
                            for chunk in chunks
                        ]
                        self.chroma_manager.add_chunks(chroma_chunks, embeddings)

                        total_chunks += len(chunks)

                        mtime = Path(file_path).stat().st_mtime
                        mtime_hash = hashlib.sha256(
                            f"{file_path}:{mtime}".encode()
                        ).hexdigest()
                        self.tracker.mark_indexed(
                            str(file_path), mtime, mtime_hash
                        )
                        success += 1

                    except Exception as e:
                        fail += 1
                        logger.warning(f"증분 처리 실패: {file_path}: {e}")
                        try:
                            self.tracker.record_error(str(file_path), str(e))
                        except Exception:
                            pass

                elapsed = time.time() - start_time

                result = {
                    "status": "완료",
                    "detected": {
                        "new_files": len(new_files),
                        "changed_files": len(changed_files),
                        "deleted_files": len(deleted_files),
                    },
                    "results": {
                        "success": success,
                        "fail": fail,
                        "deleted": deleted_count,
                        "total_chunks_added": total_chunks,
                    },
                    "elapsed_seconds": round(elapsed, 1),
                }

                # BM25 인덱스 재구축 (hybrid 모드)
                if self.retriever and self.retriever.bm25_index:
                    self.retriever.bm25_index.build_from_chroma(self.chroma_manager)
                    result["bm25_rebuilt"] = True

                return result

            # 블로킹 I/O를 executor에서 실행
            result = await asyncio.get_running_loop().run_in_executor(
                None, _do_reindex
            )

            return TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )

        except Exception as e:
            logger.error(f"인덱싱 실패: {e}")
            return self._error_response(str(e))

    async def _get_stats(self, arguments: Dict[str, Any]):
        """통계 조회"""
        try:
            chroma_manager = self.chroma_manager
            tracker = self.tracker

            def _sync_work():
                stats = {
                    "mode": "local",
                    "vectordb": chroma_manager.get_stats() if chroma_manager else {},
                }
                if tracker:
                    stats["indexing"] = tracker.get_stats()
                return stats

            stats = await asyncio.get_running_loop().run_in_executor(None, _sync_work)

            return TextContent(
                type="text",
                text=json.dumps(stats, ensure_ascii=False, indent=2)
            )

        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return self._error_response(str(e))

    async def run(self):
        """MCP 서버 실행"""
        try:
            self._load_config()
            self._initialize_components()

            logger.info("docDB MCP 서버 시작")

            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )

        except Exception as e:
            logger.error(f"서버 실행 실패: {e}")
            raise


async def main(config_path: Optional[str] = None):
    """메인 엔트리포인트"""
    server = DocDBServer(config_path=config_path)
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("서버 종료")
    except Exception as e:
        logger.error(f"치명적 오류: {e}")
        exit(1)
