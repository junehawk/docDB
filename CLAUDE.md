# CLAUDE.md

## Project Overview

**docDB**: 로컬 문서 폴더를 벡터 DB로 인덱싱하고, MCP 서버를 통해 Claude Desktop/Claude Code에서 자연어 검색하는 범용 도구.

**목표 사용자**: 외부 서버 없이 자기 PC에서 바로 사용하려는 연구자/일반 사용자
**핵심 원칙**: API 키 없이 로컬만으로 완전 동작, 설정 최소화 (문서 폴더 경로만 지정)

## 핵심 컴포넌트
- 문서 추출 파이프라인: 16개 포맷 지원 (HWP, PDF, DOCX, PPTX, XLSX, Pages 등)
- `DocumentProcessor`: 추출 → 청킹 파이프라인
- `LocalEmbedder`: BAAI/bge-m3 로컬 임베딩 (API 키 불필요)
- `ChromaManager`: ChromaDB 관리 (로컬 PersistentClient, 단일 `documents` 컬렉션)
- `Retriever` + `ResultMerger`: 벡터 검색 + RRF 병합
- `IndexTracker` + `FileScanner`: SQLite 기반 증분 인덱싱
- MCP 서버: 5개 도구 (search, get_document, list, reindex, stats)
- BM25 하이브리드 검색 + Reranker
- OCR 지원 (Vision framework / Tesseract)

## Architecture

```
[사용자 PC의 문서 폴더] → [텍스트 추출] → [청킹] → [로컬 임베딩] → [ChromaDB (로컬)]
     ~/Documents 등        16개 포맷     800자/100     bge-m3       data/chroma_db/
     어떤 구조든 OK                       오버랩     (API 키 불필요)      ↕
                                                                  [MCP Server]
                                                                      ↕
                                                              [Claude Desktop/Code]
```

### 사용 시나리오
사용자가 자기 PC의 특정 디렉토리(예: `~/Documents/연구자료`)를 지정하면,
그 아래 모든 문서를 자동으로 인덱싱하여 Claude에서 자연어로 검색/활용 가능.
외부 서버, API 키, 클라우드 연동 없이 **100% 로컬**에서 동작.

### Key Design Decisions
- **단일 컬렉션**: `documents` 컬렉션 하나. 모든 문서가 동일 컬렉션에 저장
- **로컬 전용**: BAAI/bge-m3 (1024D, cosine). 외부 API 호출 없음
- **4단계 메타데이터 추출** (아래 상세)
- **데이터 저장**: 프로젝트 폴더 내 `data/` (chroma_db, index_tracker.db, bm25_cache, logs)
- **설정 최소화**: `doc_root` 경로 하나만 지정하면 바로 사용 가능
- **디바이스 자동 감지**: MPS (Apple Silicon) → CUDA → CPU 순서로 자동 선택
- **초기 설정 자동화**: 설치 시 디바이스, 경로 등을 자동 감지하여 config 생성

### 메타데이터 추출 전략 (4단계 폴백)

기존 KISTI 방식(폴더 구조에서 연도/주제 파싱)을 제거하고, 문서 자체에서 메타데이터를 추출.

```
1단계: 문서 내장 properties
  PDF metadata, DOCX/PPTX/XLSX core_properties, HWP SummaryInformation
  → doc_title, doc_author, doc_created, doc_modified, doc_keywords, doc_subject

2단계: 첫 페이지 텍스트 패턴 매칭 (1단계가 비어있을 때 보충)
  이미 추출한 텍스트의 앞부분에서 패턴으로 제목/작성자/날짜 추출
  → 제목: 첫 의미있는 줄, "제목:", PDF 폰트크기 기반
  → 작성자: "작성자:", "담당자:" 패턴
  → 날짜: "작성일:", "2025. 3. 5" 등 날짜 패턴

3단계: 파일시스템 메타데이터 (항상)
  → file_name, file_size, fs_created, fs_modified

4단계: 경로 기반 (항상)
  → relative_path, file_type (확장자)
```

각 Extractor의 `ExtractionResult.metadata`에 1단계 정보를 포함하도록 수정.
`MetadataExtractor`가 2~4단계를 담당하고, 1단계 결과와 병합.

### Context Prefix (Contextual Retrieval)

청크 앞에 붙여 임베딩 품질 향상. 문서 메타데이터 기반으로 생성:
```
# 내부 메타데이터 있을 때
[2025-03 | 연구개발 사업계획서 | 홍길동 | 계획서.pdf]

# 없을 때 (폴백)
[연구자료/하위폴더 | 보고서.txt]
```

### MCP 검색 필터

기존 `year`, `folder_topic` → 문서 메타데이터 기반으로 변경:
- `file_type`: 파일 확장자 (유지)
- `doc_type`: 문서 유형 (보고서, 계획서 등 - 1~2단계에서 추출 가능할 때)
- `author`: 작성자 (1~2단계에서 추출 가능할 때)

## Project Structure

```
docdb/
├── CLAUDE.md
├── README.md
├── pyproject.toml              # pip install -e . 지원
├── requirements.txt
├── config/
│   └── config.yaml             # 기본 설정 (사용자가 doc_root만 수정)
├── data/                       # 런타임 데이터 (.gitignore)
│   ├── chroma_db/              # ChromaDB 벡터 DB
│   ├── index_tracker.db        # SQLite 인덱싱 추적
│   ├── bm25_cache.json          # BM25 캐시
│   └── logs/                   # 로그 파일
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI 엔트리포인트
│   ├── config.py               # 설정 로더 (PROJECT_ROOT 기준 경로 해석)
│   ├── indexing_pipeline.py    # 공통 인덱싱 파이프라인 (main.py, server.py 공유)
│   ├── document_processor/
│   │   ├── extractors/         # 16개 포맷 추출기
│   │   │   ├── base_extractor.py
│   │   │   ├── hwp_extractor.py
│   │   │   ├── pdf_extractor.py
│   │   │   ├── office_extractors.py
│   │   │   ├── text_extractors.py
│   │   │   ├── apple_extractor.py
│   │   │   └── ocr_helper.py
│   │   ├── chunking/
│   │   │   └── korean_chunker.py
│   │   ├── metadata_extractor.py   # 범용 메타데이터 (KISTI 구조 의존 제거)
│   │   └── processor.py
│   ├── embedding/
│   │   ├── local_embedder.py       # bge-m3 (유일한 임베딩 백엔드)
│   │   └── embedding_manager.py    # 임베딩 관리 (로컬 전용)
│   ├── vectorstore/
│   │   ├── chroma_manager.py       # 단일 컬렉션 기본
│   │   └── retriever.py
│   ├── search/
│   │   ├── bm25_index.py
│   │   ├── reranker.py
│   │   └── context_builder.py
│   ├── incremental/
│   │   ├── index_tracker.py
│   │   └── file_scanner.py
│   └── mcp_server/
│       └── server.py               # "docdb" MCP 서버 (stdio 전송)
└── tests/                          # (TODO)
```

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 또는 editable install
pip install -e .

# Run modes
python -m src.main --mode full_index --config config/config.yaml
python -m src.main --mode incremental_index
python -m src.main --mode mcp_server
python -m src.main --mode search --query "검색어"
python -m src.main --mode stats
```

## Conventions

- Python 3.10+
- Logging: `loguru`
- Extractors: `BaseExtractor` ABC, `extract() → ExtractionResult`
- Embedders: lazy loading (첫 `embed()` 호출 시 모델 로드)
- Korean comments/logs, English class/method names
- `chunk_id` format: `{stem}_{path_hash}_{index}`
- ChromaDB cosine distance (`hnsw:space: cosine`)
- Config: YAML, `~` 경로 확장 지원

## Migration Status

kisti-vectordb → docDB 변환 완료 상태:

### Phase 1: 복사 및 KISTI 제거 — ✅ 완료
1. [x] 소스 복사 (`api_embedder.py`, `http_transport.py` 제외)
2. [x] 모든 "KISTI", "kisti" → "docDB", "docdb" 문자열 변환
3. [x] `kisti_root` → `doc_root` 변수명 통일
4. [x] `_remap_path()` 및 SynologyDrive 관련 로직 제거
5. [x] `sensitive_folders`, 민감 라우팅 로직 제거

### Phase 2: 핵심 모듈 범용화 — ✅ 완료
6. [x] `ChromaManager` 단일 컬렉션 `documents`로 단순화
7. [x] `EmbeddingManager` 로컬 전용으로 단순화
8. [x] `Retriever` 단일 컬렉션 검색으로 단순화
9. [x] `MetadataExtractor` 4단계 폴백 범용 추출로 재작성
10. [x] 각 Extractor에 문서 내장 properties 추출 추가
11. [x] `context_builder.py` 문서 메타데이터 기반으로 변경
12. [x] `config.py` 범용화 (`DOCDB_*` 환경변수, PROJECT_ROOT 기준 경로 해석)
13. [x] 원격 ChromaDB HTTP 모드 제거

### Phase 3: 사용성 — ✅ 완료
14. [x] `config.yaml` 범용화 (doc_root만 필수, 합리적 기본값)
15. [x] 디바이스 자동 감지 (mps → cuda → cpu)
16. [x] 초기 설정 자동화 스크립트 (`setup.py`: 디바이스 감지, doc_root 입력 → config.yaml 생성)
17. [x] MCP 서버 이름/도구 설명 범용화
18. [x] MCP 검색 필터 변경 (file_type/author/doc_type)
19. [x] `pyproject.toml` 생성
20. [x] README.md 작성

### Phase 4: 품질 — 진행 중
21. [x] KoreanChunker 유지 확인
22. [ ] 테스트 작성
23. [x] 실제 문서로 인덱싱 → 검색 end-to-end 검증 (247파일, 236성공)
24. [x] 코드 품질/보안/성능 리뷰 (4라운드 에이전트 + Codex + 아키텍트 Opus 검증) → 47건 수정 완료

### 버그 수정 이력 (리팩토링 후 발견/수정)
- `chroma_manager.py`: `_sanitize_metadata()` 추가 — doc_properties nested dict → ChromaDB ValueError
- `retriever.py`: `author` → `doc_author` ChromaDB where filter 필드명 수정
- `server.py`: list_documents 필터 미적용 → ChromaDB 메타데이터 직접 쿼리로 재작성
- `server.py`: `_handle_tool_call` 반환 타입 일관성 수정 (항상 CallToolResult)
- `index_tracker.py`: `record_error` 변수 스코프 버그 수정
- `file_scanner.py`, `main.py`, `metadata_extractor.py`, `server.py`: expanduser 누락 수정
- `src/__init__.py`: 모듈 레벨 loguru import 제거
- `bm25_index.py`: 캐시 경로 `~/.docdb/` → `data/` (프로젝트 상대경로)

### v0.2.0 코드 품질/보안 개선 (4라운드 에이전트 + Codex + 아키텍트 Opus 검증, 47건)
- **C2**: `indexing_pipeline.py` 생성 — main.py/server.py의 중복 인덱싱 로직 통합
- **C3**: `add_chunks()` 실패 시 tracker에 성공 기록 방지 (반환값 확인)
- **C4**: full_index에서 stale 청크 정리 (`_cleanup_stale_chunks`)
- **C5**: 변경 파일 처리 순서 개선 — upsert 후 stale 삭제 (데이터 소실 방지)
- **C1**: BM25 캐시 pickle → JSON 전환 (임의 코드 실행 방지)
- **H1**: `get_document` path traversal 방어 강화 (doc_root 빈값 시 거부, NFC 정규화)
- **H2**: `defusedxml` 도입 — XXE/Billion Laughs 방어 (hwp, apple extractors, fallback 경고)
- **H3**: chunk ID 해시 MD5 8자 → SHA-256 12자 (충돌 확률 대폭 감소)
- **H5**: XLS + XLSX 헤더-값 매핑 버그 수정 (빈 셀 시 열 인덱스 추적)
- **H6**: asyncio.Lock `_initialize_components`로 이동 + None guard
- **M1**: MCP 파라미터 bounds 검증 (n_results, chunk_offset, chunk_limit, limit)
- **M2**: `list_documents` unbounded query → limit 상한 설정 (MAX_FETCH)
- **M3**: `record_error()` TOCTOU race condition 수정 (단일 lock 범위, is_indexed 체크)
- **M4**: 서버 종료 시 SQLite 연결 정리 (`tracker.close()`)
- **M5**: full_index 파일 스캔 rglob 10회 → 단일 순회
- **M9**: CLI 증분 인덱싱 실패 기록 누락 수정 (공통 파이프라인에서 처리)
- **M10**: 경로 NFC 정규화 통일 (macOS 한글 파일명 중복 방지)
- **M8**: 설정 파일 regex 패턴 검증 (잘못된 패턴 시 경고 후 무시, main.py + file_scanner.py)
- **L1**: 미사용 `_chunk_text` 메서드 제거
- **L2**: 루프 내 반복 해시 계산 → 루프 외 1회로 이동
- TOCTOU mtime 수정: `index_single_file`에 mtime 파라미터 추가 (처리 전 1회 확정)
- `_get_document`에서 resolved(NFC+realpath) 경로를 process_document에 전달
- 에러 메시지 내부정보 노출 제거 (5개 핸들러 모두 제네릭 메시지)
- CLI 모드 `IndexTracker` context manager 적용 (full_index, incremental, stats)
- 지원 확장자 통일 (`FileScanner.SUPPORTED`를 기본값으로 사용)
- XLSX 빈 워크시트 `max_row` 가드 추가
- `_cleanup_stale_chunks` ChromaDB 페이지네이션 적용
- BM25 IDs hash 수집 페이지네이션 적용
- BM25 중복 `collection.count()` 제거, `_save_cache(len(all_ids))` 자체 일관성
- `.md`, `.htm` 포맷 지원 추가 (총 16개 포맷)
- 로그 `retention=5` 설정 (디스크 무한 누적 방지)
- `_bm25_lock` `__init__`에서 None 초기화 (AttributeError 방지)
