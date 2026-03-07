# docDB

로컬 문서 폴더를 벡터 DB로 인덱싱하고, MCP 서버를 통해 Claude Desktop/Claude Code에서 자연어 검색하는 도구.

**100% 로컬** — 외부 서버, API 키, 클라우드 연동 없이 자기 PC에서 완전 동작.

## 특징

- **14개 문서 포맷 지원**: HWP, HWPX, PDF, DOCX, PPTX, XLSX, XLS, CSV, TXT, HTML, RTF, Pages, Numbers, Keynote
- **로컬 임베딩**: BAAI/bge-m3 (1024D) — API 키 불필요
- **하이브리드 검색**: 벡터 검색 + BM25 + Reranker
- **증분 인덱싱**: 변경된 파일만 자동 감지하여 업데이트
- **문서 메타데이터 자동 추출**: PDF/DOCX/PPTX/HWP 내장 속성 + 첫 페이지 패턴 매칭
- **MCP 서버**: Claude Desktop/Code에서 바로 사용 가능
- **디바이스 자동 감지**: Apple Silicon (MPS), NVIDIA GPU (CUDA), CPU 자동 선택

## 빠른 시작

### 1. 설치

```bash
git clone <repository-url> docdb
cd docdb

# 가상환경 생성 (Python 3.10+)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 또는 editable install
pip install -e .
```

### 2. 설정

```bash
# 자동 설정 (디바이스 감지 + 문서 폴더 경로 입력 → config.yaml 생성)
python setup.py
```

또는 `config/config.yaml`에서 직접 문서 폴더 경로를 설정:

```yaml
document_processing:
  doc_root: '~/Documents/연구자료'  # 본인의 문서 폴더 경로
```

이것만 수정하면 됩니다. 나머지는 합리적 기본값이 설정되어 있습니다.

### 3. 인덱싱

```bash
# 전체 인덱싱 (최초 1회)
python -m src.main --mode full_index --config config/config.yaml

# 증분 인덱싱 (이후 변경분만)
python -m src.main --mode incremental_index
```

### 4. Claude Desktop 연결

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) 또는
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "docdb": {
      "command": "/path/to/docdb/venv/bin/python",
      "args": ["-m", "src.main", "--mode", "mcp_server", "--config", "config/config.yaml"],
      "cwd": "/path/to/docdb"
    }
  }
}
```

Claude Desktop 재시작 후 자연어로 문서 검색:
- "최근 작성한 보고서 찾아줘"
- "예산 관련 문서 검색해줘"
- "인덱싱 상태 확인"

## CLI 사용법

```bash
# MCP 서버 실행
python -m src.main --mode mcp_server

# 전체 인덱싱
python -m src.main --mode full_index --config config/config.yaml

# 증분 인덱싱
python -m src.main --mode incremental_index

# 검색 테스트
python -m src.main --mode search --query "검색어"

# 통계 조회
python -m src.main --mode stats
```

## MCP 도구

| Tool | 설명 | 주요 파라미터 |
|------|------|---------------|
| `search_documents` | 시맨틱 검색 + 메타데이터 필터 | query, n_results, file_type, author, doc_type |
| `get_document` | 특정 문서 전체 텍스트 | file_path |
| `list_documents` | 조건별 문서 목록 | file_type, author, doc_type, limit |
| `reindex` | 증분 인덱싱 수동 트리거 | - |
| `get_stats` | 벡터 DB / 인덱싱 통계 | - |

## 데이터 저장 위치

모든 데이터는 프로젝트 폴더 내 `data/`에 저장됩니다:

| 항목 | 경로 |
|------|------|
| 벡터 DB | `data/chroma_db/` |
| 인덱싱 추적 DB | `data/index_tracker.db` |
| 로그 | `data/logs/` |
| BM25 캐시 | `data/bm25_cache.pkl` |

프로젝트 폴더를 통째로 복사하면 인덱싱 데이터도 함께 이동됩니다.

## 요구사항

- Python 3.10+
- macOS, Linux, 또는 Windows
- 디스크 여유 공간 ~10GB (벡터 DB + 추적 DB)
- LibreOffice (선택, HWP fallback용)

## 프로젝트 구조

```
docdb/
├── config/
│   └── config.yaml              # 설정 (doc_root만 수정)
├── src/
│   ├── main.py                  # CLI 엔트리포인트
│   ├── config.py                # 설정 로더
│   ├── document_processor/
│   │   ├── extractors/          # 14개 포맷 추출기
│   │   ├── chunking/            # 한국어 청킹
│   │   ├── metadata_extractor.py # 4단계 메타데이터 추출
│   │   └── processor.py         # 문서 처리 오케스트레이터
│   ├── embedding/
│   │   ├── local_embedder.py    # bge-m3 로컬 임베딩
│   │   └── embedding_manager.py # 임베딩 관리
│   ├── vectorstore/
│   │   ├── chroma_manager.py    # ChromaDB 관리
│   │   └── retriever.py         # 하이브리드 검색기
│   ├── search/
│   │   ├── bm25_index.py        # BM25 인덱스
│   │   ├── reranker.py          # 리랭커
│   │   └── context_builder.py   # 컨텍스트 접두사
│   ├── incremental/
│   │   ├── index_tracker.py     # SQLite 파일 추적
│   │   └── file_scanner.py      # 디스크 스캔
│   └── mcp_server/
│       └── server.py            # MCP 서버
├── pyproject.toml
├── requirements.txt
└── README.md
```
