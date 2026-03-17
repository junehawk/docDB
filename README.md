# 📚 docDB

> **내 PC의 문서를 AI가 읽을 수 있게 만들어주는 로컬 벡터 검색 엔진**

문서 폴더 경로 하나만 지정하면, 알아서 인덱싱하고 MCP 서버로 띄워줍니다.
Claude, ChatGPT, Cursor 등 **MCP를 지원하는 모든 AI 클라이언트**에서 자연어로 내 문서를 검색할 수 있어요.

**100% 로컬** — 외부 서버 없음. API 키 없음. 클라우드 전송 없음. 내 문서는 내 PC에만.

---

## ✨ 왜 docDB?

| 기존 방식 | docDB |
|-----------|-------|
| 파일 탐색기에서 폴더 뒤지기 | AI에게 "지난달 작성한 보고서 찾아줘" 한마디 |
| 파일명으로만 검색 | 문서 **내용** 기반 시맨틱 검색 |
| 클라우드에 문서 업로드 필요 | 100% 로컬, 내 PC에서 완결 |
| 특정 AI 도구에 종속 | MCP 표준 프로토콜 → 어떤 AI든 OK |

---

## 🔧 주요 기능

- **16개 문서 포맷**: HWP, HWPX, PDF, DOCX, PPTX, XLSX, XLS, CSV, TXT, Markdown, HTML, HTM, RTF, Pages, Numbers, Keynote
- **로컬 임베딩**: BAAI/bge-m3 (1024D) — API 키 불필요
- **하이브리드 검색**: 벡터 유사도 + BM25 키워드 + Reranker
- **증분 인덱싱**: 변경된 파일만 자동 감지 → 빠른 업데이트
- **메타데이터 자동 추출**: 문서 내장 속성 + 첫 페이지 패턴 매칭 + 파일시스템 정보
- **디바이스 자동 감지**: Apple Silicon (MPS) → NVIDIA (CUDA) → CPU 순 자동 선택
- **MCP 서버**: 표준 MCP 프로토콜로 어떤 AI 클라이언트든 연결 가능

---

## 🏗️ 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 PC                                 │
│                                                                 │
│  ┌──────────────┐                                               │
│  │  문서 폴더    │  ~/Documents, ~/Research 등                   │
│  │  (어떤 구조든) │  HWP, PDF, DOCX, PPTX, XLSX, ...            │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              docDB 인덱싱 파이프라인                    │       │
│  │                                                      │       │
│  │  📄 텍스트 추출 ──→ ✂️ 청킹 ──→ 🔢 임베딩 ──→ 💾 저장  │       │
│  │  (16개 포맷)    (800자/100)  (bge-m3)    (ChromaDB)  │       │
│  └──────────────────────────────────────────────────────┘       │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              로컬 벡터 DB (data/)                      │       │
│  │  ChromaDB + BM25 인덱스 + SQLite 트래커               │       │
│  └──────────────────────────┬───────────────────────────┘       │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              docDB MCP 서버 (stdio)                   │       │
│  │  search · get_document · list · reindex · stats      │       │
│  └──────────────────────────┬───────────────────────────┘       │
│                             │ MCP Protocol (stdin/stdout)       │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────┐       │
│  │           MCP 호환 AI 클라이언트 (아무거나!)            │       │
│  │                                                      │       │
│  │  🟤 Claude Desktop    🟢 ChatGPT Desktop             │       │
│  │  🔵 Claude Code       🟡 Cursor                      │       │
│  │  ⚫ Windsurf          🟣 Cline / Continue 등          │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 빠른 시작

### 1. 설치

```bash
git clone https://github.com/junehawk/docDB.git
cd docDB

# 가상환경 (Python 3.10+)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 또는 editable install
pip install -e .
```

### 2. 설정

```bash
python setup.py
```

자동으로 3단계를 진행합니다:

```
[1/3] 디바이스 감지 중...
  -> Apple Silicon (MPS)                    # MPS / CUDA / CPU 자동 선택

[2/3] 문서 폴더 설정
인덱싱할 문서 폴더 경로를 입력하세요.
  기본값: /Users/you/Documents              # OS별 기본 문서 폴더 자동 감지
  경로 (Enter=기본값): ~/Research/papers    # 원하는 경로 입력 또는 Enter
  -> 지원 포맷 파일 142개 발견              # 16개 포맷 자동 스캔

[3/3] 설정 파일 생성
  -> 저장 완료: config/config.yaml          # 기존 설정이 있으면 .bak 백업
```

> 수동 설정도 가능합니다 — `config/config.yaml`에서 **딱 한 줄만 바꾸면 됩니다**:
>
> ```yaml
> document_processing:
>   doc_root: '~/Documents/내문서폴더'  # 👈 여기만 수정
> ```

### 3. 인덱싱

```bash
# 전체 인덱싱 (최초 1회)
python -m src.main --mode full_index --config config/config.yaml

# 이후엔 변경분만 (증분)
python -m src.main --mode incremental_index
```

### 4. AI 클라이언트에 연결

아래 [MCP 클라이언트 설정 가이드](#-mcp-클라이언트-설정-가이드) 참고!

---

## 🔌 MCP 클라이언트 설정 가이드

docDB는 **MCP (Model Context Protocol)** 표준을 따르기 때문에, MCP를 지원하는 모든 AI 클라이언트에서 사용할 수 있습니다.

> **공통 참고사항**
> - `/absolute/path/to/docdb` 부분을 실제 docDB 설치 경로로 바꿔주세요
> - 가상환경의 Python 경로를 사용해야 합니다 (`venv/bin/python`)
> - 설정 후 클라이언트를 재시작해야 적용됩니다

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
`%APPDATA%\Claude\claude_desktop_config.json` (Windows)

```json
{
  "mcpServers": {
    "docdb": {
      "command": "/absolute/path/to/docdb/venv/bin/python",
      "args": ["-m", "src.main", "--mode", "mcp_server", "--config", "config/config.yaml"],
      "cwd": "/absolute/path/to/docdb"
    }
  }
}
```

### Claude Code

```bash
# 프로젝트 단위 설정
claude mcp add docdb \
  --scope project \
  -- /absolute/path/to/docdb/venv/bin/python \
  -m src.main --mode mcp_server --config config/config.yaml

# 또는 전역 설정
claude mcp add docdb \
  --scope user \
  -- /absolute/path/to/docdb/venv/bin/python \
  -m src.main --mode mcp_server --config config/config.yaml
```

> Claude Code에서는 `DOCDB_CONFIG` 환경변수로 config 경로를 지정할 수도 있습니다.

### ChatGPT Desktop (macOS)

ChatGPT Desktop은 MCP 서버를 지원합니다. 설정 파일 위치:

`~/Library/Application Support/com.openai.chat/mcp.json`

```json
{
  "mcpServers": {
    "docdb": {
      "command": "/absolute/path/to/docdb/venv/bin/python",
      "args": ["-m", "src.main", "--mode", "mcp_server", "--config", "config/config.yaml"],
      "cwd": "/absolute/path/to/docdb"
    }
  }
}
```

> ChatGPT Desktop → Settings → Beta features → MCP servers 토글을 켜야 합니다.

### Cursor

Cursor Settings → MCP → "Add new MCP server" 클릭 후:

```json
{
  "mcpServers": {
    "docdb": {
      "command": "/absolute/path/to/docdb/venv/bin/python",
      "args": ["-m", "src.main", "--mode", "mcp_server", "--config", "config/config.yaml"],
      "cwd": "/absolute/path/to/docdb"
    }
  }
}
```

### 기타 MCP 호환 클라이언트

Windsurf, Cline, Continue 등 MCP를 지원하는 클라이언트라면 동일한 패턴으로 연결 가능합니다.
대부분 아래 정보만 있으면 설정할 수 있어요:

| 항목 | 값 |
|------|----|
| **Command** | `/absolute/path/to/docdb/venv/bin/python` |
| **Args** | `-m src.main --mode mcp_server --config config/config.yaml` |
| **Working Directory** | `/absolute/path/to/docdb` |
| **Transport** | stdio |

---

## 💬 사용 예시

MCP 연결 후 AI에게 자연어로 물어보세요:

```
"최근 작성한 보고서 찾아줘"
"예산 관련 문서 검색해줘"
"홍길동이 작성한 PDF 문서 목록 보여줘"
"인덱싱 상태 확인해줘"
"새로 추가한 파일 인덱싱해줘"
```

---

## 🛠️ CLI 명령어

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

---

## 📡 MCP 도구 목록

| Tool | 설명 | 주요 파라미터 |
|------|------|---------------|
| `search_documents` | 시맨틱 검색 + 메타데이터 필터링 | `query`, `n_results`, `file_type`, `author`, `doc_type` |
| `get_document` | 특정 문서의 전체 텍스트 반환 | `file_path`, `chunk_offset`, `chunk_limit` |
| `list_documents` | 조건별 문서 목록 조회 | `file_type`, `author`, `doc_type`, `limit` |
| `reindex` | 증분 인덱싱 수동 트리거 | — |
| `get_stats` | 벡터 DB / 인덱싱 통계 | — |

---

## 📁 데이터 저장 위치

모든 런타임 데이터는 프로젝트 내 `data/` 폴더에 저장됩니다.
폴더 통째로 복사하면 인덱싱 데이터도 같이 이동돼요.

| 항목 | 경로 |
|------|------|
| 벡터 DB | `data/chroma_db/` |
| 인덱싱 추적 DB | `data/index_tracker.db` |
| BM25 캐시 | `data/bm25_cache.json` |
| 로그 | `data/logs/` |

---

## 📋 요구사항

- Python 3.10+
- macOS / Linux / Windows
- 디스크 여유 공간 ~10GB (벡터 DB + 모델 캐시)
- LibreOffice (선택사항, HWP fallback용)
- defusedxml (XML 파싱 보안, `pip install` 시 자동 설치)

---

## 🗂️ 프로젝트 구조

```
docdb/
├── config/
│   └── config.yaml              # 설정 (doc_root만 수정하면 됨)
├── src/
│   ├── main.py                  # CLI 엔트리포인트
│   ├── config.py                # 설정 로더
│   ├── indexing_pipeline.py     # 공통 인덱싱 파이프라인
│   ├── document_processor/
│   │   ├── extractors/          # 16개 포맷 추출기
│   │   ├── chunking/            # 한국어 청킹
│   │   ├── metadata_extractor.py # 4단계 메타데이터 추출
│   │   └── processor.py         # 문서 처리 파이프라인
│   ├── embedding/               # bge-m3 로컬 임베딩
│   ├── vectorstore/             # ChromaDB + 하이브리드 검색
│   ├── search/                  # BM25, Reranker, 컨텍스트
│   ├── incremental/             # 증분 인덱싱 (SQLite 트래커)
│   └── mcp_server/
│       └── server.py            # MCP 서버 (stdio)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 📄 라이선스

MIT License
