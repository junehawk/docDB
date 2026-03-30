# Windows Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make docDB fully functional on Windows (currently macOS-only tested) by fixing 29 identified compatibility issues across 4 priority phases.

**Architecture:** Create a `src/compat.py` platform utility module that centralizes all platform detection and OS-specific helpers. Each existing module then imports from `compat.py` instead of sprinkling `sys.platform` checks everywhere. This keeps platform logic DRY and testable.

**Tech Stack:** Python 3.10+, pathlib, sys.platform guards, subprocess.CREATE_NO_WINDOW, asyncio.WindowsSelectorEventLoopPolicy

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/compat.py` | Platform detection, path normalization, console setup, device guards |
| Create | `tests/test_compat.py` | Tests for platform utility module |
| Modify | `src/main.py` | Use compat helpers for MPS guard, asyncio policy, console encoding, logger |
| Modify | `src/mcp_server/server.py` | Use compat helpers for MPS guard, asyncio policy, path traversal, atexit |
| Modify | `src/search/context_builder.py` | OS-aware path separator in context prefix |
| Modify | `src/search/reranker.py` | Default device `'mps'` → `'auto'` |
| Modify | `src/search/bm25_index.py` | Atomic cache write via temp file + os.replace |
| Modify | `src/document_processor/processor.py` | Catch PermissionError for locked files |
| Modify | `src/document_processor/extractors/hwp_extractor.py` | CREATE_NO_WINDOW, soffice/libreoffice detection, TemporaryDirectory cleanup |
| Modify | `src/document_processor/extractors/ocr_helper.py` | Windows Tesseract path detection |
| Modify | `src/document_processor/extractors/text_extractors.py` | EUC-KR → cp949 mapping |
| Modify | `src/document_processor/metadata_extractor.py` | Safe rstrip(os.sep), path fallback improvement |
| Modify | `src/incremental/file_scanner.py` | followlinks=False, extension via splitext, Windows excluded patterns |
| Modify | `src/incremental/index_tracker.py` | SQLite timeout parameter |
| Modify | `src/indexing_pipeline.py` | Device-guarded gc.collect, platform-aware normalize_path |
| Modify | `src/embedding/local_embedder.py` | Device-guarded gc.collect in _clear_device_cache |
| Modify | `src/embedding/embedding_manager.py` | Guard MPS detection with sys.platform |
| Modify | `src/vectorstore/chroma_manager.py` | normpath for persist_dir |
| Modify | `src/config.py` | Windows USERPROFILE awareness |
| Modify | `setup.py` | Forward-slash YAML paths, Windows Defender guidance |
| Modify | `config/config.yaml` | Add Windows excluded patterns |

---

### Task 1: Platform Utility Module (`src/compat.py`)

**Files:**
- Create: `src/compat.py`
- Create: `tests/test_compat.py`

All subsequent tasks depend on this module. It centralizes every platform-specific check.

- [ ] **Step 1: Create `src/compat.py` with all platform helpers**

```python
"""
플랫폼 호환성 유틸리티
Windows/macOS/Linux 간 차이를 추상화하는 헬퍼 모음
"""
import os
import sys
import unicodedata
from pathlib import Path

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'


def normalize_path(file_path) -> str:
    """
    플랫폼 인식 경로 정규화.
    macOS: NFC 정규화 (HFS+/APFS NFD 대응)
    Windows/Linux: 문자열 변환만
    """
    s = str(file_path)
    if IS_MACOS:
        return unicodedata.normalize('NFC', s)
    return s


def safe_realpath(path: str) -> str:
    """
    플랫폼 인식 realpath + 대소문자 정규화.
    Windows: normcase 적용 (드라이브 문자 소문자화, 백슬래시 통일)
    """
    resolved = os.path.realpath(os.path.expanduser(path))
    if IS_MACOS:
        resolved = unicodedata.normalize('NFC', resolved)
    if IS_WINDOWS:
        resolved = os.path.normcase(resolved)
    return resolved


def path_is_under(child: str, parent: str) -> bool:
    """
    child 경로가 parent 경로 하위에 있는지 검증.
    Windows 드라이브 문자, 대소문자, junction 안전.
    """
    child_resolved = safe_realpath(child)
    parent_resolved = safe_realpath(parent)
    # Python 3.9+ is_relative_to
    try:
        return Path(child_resolved).is_relative_to(Path(parent_resolved))
    except AttributeError:
        # Python 3.8 폴백
        return child_resolved.startswith(parent_resolved + os.sep) or child_resolved == parent_resolved


def setup_mps_env():
    """MPS watermark 환경변수 설정 (macOS만)"""
    if IS_MACOS:
        os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')


def should_unload_model(device: str) -> bool:
    """주기적 모델 언로드가 필요한 디바이스인지 확인 (MPS만)"""
    return device == 'mps'


def setup_console_encoding():
    """Windows 콘솔 UTF-8 인코딩 설정"""
    if IS_WINDOWS:
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, OSError):
            pass


def setup_asyncio_policy():
    """Windows에서 SelectorEventLoopPolicy 설정 (MCP stdio 호환)"""
    import asyncio
    if IS_WINDOWS:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def get_subprocess_kwargs() -> dict:
    """Windows에서 콘솔 창 팝업 방지를 위한 subprocess kwargs"""
    import subprocess
    if IS_WINDOWS:
        return {'creationflags': subprocess.CREATE_NO_WINDOW}
    return {}


def find_executable(names: list) -> str | None:
    """
    여러 실행파일 이름 중 PATH에서 찾은 첫 번째 반환.
    Windows 공통 설치 경로도 탐색.
    """
    import shutil
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    # Windows 공통 설치 경로 탐색
    if IS_WINDOWS:
        program_dirs = [
            os.environ.get('ProgramFiles', r'C:\Program Files'),
            os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
            os.environ.get('LOCALAPPDATA', ''),
        ]
        search_paths = {
            'tesseract': ['Tesseract-OCR\\tesseract.exe'],
            'libreoffice': ['LibreOffice\\program\\soffice.exe'],
            'soffice': ['LibreOffice\\program\\soffice.exe'],
        }
        for name in names:
            for prog_dir in program_dirs:
                if not prog_dir:
                    continue
                for sub in search_paths.get(name, []):
                    candidate = os.path.join(prog_dir, sub)
                    if os.path.isfile(candidate):
                        return candidate
    return None


def fix_encoding_name(encoding: str) -> str:
    """chardet 인코딩 이름 보정 (EUC-KR → cp949 등)"""
    mapping = {
        'euc-kr': 'cp949',       # cp949는 EUC-KR의 상위 집합
        'euc_kr': 'cp949',
        'ks_c_5601-1987': 'cp949',
    }
    return mapping.get(encoding.lower(), encoding) if encoding else 'utf-8'


def get_file_extension(filename: str) -> str:
    """안전한 파일 확장자 추출 (dotless 파일 대응)"""
    _, ext = os.path.splitext(filename)
    return ext.lstrip('.').lower() if ext else ''
```

- [ ] **Step 2: Create `tests/test_compat.py`**

```python
"""src/compat.py 유닛 테스트"""
import os
import sys
import pytest
from src.compat import (
    normalize_path, safe_realpath, path_is_under,
    fix_encoding_name, get_file_extension, find_executable,
    get_subprocess_kwargs, should_unload_model,
)


class TestNormalizePath:
    def test_basic_string(self):
        assert normalize_path("/some/path/file.txt") == "/some/path/file.txt"

    def test_pathlib_input(self):
        from pathlib import Path
        result = normalize_path(Path("/some/path"))
        assert isinstance(result, str)

    def test_korean_filename(self):
        # NFC/NFD 정규화 확인 (macOS에서만 의미 있지만 다른 OS에서도 에러 없어야 함)
        result = normalize_path("/경로/한글파일.txt")
        assert "한글파일" in result


class TestPathIsUnder:
    def test_child_under_parent(self, tmp_path):
        parent = str(tmp_path)
        child = str(tmp_path / "sub" / "file.txt")
        os.makedirs(tmp_path / "sub", exist_ok=True)
        (tmp_path / "sub" / "file.txt").touch()
        assert path_is_under(child, parent) is True

    def test_child_outside_parent(self, tmp_path):
        import tempfile
        other = tempfile.mkdtemp()
        child = os.path.join(other, "file.txt")
        open(child, 'w').close()
        assert path_is_under(child, str(tmp_path)) is False


class TestFixEncodingName:
    def test_euc_kr_to_cp949(self):
        assert fix_encoding_name("EUC-KR") == "cp949"
        assert fix_encoding_name("euc-kr") == "cp949"

    def test_utf8_passthrough(self):
        assert fix_encoding_name("utf-8") == "utf-8"

    def test_none_fallback(self):
        assert fix_encoding_name(None) == "utf-8"
        assert fix_encoding_name("") == "utf-8"


class TestGetFileExtension:
    def test_normal(self):
        assert get_file_extension("report.pdf") == "pdf"

    def test_dotless(self):
        assert get_file_extension("README") == ""

    def test_dotfile(self):
        assert get_file_extension(".gitignore") == "gitignore"

    def test_multiple_dots(self):
        assert get_file_extension("archive.tar.gz") == "gz"

    def test_uppercase(self):
        assert get_file_extension("REPORT.PDF") == "pdf"


class TestShouldUnloadModel:
    def test_mps(self):
        assert should_unload_model("mps") is True

    def test_cpu(self):
        assert should_unload_model("cpu") is False

    def test_cuda(self):
        assert should_unload_model("cuda") is False


class TestGetSubprocessKwargs:
    def test_returns_dict(self):
        result = get_subprocess_kwargs()
        assert isinstance(result, dict)
        if sys.platform == 'win32':
            assert 'creationflags' in result
        else:
            assert result == {}
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/JL/Research/docDB && python -m pytest tests/test_compat.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/compat.py tests/test_compat.py
git commit -m "feat: add platform compatibility utility module (src/compat.py)"
```

---

### Task 2: Entry Points — asyncio, MPS Guard, Console, Logger

**Files:**
- Modify: `src/main.py:8,25-28,59,89,93,171,201-205,312-317`
- Modify: `src/mcp_server/server.py:5,591-595,697`

Fixes: C1(MPS unload guard), NEW-C1(asyncio policy), NEW-C3(MPS env), NEW-H1(ANSI logger), NEW-M4(tqdm), NEW-M5(Korean print encoding)

- [ ] **Step 1: Update `src/main.py` — import compat, replace MPS env, add console/asyncio setup**

Replace the top of `main.py` (lines 1-16):

```python
"""
docDB - 메인 엔트리포인트
"""
import argparse
import asyncio
import gc
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

from src.compat import (
    setup_mps_env, setup_console_encoding, setup_asyncio_policy,
    should_unload_model, normalize_path as _compat_normalize,
)

# MPS watermark (macOS만)
setup_mps_env()

from src.config import load_config as _load_config
from src.indexing_pipeline import index_single_file, normalize_path, compute_mtime_hash
```

- [ ] **Step 2: Update `_setup_logger` — ANSI color auto-detect**

Replace `main.py` logger.add for stderr (line 25-28):

```python
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=None,  # auto-detect (Windows legacy 터미널 대응)
    )
```

- [ ] **Step 3: Update `run_mcp_server` — add asyncio policy before asyncio.run**

Replace `main.py` `run_mcp_server` function (lines 54-66):

```python
def run_mcp_server(config_path: Optional[str] = None):
    """MCP 서버 모드 실행 (stdio only)"""
    try:
        from src.mcp_server.server import main
        logger.info("MCP 서버 모드 시작 (stdio 전송)")
        setup_asyncio_policy()
        asyncio.run(main(config_path))
    except ImportError as e:
        logger.error(f"MCP 라이브러리 임포트 실패: {e}")
        logger.error("설치 명령어: pip install 'mcp[cli]'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"MCP 서버 실행 실패: {e}")
        sys.exit(1)
```

- [ ] **Step 4: Update `run_full_index` — path comparison + tqdm + model unload guard**

In `main.py` line 89, use case-insensitive comparison:

```python
        from src.compat import safe_realpath
        if safe_realpath(expanded_root) == safe_realpath(default_docs):
```

In `main.py` line 171, add tqdm non-interactive guard:

```python
            for file_path in tqdm(target_files, desc="인덱싱", disable=not sys.stderr.isatty()):
```

In `main.py` lines 201-205, guard MPS-only unload:

```python
                _files_since_unload += 1
                if (should_unload_model(emb_manager.device)
                        and _files_since_unload >= 200
                        and emb_manager.local_embedder.is_loaded):
                    logger.info("주기적 모델 언로드 (MPS 단편화 방지)")
                    emb_manager.unload_model()
                    gc.collect()
                    _files_since_unload = 0
```

- [ ] **Step 5: Apply same MPS unload guard to `run_incremental_index` (lines 312-317)**

```python
                _files_since_unload += 1
                if (should_unload_model(emb_manager.device)
                        and _files_since_unload >= 200
                        and emb_manager.local_embedder.is_loaded):
                    logger.info("주기적 모델 언로드 (MPS 단편화 방지)")
                    emb_manager.unload_model()
                    gc.collect()
                    _files_since_unload = 0
```

- [ ] **Step 6: Update `main()` — add console encoding setup**

At the top of `main()` function (line 435), before `_setup_logger()`:

```python
def main():
    """메인 함수"""
    setup_console_encoding()
    _setup_logger()
```

- [ ] **Step 7: Update `src/mcp_server/server.py` — MPS env, asyncio policy, unload guard, atexit**

Replace server.py lines 4-5:

```python
import os
import sys
from src.compat import setup_mps_env, should_unload_model, setup_asyncio_policy
setup_mps_env()
```

Replace server.py lines 590-595 (inside `_do_reindex`):

```python
                    _files_since_unload += 1
                    if (should_unload_model(self.embedding_manager.device)
                            and _files_since_unload >= 200
                            and self.embedding_manager.local_embedder.is_loaded):
                        logger.info("주기적 모델 언로드 (MPS 단편화 방지)")
                        self.embedding_manager.unload_model()
                        _gc.collect()
                        _files_since_unload = 0
```

Add atexit safety net in `_initialize_components` (after line 102):

```python
            self.tracker = IndexTracker(db_path=tracker_db)
            import atexit
            atexit.register(lambda: self.tracker.close() if self.tracker else None)
```

Replace server.py `if __name__` block (lines 695-702):

```python
if __name__ == "__main__":
    setup_asyncio_policy()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("서버 종료")
    except Exception as e:
        logger.error(f"치명적 오류: {e}")
        exit(1)
```

- [ ] **Step 8: Verify syntax**

Run: `python -c "from src.main import main; print('main OK')" && python -c "from src.mcp_server.server import DocDBServer; print('server OK')"`
Expected: Both print OK

- [ ] **Step 9: Commit**

```bash
git add src/main.py src/mcp_server/server.py
git commit -m "fix: entry points — asyncio policy, MPS guard, console encoding, ANSI auto-detect"
```

---

### Task 3: Path Handling — context_builder, metadata_extractor, server.py, indexing_pipeline

**Files:**
- Modify: `src/search/context_builder.py:46`
- Modify: `src/document_processor/metadata_extractor.py:244,255`
- Modify: `src/mcp_server/server.py:404-409`
- Modify: `src/indexing_pipeline.py:21-23`
- Modify: `src/vectorstore/chroma_manager.py:23`
- Modify: `src/config.py:15`

Fixes: C3(context_builder `/`), C2(path traversal), M4(rstrip `C:\`), M5(relpath fallback), H5(NFC macOS-only), NEW-M1(realpath junctions), NEW-M10(chroma normpath), NEW-M8(USERPROFILE)

- [ ] **Step 1: Fix `context_builder.py` — OS-aware path splitting**

Replace `context_builder.py` line 46:

```python
        parent = os.path.dirname(rel)
```

Add `import os` at top of file (after line 5):

```python
import os
```

- [ ] **Step 2: Fix `metadata_extractor.py` — safe rstrip, better relpath fallback**

Replace `metadata_extractor.py` line 255:

```python
            doc_root = os.path.normpath(str(self.doc_root))
            if IS_MACOS:
                doc_root = unicodedata.normalize('NFC', doc_root)
```

Add import at top of the `_from_path` method or module level:

```python
from src.compat import IS_MACOS
```

Replace `metadata_extractor.py` lines 258-260:

```python
            try:
                result['relative_path'] = os.path.relpath(file_path, doc_root)
            except ValueError:
                # Windows 크로스 드라이브 등
                result['relative_path'] = os.path.basename(file_path)
```

- [ ] **Step 3: Fix `server.py` path traversal — use compat.path_is_under**

Replace `server.py` lines 403-409:

```python
            from src.compat import path_is_under, safe_realpath
            file_path = os.path.expanduser(file_path)
            resolved = safe_realpath(file_path)
            if not path_is_under(resolved, os.path.expanduser(doc_root)):
                logger.warning(f"Path traversal 차단: {file_path} (doc_root: {doc_root})")
                return self._error_response("허용되지 않은 경로입니다")
```

- [ ] **Step 4: Fix `indexing_pipeline.py` — use compat.normalize_path**

Replace `indexing_pipeline.py` lines 12-23:

```python
import gc
import hashlib
from pathlib import Path
from typing import Dict, Any, Set
from loguru import logger
from src.compat import normalize_path as _platform_normalize


def normalize_path(file_path) -> str:
    """플랫폼 인식 경로 정규화 (macOS NFC, Windows 패스스루)"""
    return _platform_normalize(file_path)
```

Remove `import unicodedata` (line 15) since it's no longer used here.

- [ ] **Step 5: Fix `chroma_manager.py` — normpath for persist_dir**

Replace `chroma_manager.py` line 23:

```python
            self.persist_dir = os.path.normpath(os.path.expanduser(persist_dir))
```

- [ ] **Step 6: Fix `config.py` — USERPROFILE awareness**

Replace `config.py` `_resolve_path` function (lines 13-18):

```python
def _resolve_path(path: str) -> str:
    """경로 확장: ~ → 홈, 상대경로 → 프로젝트 루트 기준 절대경로"""
    if path.startswith('~'):
        # Windows Git Bash에서 HOME vs USERPROFILE 불일치 대응
        if sys.platform == 'win32' and 'USERPROFILE' in os.environ:
            home = os.environ['USERPROFILE']
            path = os.path.join(home, path[2:])  # ~/ or ~\ 이후 부분
        else:
            path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return os.path.normpath(path)
```

Add `import sys` at line 3 of `config.py`.

- [ ] **Step 7: Verify**

Run: `python -c "from src.search.context_builder import build_context_prefix; print(build_context_prefix({'relative_path': 'sub\\\\report.pdf', 'file_name': 'report.pdf'}))"`
Expected: Context prefix with `sub` parent path extracted correctly

- [ ] **Step 8: Commit**

```bash
git add src/search/context_builder.py src/document_processor/metadata_extractor.py \
       src/mcp_server/server.py src/indexing_pipeline.py src/vectorstore/chroma_manager.py src/config.py
git commit -m "fix: cross-platform path handling — traversal check, NFC guard, normpath, relpath"
```

---

### Task 4: Subprocess & External Tools — HWP, OCR, LibreOffice

**Files:**
- Modify: `src/document_processor/extractors/hwp_extractor.py:462-478`
- Modify: `src/document_processor/extractors/ocr_helper.py:59-74`

Fixes: NEW-C2(CREATE_NO_WINDOW), H3(soffice/libreoffice), NEW-H2(Tesseract path), NEW-M6(TemporaryDirectory cleanup)

- [ ] **Step 1: Fix `hwp_extractor.py` — soffice detection, CREATE_NO_WINDOW, temp cleanup**

Replace `hwp_extractor.py` lines 460-478 (the `_try_libreoffice_extract` method body):

```python
        try:
            import tempfile
            import os
        except ImportError:
            return self._create_error_result("tempfile module not available")

        try:
            from src.compat import find_executable, get_subprocess_kwargs

            # LibreOffice 실행 파일 탐지 (Windows: soffice, macOS/Linux: libreoffice)
            lo_cmd = find_executable(['libreoffice', 'soffice'])
            if not lo_cmd:
                return self._create_error_result("LibreOffice not found in PATH")

            # Python 3.12+ ignore_cleanup_errors 지원
            import sys
            td_kwargs = {}
            if sys.version_info >= (3, 12):
                td_kwargs['ignore_cleanup_errors'] = True

            with tempfile.TemporaryDirectory(**td_kwargs) as tmpdir:
                output_path = Path(tmpdir) / (self.file_path.stem + ".txt")

                cmd = [
                    lo_cmd,
                    "--headless",
                    "--convert-to", "txt:Text - txt - csv (StarCalc):44,34,76,1,,1033,true,true,true,false,false",
                    "--outdir", tmpdir,
                    str(self.file_path)
                ]

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        timeout=self.timeout,
                        text=True,
                        **get_subprocess_kwargs()
                    )

                    if result.returncode == 0 and output_path.exists():
                        with open(output_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        return self._create_success_result(
                            text, {"method": "libreoffice", "doc_properties": {}}
                        )
                    else:
                        return self._create_error_result(
                            f"LibreOffice conversion failed: {result.stderr}"
                        )
                except subprocess.TimeoutExpired:
                    return self._create_error_result(
                        f"LibreOffice timeout after {self.timeout}s"
                    )

        except Exception as e:
            logger.debug(f"LibreOffice extraction failed: {e}")
            return self._create_error_result(f"LibreOffice error: {str(e)}")
```

- [ ] **Step 2: Fix `ocr_helper.py` — Tesseract path detection with compat**

Replace `ocr_helper.py` `_is_tesseract_available` method (lines 59-74):

```python
    def _is_tesseract_available(self) -> bool:
        """Tesseract OCR 사용 가능 여부"""
        if self._tesseract_available is not None:
            return self._tesseract_available

        from src.compat import find_executable, get_subprocess_kwargs

        tesseract_cmd = find_executable(['tesseract'])
        if not tesseract_cmd:
            self._tesseract_available = False
            logger.debug("Tesseract OCR 미설치 (Windows: choco install tesseract 또는 https://github.com/UB-Mannheim/tesseract/wiki)")
            return False

        try:
            result = subprocess.run(
                [tesseract_cmd, '--version'],
                capture_output=True, timeout=5,
                **get_subprocess_kwargs()
            )
            self._tesseract_available = (result.returncode == 0)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._tesseract_available = False
            logger.debug("Tesseract OCR 실행 실패")

        return self._tesseract_available
```

- [ ] **Step 3: Verify imports work**

Run: `python -c "from src.document_processor.extractors.hwp_extractor import HwpExtractor; print('OK')" && python -c "from src.document_processor.extractors.ocr_helper import OCRHelper; print('OK')"`
Expected: Both OK

- [ ] **Step 4: Commit**

```bash
git add src/document_processor/extractors/hwp_extractor.py src/document_processor/extractors/ocr_helper.py
git commit -m "fix: subprocess — CREATE_NO_WINDOW, soffice detection, Tesseract path lookup"
```

---

### Task 5: File I/O Robustness — PermissionError, BM25 Atomic Write, SQLite Timeout, Encoding

**Files:**
- Modify: `src/document_processor/processor.py:104-106`
- Modify: `src/search/bm25_index.py:182-195`
- Modify: `src/incremental/index_tracker.py:43`
- Modify: `src/document_processor/extractors/text_extractors.py:66`

Fixes: NEW-H3(PermissionError), NEW-H4(BM25 atomic write), NEW-M3(SQLite timeout), NEW-M7(EUC-KR→cp949)

- [ ] **Step 1: Fix `processor.py` — catch PermissionError for locked files**

In `processor.py`, wrap the extraction call (around line 105). Replace lines 104-112:

```python
            # 텍스트 추출
            try:
                extraction_result = extractor.extract()
            except PermissionError:
                logger.warning(
                    f"파일이 다른 프로그램에 의해 잠김: {file_path.name} "
                    "(Windows에서 파일을 닫고 재시도하세요)"
                )
                return []

            if not extraction_result.success:
                logger.warning(
                    f"Extraction failed for {file_path.name}: "
                    f"{extraction_result.error}"
                )
                return []
```

- [ ] **Step 2: Fix `bm25_index.py` — atomic cache write**

Replace `bm25_index.py` `_save_cache` method (lines 182-195):

```python
    def _save_cache(self, count: int):
        """BM25 인덱스를 로컬 캐시로 저장 (원자적 쓰기 — 크래시 시 손상 방지)"""
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            cache = {
                'count': count,
                'ids_hash': self._compute_ids_hash(self.chunk_ids),
                'chunk_ids': self.chunk_ids,
                'texts': self._texts,
            }
            # 임시 파일에 쓴 뒤 원자적으로 교체 (Windows/POSIX 모두 안전)
            import tempfile
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(self._cache_path), suffix='.tmp'
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, ensure_ascii=False)
                os.replace(tmp_path, self._cache_path)
            except BaseException:
                # 실패 시 임시 파일 정리
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning(f"BM25 캐시 저장 실패: {e}")
```

- [ ] **Step 3: Fix `index_tracker.py` — add SQLite timeout**

Replace `index_tracker.py` line 43:

```python
            self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
```

- [ ] **Step 4: Fix `text_extractors.py` — encoding name correction**

Replace `text_extractors.py` lines 65-72:

```python
            detected = chardet.detect(raw_sample)
            encoding = detected.get("encoding", "utf-8")

            if encoding is None:
                encoding = "utf-8"

            # chardet 인코딩명 보정 (EUC-KR → cp949 등)
            from src.compat import fix_encoding_name
            encoding = fix_encoding_name(encoding)

            logger.debug(f"Detected encoding for {self.file_path.name}: {encoding}")
            return encoding
```

- [ ] **Step 5: Verify**

Run: `python -c "from src.document_processor.processor import DocumentProcessor; print('OK')" && python -c "from src.search.bm25_index import BM25Index; print('OK')" && python -c "from src.incremental.index_tracker import IndexTracker; print('OK')"`
Expected: All OK

- [ ] **Step 6: Commit**

```bash
git add src/document_processor/processor.py src/search/bm25_index.py \
       src/incremental/index_tracker.py src/document_processor/extractors/text_extractors.py
git commit -m "fix: I/O robustness — PermissionError, atomic BM25 write, SQLite timeout, cp949"
```

---

### Task 6: Reranker, Embedder, EmbeddingManager — Device Defaults & GC Guards

**Files:**
- Modify: `src/search/reranker.py:12`
- Modify: `src/embedding/local_embedder.py:61-73`
- Modify: `src/embedding/embedding_manager.py:26-27`
- Modify: `src/indexing_pipeline.py:136-138`

Fixes: H4(reranker default device), NEW-L1(gc.collect guard), L4(MPS detection platform guard), M6(CPU gc overhead)

- [ ] **Step 1: Fix `reranker.py` — default device `'mps'` → `'auto'`**

Replace `reranker.py` line 12:

```python
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3', device: str = 'auto', batch_size: int = 16):
```

Add device detection in `_load_model` (replace line 22):

```python
    def _load_model(self):
        """Lazy loading: 첫 호출 시에만 모델 로드"""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            if self.device == 'auto':
                from src.embedding.embedding_manager import _detect_device
                self.device = _detect_device('auto')
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Reranker 모델 로드 완료: {self.model_name} ({self.device})")
```

- [ ] **Step 2: Fix `local_embedder.py` — device-guarded gc.collect in _clear_device_cache**

Replace `local_embedder.py` `_clear_device_cache` method (lines 61-73):

```python
    def _clear_device_cache(self):
        """GPU/MPS 메모리 캐시 해제 (CPU에서는 스킵)"""
        if self.device == 'cpu':
            return
        import gc
        gc.collect()  # ① Python 참조 해제 → 텐서 참조 끊기
        try:
            import torch
            if self.device == 'mps' and hasattr(torch, 'mps'):
                torch.mps.synchronize()   # ② GPU 작업 완료 대기
                torch.mps.empty_cache()   # ③ 해제 가능한 블록 반환
            elif self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"Device cache clear failed: {e}")
```

- [ ] **Step 3: Fix `embedding_manager.py` — guard MPS detection with platform check**

Replace `embedding_manager.py` lines 25-27 (inside `_detect_device`):

```python
    try:
        import torch
        import sys
        if sys.platform == 'darwin' and torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
```

- [ ] **Step 4: Fix `indexing_pipeline.py` — device-guarded gc.collect in finally block**

Replace `indexing_pipeline.py` lines 136-138:

```python
        finally:
            del valid_chunks
            if emb_manager.device != 'cpu':
                gc.collect()
```

- [ ] **Step 5: Verify**

Run: `python -c "from src.search.reranker import Reranker; r = Reranker(); print(f'device={r.device}')"`
Expected: `device=auto`

- [ ] **Step 6: Commit**

```bash
git add src/search/reranker.py src/embedding/local_embedder.py \
       src/embedding/embedding_manager.py src/indexing_pipeline.py
git commit -m "fix: device defaults — reranker auto, gc.collect CPU skip, MPS platform guard"
```

---

### Task 7: File Scanner — followlinks, Extension, Excluded Patterns

**Files:**
- Modify: `src/incremental/file_scanner.py:126,139,44-50`
- Modify: `config/config.yaml:54-58`

Fixes: NEW-M9(os.walk followlinks), H6(extension extraction), H7(Windows excluded patterns), L1(DS_Store only), NEW-L4(Thumbs.db)

- [ ] **Step 1: Fix `file_scanner.py` — followlinks=False**

Replace `file_scanner.py` line 126:

```python
            for root, dirs, filenames in os.walk(self.doc_root, followlinks=False):
```

- [ ] **Step 2: Fix `file_scanner.py` — extension extraction via compat**

Replace `file_scanner.py` lines 138-140:

```python
                    # 파일 확장자 확인
                    from src.compat import get_file_extension
                    ext = get_file_extension(filename)
                    if not ext or ext not in self.SUPPORTED:
                        continue
```

- [ ] **Step 3: Add Windows patterns to default excluded patterns in `file_scanner.py`**

Find the default excluded patterns list in `file_scanner.py` (around line 44-50) and add Windows patterns. Replace:

```python
        DEFAULT_EXCLUDED = [
            r'\.DS_Store',
            r'__pycache__',
            r'\.git',
            r'~\$',           # Office lock files
            r'Thumbs\.db',    # Windows 썸네일 캐시
            r'desktop\.ini',  # Windows 폴더 설정
        ]
```

- [ ] **Step 4: Update `config/config.yaml` — add Windows patterns**

Add to the excluded_patterns section (around line 54-58):

```yaml
excluded_patterns:
  - '\.DS_Store'     # macOS
  - '__pycache__'
  - '\.git'
  - '~\$'            # Office lock files
  - 'Thumbs\.db'     # Windows
  - 'desktop\.ini'   # Windows
```

- [ ] **Step 5: Verify**

Run: `python -c "from src.incremental.file_scanner import FileScanner; print('OK')"`
Expected: OK

- [ ] **Step 6: Commit**

```bash
git add src/incremental/file_scanner.py config/config.yaml
git commit -m "fix: file scanner — followlinks=False, safe extension, Windows exclusion patterns"
```

---

### Task 8: Setup Script & Documentation

**Files:**
- Modify: `setup.py:89`
- Modify: `CLAUDE.md` (Memory & Performance section)

Fixes: NEW-L2(setup.py backslash paths), M2(display path), NEW-L3(Windows Defender guidance), M11(Claude Desktop path)

- [ ] **Step 1: Fix `setup.py` — forward-slash YAML paths**

Replace `setup.py` line 89:

```python
    display_root = doc_root.replace(home, '~') if doc_root.startswith(home) else doc_root
    display_root = display_root.replace('\\', '/')  # YAML 내 경로는 항상 forward slash
```

- [ ] **Step 2: Add Windows Defender note to setup.py post-setup output**

Find the post-setup instructions section in setup.py (around line 220-228) and add after the Claude Desktop config path instructions:

```python
    if sys.platform == 'win32':
        print(f"\n  [팁] Windows Defender 실시간 보호가 인덱싱 속도를 크게 저하시킬 수 있습니다.")
        print(f"  성능 향상을 위해 아래 폴더를 제외 목록에 추가하세요:")
        print(f"    - {os.path.join(project_root, 'data')}")
        print(f"    - 설정: Windows 보안 → 바이러스 및 위협 방지 → 설정 관리 → 제외")
```

- [ ] **Step 3: Update CLAUDE.md — Windows 호환성 섹션 추가**

`CLAUDE.md`의 `## Conventions` 섹션 뒤에 추가:

```markdown
## Windows Compatibility

v0.9.2에서 Windows 호환성 개선 완료. 수정 시 아래 규칙 유지:

1. **경로**: `src/compat.py`의 `normalize_path()`, `safe_realpath()`, `path_is_under()` 사용. 하드코딩 `/` 금지
2. **subprocess**: `get_subprocess_kwargs()` 전개하여 `CREATE_NO_WINDOW` 자동 적용
3. **MPS 전용 코드**: `should_unload_model(device)` 가드 필수. 무조건 `gc.collect()` 금지 (CPU 오버헤드)
4. **외부 실행파일**: `find_executable(['libreoffice', 'soffice'])` 등 compat 함수 사용
5. **인코딩**: `fix_encoding_name()` 으로 chardet 결과 보정. 콘솔 출력 전 `setup_console_encoding()`
6. **asyncio**: MCP 서버 진입점에서 `setup_asyncio_policy()` 호출 (ProactorEventLoop 방지)
7. **파일 I/O**: `PermissionError` catch 필수 (Windows 파일 잠금). BM25 캐시는 원자적 쓰기
```

- [ ] **Step 4: Commit**

```bash
git add setup.py CLAUDE.md
git commit -m "docs: Windows compatibility notes, setup.py forward-slash paths"
```

---

### Task 9: Version Bump & Final Verification

**Files:**
- Modify: `pyproject.toml:7`

- [ ] **Step 1: Bump version to 0.9.2**

In `pyproject.toml` line 7:

```toml
version = "0.9.2"
```

- [ ] **Step 2: Full import verification**

Run:
```bash
python -c "
from src.compat import *
from src.config import load_config
from src.main import main
from src.mcp_server.server import DocDBServer
from src.search.context_builder import build_context_prefix
from src.search.reranker import Reranker
from src.search.bm25_index import BM25Index
from src.document_processor.processor import DocumentProcessor
from src.document_processor.extractors.hwp_extractor import HwpExtractor
from src.document_processor.extractors.ocr_helper import OCRHelper
from src.document_processor.extractors.text_extractors import TextExtractor
from src.document_processor.metadata_extractor import MetadataExtractor
from src.incremental.file_scanner import FileScanner
from src.incremental.index_tracker import IndexTracker
from src.indexing_pipeline import index_single_file, normalize_path
from src.embedding.local_embedder import LocalEmbedder
from src.embedding.embedding_manager import EmbeddingManager
from src.vectorstore.chroma_manager import ChromaManager
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit and tag**

```bash
git add pyproject.toml
git commit -m "chore: bump version to v0.9.2 — Windows compatibility"
git tag v0.9.2
```

---

## Summary of Changes by Issue

| Phase | Issue IDs | Task |
|-------|-----------|------|
| Phase 1 (Critical) | C1, C2, C3, NEW-C1, NEW-C2, NEW-C3 | Tasks 1-4 |
| Phase 2 (High) | H3, H4, H6, H7, NEW-H1, NEW-H2, NEW-H3, NEW-H4 | Tasks 2, 4-6 |
| Phase 3 (Medium) | M2, M4, M5, M6, NEW-M1, NEW-M3, NEW-M4, NEW-M5, NEW-M6, NEW-M7, NEW-M8, NEW-M9, NEW-M10 | Tasks 2-7 |
| Phase 4 (Polish) | L1, L4, NEW-L1, NEW-L2, NEW-L3, NEW-L4 | Tasks 6-8 |
