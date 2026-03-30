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


def find_executable(names: list):
    """
    여러 실행파일 이름 중 PATH에서 찾은 첫 번째 반환.
    Windows 공통 설치 경로도 탐색.
    """
    import shutil
    for name in names:
        found = shutil.which(name)
        if found:
            return found
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
    """chardet 인코딩 이름 보정 (EUC-KR -> cp949 등)"""
    mapping = {
        'euc-kr': 'cp949',
        'euc_kr': 'cp949',
        'ks_c_5601-1987': 'cp949',
    }
    return mapping.get(encoding.lower(), encoding) if encoding else 'utf-8'


def get_file_extension(filename: str) -> str:
    """안전한 파일 확장자 추출 (dotless 파일 대응)"""
    basename = os.path.basename(filename)
    _, ext = os.path.splitext(basename)
    if not ext and basename.startswith('.') and len(basename) > 1:
        # dotfile: .gitignore -> "gitignore"
        return basename[1:].lower()
    return ext.lstrip('.').lower() if ext else ''
