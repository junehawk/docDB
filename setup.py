#!/usr/bin/env python3
"""
docDB 초기 설정 스크립트
디바이스 자동 감지, 문서 폴더 경로 입력 → config.yaml 생성
"""
import os
import sys
import shutil


def detect_device() -> str:
    """사용 가능한 최적 디바이스 감지"""
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    return 'cpu'


def get_doc_root() -> str:
    """사용자에게 문서 폴더 경로 입력받기"""
    default = os.path.expanduser('~/Documents')

    print(f"\n인덱싱할 문서 폴더 경로를 입력하세요.")
    print(f"  기본값: {default}")
    user_input = input(f"  경로 (Enter=기본값): ").strip()

    path = user_input if user_input else default
    path = os.path.expanduser(path)

    if not os.path.isdir(path):
        print(f"\n  [!] '{path}' 디렉토리가 존재하지 않습니다.")
        create = input(f"  생성할까요? (y/N): ").strip().lower()
        if create == 'y':
            os.makedirs(path, exist_ok=True)
            print(f"  -> 생성 완료: {path}")
        else:
            print("  -> 설정을 중단합니다. 유효한 경로로 다시 시도하세요.")
            sys.exit(1)

    return path


def count_supported_files(doc_root: str) -> int:
    """지원 포맷 파일 수 카운트"""
    extensions = {
        '.hwp', '.hwpx', '.pdf', '.docx', '.pptx', '.xlsx', '.xls',
        '.csv', '.txt', '.html', '.rtf', '.pages', '.numbers', '.key'
    }
    count = 0
    for root, dirs, files in os.walk(doc_root):
        # 숨김 폴더 스킵
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            if os.path.splitext(f)[1].lower() in extensions:
                count += 1
    return count


def generate_config(doc_root: str, device: str) -> str:
    """config.yaml 내용 생성"""
    # ~ 표기 가능하면 사용 (가독성)
    home = os.path.expanduser('~')
    display_root = doc_root.replace(home, '~') if doc_root.startswith(home) else doc_root

    return f"""# docDB Configuration
# ================================

document_processing:
  doc_root: '{display_root}'    # 인덱싱할 문서 폴더
  chunk_size: 800
  chunk_overlap: 100
  max_file_size_mb: 100

  supported_extensions:
    - hwp
    - hwpx
    - pdf
    - docx
    - pptx
    - xlsx
    - xls
    - csv
    - txt
    - html
    - rtf
    - pages
    - numbers
    - key

embedding:
  model: 'BAAI/bge-m3'
  device: '{device}'
  batch_size: 32

vectorstore:
  chroma_path: 'data/chroma_db'

indexing:
  tracker_db: 'data/index_tracker.db'

search:
  mode: 'hybrid'
  vector_top_n: 50
  bm25_top_n: 50
  rrf_k: 60
  rerank_top_n: 20
  final_top_n: 10

reranker:
  enable: true
  model: 'BAAI/bge-reranker-v2-m3'
  device: '{device}'
  batch_size: 16

excluded_patterns:
  - '\\.DS_Store'
  - '__pycache__'
  - '\\.git'
  - '~\\$'

ocr:
  enabled: true
  primary_engine: 'auto'
  fallback_engine: 'tesseract'
  languages: ['eng', 'kor']
  dpi: 300
  max_pages_per_pdf: 50
  timeout_per_page: 10

contextual:
  enable: true

logging:
  level: 'INFO'
  log_dir: 'data/logs'
"""


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, 'config', 'config.yaml')

    print("=" * 50)
    print("  docDB 초기 설정")
    print("=" * 50)

    # 1. 디바이스 감지
    print("\n[1/3] 디바이스 감지 중...")
    device = detect_device()
    device_names = {'mps': 'Apple Silicon (MPS)', 'cuda': 'NVIDIA GPU (CUDA)', 'cpu': 'CPU'}
    print(f"  -> {device_names.get(device, device)}")

    # 2. 문서 폴더 경로
    print("\n[2/3] 문서 폴더 설정")
    doc_root = get_doc_root()

    file_count = count_supported_files(doc_root)
    print(f"  -> 지원 포맷 파일 {file_count}개 발견")

    if file_count == 0:
        print("  [!] 지원되는 문서 파일이 없습니다. 경로를 확인하세요.")
        proceed = input("  그래도 계속할까요? (y/N): ").strip().lower()
        if proceed != 'y':
            sys.exit(1)

    # 3. config.yaml 생성
    print(f"\n[3/3] 설정 파일 생성")

    if os.path.exists(config_path):
        backup_path = config_path + '.bak'
        shutil.copy2(config_path, backup_path)
        print(f"  -> 기존 설정 백업: {backup_path}")

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config_content = generate_config(doc_root, device)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"  -> 저장 완료: {config_path}")

    # 4. data 디렉토리 생성
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 완료 안내
    print("\n" + "=" * 50)
    print("  설정 완료!")
    print("=" * 50)
    print(f"""
다음 단계:

  # 1. 전체 인덱싱 (최초 1회, 문서 수에 따라 수분~수십분)
  python -m src.main --mode full_index

  # 2. Claude Desktop에 MCP 서버 연결
  #    ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
  #    %APPDATA%\\Claude\\claude_desktop_config.json (Windows)

  # 3. 이후 변경분만 증분 인덱싱
  python -m src.main --mode incremental_index
""")


if __name__ == '__main__':
    main()
