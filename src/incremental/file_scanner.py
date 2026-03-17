"""
파일 시스템 스캔 및 변경 감지
"""
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set
from loguru import logger


class FileScanner:
    """파일 시스템 스캔 및 변경 감지"""

    SUPPORTED = {
        'hwp', 'hwpx', 'pdf', 'docx', 'pptx', 'xlsx', 'xls',
        'csv', 'txt', 'md', 'html', 'htm', 'rtf', 'pages', 'numbers', 'key'
    }

    def __init__(
        self,
        doc_root: str,
        tracker,
        excluded_patterns: Optional[List[str]] = None,
        max_file_size_mb: int = 500
    ):
        """
        FileScanner 초기화

        Args:
            doc_root: 문서 루트 디렉토리 경로
            tracker: IndexTracker 인스턴스
            excluded_patterns: 제외할 파일 패턴 정규식 리스트
                예: [r'.*\\.temp$', r'.*__pycache__.*']
            max_file_size_mb: 최대 파일 크기 (MB)
        """
        try:
            self.doc_root = Path(os.path.expanduser(doc_root))
            self.tracker = tracker
            self.max_file_size = max_file_size_mb * 1024 * 1024

            # 기본 제외 패턴
            self.excluded_patterns = [
                r'^\..*',  # 숨김 파일
                r'^~\$.*',  # MS Office 잠금 파일
                r'.*~$',  # Backup 파일
                r'.*\.tmp$',  # 임시 파일
                r'.*__pycache__.*',  # Python cache
                r'.*\.git.*',  # Git 파일
                r'.*node_modules.*',  # Node modules
            ]

            if excluded_patterns:
                self.excluded_patterns.extend(excluded_patterns)

            self.excluded_regex = []
            for p in self.excluded_patterns:
                try:
                    self.excluded_regex.append(re.compile(p))
                except re.error as e:
                    logger.warning(f"잘못된 제외 패턴 무시: '{p}': {e}")

            logger.info(f"FileScanner 초기화: {doc_root}")

        except Exception as e:
            logger.error(f"FileScanner 초기화 실패: {e}")
            raise

    def scan_and_diff(self) -> Tuple[List[str], List[str], List[str]]:
        """
        디스크 상태와 트래커 DB 비교하여 변경사항 감지

        Returns:
            (new_files, changed_files, deleted_files)
            각각 파일 경로의 리스트
        """
        try:
            # 현재 디스크 상태 스캔
            current_disk_state = self._scan_disk()

            # 트래커 DB에서 추적 중인 파일
            tracked_files = self.tracker.get_tracked()

            # 변경사항 감지
            new_files = []
            changed_files = []

            # 현재 디스크 파일 검사
            for file_path, mtime in current_disk_state.items():
                if file_path not in tracked_files:
                    # 새로운 파일
                    new_files.append(file_path)
                else:
                    # 기존 파일 - mtime 비교
                    tracked_mtime, _ = tracked_files[file_path]
                    if mtime != tracked_mtime:
                        changed_files.append(file_path)

            # 삭제된 파일 감지
            deleted_files = []
            for file_path in tracked_files.keys():
                if file_path not in current_disk_state:
                    deleted_files.append(file_path)

            logger.info(
                f"변경사항 감지: 신규({len(new_files)}) + "
                f"변경({len(changed_files)}) + 삭제({len(deleted_files)})"
            )

            return new_files, changed_files, deleted_files

        except Exception as e:
            logger.error(f"변경사항 감지 실패: {e}")
            raise

    def _scan_disk(self) -> Dict[str, float]:
        """
        디렉토리 트리를 순회하여 지원되는 파일 목록 반환

        Returns:
            {file_path: mtime} 딕셔너리
        """
        try:
            files = {}

            for root, dirs, filenames in os.walk(self.doc_root):
                # 제외된 디렉토리는 스킵
                dirs[:] = [
                    d for d in dirs
                    if not any(pattern.search(d) for pattern in self.excluded_regex)
                ]

                for filename in filenames:
                    # 제외된 파일은 스킵
                    if any(pattern.search(filename) for pattern in self.excluded_regex):
                        continue

                    # 파일 확장자 확인
                    ext = filename.split('.')[-1].lower()
                    if ext not in self.SUPPORTED:
                        continue

                    file_path = unicodedata.normalize('NFC', os.path.join(root, filename))

                    # 파일 크기 확인
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size > self.max_file_size:
                            logger.warning(f"파일이 너무 큼: {file_path} ({file_size / 1024 / 1024:.1f}MB)")
                            continue
                    except OSError:
                        continue

                    # 수정 시간 가져오기
                    try:
                        mtime = os.path.getmtime(file_path)
                        files[file_path] = mtime
                    except OSError:
                        logger.warning(f"파일 수정 시간 읽을 수 없음: {file_path}")
                        continue

            logger.info(f"디스크 스캔 완료: {len(files)}개 파일 발견")
            return files

        except Exception as e:
            logger.error(f"디스크 스캔 실패: {e}")
            return {}

    def get_file_metadata(self, file_path: str) -> Optional[Dict]:
        """
        파일의 기본 메타데이터 추출

        Args:
            file_path: 파일 경로

        Returns:
            {
                'file_path': str,
                'file_name': str,
                'file_type': str,
                'mtime': float,
                'mtime_iso': str,
            }
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.warning(f"파일 없음: {file_path}")
                return None

            # 파일 수정 시간
            mtime = file_path.stat().st_mtime

            from datetime import datetime
            mtime_iso = datetime.fromtimestamp(mtime).isoformat()

            # 파일 확장자
            file_type = file_path.suffix.lower().lstrip('.')

            return {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_type': file_type,
                'mtime': mtime,
                'mtime_iso': mtime_iso,
            }

        except Exception as e:
            logger.error(f"메타데이터 추출 실패: {file_path}, {e}")
            return None
