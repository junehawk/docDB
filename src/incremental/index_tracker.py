"""
SQLite 기반 파일 인덱싱 상태 추적
"""
import sqlite3
import hashlib
import threading
from pathlib import Path
from typing import Dict, Tuple, Optional
from loguru import logger


class IndexTracker:
    """SQLite 기반 파일 인덱싱 상태 추적"""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS indexed_files (
        file_path TEXT PRIMARY KEY,
        file_mtime REAL NOT NULL,
        content_hash TEXT NOT NULL,
        last_indexed TEXT NOT NULL,
        is_indexed INTEGER NOT NULL,
        error_msg TEXT,
        embedding_model TEXT
    )
    """

    MIGRATION_CHECK = """
    SELECT COUNT(*) FROM pragma_table_info('indexed_files') WHERE name='is_sensitive'
    """

    def __init__(self, db_path: str = './data/index_tracker.db'):
        """
        IndexTracker 초기화

        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        try:
            self.db_path = db_path
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            self._lock = threading.Lock()
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

            self.conn.execute(self.SCHEMA)
            self.conn.commit()

            # 기존 DB에 is_sensitive 컬럼이 있으면 제거 (마이그레이션)
            self._migrate_remove_sensitive()

            logger.info(f"IndexTracker 초기화 완료: {db_path}")

        except Exception as e:
            logger.error(f"IndexTracker 초기화 실패: {e}")
            raise

    def _migrate_remove_sensitive(self):
        """기존 DB에서 is_sensitive 컬럼 제거 마이그레이션"""
        try:
            cursor = self.conn.execute(self.MIGRATION_CHECK)
            has_sensitive = cursor.fetchone()[0] > 0
            if has_sensitive:
                logger.info("is_sensitive 컬럼 마이그레이션 시작...")
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS indexed_files_new (
                        file_path TEXT PRIMARY KEY,
                        file_mtime REAL NOT NULL,
                        content_hash TEXT NOT NULL,
                        last_indexed TEXT NOT NULL,
                        is_indexed INTEGER NOT NULL,
                        error_msg TEXT,
                        embedding_model TEXT
                    )
                """)
                self.conn.execute("""
                    INSERT OR IGNORE INTO indexed_files_new
                    (file_path, file_mtime, content_hash, last_indexed, is_indexed, error_msg, embedding_model)
                    SELECT file_path, file_mtime, content_hash, last_indexed, is_indexed, error_msg, embedding_model
                    FROM indexed_files
                """)
                self.conn.execute("DROP TABLE indexed_files")
                self.conn.execute("ALTER TABLE indexed_files_new RENAME TO indexed_files")
                self.conn.commit()
                logger.info("is_sensitive 컬럼 마이그레이션 완료")
        except Exception as e:
            logger.warning(f"마이그레이션 스킵 (오류 무시): {e}")

    def record(
        self,
        file_path: str,
        mtime: float,
        content_hash: str,
        is_indexed: bool,
        embedding_model: Optional[str] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        파일 인덱싱 상태 기록 (INSERT OR REPLACE)

        Args:
            file_path: 파일 경로
            mtime: 파일 수정 시간 (unix timestamp)
            content_hash: 파일 경로+수정시간 기반 해시값 (SHA256)
            is_indexed: 인덱싱 완료 여부
            embedding_model: 사용된 임베딩 모델
            error: 에러 메시지 (실패시)

        Returns:
            성공 여부
        """
        try:
            with self._lock:
                cursor = self.conn.cursor()

                from datetime import datetime
                last_indexed = datetime.now().isoformat()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO indexed_files
                    (file_path, file_mtime, content_hash, last_indexed, is_indexed,
                     embedding_model, error_msg)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_path,
                        mtime,
                        content_hash,
                        last_indexed,
                        1 if is_indexed else 0,
                        embedding_model,
                        error
                    )
                )

                self.conn.commit()
            logger.debug(f"파일 기록됨: {file_path}")
            return True

        except Exception as e:
            logger.error(f"파일 기록 실패: {file_path}, {e}")
            return False

    def get_tracked(self) -> Dict[str, Tuple[float, str]]:
        """
        추적된 모든 파일 정보 조회

        Returns:
            {file_path: (mtime, content_hash)}
        """
        try:
            with self._lock:
                cursor = self.conn.cursor()
                cursor.execute("SELECT file_path, file_mtime, content_hash FROM indexed_files")

                tracked = {}
                for row in cursor:
                    tracked[row['file_path']] = (row['file_mtime'], row['content_hash'])

            logger.info(f"추적 중인 파일 수: {len(tracked)}")
            return tracked

        except Exception as e:
            logger.error(f"추적 정보 조회 실패: {e}")
            return {}

    def get_by_file_path(self, file_path: str) -> Optional[Dict]:
        """
        특정 파일의 추적 정보 조회

        Args:
            file_path: 파일 경로

        Returns:
            추적 정보 딕셔너리 또는 None
        """
        try:
            with self._lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT * FROM indexed_files WHERE file_path = ?",
                    (file_path,)
                )

                row = cursor.fetchone()
                result = dict(row) if row else None

            return result

        except Exception as e:
            logger.error(f"파일 정보 조회 실패: {file_path}, {e}")
            return None

    def delete(self, file_path: str) -> bool:
        """
        파일 추적 정보 삭제

        Args:
            file_path: 삭제할 파일 경로

        Returns:
            성공 여부
        """
        try:
            with self._lock:
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM indexed_files WHERE file_path = ?", (file_path,))
                self.conn.commit()

            logger.debug(f"파일 추적 정보 삭제됨: {file_path}")
            return True

        except Exception as e:
            logger.error(f"파일 삭제 실패: {file_path}, {e}")
            return False

    def get_stats(self) -> Dict:
        """
        인덱싱 통계 조회

        Returns:
            {
                'total': int,
                'indexed': int,
                'failed': int,
                'pending': int,
            }
        """
        try:
            with self._lock:
                cursor = self.conn.cursor()

                cursor.execute("SELECT COUNT(*) as count FROM indexed_files")
                total = cursor.fetchone()['count']

                cursor.execute("SELECT COUNT(*) as count FROM indexed_files WHERE is_indexed = 1")
                indexed = cursor.fetchone()['count']

                cursor.execute("SELECT COUNT(*) as count FROM indexed_files WHERE is_indexed = 0 AND error_msg IS NOT NULL")
                failed = cursor.fetchone()['count']

                stats = {
                    'total': total,
                    'indexed': indexed,
                    'failed': failed,
                    'pending': total - indexed - failed,
                }

            logger.info(f"인덱싱 통계: {stats}")
            return stats

        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {
                'total': 0,
                'indexed': 0,
                'failed': 0,
                'pending': 0,
            }

    def get_indexed_files(self) -> list:
        """
        인덱싱 완료된 모든 파일 정보 조회

        Returns:
            [{'file_path': str, 'file_mtime': float, 'last_indexed': str}, ...]
        """
        try:
            with self._lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT file_path, file_mtime, last_indexed "
                    "FROM indexed_files WHERE is_indexed = 1"
                )
                result = [dict(row) for row in cursor]
            return result
        except Exception as e:
            logger.error(f"인덱싱 파일 목록 조회 실패: {e}")
            return []

    def is_indexed(self, file_path: str, content_hash: str) -> bool:
        """파일이 이미 동일 해시로 인덱싱되었는지 확인"""
        info = self.get_by_file_path(file_path)
        if info and info['is_indexed'] and info['content_hash'] == content_hash:
            return True
        return False

    def mark_indexed(
        self, file_path: str, mtime: float, content_hash: str
    ) -> bool:
        """파일을 인덱싱 완료로 기록"""
        return self.record(
            file_path=file_path,
            mtime=mtime,
            content_hash=content_hash,
            is_indexed=True,
        )

    def record_error(self, file_path: str, error_msg: str) -> bool:
        """파일 인덱싱 오류 기록 — 기존 성공 기록이 있으면 보존하고 에러만 기록"""
        try:
            with self._lock:
                cursor = self.conn.execute(
                    "SELECT is_indexed FROM indexed_files WHERE file_path = ?",
                    (file_path,)
                )
                existing = cursor.fetchone()
                if existing and existing[0] == 1:
                    # 기존 성공 기록 보존, 에러 메시지만 업데이트
                    self.conn.execute(
                        "UPDATE indexed_files SET error_msg = ? WHERE file_path = ?",
                        (error_msg, file_path)
                    )
                    self.conn.commit()
                    return True

                # 기존 기록 없음 — 에러 기록 삽입 (lock 내에서 처리)
                from datetime import datetime
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO indexed_files
                    (file_path, file_mtime, content_hash, last_indexed, is_indexed,
                     embedding_model, error_msg)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (file_path, 0, '', datetime.now().isoformat(), 0, None, error_msg)
                )
                self.conn.commit()
                return True
        except Exception as e:
            logger.error(f"에러 기록 실패: {file_path}, {e}")
            return False

    def mark_deleted(self, file_path: str) -> bool:
        """파일을 삭제됨으로 처리"""
        return self.delete(file_path)

    def close(self):
        """데이터베이스 연결 종료"""
        try:
            with self._lock:
                self.conn.close()
            logger.info("IndexTracker 연결 종료")
        except Exception as e:
            logger.error(f"연결 종료 실패: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
