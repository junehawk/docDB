"""
Abstract base class for document extractors.
모든 문서 추출기의 기본 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger


@dataclass
class ExtractionResult:
    """결과 데이터 클래스 - 추출 결과를 저장합니다."""
    text: str
    success: bool
    extractor_used: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """메타데이터 초기화를 확인합니다."""
        if self.metadata is None:
            self.metadata = {}


class BaseExtractor(ABC):
    """
    모든 추출기의 기본 추상 클래스입니다.
    각 파일 형식별 추출기는 이 클래스를 상속받아 구현합니다.
    """

    def __init__(self, file_path: str, timeout: int = 30, ocr_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            file_path: 추출할 파일의 경로
            timeout: 추출 작업의 타임아웃 (초)
            ocr_config: OCR 설정 딕셔너리 (선택사항)
        """
        self.file_path = Path(file_path)
        self.timeout = timeout
        self.ocr_config = ocr_config
        self.extractor_name = self.__class__.__name__

    @abstractmethod
    def extract(self) -> ExtractionResult:
        """
        파일에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        pass

    def validate_file(self) -> bool:
        """
        파일이 존재하고 읽을 수 있는지 검증합니다.

        Returns:
            bool: 파일이 유효하면 True
        """
        if not self.file_path.exists():
            logger.error(f"File does not exist: {self.file_path}")
            return False

        if not self.file_path.is_file():
            logger.error(f"Path is not a file: {self.file_path}")
            return False

        if not self.file_path.stat().st_size > 0:
            logger.warning(f"File is empty: {self.file_path}")
            return False

        try:
            with open(self.file_path, "rb") as f:
                f.read(1)
        except (IOError, OSError) as e:
            logger.error(f"Cannot read file {self.file_path}: {e}")
            return False

        return True

    def _create_error_result(self, error_message: str) -> ExtractionResult:
        """
        에러 결과를 생성합니다.

        Args:
            error_message: 에러 메시지

        Returns:
            ExtractionResult: 실패 상태의 결과 객체
        """
        return ExtractionResult(
            text="",
            success=False,
            extractor_used=self.extractor_name,
            error=error_message,
            metadata={"file_path": str(self.file_path)}
        )

    def _create_success_result(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        성공 결과를 생성합니다.

        Args:
            text: 추출된 텍스트
            metadata: 추가 메타데이터

        Returns:
            ExtractionResult: 성공 상태의 결과 객체
        """
        if metadata is None:
            metadata = {}

        metadata["file_path"] = str(self.file_path)
        metadata["text_length"] = len(text)

        return ExtractionResult(
            text=text,
            success=True,
            extractor_used=self.extractor_name,
            metadata=metadata
        )
