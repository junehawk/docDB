"""
문서 처리 오케스트레이터.
DocumentProcessor는 다양한 파일 형식에 대한 추출 파이프라인을 조정합니다.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from .extractors.base_extractor import BaseExtractor, ExtractionResult
from .extractors.hwp_extractor import HWPExtractor
from .extractors.pdf_extractor import PDFExtractor
from .extractors.office_extractors import DocxExtractor, PptxExtractor, XlsExtractor, XlsxExtractor
from .extractors.text_extractors import TxtExtractor, HtmlExtractor, CsvExtractor, RtfExtractor
from .extractors.apple_extractor import PagesExtractor, NumbersExtractor, KeynoteExtractor
from .chunking.korean_chunker import KoreanChunker


class DocumentProcessor:
    """
    문서 처리 오케스트레이터.
    파일 형식을 인식하여 적절한 추출기를 선택하고 실행합니다.
    """

    # 지원하는 확장자와 해당 추출기 매핑
    EXTRACTOR_MAPPING = {
        # HWP/HWPX (한글)
        ".hwp": HWPExtractor,
        ".hwpx": HWPExtractor,

        # PDF
        ".pdf": PDFExtractor,

        # Microsoft Office
        ".docx": DocxExtractor,
        ".pptx": PptxExtractor,
        ".xls": XlsExtractor,
        ".xlsx": XlsxExtractor,

        # 텍스트 형식
        ".txt": TxtExtractor,
        ".md": TxtExtractor,
        ".html": HtmlExtractor,
        ".htm": HtmlExtractor,
        ".csv": CsvExtractor,
        ".rtf": RtfExtractor,

        # Apple 형식
        ".pages": PagesExtractor,
        ".numbers": NumbersExtractor,
        ".key": KeynoteExtractor,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        DocumentProcessor를 초기화합니다.

        Args:
            config: 설정 딕셔너리 (선택사항)
                - timeout: 추출 타임아웃 (기본값: 30초)
                - chunk_size: 청크 크기 (기본값: 1000)
                - chunk_overlap: 청크 오버랩 (기본값: 100)
        """
        self.config = config or {}
        self.timeout = self.config.get("timeout", 30)
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.ocr_config = self.config.get("ocr", {})
        self._chunker = KoreanChunker(chunk_size=self.chunk_size, overlap=self.chunk_overlap)

        logger.info(f"DocumentProcessor initialized with config: {self.config}")

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        문서를 처리하고 청크 목록을 반환합니다.

        Args:
            file_path: 처리할 파일의 경로

        Returns:
            List[Dict[str, Any]]: 청크 딕셔너리 리스트
                각 청크는 다음을 포함:
                - text: 청크 텍스트
                - chunk_id: 청크 ID
                - chunk_index: 청크 인덱스
                - metadata: 메타데이터 (파일경로, 추출기, 등)
        """
        file_path = Path(file_path)

        try:
            logger.info(f"Processing document: {file_path}")

            # 파일 존재 확인
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return []

            # 추출기 획득
            extractor = self._get_extractor(file_path)
            if extractor is None:
                logger.error(f"No extractor found for {file_path.suffix}")
                return []

            # 텍스트 추출
            extraction_result = extractor.extract()

            if not extraction_result.success:
                logger.warning(
                    f"Extraction failed for {file_path.name}: "
                    f"{extraction_result.error}"
                )
                return []

            logger.info(
                f"Successfully extracted {len(extraction_result.text)} chars "
                f"from {file_path.name}"
            )

            # 텍스트를 청크로 분할
            chunks = self._chunker.chunk(extraction_result.text)

            # 청크 리스트 생성
            import hashlib as _hashlib
            path_hash = _hashlib.sha256(str(file_path).encode()).hexdigest()[:12]
            file_stat = file_path.stat()

            chunk_dicts = []
            for chunk_index, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_size": file_stat.st_size,
                    "extractor_used": extraction_result.extractor_used,
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                }

                # 원본 메타데이터 병합
                if extraction_result.metadata:
                    chunk_metadata.update(extraction_result.metadata)

                chunk_dict = {
                    "text": chunk_text,
                    "chunk_id": f"{file_path.stem}_{path_hash}_{chunk_index}",
                    "chunk_index": chunk_index,
                    "metadata": chunk_metadata,
                }

                chunk_dicts.append(chunk_dict)

            logger.info(
                f"Created {len(chunk_dicts)} chunks from {file_path.name}"
            )
            return chunk_dicts

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}", exc_info=True)
            return []

    def _get_extractor(self, file_path: Path) -> Optional[BaseExtractor]:
        """
        파일 확장자를 기반으로 추출기를 반환합니다.

        Args:
            file_path: 파일 경로

        Returns:
            BaseExtractor: 추출기 인스턴스, 또는 None
        """
        extension = file_path.suffix.lower()

        extractor_class = self.EXTRACTOR_MAPPING.get(extension)

        if extractor_class is None:
            logger.warning(f"No extractor for extension {extension}")
            return None

        try:
            extractor = extractor_class(
                str(file_path),
                timeout=self.timeout,
                ocr_config=self.ocr_config if self.ocr_config else None,
            )
            logger.debug(f"Created {extractor_class.__name__} for {file_path.name}")
            return extractor
        except Exception as e:
            logger.error(f"Failed to create extractor for {file_path}: {e}")
            return None

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """
        지원하는 파일 확장자 목록을 반환합니다.

        Returns:
            List[str]: 지원하는 확장자 리스트
        """
        return sorted(DocumentProcessor.EXTRACTOR_MAPPING.keys())

    @staticmethod
    def is_supported(file_path: str) -> bool:
        """
        파일이 지원되는 형식인지 확인합니다.

        Args:
            file_path: 파일 경로

        Returns:
            bool: 지원되면 True
        """
        extension = Path(file_path).suffix.lower()
        return extension in DocumentProcessor.EXTRACTOR_MAPPING
