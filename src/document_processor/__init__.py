"""
docDB Document Processor Module.
다양한 문서 형식에서 텍스트를 추출하는 파이프라인입니다.
"""

from .processor import DocumentProcessor
from .extractors.base_extractor import BaseExtractor, ExtractionResult

__version__ = "0.1.0"
__all__ = [
    "DocumentProcessor",
    "BaseExtractor",
    "ExtractionResult",
]
