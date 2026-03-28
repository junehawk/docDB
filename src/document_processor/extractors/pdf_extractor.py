"""
PDF 파일 추출기 - pdfplumber (주) + PyPDF2 (폴백) + OCR (3차 폴백).
"""

from loguru import logger

from .base_extractor import BaseExtractor, ExtractionResult


class PDFExtractor(BaseExtractor):
    """
    PDF 파일 추출기.
    pdfplumber를 주 방법으로 사용하고, 실패 시 PyPDF2, 그 다음 OCR을 폴백으로 사용합니다.
    """

    def extract(self) -> ExtractionResult:
        """
        PDF 파일에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        if not self.validate_file():
            return self._create_error_result("File validation failed")

        # pdfplumber 시도 (주 방법)
        try:
            result = self._try_pdfplumber()
            if result.success:
                return result
        except Exception as e:
            logger.debug(f"pdfplumber method failed: {e}")

        # PyPDF2 시도 (폴백)
        try:
            result = self._try_pypdf2()
            if result.success:
                return result
        except Exception as e:
            logger.debug(f"PyPDF2 method failed: {e}")

        # OCR 시도 (3차 폴백 - 스캔 이미지 PDF용)
        try:
            result = self._try_ocr()
            if result.success:
                return result
        except Exception as e:
            logger.debug(f"OCR method failed: {e}")

        logger.error(f"All PDF extraction methods failed for {self.file_path.name}")
        return self._create_error_result(
            "All extraction methods failed (pdfplumber, PyPDF2, OCR)"
        )

    def _extract_pdf_properties(self) -> dict:
        """
        PDF 문서 내장 properties를 추출합니다.
        실패 시 빈 dict를 반환합니다.

        Returns:
            dict: PDF 문서 properties
        """
        try:
            import pdfplumber
            with pdfplumber.open(str(self.file_path)) as pdf:
                info = pdf.metadata or {}
                return {
                    'title': info.get('Title', '') or '',
                    'author': info.get('Author', '') or '',
                    'subject': info.get('Subject', '') or '',
                    'keywords': info.get('Keywords', '') or '',
                    'creator': info.get('Creator', '') or '',
                    'creation_date': info.get('CreationDate', '') or '',
                    'modification_date': info.get('ModDate', '') or '',
                }
        except Exception:
            pass

        try:
            from PyPDF2 import PdfReader
            with open(str(self.file_path), "rb") as f:
                reader = PdfReader(f)
                info = reader.metadata or {}
                return {
                    'title': info.get('/Title', '') or '',
                    'author': info.get('/Author', '') or '',
                    'subject': info.get('/Subject', '') or '',
                    'keywords': info.get('/Keywords', '') or '',
                    'creator': info.get('/Creator', '') or '',
                    'creation_date': info.get('/CreationDate', '') or '',
                    'modification_date': info.get('/ModDate', '') or '',
                }
        except Exception:
            pass

        return {}

    def _try_pdfplumber(self) -> ExtractionResult:
        """
        pdfplumber를 사용한 PDF 추출.
        페이지별로 텍스트를 추출하고 페이지 구분선으로 연결합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        try:
            import pdfplumber
        except ImportError:
            logger.debug("pdfplumber not installed, skipping pdfplumber method")
            return self._create_error_result("pdfplumber not installed")

        try:
            text_parts = []
            page_count = 0

            with pdfplumber.open(str(self.file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = None
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                            page_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to extract page {page_num}: {e}")
                    finally:
                        if hasattr(page, 'flush_cache'):
                            page.flush_cache()
                        page_text = None

            if text_parts:
                # 페이지 구분선으로 연결합니다
                text = "\n--- Page Break ---\n".join(text_parts)
                doc_properties = self._extract_pdf_properties()
                return self._create_success_result(
                    text, {
                        "method": "pdfplumber",
                        "pages_extracted": page_count,
                        "total_pages": len(pdf.pages),
                        "doc_properties": doc_properties,
                    }
                )
            else:
                return self._create_error_result("No text extracted from PDF")

        except Exception as e:
            logger.debug(f"pdfplumber extraction failed: {e}")
            return self._create_error_result(f"pdfplumber error: {str(e)}")

    def _try_pypdf2(self) -> ExtractionResult:
        """
        PyPDF2를 사용한 PDF 추출.
        pdfplumber 실패 시 폴백 방법입니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            logger.debug("PyPDF2 not installed, skipping PyPDF2 method")
            return self._create_error_result("PyPDF2 not installed")

        try:
            text_parts = []
            page_count = 0

            with open(str(self.file_path), "rb") as f:
                pdf_reader = PdfReader(f)

                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                            page_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to extract page {page_num + 1}: {e}")
                        continue

            if text_parts:
                # 페이지 구분선으로 연결합니다
                text = "\n--- Page Break ---\n".join(text_parts)
                doc_properties = self._extract_pdf_properties()
                return self._create_success_result(
                    text, {
                        "method": "pypdf2",
                        "pages_extracted": page_count,
                        "total_pages": len(pdf_reader.pages),
                        "doc_properties": doc_properties,
                    }
                )
            else:
                return self._create_error_result("No text extracted from PDF")

        except Exception as e:
            logger.debug(f"PyPDF2 extraction failed: {e}")
            return self._create_error_result(f"PyPDF2 error: {str(e)}")

    def _try_ocr(self) -> ExtractionResult:
        """
        OCR을 사용한 PDF 추출.
        스캔 이미지 기반 PDF에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        if not self.ocr_config or not self.ocr_config.get('enabled', False):
            return self._create_error_result("OCR disabled")

        try:
            from .ocr_helper import OCRHelper
        except ImportError:
            return self._create_error_result("OCR helper not available")

        try:
            ocr = OCRHelper(self.ocr_config)
            if not ocr.is_available():
                return self._create_error_result("No OCR engine available")

            text = ocr.ocr_pdf(str(self.file_path))
            if text and text.strip():
                engine = ocr._select_engine() or 'unknown'
                doc_properties = self._extract_pdf_properties()
                return self._create_success_result(
                    text, {"method": f"ocr_{engine}", "doc_properties": doc_properties}
                )
            else:
                return self._create_error_result("OCR produced no text")

        except Exception as e:
            logger.debug(f"OCR extraction failed: {e}")
            return self._create_error_result(f"OCR error: {str(e)}")
