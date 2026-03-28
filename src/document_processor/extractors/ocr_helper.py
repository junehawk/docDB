"""
OCR 헬퍼 - 스캔 PDF에서 텍스트 추출.
Apple Vision Framework (macOS) 우선, Tesseract 폴백.
"""

import platform
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger


class OCRHelper:
    """
    OCR 엔진 자동 감지 및 실행.
    macOS에서는 Apple Vision Framework를 우선 사용하고,
    사용 불가 시 Tesseract OCR로 폴백합니다.
    """

    def __init__(self, ocr_config: Optional[Dict[str, Any]] = None):
        self.config = ocr_config or {}
        self.enabled = self.config.get('enabled', True)
        self.primary_engine = self.config.get('primary_engine', 'auto')
        self.fallback_engine = self.config.get('fallback_engine', 'tesseract')
        self.languages = self.config.get('languages', ['eng', 'kor'])
        self.dpi = self.config.get('dpi', 300)
        self.max_pages = self.config.get('max_pages_per_pdf', 50)
        self.timeout_per_page = self.config.get('timeout_per_page', 10)

        self._vision_available = None
        self._tesseract_available = None

    def is_available(self) -> bool:
        """OCR 엔진이 하나라도 사용 가능한지 확인"""
        if not self.enabled:
            return False
        return self._is_vision_available() or self._is_tesseract_available()

    def _is_vision_available(self) -> bool:
        """Apple Vision Framework 사용 가능 여부"""
        if self._vision_available is not None:
            return self._vision_available

        if platform.system() != 'Darwin':
            self._vision_available = False
            return False

        try:
            import Vision  # noqa: F401
            import Quartz  # noqa: F401
            self._vision_available = True
        except ImportError:
            self._vision_available = False
            logger.debug("Apple Vision Framework 미설치 (pyobjc-framework-Vision)")

        return self._vision_available

    def _is_tesseract_available(self) -> bool:
        """Tesseract OCR 사용 가능 여부"""
        if self._tesseract_available is not None:
            return self._tesseract_available

        try:
            result = subprocess.run(
                ['tesseract', '--version'],
                capture_output=True, timeout=5
            )
            self._tesseract_available = (result.returncode == 0)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._tesseract_available = False
            logger.debug("Tesseract OCR 미설치")

        return self._tesseract_available

    def ocr_pdf(self, pdf_path: str) -> Optional[str]:
        """
        PDF를 OCR 처리하여 텍스트를 반환합니다.
        메모리 효율을 위해 페이지별로 처리합니다 (~35MB/page vs ~1.3GB 전체).

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            추출된 텍스트, 실패 시 None
        """
        if not self.enabled:
            return None

        try:
            from pdf2image import convert_from_path
            from pdf2image.pdf2image import pdfinfo_from_path
        except ImportError:
            logger.warning("pdf2image 미설치 — OCR 불가 (pip install pdf2image)")
            return None

        # 엔진 선택
        engine = self._select_engine()
        if engine is None:
            logger.warning("사용 가능한 OCR 엔진이 없습니다")
            return None

        # 전체 페이지 수 확인
        try:
            info = pdfinfo_from_path(str(pdf_path))
            total_pages = min(info.get('Pages', 1), self.max_pages)
        except Exception:
            total_pages = self.max_pages

        logger.debug(f"OCR 시작: {Path(pdf_path).name} ({total_pages}페이지)")

        all_texts: List[str] = []
        for page_num in range(1, total_pages + 1):
            try:
                images = convert_from_path(
                    str(pdf_path), dpi=self.dpi,
                    first_page=page_num, last_page=page_num, fmt='png'
                )
                if not images:
                    continue
                image = images[0]
                del images

                page_text = self._ocr_single_page(image, page_num, engine)
                # Vision 실패 시 Tesseract 폴백
                if not page_text and engine == 'vision' and self.fallback_engine == 'tesseract' and self._is_tesseract_available():
                    page_text = self._ocr_single_page_tesseract(image, page_num)
                del image

                if page_text:
                    all_texts.append(page_text)
            except Exception as e:
                logger.debug(f"OCR 페이지 {page_num} 실패: {e}")
                continue

        if all_texts:
            return "\n--- Page Break ---\n".join(all_texts)
        return None

    def _select_engine(self) -> Optional[str]:
        """설정과 가용성에 따라 OCR 엔진 선택"""
        if self.primary_engine == 'vision':
            if self._is_vision_available():
                return 'vision'
            return None
        elif self.primary_engine == 'tesseract':
            if self._is_tesseract_available():
                return 'tesseract'
            return None
        else:  # 'auto'
            if self._is_vision_available():
                return 'vision'
            if self._is_tesseract_available():
                return 'tesseract'
            return None

    def _ocr_single_page(self, image: Any, page_num: int, engine: str) -> Optional[str]:
        """단일 페이지 이미지를 지정된 엔진으로 OCR"""
        if engine == 'vision':
            return self._ocr_single_page_vision(image, page_num)
        elif engine == 'tesseract':
            return self._ocr_single_page_tesseract(image, page_num)
        return None

    def _ocr_single_page_vision(self, image: Any, page_num: int) -> Optional[str]:
        """Apple Vision Framework로 단일 페이지 OCR"""
        try:
            import Vision
            import Quartz
            from Foundation import NSData
            import io
        except ImportError:
            return None

        try:
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            png_data = buf.getvalue()
            del buf
            ns_data = NSData.dataWithBytes_length_(png_data, len(png_data))
            del png_data

            image_source = Quartz.CGImageSourceCreateWithData(ns_data, None)
            del ns_data
            if image_source is None:
                logger.debug(f"페이지 {page_num}: CGImageSource 생성 실패")
                return None
            cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
            del image_source
            if cg_image is None:
                logger.debug(f"페이지 {page_num}: CGImage 생성 실패")
                return None

            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setRecognitionLanguages_(['ko-KR', 'en-US'])
            request.setUsesLanguageCorrection_(True)

            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
                cg_image, None
            )
            success = handler.performRequests_error_([request], None)
            if not success[0]:
                logger.debug(f"페이지 {page_num}: Vision 요청 실패")
                return None

            results = request.results()
            if results:
                page_texts = []
                for observation in results:
                    candidates = observation.topCandidates_(1)
                    if candidates:
                        page_texts.append(candidates[0].string())
                if page_texts:
                    return '\n'.join(page_texts)
        except Exception as e:
            logger.debug(f"페이지 {page_num} Vision OCR 실패: {e}")

        return None

    def _ocr_single_page_tesseract(self, image: Any, page_num: int) -> Optional[str]:
        """Tesseract OCR로 단일 페이지 텍스트 추출"""
        try:
            import pytesseract
        except ImportError:
            logger.warning("pytesseract 미설치 (pip install pytesseract)")
            return None

        lang_map = {'eng': 'eng', 'kor': 'kor', 'jpn': 'jpn', 'chi_sim': 'chi_sim'}
        tess_langs = '+'.join(lang_map.get(lang, lang) for lang in self.languages)

        try:
            page_text = pytesseract.image_to_string(
                image,
                lang=tess_langs,
                timeout=self.timeout_per_page,
            )
            if page_text and page_text.strip():
                return page_text.strip()
        except Exception as e:
            logger.debug(f"페이지 {page_num} Tesseract OCR 실패: {e}")

        return None
