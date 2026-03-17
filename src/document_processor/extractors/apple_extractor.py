"""
Apple 형식 추출기들 (Pages, Numbers, Keynote).
Apple 앱의 파일들은 ZIP 컨테이너로 되어있습니다.
"""

import zipfile
try:
    import defusedxml.ElementTree as ET
except ImportError:
    import warnings
    warnings.warn(
        "defusedxml 미설치 — XML 파싱에 stdlib 사용 (XXE 취약). 설치: pip install defusedxml",
        stacklevel=2,
    )
    from xml.etree import ElementTree as ET
from loguru import logger

from .base_extractor import BaseExtractor, ExtractionResult


class AppleExtractor(BaseExtractor):
    """
    Apple 형식 (Pages, Numbers, Keynote) 추출기.
    이들은 모두 ZIP 컨테이너이며 내부에 XML 파일들을 포함합니다.
    """

    def extract(self) -> ExtractionResult:
        """
        Apple 파일에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        if not self.validate_file():
            return self._create_error_result("File validation failed")

        # ZIP 유효성 확인
        if not zipfile.is_zipfile(str(self.file_path)):
            return self._create_error_result("File is not a valid ZIP archive")

        # XML 파싱 시도
        try:
            result = self._try_parse_xml()
            if result.success:
                return result
        except Exception as e:
            logger.debug(f"XML parsing failed: {e}")

        # 폴백: 파일명 나열
        try:
            result = self._fallback_list_files()
            if result.success:
                return result
        except Exception as e:
            logger.debug(f"Fallback file listing failed: {e}")

        logger.error(f"All extraction methods failed for {self.file_path.name}")
        return self._create_error_result("Could not extract text from Apple format")

    def _try_parse_xml(self) -> ExtractionResult:
        """
        ZIP 내부의 XML 파일을 파싱하여 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        try:
            with zipfile.ZipFile(str(self.file_path)) as zf:
                text_parts = []
                xml_files_found = 0

                # 가능한 XML 파일 패턴들을 탐색합니다
                for file_name in zf.namelist():
                    if file_name.endswith(".xml"):
                        try:
                            xml_content = zf.read(file_name)
                            root = ET.fromstring(xml_content)

                            # XML에서 텍스트 노드 추출
                            text = self._extract_text_from_element(root)
                            if text:
                                text_parts.append(text)
                                xml_files_found += 1

                        except ET.ParseError as e:
                            logger.debug(f"Failed to parse XML {file_name}: {e}")
                            continue
                        except Exception as e:
                            logger.debug(f"Error processing {file_name}: {e}")
                            continue

                if text_parts:
                    text = " ".join(text_parts)
                    return self._create_success_result(
                        text, {
                            "method": "xml_parsing",
                            "xml_files_parsed": xml_files_found,
                            "doc_properties": {},
                        }
                    )
                else:
                    return self._create_error_result("No text found in XML files")

        except Exception as e:
            logger.debug(f"XML parsing failed: {e}")
            return self._create_error_result(f"XML parsing error: {str(e)}")

    def _fallback_list_files(self) -> ExtractionResult:
        """
        폴백: ZIP 내부 파일명들을 나열하여 구조 정보를 제공합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        try:
            with zipfile.ZipFile(str(self.file_path)) as zf:
                file_list = zf.namelist()

                # 파일 목록을 텍스트로 변환
                text_parts = []
                text_parts.append(f"Apple file structure ({len(file_list)} files):")
                text_parts.extend(file_list[:100])  # 최대 100개 파일명

                if len(file_list) > 100:
                    text_parts.append(f"... and {len(file_list) - 100} more files")

                text = "\n".join(text_parts)

                return self._create_success_result(
                    text, {
                        "method": "file_listing",
                        "total_files": len(file_list),
                        "doc_properties": {},
                    }
                )

        except Exception as e:
            logger.debug(f"File listing failed: {e}")
            return self._create_error_result(f"File listing error: {str(e)}")

    @staticmethod
    def _extract_text_from_element(element, depth=0, max_depth=100) -> str:
        """
        XML 요소에서 텍스트를 재귀적으로 추출합니다.

        Args:
            element: XML 요소 (ElementTree Element)
            depth: 현재 재귀 깊이
            max_depth: 최대 재귀 깊이

        Returns:
            str: 추출된 텍스트
        """
        if depth > max_depth:
            return ""

        text_parts = []

        # 요소의 텍스트 추출
        if element.text:
            text = element.text.strip()
            if text:
                text_parts.append(text)

        # 자식 요소 재귀 처리
        for child in element:
            child_text = AppleExtractor._extract_text_from_element(child, depth + 1, max_depth)
            if child_text:
                text_parts.append(child_text)

            # 자식 요소 뒤의 텍스트
            if child.tail:
                text = child.tail.strip()
                if text:
                    text_parts.append(text)

        return " ".join(text_parts)


class PagesExtractor(AppleExtractor):
    """
    Apple Pages 파일 추출기.
    AppleExtractor의 Pages 특화 버전입니다.
    """

    def extract(self) -> ExtractionResult:
        """
        Pages 파일에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        result = super().extract()
        # 메타데이터에 파일 형식 추가
        if result.metadata:
            result.metadata["format"] = "pages"
        return result


class NumbersExtractor(AppleExtractor):
    """
    Apple Numbers 파일 추출기.
    AppleExtractor의 Numbers 특화 버전입니다.
    """

    def extract(self) -> ExtractionResult:
        """
        Numbers 파일에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        result = super().extract()
        # 메타데이터에 파일 형식 추가
        if result.metadata:
            result.metadata["format"] = "numbers"
        return result


class KeynoteExtractor(AppleExtractor):
    """
    Apple Keynote 파일 추출기.
    AppleExtractor의 Keynote 특화 버전입니다.
    """

    def extract(self) -> ExtractionResult:
        """
        Keynote 파일에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        result = super().extract()
        # 메타데이터에 파일 형식 추가
        if result.metadata:
            result.metadata["format"] = "keynote"
        return result
