"""
단순 텍스트 형식 추출기들.
TXT, HTML, CSV, RTF 파일 추출을 담당합니다.
"""

import csv
import re
from io import StringIO
from loguru import logger

from .base_extractor import BaseExtractor, ExtractionResult


class TxtExtractor(BaseExtractor):
    """
    TXT 파일 추출기.
    chardet을 사용하여 자동 인코딩 감지합니다.
    """

    def extract(self) -> ExtractionResult:
        """
        TXT 파일에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        if not self.validate_file():
            return self._create_error_result("File validation failed")

        try:
            # 인코딩 감지
            encoding = self._detect_encoding()

            with open(self.file_path, "r", encoding=encoding) as f:
                text = f.read()

            if text.strip():
                return self._create_success_result(
                    text, {"method": "direct_read", "encoding": encoding, "doc_properties": {}}
                )
            else:
                return self._create_error_result("File is empty")

        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            return self._create_error_result(f"TXT error: {str(e)}")

    def _detect_encoding(self) -> str:
        """
        파일의 인코딩을 감지합니다.

        Returns:
            str: 감지된 인코딩 이름
        """
        try:
            import chardet
        except ImportError:
            logger.warning("chardet not installed, using utf-8 as default")
            return "utf-8"

        try:
            with open(self.file_path, "rb") as f:
                raw_sample = f.read(65536)  # 인코딩 감지에 64KB 샘플 충분

            detected = chardet.detect(raw_sample)
            encoding = detected.get("encoding", "utf-8")

            if encoding is None:
                encoding = "utf-8"

            from src.compat import fix_encoding_name
            encoding = fix_encoding_name(encoding)

            logger.debug(f"Detected encoding for {self.file_path.name}: {encoding}")
            return encoding

        except Exception as e:
            logger.debug(f"Encoding detection failed: {e}, using utf-8")
            return "utf-8"


class HtmlExtractor(BaseExtractor):
    """
    HTML 파일 추출기.
    BeautifulSoup을 사용하여 보이는 텍스트만 추출합니다.
    """

    def extract(self) -> ExtractionResult:
        """
        HTML 파일에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        if not self.validate_file():
            return self._create_error_result("File validation failed")

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 not installed")
            return self._create_error_result("beautifulsoup4 not installed")

        try:
            # 인코딩 감지 후 읽기
            encoding = "utf-8"
            try:
                import chardet
                with open(self.file_path, "rb") as bf:
                    raw_sample = bf.read(65536)
                detected = chardet.detect(raw_sample)
                if detected.get("encoding"):
                    encoding = detected["encoding"]
            except ImportError:
                pass

            with open(self.file_path, "r", encoding=encoding, errors="ignore") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # 스크립트와 스타일 태그 제거
            for script in soup(["script", "style"]):
                script.decompose()

            # 보이는 텍스트 추출
            text = soup.get_text()

            # 여러 공백 정리
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            if text.strip():
                return self._create_success_result(
                    text, {"method": "beautifulsoup", "doc_properties": {}}
                )
            else:
                return self._create_error_result("No text found in HTML")

        except Exception as e:
            logger.error(f"HTML extraction failed: {e}")
            return self._create_error_result(f"HTML error: {str(e)}")


class CsvExtractor(BaseExtractor):
    """
    CSV 파일 추출기.
    CSV를 읽고 "col1: val1, col2: val2" 형식으로 포맷합니다.
    """

    MAX_ROWS = 10000  # 최대 행 수 제한

    def extract(self) -> ExtractionResult:
        """
        CSV 파일에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        if not self.validate_file():
            return self._create_error_result("File validation failed")

        try:
            text_parts = []

            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                # 자동 방언 감지
                sample = f.read(8192)
                f.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(sample)
                except csv.Error:
                    dialect = csv.excel

                reader = csv.DictReader(f, dialect=dialect)

                if reader.fieldnames is None:
                    return self._create_error_result("No headers found in CSV")

                row_count = 0
                truncated = False
                for row in reader:
                    if row_count >= self.MAX_ROWS:
                        truncated = True
                        break
                    row_text = ", ".join(
                        f"{col}: {value}"
                        for col, value in row.items()
                        if value is not None and value.strip()
                    )
                    if row_text:
                        text_parts.append(row_text)
                    row_count += 1

                if truncated:
                    text_parts.append(f"\n[... CSV 행 {self.MAX_ROWS}개 제한으로 잘림]")

            if text_parts:
                text = "\n".join(text_parts)
                return self._create_success_result(
                    text, {
                        "method": "csv_dictreader",
                        "rows": row_count,
                        "columns": len(reader.fieldnames) if reader.fieldnames else 0,
                        "doc_properties": {},
                    }
                )
            else:
                return self._create_error_result("No data found in CSV")

        except Exception as e:
            logger.error(f"CSV extraction failed: {e}")
            return self._create_error_result(f"CSV error: {str(e)}")


class RtfExtractor(BaseExtractor):
    """
    RTF (Rich Text Format) 파일 추출기.
    RTF 제어 코드를 제거하여 평문 텍스트를 추출합니다.
    """

    def extract(self) -> ExtractionResult:
        """
        RTF 파일에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        if not self.validate_file():
            return self._create_error_result("File validation failed")

        try:
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                rtf_content = f.read()

            # RTF 제어 코드 제거
            text = self._strip_rtf_codes(rtf_content)

            if text.strip():
                return self._create_success_result(
                    text, {"method": "rtf_strip", "doc_properties": {}}
                )
            else:
                return self._create_error_result("No text found in RTF")

        except Exception as e:
            logger.error(f"RTF extraction failed: {e}")
            return self._create_error_result(f"RTF error: {str(e)}")

    @staticmethod
    def _strip_rtf_codes(rtf_text: str) -> str:
        """
        RTF 제어 코드를 제거합니다.

        Args:
            rtf_text: RTF 텍스트

        Returns:
            str: 정리된 평문
        """
        # 한국어 유니코드 이스케이프 (\uNNNNN) → 실제 문자로 변환
        text = re.sub(
            r'\\u(-?\d+)[?]?',
            lambda m: chr(int(m.group(1)) % 65536),
            rtf_text
        )

        # RTF 제어 단어 제거 (\command)
        text = re.sub(r'\\[a-z]+\d*\s?', '', text)

        # RTF 제어 기호 제거 (\*)
        text = re.sub(r'\\[^a-z0-9\s]', '', text)

        # 중괄호 제거
        text = text.replace('{', '').replace('}', '')

        # 여러 공백을 단일 공백으로 변환
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
