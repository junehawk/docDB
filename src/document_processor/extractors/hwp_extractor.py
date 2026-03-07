"""
HWP/HWPX 파일 추출기 - 5단계 폴백 체인을 사용합니다.
한글 파일은 복잡한 형식이므로 여러 추출 방법을 시도합니다.
"""

import os
import subprocess
import zlib
import re
from pathlib import Path
from typing import Optional
from loguru import logger

from .base_extractor import BaseExtractor, ExtractionResult


class HWPExtractor(BaseExtractor):
    """
    HWP/HWPX 파일 추출기 - 5단계 폴백 체인.
    OleFile -> Zipfile XML -> LibreOffice -> Binary Scan 순서로 시도합니다.
    """

    MIN_TEXT_LENGTH = 50  # 최소 텍스트 길이 기준

    def extract(self) -> ExtractionResult:
        """
        5단계 폴백 체인을 사용하여 HWP/HWPX 파일에서 텍스트를 추출합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        if not self.validate_file():
            return self._create_error_result("File validation failed")

        methods = [
            ("OleFile", self._try_olefile),
            ("ZipFile XML", self._try_zipfile_xml),
            ("LibreOffice", self._try_libreoffice),
            ("Binary Scan", self._try_binary_scan),
        ]

        for method_name, method_func in methods:
            try:
                logger.debug(f"Trying {method_name} method for {self.file_path.name}")
                result = method_func()

                if result.success and len(result.text.strip()) > self.MIN_TEXT_LENGTH:
                    logger.info(
                        f"Successfully extracted {len(result.text)} chars "
                        f"using {method_name} from {self.file_path.name}"
                    )
                    return result
                elif result.success:
                    logger.debug(
                        f"{method_name} extracted text but length "
                        f"({len(result.text)}) < {self.MIN_TEXT_LENGTH}"
                    )
            except Exception as e:
                logger.debug(f"{method_name} method failed: {e}")
                continue

        logger.error(f"All extraction methods failed for {self.file_path.name}")
        return self._create_error_result(
            "All extraction methods failed (OleFile, Zipfile, LibreOffice, Binary)"
        )

    def _extract_hwp_properties(self) -> dict:
        """
        HWP OleFile의 SummaryInformation에서 문서 내장 properties를 추출합니다.
        실패 시 빈 dict를 반환합니다.

        Returns:
            dict: HWP 문서 properties
        """
        try:
            import olefile
            if olefile.isOleFile(str(self.file_path)):
                ole = olefile.OleFileIO(str(self.file_path))
                try:
                    meta = ole.get_metadata()
                    return {
                        'title': (meta.title or b'').decode('utf-8', errors='ignore'),
                        'author': (meta.author or b'').decode('utf-8', errors='ignore'),
                        'subject': (meta.subject or b'').decode('utf-8', errors='ignore'),
                        'keywords': (meta.keywords or b'').decode('utf-8', errors='ignore'),
                        'created': str(meta.create_time) if meta.create_time else '',
                        'modified': str(meta.last_saved_time) if meta.last_saved_time else '',
                    }
                finally:
                    ole.close()
        except Exception:
            pass
        return {}

    def _try_olefile(self) -> ExtractionResult:
        """
        OleFile을 사용한 HWP 추출.
        HWP는 OLE 복합 문서 형식이며, BodyText 스트림은 zlib 압축 + UTF-16LE 인코딩.

        우선순위:
        1. BodyText/Section 스트림 (전체 본문, zlib 압축 해제 + 레코드 파싱)
        2. PrvText 스트림 (미리보기 텍스트, ~1024바이트 제한 — 폴백 전용)

        Returns:
            ExtractionResult: 추출 결과
        """
        try:
            import olefile
        except ImportError:
            logger.debug("olefile not installed, skipping OleFile method")
            return self._create_error_result("olefile not installed")

        try:
            if not olefile.isOleFile(str(self.file_path)):
                return self._create_error_result("File is not a valid OleFile")

            with olefile.OleFileIO(str(self.file_path)) as ole:

                # FileHeader에서 압축 여부 확인
                is_compressed = True  # 기본값: 압축됨
                try:
                    if ole.exists("FileHeader"):
                        header_data = ole.openstream("FileHeader").read()
                        if len(header_data) >= 40:
                            # 오프셋 36에 문서 속성 플래그 (4바이트, little-endian)
                            flags = int.from_bytes(header_data[36:40], byteorder="little")
                            is_compressed = bool(flags & 0x01)
                            logger.debug(
                                f"HWP FileHeader flags=0x{flags:08X}, "
                                f"compressed={is_compressed}"
                            )
                except Exception as e:
                    logger.debug(f"Failed to read FileHeader: {e}")

                # 1단계: BodyText/Section 스트림에서 전체 본문 추출
                text_parts = []

                # Section 스트림을 번호 순서대로 정렬하여 처리
                section_entries = []
                for entry in ole.listdir():
                    entry_name = "/".join(entry)
                    if entry_name.startswith("BodyText/Section"):
                        section_entries.append(entry)

                # Section 번호 기준 정렬 (Section0, Section1, ...)
                section_entries.sort(
                    key=lambda e: int(
                        re.search(r"Section(\d+)", "/".join(e)).group(1)
                    )
                    if re.search(r"Section(\d+)", "/".join(e))
                    else 0
                )

                for entry in section_entries:
                    entry_name = "/".join(entry)
                    try:
                        stream_data = ole.openstream(entry).read()

                        # zlib 압축 해제 (FileHeader 플래그에 따라)
                        data = stream_data
                        if is_compressed:
                            decompressed = None
                            for wbits in (-15, 15, -zlib.MAX_WBITS, zlib.MAX_WBITS):
                                try:
                                    decompressed = zlib.decompress(stream_data, wbits)
                                    break
                                except zlib.error:
                                    continue

                            if decompressed:
                                data = decompressed
                            else:
                                logger.debug(
                                    f"zlib decompression failed for {entry_name}, "
                                    f"using raw data"
                                )

                        # HWP 레코드 구조에서 텍스트 추출
                        section_text = self._extract_text_from_hwp_body(data)
                        if section_text:
                            text_parts.append(section_text)
                            logger.debug(
                                f"Extracted {len(section_text)} chars from {entry_name}"
                            )

                    except Exception as e:
                        logger.debug(f"Failed to process stream {entry_name}: {e}")
                        continue

                if text_parts:
                    text = "\n".join(text_parts)
                    if len(text.strip()) > self.MIN_TEXT_LENGTH:
                        doc_properties = self._extract_hwp_properties()
                        return self._create_success_result(
                            text,
                            {
                                "method": "olefile_bodytext",
                                "streams_processed": len(text_parts),
                                "doc_properties": doc_properties,
                            },
                        )
                    else:
                        logger.debug(
                            f"BodyText extracted only {len(text.strip())} chars, "
                            f"falling back to PrvText"
                        )

                # 2단계: PrvText 폴백 (미리보기 텍스트, ~1024바이트 제한)
                try:
                    if ole.exists("PrvText"):
                        prv_data = ole.openstream("PrvText").read()
                        prv_text = prv_data.decode("utf-16-le", errors="ignore").strip()
                        prv_text = prv_text.replace("\x00", "")
                        if len(prv_text) > self.MIN_TEXT_LENGTH:
                            doc_properties = self._extract_hwp_properties()
                            return self._create_success_result(
                                prv_text, {"method": "olefile_prvtext", "doc_properties": doc_properties}
                            )
                except Exception as e:
                    logger.debug(f"PrvText extraction failed: {e}")

                return self._create_error_result("No readable text in OleFile streams")

        except Exception as e:
            logger.debug(f"OleFile extraction failed: {e}")
            return self._create_error_result(f"OleFile error: {str(e)}")

    @staticmethod
    def _extract_text_from_hwp_body(data: bytes) -> str:
        """
        HWP BodyText 스트림의 레코드 구조를 파싱하여 텍스트를 추출합니다.

        HWP5 레코드 구조:
        - 각 레코드는 4바이트 헤더: [TagID(10bit) | Level(10bit) | Size(12bit)]
        - Size가 0xFFF(4095)이면 추가 4바이트에 실제 크기가 있음
        - HWPTAG_PARA_TEXT (TagID=67, 0x0043 + 0x0040 기본 오프셋 = 실제 tag 0x43)

        텍스트 레코드 내부:
        - UTF-16LE 인코딩
        - 특수 제어 문자 (char codes < 32)는 HWP 내부 제어 코드
        - char code 0~3: 가변 길이 인라인 제어 (추가 12바이트 스킵)
        - char code 4~9: 가변 길이 인라인 제어 (추가 12바이트 스킵)
        - char code 10: 줄바꿈
        - char code 13: 문단 끝
        - char code 11~12, 14~23: 고정 길이 인라인 제어 (추가 바이트 없음)
        - char code 24~31: 기타 제어 문자 (무시)

        Args:
            data: 압축 해제된 BodyText 스트림 바이너리 데이터

        Returns:
            str: 추출된 텍스트
        """
        # HWP5 태그 ID 상수
        # 레코드 헤더의 하위 10비트가 태그 ID
        # HWPTAG_BEGIN = 0x0040 (64), HWPTAG_PARA_TEXT = HWPTAG_BEGIN + 51 = 67
        HWPTAG_PARA_TEXT = 67

        paragraphs = []
        pos = 0

        while pos + 4 <= len(data):
            try:
                # 4바이트 레코드 헤더 읽기 (little-endian)
                header = int.from_bytes(data[pos:pos + 4], byteorder="little")
                tag_id = header & 0x3FF          # 하위 10비트: 태그 ID
                # level = (header >> 10) & 0x3FF  # 중간 10비트: 레벨 (미사용)
                size = (header >> 20) & 0xFFF     # 상위 12비트: 크기

                pos += 4

                # 크기가 0xFFF이면 다음 4바이트에서 실제 크기를 읽음
                if size == 0xFFF:
                    if pos + 4 > len(data):
                        break
                    size = int.from_bytes(data[pos:pos + 4], byteorder="little")
                    pos += 4

                # 레코드 데이터 범위 확인
                if pos + size > len(data):
                    # 손상된 레코드 — 남은 데이터로 최선을 다함
                    size = len(data) - pos

                record_data = data[pos:pos + size]
                pos += size

                # HWPTAG_PARA_TEXT 레코드인지 확인 (tag_id == 67)
                if tag_id != 67:
                    continue

                # UTF-16LE 문자 단위로 파싱
                para_chars = []
                i = 0
                while i + 1 < len(record_data):
                    char_code = int.from_bytes(
                        record_data[i:i + 2], byteorder="little"
                    )

                    if char_code == 0:
                        # NULL — 무시
                        i += 2
                    elif 1 <= char_code <= 3:
                        # 확장 제어 문자: 추가 12바이트 (총 14바이트 = 2 + 12)
                        i += 2 + 12
                    elif 4 <= char_code <= 9:
                        # 확장 제어 문자: 추가 12바이트
                        i += 2 + 12
                    elif char_code == 10:
                        # 줄바꿈
                        para_chars.append("\n")
                        i += 2
                    elif char_code == 13:
                        # 문단 끝 (줄바꿈으로 치환)
                        para_chars.append("\n")
                        i += 2
                    elif 11 <= char_code <= 12:
                        # 고정 길이 제어 문자 (탭 등) — 공백으로
                        para_chars.append(" ")
                        i += 2
                    elif 14 <= char_code <= 23:
                        # 고정 길이 제어 문자 — 무시
                        i += 2
                    elif 24 <= char_code <= 31:
                        # 기타 제어 문자 — 무시
                        i += 2
                    elif 0xD800 <= char_code <= 0xDBFF:
                        # UTF-16 High surrogate: 다음 2바이트와 쌍으로 디코딩
                        if i + 3 < len(record_data):
                            low_code = int.from_bytes(
                                record_data[i + 2:i + 4], byteorder="little"
                            )
                            if 0xDC00 <= low_code <= 0xDFFF:
                                # 유효한 surrogate pair → 실제 문자로 디코딩
                                codepoint = (
                                    0x10000
                                    + (char_code - 0xD800) * 0x400
                                    + (low_code - 0xDC00)
                                )
                                para_chars.append(chr(codepoint))
                                i += 4
                                continue
                        # Lone high surrogate — 무시
                        i += 2
                    elif 0xDC00 <= char_code <= 0xDFFF:
                        # Lone low surrogate — 무시
                        i += 2
                    else:
                        # 일반 문자 (한글, 영문, 숫자, 특수문자 등)
                        try:
                            char = chr(char_code)
                            para_chars.append(char)
                        except (ValueError, OverflowError):
                            pass
                        i += 2

                para_text = "".join(para_chars).strip()
                if para_text:
                    paragraphs.append(para_text)

            except Exception as e:
                logger.debug(f"Error parsing HWP record at pos {pos}: {e}")
                pos += 4  # 최소 레코드 헤더 크기만큼 전진 후 재시도
                continue

        if not paragraphs:
            return ""

        full_text = "\n".join(paragraphs)

        # 정리: 연속 줄바꿈 정리, 앞뒤 공백 제거
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        full_text = re.sub(r"[ \t]+", " ", full_text)
        full_text = full_text.strip()

        return full_text

    def _try_zipfile_xml(self) -> ExtractionResult:
        """
        ZIP 형식으로 HWP/HWPX 파일을 열어 XML을 파싱합니다.
        HWPX 및 일부 HWP5 파일은 ZIP 컨테이너입니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        try:
            import zipfile
            from xml.etree import ElementTree as ET
        except ImportError:
            return self._create_error_result("zipfile or xml modules not available")

        try:
            if not zipfile.is_zipfile(str(self.file_path)):
                return self._create_error_result("File is not a valid ZIP")

            with zipfile.ZipFile(str(self.file_path)) as zf:
                text_parts = []

                # 가능한 XML 파일들을 탐색합니다
                xml_files = [
                    f for f in zf.namelist()
                    if f.endswith(".xml") and ("content" in f.lower() or "hpf" in f.lower())
                ]

                if not xml_files:
                    # 기본 content.xml 시도
                    if "Contents/content.xml" in zf.namelist():
                        xml_files = ["Contents/content.xml"]
                    elif "content.xml" in zf.namelist():
                        xml_files = ["content.xml"]

                for xml_file in xml_files:
                    try:
                        xml_content = zf.read(xml_file).decode("utf-8")
                        root = ET.fromstring(xml_content)

                        # 모든 텍스트 노드를 추출합니다
                        text = self._extract_text_from_xml(root)
                        if text:
                            text_parts.append(text)
                    except Exception as e:
                        logger.debug(f"Failed to parse {xml_file}: {e}")
                        continue

            if text_parts:
                text = "\n".join(text_parts)
                return self._create_success_result(
                    text, {"method": "zipfile_xml", "xml_files_processed": len(xml_files), "doc_properties": {}}
                )
            else:
                return self._create_error_result("No text found in XML files")

        except Exception as e:
            logger.debug(f"ZipFile XML extraction failed: {e}")
            return self._create_error_result(f"ZipFile error: {str(e)}")

    def _try_libreoffice(self) -> ExtractionResult:
        """
        LibreOffice CLI를 사용한 HWP 변환.
        로컬 환경에 LibreOffice가 설치되어 있어야 합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        try:
            import tempfile
            import os
        except ImportError:
            return self._create_error_result("tempfile module not available")

        try:
            # 임시 디렉토리에 TXT 파일로 변환합니다
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / (self.file_path.stem + ".txt")

                cmd = [
                    "libreoffice",
                    "--headless",
                    "--convert-to", "txt:Text - txt - csv (StarCalc):44,34,76,1,,1033,true,true,true,false,false",
                    "--outdir", tmpdir,
                    str(self.file_path)
                ]

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        timeout=self.timeout,
                        text=True
                    )

                    if result.returncode == 0 and output_path.exists():
                        with open(output_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        return self._create_success_result(
                            text, {"method": "libreoffice", "doc_properties": {}}
                        )
                    else:
                        return self._create_error_result(
                            f"LibreOffice conversion failed: {result.stderr}"
                        )
                except subprocess.TimeoutExpired:
                    return self._create_error_result(
                        f"LibreOffice timeout after {self.timeout}s"
                    )

        except Exception as e:
            logger.debug(f"LibreOffice extraction failed: {e}")
            return self._create_error_result(f"LibreOffice error: {str(e)}")

    def _try_binary_scan(self) -> ExtractionResult:
        """
        이진 파일을 스캔하여 한글 UTF-8 텍스트를 추출합니다.
        0xEA-0xED 범위의 바이트를 한글로 인식합니다.

        Returns:
            ExtractionResult: 추출 결과
        """
        file_size = os.path.getsize(str(self.file_path))
        if file_size > 10 * 1024 * 1024:  # 10MB
            logger.warning(
                f"Binary scan 스킵 (파일 크기 {file_size / 1024 / 1024:.1f}MB > 10MB): "
                f"{self.file_path}"
            )
            return ExtractionResult(
                text='', success=False, extractor_used='binary_scan',
                error='파일 크기 초과'
            )

        try:
            with open(self.file_path, "rb") as f:
                binary_data = f.read()

            text_parts = []
            current_text = []

            i = 0
            while i < len(binary_data):
                byte = binary_data[i]

                # UTF-8 한글 범위 (0xEA-0xED)
                if 0xEA <= byte <= 0xED:
                    if i + 2 < len(binary_data):
                        # 3바이트 UTF-8 한글 문자
                        try:
                            korean_char = binary_data[i:i+3].decode("utf-8")
                            current_text.append(korean_char)
                            i += 3
                            continue
                        except (UnicodeDecodeError, IndexError):
                            pass

                # ASCII 문자 또는 공백
                if 32 <= byte <= 126:
                    try:
                        char = chr(byte)
                        current_text.append(char)
                    except ValueError:
                        pass
                elif byte in (9, 10, 13):  # tab, newline, carriage return
                    current_text.append(chr(byte))
                else:
                    # 텍스트 클러스터 끝
                    if current_text:
                        text_cluster = "".join(current_text).strip()
                        if len(text_cluster) > 2:
                            text_parts.append(text_cluster)
                        current_text = []

                i += 1

            # 마지막 텍스트 클러스터
            if current_text:
                text_cluster = "".join(current_text).strip()
                if len(text_cluster) > 2:
                    text_parts.append(text_cluster)

            if text_parts:
                text = " ".join(text_parts)
                return self._create_success_result(
                    text, {"method": "binary_scan", "clusters_found": len(text_parts), "doc_properties": {}}
                )
            else:
                return self._create_error_result("No text found in binary scan")

        except Exception as e:
            logger.debug(f"Binary scan extraction failed: {e}")
            return self._create_error_result(f"Binary scan error: {str(e)}")

    @staticmethod
    def _clean_hwp_text(raw_text: str) -> str:
        """
        HWP에서 디코딩된 원시 텍스트를 정제합니다.
        제어 문자를 제거하고 의미 있는 텍스트만 추출합니다.

        Args:
            raw_text: UTF-16LE 디코딩된 원시 텍스트

        Returns:
            str: 정제된 텍스트
        """
        # 제어 문자 제거 (탭, 줄바꿈은 유지)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw_text)

        # 한글, 영문, 숫자, 기본 구두점, 공백만 추출
        segments = re.findall(
            r"[가-힣a-zA-Z0-9\s.,;:!?()（）\[\]{}<>@#$%&*+\-=/·…\"'~\t\n]+",
            cleaned,
        )

        # 의미 있는 세그먼트만 결합 (2자 이상)
        meaningful = [seg.strip() for seg in segments if len(seg.strip()) >= 2]

        result = " ".join(meaningful)
        # 연속 공백 정리
        result = re.sub(r"\s+", " ", result).strip()
        return result

    @staticmethod
    def _extract_korean_text_from_bytes(data: bytes) -> str:
        """
        바이너리 데이터에서 한글 UTF-8 텍스트를 추출합니다 (바이너리 스캔 용도).

        Args:
            data: 바이너리 데이터

        Returns:
            str: 추출된 텍스트
        """
        text_parts = []
        i = 0

        while i < len(data):
            byte = data[i]

            # UTF-8 한글 범위 (3바이트: 0xEA-0xED)
            if 0xEA <= byte <= 0xED and i + 2 < len(data):
                try:
                    korean_char = data[i:i+3].decode("utf-8")
                    text_parts.append(korean_char)
                    i += 3
                    continue
                except (UnicodeDecodeError, IndexError):
                    pass

            # ASCII 문자
            if 32 <= byte <= 126:
                text_parts.append(chr(byte))
            elif byte in (9, 10, 13):
                text_parts.append(chr(byte))

            i += 1

        return "".join(text_parts)

    @staticmethod
    def _extract_text_from_xml(element, depth=0) -> str:
        """
        XML 요소에서 텍스트를 재귀적으로 추출합니다.

        Args:
            element: XML 요소
            depth: 현재 재귀 깊이 (최대 100)

        Returns:
            str: 추출된 텍스트
        """
        if depth > 100:
            return ''

        text_parts = []

        if element.text:
            text_parts.append(element.text.strip())

        for child in element:
            child_text = HWPExtractor._extract_text_from_xml(child, depth + 1)
            if child_text:
                text_parts.append(child_text)

            if child.tail:
                text_parts.append(child.tail.strip())

        return " ".join([t for t in text_parts if t])
