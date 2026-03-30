"""
범용 문서 메타데이터 추출기 (4단계 폴백)
"""
import os
import re
import unicodedata
from datetime import datetime
from typing import Dict, Optional

from loguru import logger
from src.compat import IS_MACOS


class MetadataExtractor:
    """
    범용 문서 메타데이터 추출기

    4단계 폴백:
    1. 문서 내장 properties (PDF metadata, DOCX core_properties 등)
    2. 첫 페이지 텍스트 패턴 매칭 (제목, 작성자, 날짜)
    3. 파일시스템 메타데이터 (파일명, 크기, 생성일, 수정일)
    4. 경로 기반 (상대경로, 확장자)
    """

    # 1단계: 문서 내장 properties 키 매핑 (소문자 정규화 후 매칭)
    _PROPERTY_MAP = {
        'doc_title': ['title', '제목'],
        'doc_author': ['author', 'creator', '작성자'],
        'doc_created': ['created', 'creationdate', 'creation_date'],
        'doc_modified': ['modified', 'moddate', 'modification_date'],
        'doc_keywords': ['keywords', '키워드'],
        'doc_subject': ['subject', '주제'],
    }

    # 2단계: 첫 페이지 텍스트 패턴
    _AUTHOR_PATTERNS = re.compile(
        r'(?:작성자|담당자|발표자|작성)\s*[:：]\s*(.+?)(?:\n|$)', re.MULTILINE
    )
    _DATE_PATTERNS = re.compile(
        r'(?:작성일|일시)\s*[:：]\s*(.+?)(?:\n|$)', re.MULTILINE
    )
    _DATE_STANDALONE = re.compile(
        r'(\d{4})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})[일]?'
    )
    _TITLE_LABEL = re.compile(
        r'(?:제목|건명)\s*[:：]\s*(.+?)(?:\n|$)', re.MULTILINE
    )
    _DOC_TYPE_KEYWORDS = [
        '계획서', '보고서', '회의록', '발표자료', '제안서',
        '지침', '매뉴얼', '안내서', '결과보고', '실적보고',
        '사업계획', '연구계획', '백서', '가이드',
    ]

    def __init__(self, doc_root: str = '', **kwargs):
        """
        MetadataExtractor 초기화

        Args:
            doc_root: 문서 루트 디렉토리 (상대경로 계산 기준)
            **kwargs: 하위 호환 (kisti_root 등)
        """
        # 하위 호환: kisti_root 키워드 지원
        if not doc_root and 'kisti_root' in kwargs:
            doc_root = kwargs['kisti_root']
        self.doc_root = os.path.expanduser(doc_root) if doc_root else ''
        logger.debug(f"MetadataExtractor 초기화: doc_root={doc_root}")

    def extract(
        self,
        file_path: str,
        extracted_text: str = '',
        doc_properties: Optional[dict] = None,
    ) -> Dict:
        """
        파일에서 메타데이터 추출 (4단계 폴백)

        Args:
            file_path: 파일 절대 경로
            extracted_text: 추출된 전체 텍스트 (첫 페이지 패턴 매칭용)
            doc_properties: Extractor가 전달하는 문서 내장 properties

        Returns:
            메타데이터 딕셔너리
        """
        try:
            file_path = unicodedata.normalize('NFC', str(file_path))
            meta = self._empty_metadata()

            # 4단계: 경로 기반 (항상 적용)
            meta.update(self._from_path(file_path))

            # 3단계: 파일시스템 메타데이터 (항상 적용)
            meta.update(self._from_filesystem(file_path))

            # 1단계: 문서 내장 properties (있으면 덮어씀)
            if doc_properties:
                props = self._from_properties(doc_properties)
                # 빈 값이 아닌 필드만 덮어쓰기
                for k, v in props.items():
                    if v:
                        meta[k] = v

            # 2단계: 첫 페이지 텍스트 패턴 (빈 필드만 보충)
            if extracted_text:
                self._fill_from_first_page(meta, extracted_text)

            logger.debug(
                f"메타데이터 추출 완료: {meta.get('file_name', '')} "
                f"(title={meta.get('doc_title', '')[:30]})"
            )
            return meta

        except Exception as e:
            logger.error(f"메타데이터 추출 오류 ({file_path}): {e}")
            return self._empty_metadata()

    # ── 1단계: 문서 내장 properties ──

    def _from_properties(self, doc_properties: dict) -> Dict:
        """
        문서 내장 properties에서 표준 필드 매핑

        키 이름은 Extractor마다 다를 수 있으므로 유연하게 매핑.
        """
        result = {}
        # 소문자 정규화된 properties
        normalized = {k.lower().strip(): v for k, v in doc_properties.items() if v}

        for target_field, source_keys in self._PROPERTY_MAP.items():
            for src_key in source_keys:
                if src_key in normalized:
                    value = str(normalized[src_key]).strip()
                    if value:
                        result[target_field] = value
                        break

        logger.debug(f"문서 properties에서 {len(result)}개 필드 추출")
        return result

    # ── 2단계: 첫 페이지 텍스트 패턴 매칭 ──

    def _fill_from_first_page(self, meta: dict, text: str) -> None:
        """
        첫 1000자에서 빈 필드만 보충 (이미 채워진 필드는 건너뜀)
        """
        first_page = text[:1000]

        # doc_title: 빈 경우만
        if not meta.get('doc_title'):
            meta['doc_title'] = self._extract_title_from_text(first_page)

        # doc_author
        if not meta.get('doc_author'):
            match = self._AUTHOR_PATTERNS.search(first_page)
            if match:
                meta['doc_author'] = match.group(1).strip()

        # doc_created
        if not meta.get('doc_created'):
            meta['doc_created'] = self._extract_date_from_text(first_page)

        # doc_type
        if not meta.get('doc_type'):
            meta['doc_type'] = self._extract_doc_type(first_page)

    def _extract_title_from_text(self, text: str) -> str:
        """첫 페이지에서 제목 추출"""
        # "제목:", "건명:" 라벨 우선
        match = self._TITLE_LABEL.search(text)
        if match:
            title = match.group(1).strip()
            if len(title) >= 2:
                return title

        # 첫 의미있는 줄 (5자 이상, boilerplate 아닌 것)
        boilerplate = {'보안', '대외비', '비밀', '일반', 'confidential', 'internal'}
        for line in text.split('\n'):
            line = line.strip()
            if len(line) >= 5 and not any(bp in line.lower() for bp in boilerplate):
                # 너무 긴 줄은 제목이 아닐 가능성 높음
                if len(line) <= 100:
                    return line
        return ''

    def _extract_date_from_text(self, text: str) -> str:
        """첫 페이지에서 날짜 추출 (ISO format 반환)"""
        # "작성일:", "일시:" 라벨 우선
        match = self._DATE_PATTERNS.search(text)
        if match:
            date_str = match.group(1).strip()
            parsed = self._parse_date_string(date_str)
            if parsed:
                return parsed

        # 독립 날짜 패턴
        match = self._DATE_STANDALONE.search(text)
        if match:
            y, m, d = match.group(1), match.group(2), match.group(3)
            try:
                dt = datetime(int(y), int(m), int(d))
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass
        return ''

    def _parse_date_string(self, date_str: str) -> str:
        """날짜 문자열을 ISO format으로 파싱"""
        match = self._DATE_STANDALONE.search(date_str)
        if match:
            y, m, d = match.group(1), match.group(2), match.group(3)
            try:
                dt = datetime(int(y), int(m), int(d))
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass
        return ''

    def _extract_doc_type(self, text: str) -> str:
        """첫 페이지에서 문서 유형 키워드 추출"""
        for keyword in self._DOC_TYPE_KEYWORDS:
            if keyword in text:
                return keyword
        return ''

    # ── 3단계: 파일시스템 메타데이터 ──

    def _from_filesystem(self, file_path: str) -> Dict:
        """os.stat으로 파일 크기, 생성일, 수정일 추출"""
        result = {}
        try:
            stat = os.stat(file_path)
            result['file_size'] = stat.st_size
            result['fs_modified'] = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            # macOS: st_birthtime, Linux: st_ctime (폴백)
            ctime = getattr(stat, 'st_birthtime', stat.st_ctime)
            result['fs_created'] = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
        except OSError as e:
            logger.warning(f"파일 stat 실패 ({file_path}): {e}")
        return result

    # ── 4단계: 경로 기반 ──

    def _from_path(self, file_path: str) -> Dict:
        """경로에서 파일명, 확장자, 상대경로 추출"""
        if IS_MACOS:
            file_path = unicodedata.normalize('NFC', file_path)
        result = {}

        basename = os.path.basename(file_path)
        result['file_name'] = basename

        _, ext = os.path.splitext(basename)
        result['file_type'] = ext.lower().lstrip('.') if ext else 'unknown'

        # doc_root 기준 상대 경로
        if self.doc_root:
            doc_root = os.path.normpath(str(self.doc_root))
            if IS_MACOS:
                doc_root = unicodedata.normalize('NFC', doc_root)
            try:
                result['relative_path'] = os.path.relpath(file_path, doc_root)
            except ValueError:
                # Windows 크로스 드라이브 등
                result['relative_path'] = os.path.basename(file_path)
        else:
            result['relative_path'] = file_path

        return result

    # ── 유틸리티 ──

    def _empty_metadata(self) -> Dict:
        """모든 필드를 빈 기본값으로 초기화"""
        return {
            'doc_title': '',
            'doc_author': '',
            'doc_created': '',
            'doc_modified': '',
            'doc_keywords': '',
            'doc_subject': '',
            'doc_type': '',
            'file_name': '',
            'file_type': 'unknown',
            'file_size': 0,
            'fs_created': '',
            'fs_modified': '',
            'relative_path': '',
        }
