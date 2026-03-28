"""
BM25 키워드 검색 인덱스
ChromaDB의 벡터 검색과 병행하여 하이브리드 검색 지원
"""
import hashlib
import json
import os
import time
from typing import List, Dict, Optional
from loguru import logger

from src.config import PROJECT_ROOT
DEFAULT_CACHE_PATH = os.path.join(PROJECT_ROOT, 'data', 'bm25_cache.json')


class BM25Index:
    """BM25 키워드 검색 인덱스 (인메모리, 로컬 캐싱 지원)"""

    _kiwi = None  # 클래스 수준 lazy-loaded singleton

    @classmethod
    def _get_kiwi(cls):
        """Kiwi 형태소 분석기 lazy 로드"""
        if cls._kiwi is None:
            try:
                from kiwipiepy import Kiwi
                cls._kiwi = Kiwi()
                logger.info("Kiwi 형태소 분석기 로드 완료")
            except ImportError:
                logger.warning("kiwipiepy 미설치 — 공백 기반 토크나이징 사용")
                cls._kiwi = False  # sentinel: 재시도 방지
        return cls._kiwi if cls._kiwi is not False else None

    def __init__(self, cache_path: Optional[str] = None):
        self.index = None
        self.chunk_ids: List[str] = []
        self._texts: List[List[str]] = []
        self._cache_path = cache_path or DEFAULT_CACHE_PATH

    def build(self, chunk_ids: List[str], texts: List[str]):
        """
        BM25 인덱스 구축

        Args:
            chunk_ids: 청크 ID 리스트
            texts: 청크 텍스트 리스트 (chunk_ids와 1:1 대응)
        """
        from rank_bm25 import BM25Plus

        self.chunk_ids = list(chunk_ids)
        self._texts = [self._tokenize(t) for t in texts]
        self.index = BM25Plus(self._texts)
        logger.info(f"BM25 인덱스 구축 완료: {len(self.chunk_ids)}개 청크")

    def build_from_chroma(self, chroma_manager):
        """
        ChromaDB에서 BM25 인덱스 구축 (캐시 우선 로드)

        1. ChromaDB 청크 수 확인 (가벼운 count() 호출)
        2. 로컬 캐시와 수가 일치하면 캐시에서 로드 (네트워크 전송 없음)
        3. 불일치 시 ChromaDB에서 전체 documents 다운로드 후 재구축 + 캐시 저장

        Args:
            chroma_manager: ChromaManager 인스턴스
        """
        # ChromaDB 청크 수 확인 (가벼운 API 호출)
        total_count = chroma_manager.collection.count()

        if total_count == 0:
            logger.warning("ChromaDB에 문서가 없어 BM25 인덱스를 구축하지 않았습니다")
            return

        # chunk_ids 전체 수집하여 해시 계산 (배치 페이지네이션)
        all_ids_for_hash = []
        HASH_BATCH = 10000
        collection = chroma_manager.collection
        for hash_offset in range(0, total_count, HASH_BATCH):
            result = collection.get(
                include=[],
                limit=HASH_BATCH,
                offset=hash_offset,
            )
            if result['ids']:
                all_ids_for_hash.extend(result['ids'])
            if not result['ids'] or len(result['ids']) < HASH_BATCH:
                break

        ids_hash = self._compute_ids_hash(all_ids_for_hash)
        del all_ids_for_hash  # 해시 계산 후 트랜지언트 리스트 즉시 해제

        # 캐시에서 로드 시도 (count + ids_hash 검증)
        if self._load_cache(total_count, expected_ids_hash=ids_hash):
            return

        # 캐시 미스: ChromaDB에서 전체 documents 배치 다운로드
        logger.info(f"BM25 인덱스 구축 시작 ({total_count}개 청크 다운로드)...")
        start = time.time()
        all_ids = []
        all_texts = []

        BATCH_SIZE = 5000
        collection = chroma_manager.collection
        for offset in range(0, total_count, BATCH_SIZE):
            result = collection.get(
                include=['documents'],
                limit=BATCH_SIZE,
                offset=offset,
            )
            if result['ids'] and result['documents']:
                all_ids.extend(result['ids'])
                all_texts.extend(result['documents'])

        if all_ids:
            self.build(all_ids, all_texts)
            del all_ids, all_texts  # 트랜지언트 원본 리스트 즉시 해제 (build()가 내부에 복사본 보관)
            self._save_cache(len(self.chunk_ids))
            elapsed = time.time() - start
            logger.info(f"BM25 캐시 저장 완료 ({self._cache_path}, {elapsed:.1f}초)")

    def _load_cache(self, expected_count: int, expected_ids_hash: str = None) -> bool:
        """로컬 캐시에서 BM25 인덱스 로드 (JSON 직렬화 사용)"""
        try:
            if not os.path.exists(self._cache_path):
                return False

            start = time.time()
            with open(self._cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)

            if cache.get('count') != expected_count:
                logger.info(
                    f"BM25 캐시 무효 — 수량 불일치 (캐시: {cache.get('count')}개, "
                    f"ChromaDB: {expected_count}개)"
                )
                return False

            # chunk_ids 해시 검증 (머신 간 동기화 불일치 감지)
            if expected_ids_hash and cache.get('ids_hash') != expected_ids_hash:
                logger.info(
                    f"BM25 캐시 무효 — chunk_ids 해시 불일치 "
                    f"(캐시: {cache.get('ids_hash', 'N/A')[:12]}…, "
                    f"ChromaDB: {expected_ids_hash[:12]}…)"
                )
                return False

            from rank_bm25 import BM25Plus

            self.chunk_ids = cache['chunk_ids']
            self._texts = cache['texts']
            cached_count = cache['count']
            del cache  # 파싱된 JSON dict 즉시 해제 (chunk_ids/texts는 이미 self에 이전됨)

            # 무결성 검증: 캐시 데이터 일관성 확인
            if len(self.chunk_ids) != len(self._texts) or len(self.chunk_ids) != cached_count:
                logger.warning("BM25 캐시 무결성 오류 — 데이터 길이 불일치")
                return False

            self.index = BM25Plus(self._texts)
            elapsed = time.time() - start
            logger.info(
                f"BM25 캐시 로드 완료: {len(self.chunk_ids)}개 청크 ({elapsed:.1f}초)"
            )
            return True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"BM25 캐시 손상 — 삭제 후 재구축: {e}")
            try:
                os.remove(self._cache_path)
            except OSError:
                pass
            return False
        except Exception as e:
            logger.warning(f"BM25 캐시 로드 실패 (재구축 예정): {e}")
            return False

    @staticmethod
    def _compute_ids_hash(chunk_ids: List[str]) -> str:
        """정렬된 chunk_ids의 SHA-256 해시 계산"""
        joined = '\n'.join(sorted(chunk_ids))
        return hashlib.sha256(joined.encode()).hexdigest()

    def _save_cache(self, count: int):
        """BM25 인덱스를 로컬 캐시로 저장 (JSON 직렬화 사용)"""
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            cache = {
                'count': count,
                'ids_hash': self._compute_ids_hash(self.chunk_ids),
                'chunk_ids': self.chunk_ids,
                'texts': self._texts,
            }
            with open(self._cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"BM25 캐시 저장 실패: {e}")

    def search(self, query: str, n_results: int = 50) -> List[Dict]:
        """
        BM25 키워드 검색

        Args:
            query: 검색 쿼리
            n_results: 반환 결과 수

        Returns:
            [{'chunk_id': str, 'score': float}, ...]
        """
        if self.index is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        scored = [
            (self.chunk_ids[i], float(scores[i]))
            for i in range(len(scores))
            if scores[i] > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {'chunk_id': cid, 'score': score}
            for cid, score in scored[:n_results]
        ]

    def add_chunks(self, chunk_ids: List[str], texts: List[str]):
        """
        증분 인덱싱: 기존 인덱스에 새 청크 추가
        rank_bm25는 동적 추가를 지원하지 않으므로 전체 재구축
        """
        from rank_bm25 import BM25Plus

        self.chunk_ids.extend(chunk_ids)
        new_tokenized = [self._tokenize(t) for t in texts]
        self._texts.extend(new_tokenized)
        self.index = BM25Plus(self._texts)
        logger.debug(f"BM25 인덱스 증분 갱신: 총 {len(self.chunk_ids)}개 청크")

    def get_stats(self) -> Dict:
        """BM25 인덱스 통계"""
        return {
            'indexed': self.index is not None,
            'total_chunks': len(self.chunk_ids),
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """한국어 형태소 분석 기반 토크나이징 (kiwipiepy 사용, 폴백: 공백 분할)"""
        kiwi = BM25Index._get_kiwi()
        if kiwi is None:
            return text.split()

        try:
            tokens = kiwi.tokenize(text)
            # 형태소 추출, 1글자 조사/어미/접미사 제외 (의미 있는 형태소만)
            result = []
            for token in tokens:
                form = token.form.strip()
                if not form:
                    continue
                # 1글자 한글 조사/어미 제외 (을, 를, 이, 가, 은, 는, 에, 의, 로, 와, 과, 도, 만 등)
                if len(form) == 1 and token.tag.startswith(('J', 'E')):
                    continue
                result.append(form)
            return result if result else text.split()
        except Exception:
            return text.split()
