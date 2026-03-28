"""
하이브릿 검색기 - RRF (Reciprocal Rank Fusion)를 이용한 벡터+BM25 병합
"""
from typing import List, Dict, Optional
from loguru import logger


class Retriever:
    """하이브릿 검색기 - 단일 컬렉션 검색 + BM25 + Reranker"""

    def __init__(self, chroma_manager, embedding_manager, merger=None, bm25_index=None, reranker=None,
                 vector_top_n=50, bm25_top_n=50, rerank_top_n=20):
        """
        검색기 초기화

        Args:
            chroma_manager: ChromaManager 인스턴스
            embedding_manager: EmbeddingManager 인스턴스 (쿼리 임베딩용)
            merger: 결과 병합 전략 (기본값: RRF)
            bm25_index: BM25Index 인스턴스 (None이면 벡터 전용 검색)
            reranker: Reranker 인스턴스 (None이면 리랭킹 건너뜀)
            vector_top_n: 벡터 검색 후보 수 (기본: 50)
            bm25_top_n: BM25 검색 후보 수 (기본: 50)
            rerank_top_n: 리랭킹 입력 후보 수 (기본: 20)
        """
        self.chroma = chroma_manager
        self.embedder = embedding_manager
        self.merger = merger or ResultMerger()
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.vector_top_n = vector_top_n
        self.bm25_top_n = bm25_top_n
        self.rerank_top_n = rerank_top_n
        logger.info(
            f"Retriever 초기화 완료 "
            f"(BM25={'활성' if bm25_index else '비활성'}, "
            f"Reranker={'활성' if reranker else '비활성'}, "
            f"vector_top_n={vector_top_n}, bm25_top_n={bm25_top_n}, rerank_top_n={rerank_top_n})"
        )

    def search(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        하이브릿 검색: 벡터 + BM25 + 리랭킹 파이프라인

        Args:
            query: 검색 쿼리 (자연어)
            n_results: 반환할 결과 수
            filters: 메타데이터 필터 딕셔너리
                {
                    'file_type': str,
                    'author': str,
                    'doc_type': str
                }

        Returns:
            병합된 검색 결과 리스트
            {
                'chunk_id': str,
                'text': str,
                'score': float (0~1, 높을수록 관련성 높음),
                'metadata': dict,
                'rrf_score': float
            }
        """
        try:
            import time as _time

            # 1. 쿼리 임베딩
            t0 = _time.monotonic()
            embedding = self._embed_query(query)
            t1 = _time.monotonic()
            logger.info(f"[타이밍] 쿼리 임베딩: {t1 - t0:.2f}초")
            if embedding is None:
                logger.warning(f"쿼리 임베딩 실패: {query}")
                return []

            # 2. 벡터 검색 (단일 컬렉션)
            where_filter = self._build_where_filter(filters)
            vector_results = self._search_collection(
                embedding, n_results=self.vector_top_n, where_filter=where_filter
            )
            t2 = _time.monotonic()
            logger.info(f"[타이밍] 벡터 검색: {t2 - t1:.2f}초")

            # distance → score 변환
            vector_merged = self.merger.merge(vector_results, n_results=self.vector_top_n)

            # 3. BM25 하이브리드 검색
            bm25_results = []
            if self.bm25_index:
                bm25_results = self.bm25_index.search(query, n_results=self.bm25_top_n)
                hybrid_merged = self.merger.merge_hybrid(
                    vector_merged, bm25_results, n_results=self.rerank_top_n
                )
            else:
                hybrid_merged = vector_merged[:self.rerank_top_n]
            t3 = _time.monotonic()
            logger.info(f"[타이밍] BM25 검색+병합: {t3 - t2:.2f}초")

            # BM25 전용 결과에 text/metadata 보충
            self._fill_missing_text(hybrid_merged, vector_merged)

            # 4. Reranking
            if self.reranker and hybrid_merged:
                final = self.reranker.rerank(query, hybrid_merged, top_n=n_results)
            else:
                final = hybrid_merged[:n_results]
            t4 = _time.monotonic()
            logger.info(f"[타이밍] 리랭킹: {t4 - t3:.2f}초")

            logger.info(
                f"검색 완료: 벡터({len(vector_merged)}) + "
                f"BM25({len(bm25_results)}) → "
                f"최종({len(final)}) "
                f"[총 {t4 - t0:.2f}초]"
            )
            return final

        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []

    def _fill_missing_text(self, hybrid_results: List[Dict], vector_results: List[Dict]):
        """BM25 전용 결과에 text/metadata 보충 (벡터 결과 → ChromaDB 순으로 조회)"""
        text_map = {r['chunk_id']: r for r in vector_results}
        missing_ids = []

        for r in hybrid_results:
            if 'text' not in r:
                if r['chunk_id'] in text_map:
                    source = text_map[r['chunk_id']]
                    r['text'] = source['text']
                    r['metadata'] = source.get('metadata', {})
                else:
                    missing_ids.append(r['chunk_id'])

        # 벡터 결과에 없는 BM25 전용 청크는 ChromaDB에서 직접 조회
        if missing_ids:
            chroma_texts = self._fetch_texts_from_chroma(missing_ids)
            for r in hybrid_results:
                if 'text' not in r and r['chunk_id'] in chroma_texts:
                    source = chroma_texts[r['chunk_id']]
                    r['text'] = source['text']
                    r['metadata'] = source.get('metadata', {})

        # text가 여전히 없는 결과 제거
        before = len(hybrid_results)
        hybrid_results[:] = [r for r in hybrid_results if 'text' in r]
        dropped = before - len(hybrid_results)
        if dropped > 0:
            logger.warning(f"BM25 결과 {dropped}개 텍스트 없음 (stale cache?)")

    def _fetch_texts_from_chroma(self, chunk_ids: List[str]) -> Dict[str, Dict]:
        """ChromaDB에서 chunk_id로 텍스트와 메타데이터 직접 조회"""
        result_map = {}
        try:
            result = self.chroma.collection.get(ids=chunk_ids, include=['documents', 'metadatas'])
            if result and result['ids']:
                for i, cid in enumerate(result['ids']):
                    result_map[cid] = {
                        'text': result['documents'][i] if result['documents'] else '',
                        'metadata': result['metadatas'][i] if result['metadatas'] else {},
                    }
        except Exception as e:
            logger.warning(f"ChromaDB 텍스트 조회 실패: {e}")
        return result_map

    def _embed_query(self, query: str) -> Optional[List[float]]:
        """
        쿼리를 임베딩 모델로 임베딩

        Args:
            query: 쿼리 텍스트

        Returns:
            임베딩 벡터 또는 None
        """
        try:
            return self.embedder.embed_query(query)
        except Exception as e:
            logger.error(f"쿼리 임베딩 실패: {e}")
            return None

    def _search_collection(
        self,
        query_embedding: List[float],
        n_results: int,
        where_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        단일 컬렉션에서 검색

        Args:
            query_embedding: 임베딩 벡터
            n_results: 결과 수
            where_filter: 메타데이터 필터

        Returns:
            검색 결과 리스트
        """
        try:
            results = self.chroma.search(
                query_embedding=query_embedding,
                n_results=n_results,
                where_filter=where_filter
            )
            return results

        except Exception as e:
            logger.error(f"컬렉션 검색 실패: {e}")
            return []

    def _build_where_filter(self, filters: Optional[Dict]) -> Optional[Dict]:
        """
        필터 딕셔너리를 Chroma where 문법으로 변환

        Args:
            filters: {'file_type': str, 'author': str, 'doc_type': str}

        Returns:
            Chroma where filter 또는 None
        """
        if not filters:
            return None

        conditions = []

        if 'file_type' in filters:
            conditions.append({'file_type': {'$eq': filters['file_type']}})

        if 'author' in filters:
            conditions.append({'doc_author': {'$eq': filters['author']}})

        if 'doc_type' in filters:
            conditions.append({'doc_type': {'$eq': filters['doc_type']}})

        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {'$and': conditions}


class ResultMerger:
    """
    검색 결과 병합기 - 유사도 점수 기반

    cosine distance를 cosine similarity로 변환하여 정렬합니다.
    """

    def __init__(self, k: int = 60, min_score: float = 0.0):
        """
        Args:
            k: RRF 상수 (중복 문서가 있을 때 사용)
            min_score: 최소 유사도 임계값 (0~1, 이하 결과 제외)
        """
        self.k = k
        self.min_score = min_score

    def merge(
        self,
        results: List[Dict],
        n_results: int = 10
    ) -> List[Dict]:
        """
        단일 컬렉션 결과를 cosine similarity로 변환하여 정렬

        ChromaDB cosine distance → similarity 변환: score = 1 - distance

        Args:
            results: 벡터 검색 결과
            n_results: 최종 반환 결과 수

        Returns:
            score 내림차순 정렬된 결과 (상위 n_results개)
        """
        try:
            all_results = []

            for result in results:
                score = self._distance_to_score(result.get('distance', 1.0))
                all_results.append({**result, 'score': score})

            # 최소 유사도 임계값 필터링
            if self.min_score > 0:
                all_results = [r for r in all_results if r['score'] >= self.min_score]

            # cosine similarity 점수로 정렬 (높을수록 유사)
            all_results.sort(key=lambda x: x['score'], reverse=True)

            return all_results[:n_results]

        except Exception as e:
            logger.error(f"결과 병합 실패: {e}")
            return []

    def merge_hybrid(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        n_results: int = 20
    ) -> List[Dict]:
        """
        벡터 검색 결과와 BM25 결과를 RRF로 병합

        RRF score = Σ 1/(k + rank_i) for each ranking list

        Args:
            vector_results: 벡터 검색 결과 (score 내림차순)
            bm25_results: BM25 검색 결과 (score 내림차순)
            n_results: 반환 결과 수

        Returns:
            RRF score 기준 정렬된 병합 결과
        """
        rrf_scores = {}
        result_map = {}

        # 벡터 결과 RRF 점수
        for rank, r in enumerate(vector_results):
            cid = r['chunk_id']
            rrf_scores[cid] = 1.0 / (self.k + rank + 1)
            result_map[cid] = r

        # BM25 결과 RRF 점수 누적
        for rank, r in enumerate(bm25_results):
            cid = r['chunk_id']
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self.k + rank + 1)
            if cid not in result_map:
                result_map[cid] = r

        # RRF score 기준 정렬
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        merged = []
        for cid in sorted_ids[:n_results]:
            result = {**result_map[cid], 'rrf_score': rrf_scores[cid]}
            merged.append(result)

        return merged

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """
        ChromaDB cosine distance → cosine similarity 변환

        cosine distance = 1 - cosine_similarity (범위: 0~2)
        cosine similarity = 1 - cosine_distance (범위: -1~1)
        """
        # 의도적 클램핑: 코사인 거리 > 1 (반대 방향)인 결과를 0점으로 처리
        return max(0.0, 1.0 - distance)
