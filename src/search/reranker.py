"""
Cross-Encoder 리랭커
벡터+BM25 병합 결과의 상위 N개를 재평가하여 정밀도 향상
"""
from typing import List, Dict
from loguru import logger


class Reranker:
    """BAAI/bge-reranker-v2-m3 기반 Cross-Encoder 리랭커"""

    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3', device: str = 'auto', batch_size: int = 16):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        """Lazy loading: 첫 호출 시에만 모델 로드"""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            if self.device == 'auto':
                from src.embedding.embedding_manager import _detect_device
                self.device = _detect_device('auto')
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Reranker 모델 로드 완료: {self.model_name} ({self.device})")

    def rerank(self, query: str, results: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        Cross-Encoder로 query-chunk 쌍의 관련성을 재평가하여 재정렬

        Args:
            query: 검색 쿼리
            results: 병합된 검색 결과 리스트 (각 항목에 'text' 필수)
            top_n: 반환할 상위 결과 수

        Returns:
            rerank_score가 추가된 재정렬된 결과 리스트 (원본 미변경)
        """
        if not results:
            return []

        try:
            self._load_model()

            pairs = [(query, r['text']) for r in results]
            scores = self._model.predict(pairs, batch_size=self.batch_size)

            # 원본 리스트를 변경하지 않고 복사본 생성
            scored_results = []
            for i, result in enumerate(results):
                scored = dict(result)
                scored['rerank_score'] = float(scores[i])
                scored_results.append(scored)

            scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)

            return scored_results[:top_n]

        except Exception as e:
            logger.warning(f"Reranker 실패, 리랭킹 없이 상위 {top_n}개 반환: {e}")
            return results[:top_n]
