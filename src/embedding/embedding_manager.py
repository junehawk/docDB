"""
로컬 임베딩 매니저 모듈
로컬 모델만을 사용한 단순화된 임베딩 처리
"""
from typing import Dict, List
from loguru import logger

from .local_embedder import LocalEmbedder


def _detect_device(device_config: str) -> str:
    """
    디바이스 자동 감지

    Args:
        device_config: 설정값 ('auto', 'cpu', 'cuda', 'mps' 등)

    Returns:
        실제 사용할 디바이스 문자열
    """
    if device_config != 'auto':
        return device_config
    try:
        import torch
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    return 'cpu'


class EmbeddingManager:
    """
    로컬 임베딩 매니저

    특징:
    - 로컬 모델 전용 (프라이버시 보장)
    - 디바이스 자동 감지 (MPS / CUDA / CPU)
    - 배치 처리 지원
    - 설정 기반 유연성
    """

    def __init__(self, config: Dict):
        """
        EmbeddingManager 초기화

        Args:
            config: 설정 딕셔너리
            {
                'model': str,   # 모델명 (기본: 'BAAI/bge-m3')
                'device': str,  # 디바이스 (기본: 'auto')
            }

        Raises:
            ValueError: 임베더 초기화 실패
            ImportError: 필요 라이브러리 미설치
        """
        self.config = config

        model_name = config.get('model', 'BAAI/bge-m3')
        device = _detect_device(config.get('device', 'auto'))

        try:
            self.local_embedder = LocalEmbedder(
                model_name=model_name,
                device=device,
            )
            logger.info(f"LocalEmbedder initialized: model={model_name}, device={device}")
        except Exception as e:
            logger.error(f"Failed to initialize LocalEmbedder: {e}")
            raise ValueError(f"Embedding backend initialization failed: {e}") from e

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        여러 텍스트를 배치로 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기 (기본 64)

        Returns:
            임베딩 벡터 리스트

        Raises:
            ValueError: 텍스트 리스트가 비어있는 경우
        """
        if not texts:
            raise ValueError("Texts list must not be empty")

        # Surrogate 문자 제거
        cleaned = []
        for t in texts:
            t = t.strip()
            t = t.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
            if not t.strip():
                logger.warning("서로게이트 제거 후 빈 텍스트 — 스킵")
                cleaned.append(None)
            else:
                cleaned.append(t)

        # None 슬롯을 제로 벡터로 채우며 실제 텍스트만 임베딩
        valid_texts = [t for t in cleaned if t is not None]
        if not valid_texts:
            dim = self.local_embedder.dimension
            return [[0.0] * dim for _ in texts]

        valid_embeddings = self.local_embedder.embed_batch(valid_texts, batch_size=batch_size, show_progress=False)

        # 원래 순서 복원: None 위치에 제로 벡터 삽입
        dim = len(valid_embeddings[0])
        result = []
        valid_iter = iter(valid_embeddings)
        for t in cleaned:
            if t is None:
                result.append([0.0] * dim)
            else:
                result.append(next(valid_iter))
        return result

    def embed_text(self, text: str) -> List[float]:
        """
        단일 텍스트를 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (List[float])

        Raises:
            ValueError: 텍스트가 비어있는 경우
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        embedding = self.local_embedder.embed(text)
        logger.debug(f"Embedded text (dim={len(embedding)})")
        return embedding

    def embed_query(self, query: str) -> List[float]:
        """
        쿼리 텍스트를 임베딩

        Args:
            query: 쿼리 텍스트

        Returns:
            임베딩 벡터 (List[float])

        Raises:
            ValueError: 쿼리가 비어있는 경우
            RuntimeError: 임베딩 실패 시
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        try:
            embedding = self.local_embedder.embed(query)
            logger.debug(f"Query embedded (dim={len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Failed to embed query: {e}") from e

    @property
    def is_loaded(self) -> bool:
        """임베딩 모델이 로드되었는지 확인"""
        return self.local_embedder.is_loaded

    def get_embedding_info(self) -> Dict:
        """
        현재 임베딩 설정 정보 반환

        Returns:
            {
                'model': str,
                'dimension': int,
                'device': str,
            }
        """
        return {
            'model': self.local_embedder.model_name,
            'dimension': self.local_embedder.dimension,
            'device': self.local_embedder.device,
        }
