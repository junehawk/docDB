"""
로컬 한국어 임베딩 모듈
sentence-transformers를 사용한 한국어 특화 임베딩
"""
from typing import List, Union
from loguru import logger


class LocalEmbedder:
    """
    로컬 한국어 임베딩 - ko-sroberta-multitask

    특징:
    - Lazy loading: 첫 embed() 호출 시에만 모델 로드
    - sentence-transformers 사용
    - 한국어 문장 인코딩 특화
    - 배치 처리 지원
    """

    def __init__(self, model_name: str = 'BAAI/bge-m3', device: str = 'cpu'):
        """
        LocalEmbedder 초기화 (모델은 lazy load됨)

        Args:
            model_name: HuggingFace 모델 이름
            device: 실행 디바이스 ('cpu', 'cuda', 'mps' 등)
        """
        self._model = None
        self.model_name = model_name
        self.device = device
        logger.debug(
            f"LocalEmbedder initialized with model={model_name}, device={device} "
            "(model will be loaded on first use)"
        )

    def _load_model(self):
        """
        Lazy load: 첫 사용 시 모델 로드

        Raises:
            ImportError: sentence-transformers가 설치되지 않은 경우
            Exception: 모델 로드 실패 시
        """
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully on device: {self.device}")

        except ImportError as e:
            logger.error(
                f"sentence-transformers not installed: {e}. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _clear_device_cache(self):
        """GPU/MPS 메모리 캐시 해제 (CPU에서는 스킵)"""
        if self.device == 'cpu':
            return
        import gc
        gc.collect()  # ① Python 참조 해제 → 텐서 참조 끊기
        try:
            import torch
            if self.device == 'mps' and hasattr(torch, 'mps'):
                torch.mps.synchronize()   # ② GPU 작업 완료 대기
                torch.mps.empty_cache()   # ③ 해제 가능한 블록 반환
            elif self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"Device cache clear failed: {e}")

    def embed(self, text: str):
        """
        단일 텍스트를 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (float 리스트)

        Raises:
            ValueError: 텍스트가 비어있는 경우
            ImportError: sentence-transformers 미설치
        """
        if not isinstance(text, str) or not text.strip():
            logger.warning(f"Invalid text input: {type(text)}")
            raise ValueError("Text must be a non-empty string")

        text = text.strip()
        # Surrogate 문자 제거 (Rust tokenizer 호환)
        text = text.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
        if not text:
            raise ValueError("Text must be a non-empty string")

        if self._model is None:
            self._load_model()

        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise

    def embed_batch(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> 'np.ndarray':
        """
        여러 텍스트를 배치로 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기 (메모리 효율성)
            show_progress: 진행바 표시 여부

        Returns:
            numpy.ndarray, shape (len(texts), dim), dtype float32

        Raises:
            ValueError: 텍스트 리스트가 비어있거나 유효하지 않은 경우
            ImportError: sentence-transformers 미설치
        """
        if not texts or not isinstance(texts, list):
            logger.warning(f"Invalid texts input: {type(texts)}")
            raise ValueError("Texts must be a non-empty list of strings")

        if not self.is_loaded:
            self._load_model()

        import numpy as np
        dim = self.dimension
        result = np.empty((len(texts), dim), dtype=np.float32)

        try:
            logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
            for i in range(0, len(texts), batch_size):
                sub_texts = texts[i:i + batch_size]
                try:
                    sub_emb = self._model.encode(
                        sub_texts,
                        batch_size=batch_size,
                        show_progress_bar=show_progress and i == 0,
                        convert_to_numpy=True,
                    )
                except Exception as batch_err:
                    if 'buffer size' not in str(batch_err).lower():
                        raise
                    # MPS 메모리 에러: 캐시 정리 후 건별 재시도
                    logger.warning(
                        f"MPS 메모리 에러 (batch {i}~{i+len(sub_texts)}), "
                        f"건별 재시도: {batch_err}"
                    )
                    self._clear_device_cache()
                    sub_emb = np.empty((len(sub_texts), dim), dtype=np.float32)
                    for j, txt in enumerate(sub_texts):
                        sub_emb[j] = self._model.encode(
                            txt, convert_to_numpy=True,
                        )
                        self._clear_device_cache()

                result[i:i + len(sub_texts)] = sub_emb
                del sub_emb
                self._clear_device_cache()
            logger.info(f"Successfully embedded {len(texts)} texts")
            return result
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise

    @property
    def dimension(self) -> int:
        """
        임베딩 차원 (모델별 고정값)

        Returns:
            임베딩 차원
        """
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        KNOWN_DIMENSIONS = {
            'BAAI/bge-m3': 1024,
            'jhgan/ko-sroberta-multitask': 768,
            'intfloat/multilingual-e5-large': 1024,
            'intfloat/multilingual-e5-base': 768,
        }
        return KNOWN_DIMENSIONS.get(self.model_name, 768)

    @property
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인"""
        return self._model is not None

    def unload_model(self):
        """메모리 절약을 위해 모델 언로드"""
        if self._model is not None:
            del self._model
            self._model = None
            self._clear_device_cache()
            logger.info("Model unloaded from memory")
