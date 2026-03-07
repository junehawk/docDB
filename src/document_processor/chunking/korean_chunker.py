"""
한국어 텍스트 청킹 모듈
문단 및 문장 경계를 존중하는 한국어 특화 텍스트 분할
"""
from typing import List
import re
from loguru import logger


class KoreanChunker:
    """
    한국어 텍스트 청커 - 문단/문장 경계를 존중하는 분할

    특징:
    - 문단(double newline) 경계 존중
    - 한국어 문장 종결어미 인식
    - 청크 간 overlap 지원
    - 문자 기반 크기 제한 (토큰 기반 아님)
    """

    # 한국어 문장 종결 패턴
    KOREAN_SENTENCE_ENDINGS = [
        r'다\.',
        r'요\.',
        r'까\?',
        r'나\?',
        r'세요\.',
        r'습니다\.',
        r'입니다\.',
        r'합니다\.',
        r'[.!?]',  # 표준 마침표, 느낌표, 물음표
    ]

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        """
        KoreanChunker 초기화

        Args:
            chunk_size: 청크의 대략적 문자 개수 (토큰 아님, 한글 문자 1개 ≈ 1토큰)
            overlap: 청크 간 겹치는 문자 개수
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.sentence_pattern = '|'.join(self.KOREAN_SENTENCE_ENDINGS)
        logger.debug(f"KoreanChunker initialized with chunk_size={chunk_size}, overlap={overlap}")

    def chunk(self, text: str) -> List[str]:
        """
        텍스트를 한국어 친화적으로 분할

        알고리즘:
        1. 이중 줄바꿈(\\n\\n)으로 문단 분할
        2. 청크_크기를 초과하는 문단은 문장 경계(한국어 종결어미)로 재분할
        3. 소규모 문단들을 청크_크기 한계를 존중하며 결합
        4. 청크 간 overlap 추가

        Args:
            text: 분할할 원본 텍스트

        Returns:
            청크된 텍스트 리스트
        """
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text input: {type(text)}")
            return []

        # Step 1: 문단 분할 (이중 줄바꿈 기준)
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        logger.debug(f"Split into {len(paragraphs)} paragraphs")

        if not paragraphs:
            return []

        # Step 2: 청크_크기를 초과하는 문단은 문장 단위로 재분할
        sentences = []
        for para_idx, paragraph in enumerate(paragraphs):
            if len(paragraph) <= self.chunk_size:
                sentences.append(paragraph)
            else:
                # 문장 단위로 분할
                para_sentences = self._split_by_sentences(paragraph)
                sentences.extend(para_sentences)
                logger.debug(
                    f"Paragraph {para_idx} (len={len(paragraph)}) split into "
                    f"{len(para_sentences)} sentences"
                )

        # Step 3: 문장들을 청크_크기 한계를 존중하며 결합
        chunks = self._combine_sentences_into_chunks(sentences)

        # Step 4: Overlap 추가
        chunks = self._add_overlap(chunks)

        logger.info(f"Created {len(chunks)} chunks from text (total chars: {len(text)})")
        return chunks

    def _split_by_sentences(self, paragraph: str) -> List[str]:
        """
        문단을 문장으로 분할 (한국어 문장 종결 패턴 사용)

        Args:
            paragraph: 분할할 문단

        Returns:
            문장 리스트
        """
        # 정규식으로 문장 경계 찾기
        parts = re.split(f'({self.sentence_pattern})', paragraph)

        sentences = []
        for i in range(0, len(parts), 2):
            if i < len(parts):
                text = parts[i].strip()
                ending = parts[i + 1].strip() if i + 1 < len(parts) else ''

                sentence = text + ending
                if sentence.strip():
                    sentences.append(sentence.strip())

        return sentences if sentences else [paragraph.strip()]

    def _combine_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """
        문장들을 청크_크기 한계를 존중하며 결합

        Args:
            sentences: 분할된 문장 리스트

        Returns:
            청크 리스트
        """
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # 현재 청크에 문장을 추가했을 때 크기 초과 확인
            test_chunk = current_chunk + sentence if current_chunk else sentence

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # 현재 청크 저장 및 새로운 청크 시작
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        # 남은 청크 추가
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        청크 간 overlap 추가

        각 청크의 끝부터 다음 청크의 시작에 overlap_크기만큼 겹침

        Args:
            chunks: 원본 청크 리스트

        Returns:
            Overlap이 추가된 청크 리스트
        """
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # 이전 청크 끝에서 overlap_크기만큼 추출
            overlap_text = prev_chunk[-self.overlap:] if len(prev_chunk) > self.overlap else prev_chunk

            # overlap 텍스트 + 현재 청크
            overlapped_chunk = overlap_text + curr_chunk
            overlapped_chunks.append(overlapped_chunk)

        logger.debug(
            f"Added overlap ({self.overlap} chars) to {len(overlapped_chunks)} chunks"
        )
        return overlapped_chunks
