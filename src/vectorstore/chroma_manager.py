"""
ChromaDB 벡터 스토어 관리 - 단일 컬렉션 (documents)
"""
from typing import List, Dict, Optional, Union
import chromadb
from loguru import logger


class ChromaManager:
    """ChromaDB 벡터 스토어 관리 - 단일 documents 컬렉션"""

    COLLECTION_NAME = 'documents'

    def __init__(self, persist_dir: str = '~/.docdb/chroma_db'):
        """
        ChromaDB 클라이언트 초기화 및 컬렉션 생성

        Args:
            persist_dir: ChromaDB 데이터 디렉토리
        """
        import os
        try:
            self.persist_dir = os.path.normpath(os.path.expanduser(persist_dir))
            self.client = chromadb.PersistentClient(path=self.persist_dir)

            # 단일 컬렉션 생성 또는 가져오기 (cosine 거리 사용)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={
                    'description': '문서 벡터 저장소',
                    'hnsw:space': 'cosine',
                }
            )

            logger.info(f"ChromaManager 초기화 완료: 로컬({self.persist_dir})")

        except Exception as e:
            logger.error(f"ChromaManager 초기화 실패: {e}")
            raise

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: Union[List[List[float]], 'np.ndarray'],
    ) -> bool:
        """
        여러 청크를 컬렉션에 추가

        Args:
            chunks: 청크 딕셔너리 리스트
                {
                    'id': str,
                    'text': str,
                    'metadata': {
                        'file_type': str,
                        'file_path': str,
                        'mtime_iso': str,
                        'chunk_index': int
                    }
                }
            embeddings: 임베딩 벡터 리스트

        Returns:
            성공 여부
        """
        try:
            if not chunks or (hasattr(embeddings, '__len__') and len(embeddings) == 0):
                logger.warning("빈 청크 또는 임베딩 리스트")
                return False

            if len(chunks) != len(embeddings):
                logger.error(f"청크와 임베딩 개수 불일치: {len(chunks)} vs {len(embeddings)}")
                return False

            ids = [chunk['id'] for chunk in chunks]
            documents = [chunk['text'] for chunk in chunks]
            metadatas = [self._sanitize_metadata(chunk.get('metadata', {})) for chunk in chunks]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"컬렉션에 {len(chunks)}개 청크 추가")
            return True

        except Exception as e:
            logger.error(f"청크 추가 실패: {e}")
            return False

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """ChromaDB 호환 메타데이터로 변환 (str/int/float/bool만 허용)"""
        return {
            k: v for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool))
        }

    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict,
    ) -> bool:
        """단일 청크를 컬렉션에 추가"""
        try:
            self.collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[self._sanitize_metadata(metadata)]
            )
            return True
        except Exception as e:
            logger.error(f"청크 추가 실패: {chunk_id}: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        컬렉션에서 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            n_results: 반환 결과 수
            where_filter: 메타데이터 필터 (Chroma where 문법)

        Returns:
            검색 결과 딕셔너리 리스트
            {
                'chunk_id': str,
                'text': str,
                'distance': float,
                'metadata': dict
            }
        """
        try:
            # n_results가 컬렉션 크기를 초과하면 ChromaDB 에러 발생 → 클램핑
            collection_count = self.collection.count()
            if collection_count == 0:
                return []
            n_results = min(n_results, collection_count)

            kwargs = {
                'query_embeddings': [query_embedding],
                'n_results': n_results,
            }

            if where_filter:
                kwargs['where'] = where_filter

            results = self.collection.query(**kwargs)

            # 결과를 표준 형식으로 변환
            if not results['ids'] or not results['ids'][0]:
                return []

            formatted_results = []
            for i, chunk_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'chunk_id': chunk_id,
                    'text': results['documents'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                })

            return formatted_results

        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []

    def delete_by_file(self, file_path: str) -> bool:
        """
        특정 파일의 모든 청크 삭제 (file_path 메타데이터 필터 사용)

        Args:
            file_path: 삭제할 파일 경로

        Returns:
            성공 여부
        """
        try:
            where_filter = {'file_path': {'$eq': file_path}}
            self.collection.delete(where=where_filter)
            logger.info(f"파일 삭제됨: {file_path}")
            return True

        except Exception as e:
            logger.error(f"파일 삭제 실패: {file_path}, {e}")
            return False

    def get_stats(self) -> Dict:
        """
        컬렉션의 통계 정보 반환

        Returns:
            {'documents': int}
        """
        try:
            count = self.collection.count()
            stats = {'documents': count}
            logger.info(f"벡터 DB 통계: {stats}")
            return stats

        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {'documents': 0}
