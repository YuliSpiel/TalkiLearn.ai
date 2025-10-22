from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingService:
    """임베딩 생성 서비스 (sentence-transformers 사용)"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: sentence-transformers 모델명
                       기본값: all-MiniLM-L6-v2 (빠르고 효율적인 다국어 모델)
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully.")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        텍스트 리스트를 임베딩 벡터로 변환

        Args:
            texts: 텍스트 목록
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부

        Returns:
            임베딩 벡터 목록 (L2 정규화 적용)
        """
        # L2 정규화를 통해 코사인 유사도 최적화
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalization
            convert_to_numpy=True
        )

        # numpy array를 리스트로 변환
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        """
        단일 쿼리 텍스트를 임베딩 벡터로 변환

        Args:
            query: 쿼리 텍스트

        Returns:
            임베딩 벡터
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.model.get_sentence_embedding_dimension()


# 싱글톤 인스턴스 (전역에서 재사용)
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """임베딩 서비스 싱글톤 인스턴스 반환"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
