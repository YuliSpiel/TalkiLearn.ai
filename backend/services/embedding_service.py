from sentence_transformers import SentenceTransformer
from typing import List, Callable, Optional
import numpy as np
import torch


class EmbeddingService:
    """임베딩 생성 서비스 (sentence-transformers 사용)"""

    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2"):
        """
        Args:
            model_name: sentence-transformers 모델명
                       기본값: paraphrase-MiniLM-L3-v2 (매우 빠른 경량 모델, 2-3배 속도 향상)
        """
        self.model_name = model_name

        # GPU 디바이스 설정 (Apple Silicon MPS 또는 CUDA)
        if torch.backends.mps.is_available():
            self.device = "mps"
            print(f"🚀 Using Apple Silicon GPU (MPS) for acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"🚀 Using CUDA GPU for acceleration")
        else:
            self.device = "cpu"
            print(f"⚠️ Using CPU (no GPU available)")

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Embedding model loaded successfully on {self.device.upper()}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,  # 배치 크기 증가 (32 -> 64)
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[List[float]]:
        """
        텍스트 리스트를 임베딩 벡터로 변환

        Args:
            texts: 텍스트 목록
            batch_size: 배치 크기 (기본값 64로 증가)
            show_progress: 진행률 표시 여부
            progress_callback: 진행률 콜백 함수 (current, total)

        Returns:
            임베딩 벡터 목록 (L2 정규화 적용)
        """
        if progress_callback:
            # 배치 단위로 처리하며 진행률 보고
            all_embeddings = []
            total_texts = len(texts)

            for i in range(0, total_texts, batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                all_embeddings.extend(batch_embeddings)

                # 진행률 콜백 호출
                progress_callback(min(i + batch_size, total_texts), total_texts)

            return np.array(all_embeddings).tolist()
        else:
            # 기존 방식 (진행률 콜백 없음)
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,  # L2 normalization
                convert_to_numpy=True
            )
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
