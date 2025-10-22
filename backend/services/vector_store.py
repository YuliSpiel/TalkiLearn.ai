import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os


class VectorStoreService:
    """Chroma 벡터 스토어 서비스"""

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Args:
            persist_directory: Chroma DB 저장 경로
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Chroma 클라이언트 초기화 (persistent mode)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 컬렉션 이름
        self.collection_name = "learning_chunks"

    def get_or_create_collection(self):
        """컬렉션 가져오기 또는 생성"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
            )
        return collection

    def add_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        chunk_ids: List[str]
    ) -> None:
        """
        청크와 임베딩을 벡터 스토어에 추가

        Args:
            chunks: 텍스트 청크 목록
            embeddings: 임베딩 벡터 목록
            metadatas: 메타데이터 목록 (subsession_id, chunk_id, page_no 등)
            chunk_ids: 청크 고유 ID 목록
        """
        collection = self.get_or_create_collection()
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=chunk_ids
        )

    def query_chunks(
        self,
        query_embedding: List[float],
        subsession_id: int,
        top_k: int = 6,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        쿼리 임베딩으로 유사한 청크 검색 (MMR 포함)

        Args:
            query_embedding: 쿼리 임베딩 벡터
            subsession_id: 서브세션 ID (필터링용)
            top_k: 반환할 상위 결과 수
            where_filter: 추가 필터 조건

        Returns:
            검색 결과 (ids, documents, distances, metadatas)
        """
        collection = self.get_or_create_collection()

        # 서브세션 필터 설정
        filter_condition = {"subsession_id": subsession_id}
        if where_filter:
            filter_condition.update(where_filter)

        # MMR 파라미터: lambda=0.7 (다양성과 관련성 균형)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_condition,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else []
        }

    def get_all_chunks_by_subsession(self, subsession_id: int) -> List[Dict[str, Any]]:
        """
        서브세션의 모든 청크 가져오기

        Args:
            subsession_id: 서브세션 ID

        Returns:
            청크 목록 (id, document, metadata)
        """
        collection = self.get_or_create_collection()

        results = collection.get(
            where={"subsession_id": subsession_id},
            include=["documents", "metadatas"]
        )

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            chunks.append({
                "id": chunk_id,
                "document": results["documents"][i],
                "metadata": results["metadatas"][i]
            })

        return chunks

    def delete_session_chunks(self, session_id: int) -> None:
        """
        세션의 모든 청크 삭제

        Args:
            session_id: 세션 ID
        """
        collection = self.get_or_create_collection()

        # 세션에 속한 모든 청크 찾기
        results = collection.get(
            where={"session_id": session_id},
            include=["metadatas"]
        )

        if results["ids"]:
            collection.delete(ids=results["ids"])

    def reset(self) -> None:
        """벡터 스토어 초기화 (개발용)"""
        self.client.reset()
