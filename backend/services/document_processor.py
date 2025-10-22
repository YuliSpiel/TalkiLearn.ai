from typing import List, Tuple, Dict, Any
import re
from pypdf import PdfReader
from sklearn.cluster import KMeans
import numpy as np


class DocumentProcessor:
    """문서 처리 (파싱, 청킹, 토픽 클러스터링)"""

    def __init__(self, chunk_size: int = 800, overlap_ratio: float = 0.2):
        """
        Args:
            chunk_size: 청크 크기 (문자 단위)
            overlap_ratio: 오버랩 비율 (0.2 = 20%)
        """
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)

    def extract_text_from_file(self, file_path: str, file_type: str) -> str:
        """
        파일에서 텍스트 추출

        Args:
            file_path: 파일 경로
            file_type: 파일 타입 (txt, pdf)

        Returns:
            추출된 텍스트
        """
        if file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_type == "pdf":
            text = self._extract_text_from_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return text

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF에서 텍스트 추출"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text

    def chunk_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할 (슬라이딩 윈도우 방식)

        Args:
            text: 원본 텍스트

        Returns:
            청크 리스트
        """
        # 텍스트 정리 (과도한 공백 제거)
        text = re.sub(r'\n\s*\n', '\n\n', text)  # 연속된 빈 줄 정리
        text = re.sub(r' +', ' ', text)  # 연속된 공백 정리

        chunks = []
        start = 0

        while start < len(text):
            # 청크 끝 위치 계산
            end = start + self.chunk_size

            # 텍스트 끝을 넘지 않도록
            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            # 문장 경계에서 자르기 (마침표, 느낌표, 물음표 찾기)
            chunk_text = text[start:end]
            last_sentence_end = max(
                chunk_text.rfind('. '),
                chunk_text.rfind('! '),
                chunk_text.rfind('? '),
                chunk_text.rfind('.\n'),
                chunk_text.rfind('!\n'),
                chunk_text.rfind('?\n')
            )

            if last_sentence_end > 0:
                end = start + last_sentence_end + 1
                chunks.append(text[start:end].strip())
                start = end - self.overlap  # 오버랩 적용
            else:
                # 문장 경계를 못 찾으면 그냥 자르기
                chunks.append(chunk_text.strip())
                start = end - self.overlap

        # 빈 청크 제거
        chunks = [c for c in chunks if len(c.strip()) > 0]

        return chunks

    def cluster_chunks_into_subsessions(
        self,
        chunk_embeddings: List[List[float]],
        min_clusters: int = 3,
        max_clusters: int = 8
    ) -> Tuple[List[int], int]:
        """
        청크 임베딩을 클러스터링하여 서브세션으로 분할

        Args:
            chunk_embeddings: 청크 임베딩 벡터 목록
            min_clusters: 최소 클러스터 수
            max_clusters: 최대 클러스터 수

        Returns:
            (cluster_labels, num_clusters)
            - cluster_labels: 각 청크의 클러스터 레이블
            - num_clusters: 총 클러스터 수
        """
        num_chunks = len(chunk_embeddings)

        # 청크 수가 적으면 클러스터 수 조정
        num_clusters = min(max(min_clusters, num_chunks // 5), max_clusters)
        num_clusters = min(num_clusters, num_chunks)  # 청크 수보다 많을 수 없음

        if num_clusters <= 1:
            # 클러스터링이 의미 없는 경우
            return [0] * num_chunks, 1

        # K-Means 클러스터링
        embeddings_array = np.array(chunk_embeddings)
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(embeddings_array)

        return cluster_labels.tolist(), num_clusters

    def generate_subsession_title(self, chunks: List[str]) -> str:
        """
        서브세션 제목 생성 (첫 문장 또는 키워드 기반)

        Args:
            chunks: 서브세션에 속한 청크 목록

        Returns:
            서브세션 제목
        """
        if not chunks:
            return "Untitled Subsession"

        # 첫 번째 청크의 첫 문장 추출
        first_chunk = chunks[0]
        sentences = re.split(r'[.!?]\s+', first_chunk)

        if sentences:
            title = sentences[0].strip()
            # 너무 긴 제목은 자르기
            if len(title) > 60:
                title = title[:57] + "..."
            return title

        return "Untitled Subsession"


def extract_text_from_bytes(file_bytes: bytes, file_type: str) -> str:
    """
    바이트 데이터에서 텍스트 추출 (Streamlit 업로드 파일용)

    Args:
        file_bytes: 파일 바이트 데이터
        file_type: 파일 타입 (txt, pdf)

    Returns:
        추출된 텍스트
    """
    if file_type == "txt":
        text = file_bytes.decode("utf-8")
    elif file_type == "pdf":
        from io import BytesIO
        reader = PdfReader(BytesIO(file_bytes))
        text = ""
        print(f"📄 PDF has {len(reader.pages)} pages")

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
                print(f"✅ Page {i+1}: extracted {len(page_text)} characters")
            else:
                print(f"⚠️ Page {i+1}: no text extracted")

        print(f"📝 Total extracted text: {len(text)} characters")

        # 텍스트가 충분하지 않으면 OCR 시도
        min_chars_per_page = 100  # 페이지당 최소 100자 기대
        if len(text) < len(reader.pages) * min_chars_per_page:
            print(f"⚠️ Low text density detected. Attempting OCR...")
            ocr_text = _extract_text_with_ocr(file_bytes)
            if ocr_text and len(ocr_text) > len(text):
                print(f"✅ OCR extracted {len(ocr_text)} characters (better than {len(text)})")
                text = ocr_text
            else:
                print(f"❌ OCR failed or produced less text")

        if not text.strip():
            print("❌ WARNING: No text extracted from PDF!")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return text


def _extract_text_with_ocr(pdf_bytes: bytes) -> str:
    """
    OCR을 사용하여 PDF에서 텍스트 추출 (이미지 기반 PDF용)

    Args:
        pdf_bytes: PDF 바이트 데이터

    Returns:
        OCR로 추출된 텍스트
    """
    try:
        from io import BytesIO
        from pdf2image import convert_from_bytes
        import pytesseract

        print("🔍 Starting OCR process...")

        # PDF를 이미지로 변환
        images = convert_from_bytes(pdf_bytes, dpi=200)
        print(f"📷 Converted {len(images)} pages to images")

        text = ""
        for i, image in enumerate(images):
            # OCR 수행
            page_text = pytesseract.image_to_string(image, lang='eng+deu')  # 영어+독일어
            if page_text.strip():
                text += page_text + "\n\n"
                print(f"✅ OCR Page {i+1}: extracted {len(page_text)} characters")
            else:
                print(f"⚠️ OCR Page {i+1}: no text extracted")

            # 메모리 절약을 위해 이미지 해제
            image.close()

        print(f"📝 Total OCR text: {len(text)} characters")
        return text

    except ImportError as e:
        print(f"❌ OCR libraries not installed: {e}")
        print("💡 Install with: pip install pdf2image pytesseract")
        print("💡 Also install tesseract: brew install tesseract (macOS)")
        return ""
    except Exception as e:
        print(f"❌ OCR failed: {str(e)}")
        return ""
