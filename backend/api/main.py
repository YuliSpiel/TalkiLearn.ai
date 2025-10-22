from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional
import os
import uuid
from datetime import datetime
import json

from ..models import (
    Notebook,
    Session,
    Subsession,
    UserProfile,
    SessionStatus,
    ChatRequest,
    ChatResponse,
    QuizGenerateRequest,
    QuizGenerateResponse,
    QuizSubmitRequest,
    QuizSubmitResponse,
    QuizResult,
    SummaryRequest,
    Summary,
    CompleteSubsessionRequest,
)
from ..services import (
    VectorStoreService,
    EmbeddingService,
    LLMService,
    DocumentProcessor,
    get_embedding_service,
    get_llm_service,
)
from ..utils import get_database

# FastAPI 앱 초기화
app = FastAPI(
    title="TalkiLearn API",
    description="학습보조 챗봇 백엔드 API",
    version="1.0.0"
)

# CORS 설정 (Streamlit 프론트엔드와 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용 - 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 인스턴스 (전역)
vector_store = VectorStoreService()
db = get_database()


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    print("🚀 TalkiLearn API starting up...")
    print("📦 Loading embedding model...")
    # 임베딩 모델 사전 로드
    get_embedding_service()
    print("✅ Embedding model loaded successfully")
    print("🤖 LLM service initialized")
    get_llm_service()
    print("✅ TalkiLearn API ready!")


# ========== Health Check ==========

@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "healthy",
        "service": "TalkiLearn API",
        "version": "1.0.0"
    }


# ========== 1. Notebooks ==========

@app.post("/notebooks", response_model=Notebook)
async def create_notebook(title: str, user_id: str = "default_user"):
    """
    노트북 생성

    Args:
        title: 과목명
        user_id: 사용자 ID
    """
    notebook = db.create_notebook(title=title, user_id=user_id)
    return notebook


@app.get("/notebooks", response_model=List[Notebook])
async def get_notebooks(user_id: str = "default_user"):
    """
    노트북 목록 조회

    Args:
        user_id: 사용자 ID
    """
    notebooks = db.get_all_notebooks(user_id=user_id)
    return notebooks


@app.get("/notebooks/{notebook_id}", response_model=Notebook)
async def get_notebook(notebook_id: int):
    """
    노트북 상세 조회

    Args:
        notebook_id: 노트북 ID
    """
    notebook = db.get_notebook(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return notebook


@app.delete("/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: int):
    """
    노트북 삭제

    Args:
        notebook_id: 노트북 ID
    """
    notebook = db.get_notebook(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    # 노트북의 모든 세션에 대한 벡터 스토어 청크 삭제
    for session in notebook.sessions:
        try:
            vector_store.delete_session_chunks(session.session_id)
        except Exception as e:
            print(f"Warning: Failed to delete chunks for session {session.session_id}: {e}")

    # 데이터베이스에서 노트북 삭제
    success = db.delete_notebook(notebook_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete notebook")

    return {"message": "Notebook deleted successfully", "notebook_id": notebook_id}


# ========== 2. Session Upload ==========

@app.post("/sessions:upload")
async def upload_session(
    notebook_id: int = Form(...),
    file: UploadFile = File(...)
):
    """
    파일 업로드 및 세션 생성

    1. 파일 파싱 (txt/pdf)
    2. 청킹 (size=800, overlap=0.2)
    3. 임베딩 생성
    4. Chroma에 저장
    5. 토픽 클러스터링 → Subsession 분할
    6. 세션 상태를 indexed로 변경

    Args:
        notebook_id: 노트북 ID
        file: 업로드 파일
    """
    # 노트북 존재 확인
    notebook = db.get_notebook(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    # 파일 타입 확인
    filename = file.filename
    file_type = filename.split(".")[-1].lower()
    if file_type not in ["txt", "pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only txt and pdf are allowed.")

    # 세션 생성
    max_session_id = max([s.session_id for s in notebook.sessions], default=0)
    new_session_id = max_session_id + 1

    session = Session(
        session_id=new_session_id,
        notebook_id=notebook_id,
        filename=filename,
        file_type=file_type,
        status=SessionStatus.PROCESSING
    )

    db.add_session_to_notebook(notebook_id, session)

    try:
        # 1. 파일 읽기
        file_bytes = await file.read()

        # 2. 텍스트 추출
        from ..services.document_processor import extract_text_from_bytes
        text = extract_text_from_bytes(file_bytes, file_type)

        # 3. 청킹
        processor = DocumentProcessor(chunk_size=800, overlap_ratio=0.2)
        chunks = processor.chunk_text(text)
        session.total_chunks = len(chunks)

        # 4. 임베딩 생성
        embedding_service = get_embedding_service()
        embeddings = embedding_service.encode(chunks, show_progress=False)

        # 5. 토픽 클러스터링
        cluster_labels, num_clusters = processor.cluster_chunks_into_subsessions(
            embeddings,
            min_clusters=3,
            max_clusters=8
        )

        # 6. Subsession 생성
        subsessions = []
        # 고유 ID 생성: notebook_id와 session_id를 조합하여 전역적으로 고유하게 만듦
        subsession_id_base = notebook_id * 100000 + new_session_id * 1000

        for cluster_idx in range(num_clusters):
            # 해당 클러스터의 청크 인덱스 찾기
            cluster_chunk_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_idx]

            # 청크 ID 리스트
            chunk_ids = list(range(subsession_id_base + cluster_idx * 100, subsession_id_base + cluster_idx * 100 + len(cluster_chunk_indices)))

            # 서브세션 제목 생성 (LLM 사용)
            cluster_chunks = [chunks[i] for i in cluster_chunk_indices]
            llm_service = get_llm_service()
            title = llm_service.generate_subsession_title(cluster_chunks)

            subsession = Subsession(
                subsession_id=subsession_id_base + cluster_idx,
                session_id=new_session_id,
                index=cluster_idx + 1,
                title=title,
                chunk_ids=chunk_ids
            )
            subsessions.append(subsession)

            # 7. Chroma에 청크 저장
            chunk_embeddings = [embeddings[i] for i in cluster_chunk_indices]
            chunk_documents = cluster_chunks
            chunk_metadatas = [
                {
                    "notebook_id": notebook_id,
                    "session_id": new_session_id,
                    "subsession_id": subsession.subsession_id,
                    "chunk_id": chunk_id,
                    "cluster": cluster_idx
                }
                for chunk_id in chunk_ids
            ]
            chunk_id_strings = [str(chunk_id) for chunk_id in chunk_ids]

            vector_store.add_chunks(
                chunks=chunk_documents,
                embeddings=chunk_embeddings,
                metadatas=chunk_metadatas,
                chunk_ids=chunk_id_strings
            )

        # 8. 세션 업데이트
        session.subsessions = subsessions
        session.status = SessionStatus.INDEXED
        session.indexed_at = datetime.now()

        db.update_session(notebook_id, session)

        return {
            "message": "File uploaded and processed successfully",
            "session_id": new_session_id,
            "total_chunks": len(chunks),
            "num_subsessions": num_clusters,
            "subsessions": [
                {
                    "subsession_id": sub.subsession_id,
                    "index": sub.index,
                    "title": sub.title,
                    "num_chunks": len(sub.chunk_ids)
                }
                for sub in subsessions
            ]
        }

    except Exception as e:
        # 에러 발생 시 세션 상태 업데이트
        session.status = SessionStatus.ERROR
        db.update_session(notebook_id, session)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# ========== 3. Learning - Chat ==========

@app.post("/learn/chat/intro")
async def get_subsession_intro(subsession_id: int):
    """
    서브세션 시작 시 AI 튜터의 소개 메시지 생성

    Args:
        subsession_id: 서브세션 ID

    Returns:
        {
            "greeting": str,
            "topic_overview": str,
            "check_question": str,
            "full_message": str
        }
    """
    # 서브세션의 모든 청크 가져오기
    chunks = vector_store.get_all_chunks_by_subsession(subsession_id)

    if not chunks:
        raise HTTPException(status_code=404, detail="No content found for this subsession")

    # 청크 문서와 메타데이터 추출
    chunk_documents = [chunk["document"] for chunk in chunks]

    # 서브세션 제목 (첫 번째 청크의 메타데이터에서 가져오거나 기본값)
    # 실제로는 DB에서 가져와야 하지만 여기서는 간단히 처리
    subsession_title = "학습 주제"  # TODO: DB에서 실제 제목 가져오기

    # LLM으로 소개 메시지 생성
    llm_service = get_llm_service()
    intro = llm_service.generate_subsession_intro(
        subsession_title=subsession_title,
        context_chunks=chunk_documents
    )

    # 전체 메시지 조합
    full_message = f"{intro['greeting']}\n\n{intro['topic_overview']}\n\n{intro['check_question']}"

    return {
        "greeting": intro["greeting"],
        "topic_overview": intro["topic_overview"],
        "check_question": intro["check_question"],
        "full_message": full_message
    }


@app.post("/learn/chat", response_model=ChatResponse)
async def learn_chat(request: ChatRequest):
    """
    채팅 기반 학습

    Args:
        request: ChatRequest (subsession_id, user_msg, chat_history)

    Returns:
        ChatResponse (explanation, prompt_to_user, covered_chunk_ids)
    """
    # 1. 사용자 메시지 임베딩
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.encode_query(request.user_msg)

    # 2. 벡터 검색 (Top-K=6)
    results = vector_store.query_chunks(
        query_embedding=query_embedding,
        subsession_id=request.subsession_id,
        top_k=6
    )

    if not results["documents"]:
        raise HTTPException(status_code=404, detail="No content found for this subsession")

    # 3. LLM 응답 생성
    llm_service = get_llm_service()
    chat_history = [
        {"role": msg.role, "content": msg.content}
        for msg in request.chat_history
    ]

    response = llm_service.generate_chat_response(
        user_msg=request.user_msg,
        context_chunks=results["documents"],
        chat_history=chat_history
    )

    # covered_chunk_ids는 검색된 청크의 ID
    covered_ids = [int(id_str) for id_str in results["ids"]]
    response["covered_chunk_ids"] = covered_ids

    return ChatResponse(**response)


# ========== 4. Learning - Quiz Generate ==========

@app.post("/learn/quiz:generate", response_model=QuizGenerateResponse)
async def generate_quiz(request: QuizGenerateRequest):
    """
    퀴즈 생성

    Args:
        request: QuizGenerateRequest (subsession_id, num_questions)

    Returns:
        QuizGenerateResponse (questions)
    """
    print(f"[DEBUG] Quiz generation requested for subsession_id={request.subsession_id}, num_questions={request.num_questions}")

    # 서브세션의 모든 청크 가져오기
    chunks = vector_store.get_all_chunks_by_subsession(request.subsession_id)

    if not chunks:
        raise HTTPException(status_code=404, detail="No content found for this subsession")

    # 청크 문서 추출
    chunk_documents = [chunk["document"] for chunk in chunks]

    # LLM 퀴즈 생성
    llm_service = get_llm_service()
    questions = llm_service.generate_quiz(
        context_chunks=chunk_documents,
        num_questions=request.num_questions
    )

    # 디버그: 생성된 퀴즈 확인
    print(f"[DEBUG] Generated {len(questions)} questions")
    for i, q in enumerate(questions):
        print(f"[DEBUG] Question {i+1}: {q.get('question', 'NO QUESTION')[:50]}...")
        print(f"[DEBUG]   - Has 'options' key: {'options' in q}")
        if 'options' in q:
            print(f"[DEBUG]   - Number of options: {len(q['options'])}")
            print(f"[DEBUG]   - Options: {q['options']}")
        print(f"[DEBUG]   - correct_answer: {q.get('correct_answer', 'MISSING')}")

    return QuizGenerateResponse(
        questions=questions,
        subsession_id=request.subsession_id
    )


# ========== 5. Learning - Quiz Submit ==========

@app.post("/learn/quiz:submit", response_model=QuizSubmitResponse)
async def submit_quiz(request: QuizSubmitRequest):
    """
    퀴즈 제출 및 채점 (3지선다)

    Args:
        request: QuizSubmitRequest (subsession_id, answers, questions)

    Returns:
        QuizSubmitResponse (score, total, percentage, details)
    """
    results = []
    correct_count = 0

    # 답변 매칭
    answer_map = {ans.question_index: ans.selected_option for ans in request.answers}

    for i, question in enumerate(request.questions):
        selected_option = answer_map.get(i, -1)  # 선택 안한 경우 -1

        # 정답 확인 (선택지 인덱스 비교)
        is_correct = selected_option == question.correct_answer

        if is_correct:
            correct_count += 1

        results.append(QuizResult(
            question=question.question,
            options=question.options,
            selected_option=selected_option,
            correct_answer=question.correct_answer,
            is_correct=is_correct
        ))

    total = len(request.questions)
    percentage = (correct_count / total * 100) if total > 0 else 0

    return QuizSubmitResponse(
        score=correct_count,
        total=total,
        percentage=round(percentage, 2),
        details=results
    )


# ========== 6. Learning - Summary ==========

@app.post("/learn/summary", response_model=Summary)
async def generate_summary(request: SummaryRequest):
    """
    학습 요약 생성

    Args:
        request: SummaryRequest (subsession_id)

    Returns:
        Summary (summary, pitfalls, next_topics, links)
    """
    # 서브세션의 모든 청크 가져오기
    chunks = vector_store.get_all_chunks_by_subsession(request.subsession_id)

    if not chunks:
        raise HTTPException(status_code=404, detail="No content found for this subsession")

    # 청크 문서 추출
    chunk_documents = [chunk["document"] for chunk in chunks]

    # LLM 요약 생성
    llm_service = get_llm_service()
    summary = llm_service.generate_summary(context_chunks=chunk_documents)

    return Summary(**summary)


# ========== 7. Subsession Complete ==========

@app.post("/subsessions/{subsession_id}/complete")
async def complete_subsession(
    subsession_id: int,
    request: CompleteSubsessionRequest
):
    """
    서브세션 학습 완료 처리

    Args:
        subsession_id: 서브세션 ID
        request: CompleteSubsessionRequest (proficiency_increase)

    Returns:
        {
            "message": str,
            "subsession_id": int,
            "new_proficiency": float,
            "new_study_count": int
        }
    """
    # 서브세션 찾기 (notebook_id, session_id 모르는 상태)
    result = db.find_subsession_by_id(subsession_id)
    if not result:
        raise HTTPException(status_code=404, detail="Subsession not found")

    notebook_id, session_id, subsession = result

    # 숙련도 업데이트 (100% 상한선)
    new_proficiency = min(100.0, subsession.proficiency + request.proficiency_increase)
    subsession.proficiency = new_proficiency

    # 학습 횟수 증가
    subsession.study_count += 1

    # 서브세션 업데이트
    db.update_subsession(notebook_id, session_id, subsession)

    # 노트북 통계 업데이트
    notebook = db.get_notebook(notebook_id)
    if notebook:
        # 총 학습 횟수 증가
        notebook.total_study_count += 1

        # 학습 일수 업데이트 (날짜가 바뀌었는지 확인)
        today = datetime.now().date()
        if notebook.last_studied_at:
            # 기존 마지막 학습 날짜 가져오기
            last_date = notebook.last_studied_at.date() if hasattr(notebook.last_studied_at, 'date') else datetime.fromisoformat(str(notebook.last_studied_at)).date()

            # 오늘 처음 학습하는 경우
            if last_date != today:
                notebook.total_study_days += 1

                # 연속 학습 일수 계산
                if (today - last_date).days == 1:
                    # 어제 학습했으면 연속 증가
                    notebook.streak_days += 1
                else:
                    # 중간에 끊겼으면 1로 리셋
                    notebook.streak_days = 1
        else:
            # 첫 학습인 경우
            notebook.total_study_days = 1
            notebook.streak_days = 1

        # 최근 학습 정보 업데이트 (일수 계산 후에 업데이트)
        notebook.last_studied_at = datetime.now()
        notebook.last_session_title = subsession.title

        # 노트북 저장
        db.update_notebook(notebook)

    return {
        "message": "Learning session completed successfully",
        "subsession_id": subsession_id,
        "new_proficiency": new_proficiency,
        "new_study_count": subsession.study_count
    }


# ========== 8. User Profile ==========

@app.post("/profile", response_model=UserProfile)
async def create_profile(profile: UserProfile):
    """프로필 생성/업데이트"""
    db.save_profile(profile)
    return profile


@app.get("/profile", response_model=Optional[UserProfile])
async def get_profile(user_id: str = "default_user"):
    """프로필 조회"""
    profile = db.get_profile(user_id)
    return profile


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
