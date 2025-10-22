from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ChatMessage(BaseModel):
    """채팅 메시지"""

    role: str = Field(..., description="메시지 역할 (tutor/user)")
    content: str = Field(..., description="메시지 내용")
    timestamp: datetime = Field(default_factory=datetime.now, description="메시지 시간")


class ChatRequest(BaseModel):
    """채팅 요청"""

    subsession_id: int = Field(..., description="서브세션 ID")
    user_msg: str = Field(..., description="사용자 메시지")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="채팅 히스토리")
    current_chunk_index: int = Field(default=0, ge=0, description="현재 진행 중인 청크 인덱스 (0부터 시작)")


class ChatResponse(BaseModel):
    """채팅 응답"""

    explanation: str = Field(..., description="튜터 설명 (3-5문장)")
    prompt_to_user: str = Field(..., description="사용자에게 던질 질문 (1문장)")
    covered_chunk_ids: List[int] = Field(default_factory=list, description="이번에 다룬 청크 ID들")
    is_complete: bool = Field(default=False, description="모든 내용 학습 완료 여부")
    next_chunk_index: int = Field(default=0, description="다음에 학습할 청크 인덱스")
    total_chunks: int = Field(default=0, description="전체 청크 수")

    class Config:
        json_schema_extra = {
            "example": {
                "explanation": "독일어에서 정관사는 성(gender)에 따라 der, die, das로 구분됩니다. 남성명사는 der, 여성명사는 die, 중성명사는 das를 사용하죠. 예를 들어 'der Mann(남자)', 'die Frau(여자)', 'das Kind(아이)'처럼 말이에요.",
                "prompt_to_user": "'책(Buch)'은 중성명사인데, 정관사를 붙이면 어떻게 될까요?",
                "covered_chunk_ids": [1, 2, 3],
                "is_complete": False,
                "next_chunk_index": 1,
                "total_chunks": 5
            }
        }


class QuizQuestion(BaseModel):
    """퀴즈 문제 (3지선다)"""

    question: str = Field(..., description="문제 내용")
    options: List[str] = Field(..., description="선택지 3개", min_length=3, max_length=3)
    correct_answer: int = Field(..., description="정답 인덱스 (0, 1, 2)", ge=0, le=2)
    difficulty: str = Field(default="medium", description="난이도 (easy/medium/hard)")
    chunk_id: int = Field(default=0, description="관련 청크 ID")


class QuizGenerateRequest(BaseModel):
    """퀴즈 생성 요청"""

    subsession_id: int = Field(..., description="서브세션 ID")
    num_questions: int = Field(default=6, ge=5, le=7, description="문제 수 (5-7개)")


class QuizGenerateResponse(BaseModel):
    """퀴즈 생성 응답"""

    questions: List[QuizQuestion] = Field(..., description="생성된 문제 목록")
    subsession_id: int = Field(..., description="서브세션 ID")


class QuizAnswer(BaseModel):
    """사용자 답변 (3지선다)"""

    question_index: int = Field(..., description="문제 번호 (0부터)")
    selected_option: int = Field(..., description="선택한 선택지 인덱스 (0, 1, 2)", ge=0, le=2)


class QuizSubmitRequest(BaseModel):
    """퀴즈 제출 요청"""

    subsession_id: int = Field(..., description="서브세션 ID")
    answers: List[QuizAnswer] = Field(..., description="답변 목록")
    questions: List[QuizQuestion] = Field(..., description="문제 목록 (채점용)")


class QuizResult(BaseModel):
    """개별 문제 결과 (3지선다)"""

    question: str = Field(..., description="문제")
    options: List[str] = Field(..., description="선택지 3개")
    selected_option: int = Field(..., description="선택한 선택지 인덱스")
    correct_answer: int = Field(..., description="정답 인덱스")
    is_correct: bool = Field(..., description="정답 여부")


class QuizSubmitResponse(BaseModel):
    """퀴즈 제출 응답"""

    score: int = Field(..., description="맞은 개수")
    total: int = Field(..., description="전체 문제 수")
    percentage: float = Field(..., description="정답률 (0-100)")
    details: List[QuizResult] = Field(..., description="상세 결과")


class SummaryRequest(BaseModel):
    """요약 생성 요청"""

    subsession_id: int = Field(..., description="서브세션 ID")


class Summary(BaseModel):
    """요약 정보"""

    summary: str = Field(..., description="핵심 개념 요약 (5-7줄)")
    pitfalls: List[str] = Field(default_factory=list, description="혼동 포인트")
    next_topics: List[str] = Field(default_factory=list, description="다음 추천 주제")
    links: List[str] = Field(default_factory=list, description="참고 링크")

    class Config:
        json_schema_extra = {
            "example": {
                "summary": "독일어 정관사는 명사의 성(gender)에 따라 결정됩니다.\n- 남성: der (예: der Mann)\n- 여성: die (예: die Frau)\n- 중성: das (예: das Kind)\n복수형은 모두 die를 사용합니다.",
                "pitfalls": [
                    "영어와 달리 독일어는 모든 명사가 성을 가집니다",
                    "명사의 성은 의미와 무관할 수 있습니다 (예: das Mädchen은 중성이지만 '소녀'를 의미)"
                ],
                "next_topics": [
                    "부정관사 (ein, eine, ein)",
                    "격변화 (Nominativ, Akkusativ, Dativ, Genitiv)"
                ],
                "links": []
            }
        }


class CompleteSubsessionRequest(BaseModel):
    """서브세션 학습 완료 요청"""

    proficiency_increase: float = Field(
        default=20.0,
        ge=0,
        le=100,
        description="숙련도 증가량 (0-100)"
    )
