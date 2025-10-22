from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class SessionStatus(str, Enum):
    """세션 상태"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    INDEXED = "indexed"
    ERROR = "error"


class StudyCycle(BaseModel):
    """학습 사이클 (채팅 -> 퀴즈 -> 요약)"""

    cycle_id: str = Field(..., description="사이클 고유 ID")
    subsession_id: int = Field(..., description="서브세션 ID")
    chat_completed: bool = Field(default=False, description="채팅 학습 완료 여부")
    quiz_completed: bool = Field(default=False, description="퀴즈 완료 여부")
    summary_completed: bool = Field(default=False, description="요약 읽기 완료 여부")
    quiz_score: Optional[float] = Field(default=None, description="퀴즈 점수 (0-100)")
    started_at: datetime = Field(default_factory=datetime.now, description="학습 시작 시간")
    completed_at: Optional[datetime] = Field(default=None, description="학습 완료 시간")


class Subsession(BaseModel):
    """서브세션 (주제별 학습 단위)"""

    subsession_id: int = Field(..., description="서브세션 고유 ID")
    session_id: int = Field(..., description="부모 세션 ID")
    index: int = Field(..., description="서브세션 순서 (1부터 시작)")
    title: str = Field(..., description="서브세션 제목 (자동 생성)")
    chunk_ids: List[int] = Field(default_factory=list, description="포함된 청크 ID 목록")
    covered_chunk_ids: List[int] = Field(default_factory=list, description="학습 완료된 청크 ID")
    proficiency: float = Field(default=0.0, description="숙련도 (0-100)")
    study_count: int = Field(default=0, description="학습 반복 횟수")
    avg_quiz_score: float = Field(default=0.0, description="평균 퀴즈 점수")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")


class Session(BaseModel):
    """세션 (업로드한 파일 단위)"""

    session_id: int = Field(..., description="세션 고유 ID")
    notebook_id: int = Field(..., description="부모 노트북 ID")
    filename: str = Field(..., description="업로드 파일명")
    file_type: str = Field(..., description="파일 타입 (txt, pdf)")
    status: SessionStatus = Field(default=SessionStatus.UPLOADING, description="세션 상태")
    total_chunks: int = Field(default=0, description="총 청크 수")
    subsessions: List[Subsession] = Field(default_factory=list, description="서브세션 목록")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="업로드 시간")
    indexed_at: Optional[datetime] = Field(default=None, description="인덱싱 완료 시간")


class Notebook(BaseModel):
    """노트북 (과목 단위)"""

    notebook_id: int = Field(..., description="노트북 고유 ID")
    title: str = Field(..., description="과목명")
    user_id: str = Field(default="default_user", description="사용자 ID")
    sessions: List[Session] = Field(default_factory=list, description="세션 목록")
    total_study_count: int = Field(default=0, description="총 학습 횟수")
    total_study_days: int = Field(default=0, description="총 학습 일수")
    streak_days: int = Field(default=0, description="연속 학습 일수")
    last_studied_at: Optional[datetime] = Field(default=None, description="마지막 학습 시간")
    last_session_title: Optional[str] = Field(default=None, description="마지막 학습 세션")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")

    def calc_growth_emoji(self) -> str:
        """성장 이모지 계산"""
        score = self.total_study_days + self.streak_days * 0.5
        if score >= 8:
            return "🐓"
        elif score >= 4:
            return "🐥"
        elif score >= 1:
            return "🐣"
        else:
            return "🥚"

    class Config:
        json_schema_extra = {
            "example": {
                "notebook_id": 1,
                "title": "독일어 문법",
                "user_id": "default_user",
                "sessions": [],
                "total_study_count": 5,
                "total_study_days": 3,
                "streak_days": 2,
                "last_studied_at": "2025-10-21T15:30:00",
                "last_session_title": "관사와 격변화",
                "created_at": "2025-10-15T10:00:00"
            }
        }
