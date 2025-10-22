from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class UserProfile(BaseModel):
    """사용자 프로필 모델"""

    user_id: str = Field(default="default_user", description="사용자 ID (기본: 단일 사용자)")
    icon: str = Field(..., description="선택한 아이콘 (이모지 또는 이미지 경로)")
    background_color: str = Field(..., description="배경색 (hex color code)")
    interests: List[str] = Field(default_factory=list, description="관심 주제 목록")
    created_at: datetime = Field(default_factory=datetime.now, description="프로필 생성 시간")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "default_user",
                "icon": "🎓",
                "background_color": "#4A90E2",
                "interests": ["언어 학습", "프로그래밍", "역사"],
                "created_at": "2025-10-21T12:00:00"
            }
        }
