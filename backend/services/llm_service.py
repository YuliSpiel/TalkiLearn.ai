from openai import OpenAI
from typing import List, Dict, Any
import json
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv("./config/.env")


class LLMService:
    """LLM 서비스 (OpenAI API 사용)"""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Args:
            model: OpenAI 모델명 (기본: gpt-4o-mini)
        """
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=api_key)

    def generate_chat_response(
        self,
        user_msg: str,
        context_chunks: List[str],
        chat_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        채팅 학습 응답 생성

        Args:
            user_msg: 사용자 메시지
            context_chunks: 검색된 컨텍스트 청크 목록
            chat_history: 채팅 히스토리 (role, content)

        Returns:
            {
                "explanation": str,
                "prompt_to_user": str,
                "covered_chunk_ids": List[int]
            }
        """
        # 컨텍스트 결합
        context = "\n\n".join(context_chunks)

        # 시스템 프롬프트
        system_prompt = f"""당신은 친절하고 전문적인 학습 튜터입니다.
사용자가 업로드한 학습 자료를 바탕으로 학습을 돕습니다.

**역할:**
1. 제공된 컨텍스트를 바탕으로 3-5문장으로 개념을 설명합니다.
2. 설명 마지막에 사용자의 이해를 확인할 수 있는 짧은 질문(1문장)을 제시합니다.
3. 응답은 반드시 JSON 형식으로 출력해야 합니다.

**컨텍스트:**
{context}

**JSON 출력 형식:**
{{
  "explanation": "개념 설명 (3-5문장)",
  "prompt_to_user": "사용자에게 던질 질문 (1문장)"
}}

반드시 위 형식의 JSON만 출력하세요."""

        # 메시지 구성
        messages = [{"role": "system", "content": system_prompt}]

        # 채팅 히스토리 추가 (최근 5개만)
        for msg in chat_history[-5:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # 현재 사용자 메시지 추가
        messages.append({"role": "user", "content": user_msg})

        # LLM 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        # JSON 파싱
        result = json.loads(response.choices[0].message.content)

        return {
            "explanation": result.get("explanation", ""),
            "prompt_to_user": result.get("prompt_to_user", ""),
            "covered_chunk_ids": [],  # 추후 벡터 검색 결과에서 채울 수 있음
        }

    def generate_quiz(
        self, context_chunks: List[str], num_questions: int = 6
    ) -> List[Dict[str, Any]]:
        """
        퀴즈 문제 생성 (3지선다)

        Args:
            context_chunks: 서브세션의 모든 청크
            num_questions: 생성할 문제 수 (5-7개)

        Returns:
            [
                {
                    "question": str,
                    "options": List[str],  # 3개
                    "correct_answer": int,  # 0, 1, 2
                    "difficulty": str,
                    "chunk_id": int
                },
                ...
            ]
        """
        context = "\n\n".join(context_chunks)

        system_prompt = f"""당신은 학습 퀴즈를 생성하는 전문가입니다.
제공된 학습 자료를 바탕으로 **3지선다 객관식 문제**를 생성합니다.

**중요: 각 문제는 반드시 정확히 3개의 선택지를 가져야 합니다!**

**학습 자료:**
{context}

**출력 형식 예시:**
{{
  "questions": [
    {{
      "question": "Large Language Model(LLM)의 주요 특징은 무엇인가요?",
      "options": [
        "대규모 텍스트 데이터로 사전 학습된 모델",
        "작은 데이터셋으로만 학습 가능한 모델",
        "이미지 처리 전용 모델"
      ],
      "correct_answer": 0,
      "difficulty": "easy"
    }},
    {{
      "question": "Transformer 아키텍처의 핵심 메커니즘은?",
      "options": [
        "순환 신경망(RNN)",
        "Self-Attention 메커니즘",
        "합성곱 신경망(CNN)"
      ],
      "correct_answer": 1,
      "difficulty": "medium"
    }}
  ]
}}

**필수 요구사항:**
1. 총 {num_questions}개의 문제를 생성합니다
2. 각 문제는 학습 자료의 핵심 개념을 확인합니다
3. **각 문제마다 정확히 3개의 선택지를 제공합니다** (절대 2개나 4개가 아님!)
4. correct_answer는 0, 1, 2 중 하나의 인덱스입니다 (0=첫 번째 선택지, 1=두 번째, 2=세 번째)
5. 오답 선택지는 그럴듯하지만 명확히 틀린 내용입니다
6. difficulty는 "easy", "medium", "hard" 중 하나입니다

위 형식의 JSON만 출력하세요."""

        # LLM 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"학습 자료를 기반으로 {num_questions}개의 3지선다형 객관식 퀴즈를 만들어주세요. 각 문제는 반드시 정확히 3개의 선택지를 가져야 합니다.",
                },
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        # JSON 파싱
        result = json.loads(response.choices[0].message.content)
        questions = result.get("questions", [])

        # chunk_id 추가 및 유효성 검사
        for i, q in enumerate(questions):
            q["chunk_id"] = i
            # 선택지가 3개인지 확인
            if "options" not in q or len(q["options"]) != 3:
                q["options"] = ["옵션 1", "옵션 2", "옵션 3"]
            # correct_answer가 0-2 범위인지 확인
            if "correct_answer" not in q or not (0 <= q["correct_answer"] <= 2):
                q["correct_answer"] = 0

        return questions[:num_questions]

    def generate_subsession_title(self, chunks: List[str]) -> str:
        """
        서브세션 제목 생성 (LLM 사용)

        Args:
            chunks: 서브세션에 속한 청크 목록

        Returns:
            간결한 서브세션 제목
        """
        if not chunks:
            return "학습 주제"

        # 청크들을 샘플링 (너무 길면 처음 3개만)
        sample_chunks = chunks[:3]
        context = "\n\n".join(sample_chunks)

        # 컨텍스트가 너무 길면 자르기
        if len(context) > 2000:
            context = context[:2000] + "..."

        system_prompt = f"""당신은 학습 자료의 주제를 파악하는 전문가입니다.
제공된 텍스트를 읽고, 이 내용이 다루는 핵심 주제를 **짧고 명확한 제목**으로 만들어주세요.

**요구사항:**
1. 제목은 **5-10단어** 이내로 작성합니다
2. 핵심 개념이나 주제를 명확히 표현합니다
3. 자연스러운 한국어 제목으로 작성합니다
4. "에 대해", "관련" 같은 불필요한 말은 빼고 간결하게 작성합니다

**텍스트:**
{context}

**출력 형식:**
간결한 제목만 출력하세요. 설명이나 추가 텍스트 없이 제목만 반환합니다.

**예시:**
- "Transformer 아키텍처의 이해"
- "Python 리스트 컴프리헨션"
- "HTTP 요청과 응답 처리"
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "위 텍스트의 핵심 주제를 간결한 제목으로 만들어주세요."}
                ],
                temperature=0.5,
                max_tokens=50
            )

            title = response.choices[0].message.content.strip()

            # 따옴표 제거
            title = title.strip('"').strip("'")

            # 너무 긴 제목은 자르기
            if len(title) > 50:
                title = title[:47] + "..."

            return title

        except Exception as e:
            print(f"Warning: Failed to generate title with LLM: {e}")
            # 폴백: 첫 문장 사용
            import re
            first_chunk = chunks[0]
            sentences = re.split(r'[.!?]\s+', first_chunk)
            if sentences:
                fallback_title = sentences[0].strip()[:50]
                return fallback_title
            return "학습 주제"

    def generate_summary(self, context_chunks: List[str]) -> Dict[str, Any]:
        """
        학습 요약 생성

        Args:
            context_chunks: 서브세션의 모든 청크

        Returns:
            {
                "summary": str,
                "pitfalls": List[str],
                "next_topics": List[str],
                "links": List[str]
            }
        """
        context = "\n\n".join(context_chunks)

        system_prompt = f"""당신은 학습 내용을 요약하는 전문가입니다.
제공된 학습 자료를 바탕으로 핵심 개념을 요약합니다.

**요구사항:**
1. 핵심 개념을 반드시 포함하여 요약합니다.
- 이에 대한 예문도 이해를 도울 수 있다면 포함합니다.
2. 학습자가 혼동하기 쉬운 포인트 2-3개를 제시합니다.
3. 다음 학습 주제 2-3개를 추천합니다.
4. 응답은 반드시 JSON 형식으로 출력해야 합니다.
5. 학습 자료에 없는 내용은 포함하지 않습니다.

**학습 자료:**
{context}

**JSON 출력 형식:**
{{
  "summary": "핵심 개념 요약 (5-7줄, 줄바꿈은 \\n 사용)",
  "pitfalls": ["혼동 포인트1", "혼동 포인트2"],
  "next_topics": ["다음 주제1", "다음 주제2"]
}}

반드시 위 형식의 JSON만 출력하세요."""

        # LLM 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "학습 내용을 요약해주세요."},
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
        )

        # JSON 파싱
        result = json.loads(response.choices[0].message.content)

        return {
            "summary": result.get("summary", ""),
            "pitfalls": result.get("pitfalls", []),
            "next_topics": result.get("next_topics", []),
            "links": [],  # 링크는 추후 추가 가능
        }

    def generate_subsession_intro(
        self, subsession_title: str, context_chunks: List[str], user_nickname: str = "학습자"
    ) -> Dict[str, Any]:
        """
        서브세션 시작 시 AI 튜터의 소개 메시지 생성

        Args:
            subsession_title: 서브세션 제목
            context_chunks: 서브세션의 학습 내용 청크들
            user_nickname: 사용자 닉네임

        Returns:
            {
                "greeting": str,
                "topic_overview": str,
                "check_question": str
            }
        """
        # 컨텍스트 결합 (처음 몇 개만 사용)
        context = "\n\n".join(context_chunks[:3])  # 처음 3개 청크만 사용

        system_prompt = f"""당신은 친절하고 전문적인 학습 튜터입니다.
새로운 학습 주제를 시작하는 학생에게 인사하고 오늘 배울 내용을 소개합니다.

**학생 정보:**
- 닉네임: {user_nickname}

**역할:**
1. 학생의 닉네임을 사용하여 따뜻하게 인사하고 오늘 배울 주제를 소개합니다.
2. 이번 세션에서 배울 내용을 3~5문장으로 간략히 요약합니다.
3. 학생이 이 주제에 대해 얼마나 알고 있는지 확인하는 친근한 질문을 던집니다.
4. 응답은 반드시 JSON 형식으로 출력해야 합니다.

**주제:** {subsession_title}

**학습 내용:**
{context}

**JSON 출력 형식:**
{{
  "greeting": "닉네임을 포함한 인사말 (1-2문장, 예: '{user_nickname}님, 안녕하세요!')",
  "topic_overview": "오늘 배울 내용 요약 (2-3문장)",
  "check_question": "사전 지식 확인 질문 (1문장, 예: '이 주제에 대해 들어본 적 있나요?', '어느 정도 알고 계신가요?')"
}}

반드시 위 형식의 JSON만 출력하세요. 친근하고 격려하는 톤을 사용하세요."""

        # LLM 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"'{subsession_title}' 주제로 학습을 시작하려고 합니다. 학생에게 인사하고 소개해주세요.",
                },
            ],
            temperature=0.8,  # 좀 더 자연스러운 대화를 위해
            response_format={"type": "json_object"},
        )

        # JSON 파싱
        result = json.loads(response.choices[0].message.content)

        return {
            "greeting": result.get("greeting", "안녕하세요! 함께 공부해볼까요?"),
            "topic_overview": result.get("topic_overview", ""),
            "check_question": result.get(
                "check_question", "이 주제에 대해 들어본 적 있나요?"
            ),
        }

    @staticmethod
    def normalize_answer(text: str) -> str:
        """
        답변 정규화 (채점용)

        Args:
            text: 원본 텍스트

        Returns:
            정규화된 텍스트 (소문자, 특수문자 제거)
        """
        import re

        # 특수문자 제거, 소문자 변환
        normalized = re.sub(r"[^0-9a-z가-힣]", "", text.lower())
        return normalized.strip()


# 싱글톤 인스턴스
_llm_service = None


def get_llm_service() -> LLMService:
    """LLM 서비스 싱글톤 인스턴스 반환"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
