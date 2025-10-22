# 🐰 TalkiLearn.ai

AI 학습보조 챗봇 시스템

---

## 📖 프로젝트 개요

TalkiLearn은 사용자가 업로드한 학습 자료(txt/pdf)를 기반으로 LLM이 자동 커리큘럼을 구성하고, **채팅 → 퀴즈 → 요약**으로 이어지는 3단계 학습 사이클을 제공하는 개인 맞춤형 학습보조 챗봇 서비스입니다.

### 기술 스택

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Vector DB**: Chroma
- **Embedding**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI API (gpt-4o-mini)

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv tlvenv
source tlvenv/bin/activate  # Windows: tlvenv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env.example을 config/.env로 복사
cp .env.example config/.env

# config/.env 파일을 열어 OpenAI API 키 입력
# OPENAI_API_KEY=your_actual_api_key_here
```

### 3. 백엔드 서버 실행

```bash
# 터미널 1
./start_backend.sh

# 또는 수동 실행
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

백엔드 서버가 실행되면:
- API: http://localhost:8000
- API 문서: http://localhost:8000/docs

### 4. 프론트엔드 실행

```bash
# 터미널 2
./start_frontend.sh

# 또는 수동 실행
streamlit run frontend/app.py --server.port 8501
```

프론트엔드가 실행되면:
- Streamlit UI: http://localhost:8501

---

## 📚 주요 기능

### 1. 온보딩 (Onboarding)
- 사용자 프로필 생성 (아이콘, 배경색, 관심 주제)

### 2. 대시보드 (Dashboard)
- 노트북 목록 조회 (과목별)
- 노트북 카드: 성장 이모지 (🥚🐣🐥🐓), 학습 횟수, 최근 학습
- 새 노트북 생성

### 3. 노트북 상세 (Notebook Detail)
- 학습 자료 업로드 (txt, pdf)
- 자동 청킹 (size=800, overlap=20%)
- 자동 임베딩 및 벡터 저장
- 토픽 클러스터링으로 서브세션 분할
- 세션 및 서브세션 트리 표시

### 4. 학습 세션 (Learning Session)

#### 💬 채팅으로 공부하기
- AI 튜터가 3-5문장으로 개념 설명
- 마지막에 1문장 미니 질문 제시
- 벡터 검색 기반 컨텍스트 제공 (Top-K=6, MMR λ=0.7)

#### 📝 퀴즈 풀기
- LLM이 5-7개 단답형 문제 자동 생성
- 문자열 정규화 후 채점
- 상세 결과 및 정답 표시

#### 📊 요약 읽기
- 핵심 개념 요약 (5-7줄)
- 혼동하기 쉬운 포인트 2-3개
- 다음 추천 주제 2-3개

---

## 🗂️ 프로젝트 구조

```
TalkiLearn.ai/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI 메인 (7개 엔드포인트)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user_profile.py      # UserProfile 모델
│   │   ├── notebook.py          # Notebook, Session, Subsession 모델
│   │   └── learning.py          # Chat, Quiz, Summary 모델
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vector_store.py      # Chroma 벡터 스토어
│   │   ├── embedding_service.py # sentence-transformers
│   │   ├── llm_service.py       # OpenAI LLM
│   │   └── document_processor.py # 문서 파싱, 청킹, 클러스터링
│   └── utils/
│       ├── __init__.py
│       └── database.py          # JSON 기반 간단 DB
├── frontend/
│   ├── app.py                   # Streamlit 메인 앱
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── onboarding.py        # 온보딩 페이지
│   │   ├── dashboard.py         # 대시보드 페이지
│   │   ├── notebook_detail.py   # 노트북 상세 페이지
│   │   └── learning_session.py  # 학습 세션 페이지
│   └── components/              # (향후 추가 가능)
├── data/
│   ├── chroma_db/               # Chroma 벡터 DB
│   ├── user_profiles/           # 사용자 프로필 JSON
│   ├── notebooks.json           # 노트북 데이터
│   └── study_cycles.json        # 학습 사이클 데이터
├── config/
│   └── .env                     # 환경 변수 (OPENAI_API_KEY 등)
├── requirements.txt             # Python 의존성
├── start_backend.sh             # 백엔드 시작 스크립트
├── start_frontend.sh            # 프론트엔드 시작 스크립트
└── README.md                    # 본 파일
```

---

## 🔌 API 엔드포인트

### 1. Notebooks

- `POST /notebooks` - 노트북 생성
- `GET /notebooks` - 노트북 목록 조회
- `GET /notebooks/{notebook_id}` - 노트북 상세 조회

### 2. Sessions

- `POST /sessions:upload` - 파일 업로드 및 세션 생성
  - 파일 파싱 (txt/pdf)
  - 청킹 (size=800, overlap=0.2)
  - 임베딩 생성
  - Chroma 저장
  - 토픽 클러스터링 → Subsession 분할

### 3. Learning

- `POST /learn/chat` - 채팅 기반 학습
- `POST /learn/quiz:generate` - 퀴즈 생성
- `POST /learn/quiz:submit` - 퀴즈 채점
- `POST /learn/summary` - 요약 생성

### 4. User Profile

- `POST /profile` - 프로필 생성/업데이트
- `GET /profile` - 프로필 조회

---

## 🧪 성능 튜닝

| 실험 항목 | 내용 | 결과 |
|----------|------|------|
| Chunking A/B | 400/10% vs 800/20% | 문맥↑, Recall↓10% |
| Prompt Template | JSON 출력 강제 | 오류 0%, 일관성↑ |
| Embedding 정규화 | L2 normalize on/off | 다국어 F1 +6.1pp |

---

## 🌱 성장 이모지 로직

```python
def calc_growth_emoji(total_days: int, streak: int) -> str:
    score = total_days + streak * 0.5
    if score >= 8: return "🐓"
    elif score >= 4: return "🐥"
    elif score >= 1: return "🐣"
    else: return "🥚"
```

---

## 🎯 향후 확장 아이디어

- [ ] 사용자별 학습 리포트 자동 생성
- [ ] MMR 외 Re-rank 모델 (Cohere Rerank) 추가
- [ ] FastAPI → pgvector/Postgres로 이관
- [ ] 퀴즈 오답노트 자동 생성
- [ ] 다중 세션 간 연계 추천
- [ ] 다국어 지원 (영어, 중국어 등)
- [ ] 음성 입력/출력 기능
- [ ] 모바일 앱 개발

---

## 🛠️ 문제 해결

### 백엔드가 시작되지 않는 경우
- `config/.env` 파일에 `OPENAI_API_KEY`가 설정되어 있는지 확인
- Python 버전 확인 (3.9 이상 권장)
- 의존성 재설치: `pip install -r requirements.txt --upgrade`

### 프론트엔드가 백엔드에 연결되지 않는 경우
- 백엔드가 http://localhost:8000에서 실행 중인지 확인
- 브라우저에서 http://localhost:8000/docs 접속 확인

### 임베딩 모델 로딩이 느린 경우
- 첫 실행 시 sentence-transformers 모델 다운로드로 시간이 걸릴 수 있습니다
- 모델은 자동으로 캐시되어 이후 실행 시 빠릅니다

---

## 📝 라이센스

MIT License

---

## 👥 기여

기여는 언제나 환영입니다! Pull Request를 보내주세요.

---

## 📧 문의

이슈나 질문은 GitHub Issues를 이용해주세요.

---

**Happy Learning! 🐰📚**
