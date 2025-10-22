# 🐰 TalkiLearn 설치 및 실행 가이드

이 문서는 TalkiLearn을 처음 설치하고 실행하는 방법을 단계별로 안내합니다.

---

## 📋 사전 요구사항

### 필수 항목
1. **Python 3.9 이상**
   ```bash
   python --version  # Python 3.9 이상인지 확인
   ```

2. **OpenAI API Key**
   - OpenAI 계정 생성: https://platform.openai.com/
   - API Key 발급: https://platform.openai.com/api-keys
   - 요금제: gpt-4o-mini 사용 (비용 효율적)

3. **충분한 디스크 공간**
   - 임베딩 모델 다운로드: ~400MB
   - 기타 의존성: ~2GB

### 권장 항목
- Git (버전 관리용)
- VS Code 또는 PyCharm (개발 환경)

---

## 🚀 설치 단계

### Step 1: 프로젝트 클론 또는 다운로드

```bash
# Git을 사용하는 경우
git clone <repository-url>
cd TalkiLearn.ai

# 또는 ZIP 파일 다운로드 후 압축 해제
```

### Step 2: 가상환경 생성

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# macOS/Linux:
source .venv/bin/activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Windows (Command Prompt):
.venv\Scripts\activate.bat
```

가상환경이 활성화되면 터미널 프롬프트 앞에 `(.venv)`가 표시됩니다.

### Step 3: 의존성 설치

```bash
# pip 업그레이드
pip install --upgrade pip

# requirements.txt에서 의존성 설치
pip install -r requirements.txt
```

**주의**:
- PyTorch 설치 시간이 오래 걸릴 수 있습니다 (~5-10분)
- sentence-transformers 첫 실행 시 모델 다운로드가 진행됩니다

### Step 4: 환경 변수 설정

```bash
# .env.example을 config/.env로 복사
cp .env.example config/.env

# config/.env 파일 편집
# macOS/Linux:
nano config/.env
# 또는
vim config/.env

# Windows:
notepad config/.env
```

**config/.env 파일 내용:**
```bash
# OpenAI API Key를 실제 키로 변경하세요
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 나머지 설정은 기본값 사용 가능
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_PORT=8501
DATA_DIR=./data
CHROMA_DB_DIR=./data/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=gpt-4o-mini
CHUNK_SIZE=800
CHUNK_OVERLAP_RATIO=0.2
MIN_CLUSTERS=3
MAX_CLUSTERS=8
```

**중요**: `OPENAI_API_KEY`를 반드시 실제 API 키로 변경해야 합니다!

### Step 5: 데이터 디렉토리 확인

```bash
# 데이터 디렉토리가 자동 생성되었는지 확인
ls -la data/

# 없다면 수동 생성
mkdir -p data/chroma_db data/user_profiles
```

---

## 🎮 실행 방법

### 방법 1: 스크립트 사용 (권장)

**터미널 1 - 백엔드 실행:**
```bash
# 실행 권한 부여 (최초 1회)
chmod +x start_backend.sh

# 백엔드 시작
./start_backend.sh
```

**터미널 2 - 프론트엔드 실행:**
```bash
# 실행 권한 부여 (최초 1회)
chmod +x start_frontend.sh

# 프론트엔드 시작
./start_frontend.sh
```

### 방법 2: 수동 실행

**터미널 1 - 백엔드:**
```bash
ㅊ
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**터미널 2 - 프론트엔드:**
```bash
streamlit run frontend/app.py --server.port 8501
```

---

## ✅ 실행 확인

### 백엔드 확인

1. 터미널에서 다음 메시지를 확인하세요:
   ```
   🚀 TalkiLearn API starting up...
   📦 Loading embedding model...
   ✅ Embedding model loaded successfully
   🤖 LLM service initialized
   ✅ TalkiLearn API ready!
   ```

2. 브라우저에서 접속:
   - API 서버: http://localhost:8000
   - API 문서: http://localhost:8000/docs (Swagger UI)

3. API 헬스 체크:
   ```bash
   curl http://localhost:8000
   ```
   응답: `{"status":"healthy","service":"TalkiLearn API","version":"1.0.0"}`

### 프론트엔드 확인

1. 브라우저가 자동으로 열리면서 http://localhost:8501 접속
2. TalkiLearn 온보딩 페이지가 표시됩니다

---

## 🧪 첫 실행 테스트

### 1. 프로필 생성
1. 온보딩 페이지에서:
   - 아이콘 선택 (예: 🎓)
   - 배경색 선택 (예: 파란색)
   - 관심 주제 입력 (예: "언어 학습, 프로그래밍")
2. "프로필 완성하기" 클릭

### 2. 노트북 생성
1. 대시보드에서 "새 노트북 만들기" 클릭
2. 과목명 입력 (예: "독일어 문법")
3. "생성" 클릭

### 3. 학습 자료 업로드
1. 생성된 노트북 카드의 "열기" 클릭
2. "새 학습 자료 업로드" 섹션 펼치기
3. 텍스트 파일(.txt) 또는 PDF 파일(.pdf) 선택
4. "학습 시작하기" 클릭
5. 처리 완료까지 대기 (1-2분 소요)

### 4. 학습 시작
1. 생성된 서브세션 중 하나의 "시작" 버튼 클릭
2. "채팅으로 공부하기" 탭에서 AI 튜터와 대화
3. "퀴즈 풀기" 탭에서 퀴즈 생성 및 풀이
4. "요약 읽기" 탭에서 학습 내용 요약 확인

---

## 🐛 문제 해결

### 문제 1: ModuleNotFoundError

**증상:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**해결:**
```bash
# 가상환경이 활성화되어 있는지 확인
which python  # .venv/bin/python이어야 함

# 의존성 재설치
pip install -r requirements.txt --upgrade
```

### 문제 2: OpenAI API 오류

**증상:**
```
ValueError: OPENAI_API_KEY not found in environment variables
```

**해결:**
1. `config/.env` 파일이 존재하는지 확인
2. `OPENAI_API_KEY=sk-proj-...` 형식으로 올바르게 설정되었는지 확인
3. API 키에 따옴표가 없어야 합니다 (잘못: `"sk-proj-..."`)

### 문제 3: Chroma 데이터베이스 오류

**증상:**
```
chromadb.errors.ChromaError: ...
```

**해결:**
```bash
# Chroma DB 초기화 (주의: 모든 데이터 삭제됨)
rm -rf data/chroma_db
mkdir -p data/chroma_db

# 백엔드 재시작
```

### 문제 4: 포트가 이미 사용 중

**증상:**
```
ERROR: [Errno 48] Address already in use
```

**해결:**
```bash
# 8000번 포트 사용 중인 프로세스 찾기
lsof -ti:8000

# 프로세스 종료
kill -9 <PID>

# 또는 다른 포트 사용
python -m uvicorn api.main:app --port 8001
```

### 문제 5: 임베딩 모델 다운로드 실패

**증상:**
```
OSError: Can't load model for 'all-MiniLM-L6-v2'
```

**해결:**
```bash
# 인터넷 연결 확인
ping www.google.com

# 모델 캐시 삭제 후 재다운로드
rm -rf ~/.cache/torch/sentence_transformers

# 백엔드 재시작
```

---

## 🔄 업데이트 및 유지보수

### 의존성 업데이트
```bash
pip install -r requirements.txt --upgrade
```

### 데이터 백업
```bash
# 중요 데이터 백업
cp -r data/ data_backup_$(date +%Y%m%d)/
```

### 데이터베이스 초기화 (리셋)
```bash
# 주의: 모든 학습 데이터가 삭제됩니다!
rm -rf data/chroma_db
rm -f data/notebooks.json
rm -f data/study_cycles.json
rm -rf data/user_profiles

# 디렉토리 재생성
mkdir -p data/chroma_db data/user_profiles
```

---

## 📚 추가 학습 자료

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [sentence-transformers 문서](https://www.sbert.net/)
- [OpenAI API 문서](https://platform.openai.com/docs/)

---

## 🎯 다음 단계

1. 다양한 학습 자료를 업로드해보세요 (PDF, TXT)
2. 여러 노트북을 만들어 과목별로 관리해보세요
3. 학습 사이클(채팅→퀴즈→요약)을 완료하며 성장 이모지를 키워보세요!
4. 코드를 커스터마이징하여 자신만의 기능을 추가해보세요

---

**Happy Learning! 🐰📚**

문제가 해결되지 않으면 GitHub Issues에 질문을 남겨주세요!
