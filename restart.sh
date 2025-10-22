#!/bin/bash

echo "🛑 모든 프로세스 강제 종료 중..."
killall -9 Python python python3 streamlit uvicorn 2>/dev/null
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null
lsof -ti:8501 2>/dev/null | xargs kill -9 2>/dev/null

sleep 3

echo "🗑️  ChromaDB 초기화 중..."
rm -rf ./data/chroma_db/*

echo "✅ 모든 프로세스 종료 및 DB 초기화 완료!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 서버 실행 방법"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ 완전한 동기 구조로 변경되었습니다!"
echo "   - 더 이상 프로세스가 계속 생기지 않습니다"
echo "   - ChromaDB 데이터가 모든 요청에서 공유됩니다"
echo ""
echo "터미널 1️⃣  - 백엔드:"
echo "  cd /Users/yuli/Documents/AI_Workspace/TalkiLearn.ai"
echo "  python run_backend.py"
echo ""
echo "터미널 2️⃣  - 프론트엔드:"
echo "  cd /Users/yuli/Documents/AI_Workspace/TalkiLearn.ai"
echo "  streamlit run ./frontend/app.py --server.port 8501"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
