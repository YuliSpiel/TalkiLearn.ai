#!/bin/bash

# TalkiLearn 백엔드 서버 시작 스크립트

echo "🚀 Starting TalkiLearn Backend Server..."

# 가상환경 활성화 (필요시)
# source tlvenv/bin/activate

# FastAPI 서버 실행
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

echo "✅ Backend server is running on http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
