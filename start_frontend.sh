#!/bin/bash

# TalkiLearn 프론트엔드 서버 시작 스크립트

echo "🚀 Starting TalkiLearn Frontend (Streamlit)..."

# 가상환경 활성화 (필요시)
# source tlvenv/bin/activate

# Streamlit 앱 실행
streamlit run frontend/app.py --server.port 8501

echo "✅ Frontend is running on http://localhost:8501"
