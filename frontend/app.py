import streamlit as st
import sys
import os

# 백엔드 모듈 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 페이지 설정
st.set_page_config(
    page_title="TalkiLearn",
    page_icon="🐰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 세션 상태 초기화
if "page" not in st.session_state:
    st.session_state.page = "onboarding"

if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

if "selected_notebook_id" not in st.session_state:
    st.session_state.selected_notebook_id = None

if "selected_subsession_id" not in st.session_state:
    st.session_state.selected_subsession_id = None

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"


# 페이지 네비게이션 함수
def navigate_to(page_name: str):
    """페이지 이동"""
    st.session_state.page = page_name
    st.rerun()


# 페이지 라우팅
from pages import onboarding, dashboard, notebook_detail, learning_session

page = st.session_state.page

if page == "onboarding":
    if st.session_state.user_profile is None:
        onboarding.show()
    else:
        navigate_to("dashboard")

elif page == "dashboard":
    dashboard.show()

elif page == "notebook_detail":
    notebook_detail.show()

elif page == "learning_session":
    learning_session.show()

else:
    st.error("Unknown page")
