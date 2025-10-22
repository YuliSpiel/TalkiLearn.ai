import streamlit as st
import httpx


def show():
    """온보딩 페이지"""

    # 중앙 정렬을 위한 컬럼 레이아웃
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # 로고 및 제목
        st.markdown("<h1 style='text-align: center;'>🐰 TalkiLearn</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>AI 학습보조 챗봇</h3>", unsafe_allow_html=True)
        st.markdown("---")

        # 앱 소개
        st.markdown("""
        ### 환영합니다!

        **TalkiLearn**은 여러분의 학습을 도와주는 AI 챗봇입니다.

        **주요 기능:**
        - 📚 학습 자료 업로드 (PDF, TXT)
        - 💬 AI 튜터와 대화하며 학습
        - 📝 자동 생성되는 퀴즈로 복습
        - 📊 학습 내용 요약 및 리포트

        **학습 사이클:**
        1. **채팅으로 공부하기** - AI 튜터가 개념을 설명하고 질문합니다
        2. **퀴즈 풀기** - 학습한 내용을 문제로 확인합니다
        3. **요약 읽기** - 핵심 개념을 정리하고 다음 주제를 추천받습니다

        시작하려면 프로필을 만들어주세요!
        """)

        st.markdown("---")

        # 프로필 생성 폼
        with st.form("profile_form"):
            st.subheader("프로필 만들기")

            # 1. 아이콘 선택
            st.write("**1. 아이콘을 선택하세요**")
            icons = ["🎓", "📚", "🦊", "🐱", "🐶", "🐼", "🐨", "🦁", "🐯", "🐸"]
            selected_icon = st.radio(
                "아이콘",
                icons,
                horizontal=True,
                label_visibility="collapsed"
            )

            st.write("")

            # 2. 배경색 선택
            st.write("**2. 배경색을 선택하세요**")
            colors = {
                "파란색": "#4A90E2",
                "초록색": "#7ED321",
                "보라색": "#9013FE",
                "핑크색": "#F5A623",
                "빨간색": "#D0021B"
            }
            selected_color_name = st.radio(
                "배경색",
                list(colors.keys()),
                horizontal=True,
                label_visibility="collapsed"
            )
            selected_color = colors[selected_color_name]

            st.write("")

            # 3. 닉네임 입력
            st.write("**3. 닉네임을 입력하세요**")
            nickname = st.text_input(
                "닉네임",
                placeholder="예: 학습왕, 공부벌레",
                label_visibility="collapsed",
                max_chars=20
            )

            # 제출 버튼
            submitted = st.form_submit_button("프로필 완성하기", use_container_width=True)

            if submitted:
                if not nickname.strip():
                    st.error("닉네임을 입력해주세요.")
                else:
                    # API 호출하여 프로필 저장
                    try:
                        api_url = st.session_state.api_url
                        profile_data = {
                            "user_id": "default_user",
                            "nickname": nickname.strip(),
                            "icon": selected_icon,
                            "background_color": selected_color
                        }

                        with httpx.Client() as client:
                            response = client.post(f"{api_url}/profile", json=profile_data)

                        if response.status_code == 200:
                            st.session_state.user_profile = response.json()
                            st.success("프로필이 생성되었습니다!")
                            st.balloons()

                            # 대시보드로 이동
                            import time
                            time.sleep(1)
                            st.session_state.page = "dashboard"
                            st.rerun()
                        else:
                            st.error(f"프로필 생성에 실패했습니다: {response.text}")

                    except Exception as e:
                        st.error(f"API 연결 오류: {str(e)}")
                        st.info("백엔드 서버가 실행 중인지 확인해주세요 (http://localhost:8000)")
