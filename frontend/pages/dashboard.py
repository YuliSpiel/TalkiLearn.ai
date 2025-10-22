import streamlit as st
import httpx
from datetime import datetime


def calc_growth_emoji(total_days: int, streak: int) -> str:
    """성장 이모지 계산"""
    score = total_days + streak * 0.5
    if score >= 8:
        return "🐓"
    elif score >= 4:
        return "🐥"
    elif score >= 1:
        return "🐣"
    else:
        return "🥚"


def show():
    """대시보드 페이지"""

    # 프로필 확인
    profile = st.session_state.user_profile
    if not profile:
        # 프로필이 없으면 API에서 로드 시도
        try:
            api_url = st.session_state.api_url
            with httpx.Client() as client:
                response = client.get(f"{api_url}/profile")

            if response.status_code == 200:
                st.session_state.user_profile = response.json()
                profile = st.session_state.user_profile
            else:
                st.session_state.page = "onboarding"
                st.rerun()
        except:
            st.session_state.page = "onboarding"
            st.rerun()

    # 헤더
    st.markdown(f"<h1 style='text-align: center;'>🐰 TalkiLearn</h1>", unsafe_allow_html=True)

    # 사용자 프로필 표시
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.markdown(
            f"<div style='text-align: center; font-size: 50px;'>{profile['icon']}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align: center; color: gray;'>관심 주제: {', '.join(profile['interests'])}</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 노트북 목록 조회
    try:
        api_url = st.session_state.api_url
        with httpx.Client() as client:
            response = client.get(f"{api_url}/notebooks")

        if response.status_code == 200:
            notebooks = response.json()
        else:
            st.error("노트북 목록을 불러올 수 없습니다.")
            notebooks = []

    except Exception as e:
        st.error(f"API 연결 오류: {str(e)}")
        notebooks = []

    # 새 노트북 만들기
    with st.expander("📗 새 노트북 만들기", expanded=False):
        with st.form("create_notebook_form"):
            notebook_title = st.text_input("과목명", placeholder="예: 독일어 문법")
            submitted = st.form_submit_button("생성", use_container_width=True)

            if submitted:
                if not notebook_title.strip():
                    st.error("과목명을 입력해주세요.")
                else:
                    try:
                        with httpx.Client() as client:
                            response = client.post(
                                f"{api_url}/notebooks",
                                params={"title": notebook_title.strip()}
                            )

                        if response.status_code == 200:
                            st.success(f"'{notebook_title}' 노트북이 생성되었습니다!")
                            st.rerun()
                        else:
                            st.error(f"생성 실패: {response.text}")

                    except Exception as e:
                        st.error(f"API 오류: {str(e)}")

    st.markdown("---")

    # 노트북 카드 표시 (2열 그리드)
    if notebooks:
        st.subheader("📚 내 노트북")

        # 2열 그리드
        for i in range(0, len(notebooks), 2):
            cols = st.columns(2)

            for j in range(2):
                if i + j < len(notebooks):
                    notebook = notebooks[i + j]

                    with cols[j]:
                        # 성장 이모지 계산
                        growth_emoji = calc_growth_emoji(
                            notebook.get("total_study_days", 0),
                            notebook.get("streak_days", 0)
                        )

                        # 컨테이너로 카드 생성
                        with st.container():
                            # 카드 스타일
                            card_html = f"""
                            <div style="
                                border: 2px solid #e0e0e0;
                                border-radius: 10px;
                                padding: 20px;
                                margin-bottom: 10px;
                                background-color: #f9f9f9;
                            ">
                                <div style="text-align: center; font-size: 60px; margin-bottom: 10px;">
                                    {growth_emoji}
                                </div>
                                <h3 style="text-align: center; margin-bottom: 10px;">
                                    {notebook['title']}
                                </h3>
                                <p style="text-align: center; color: gray; font-size: 14px;">
                                    학습 횟수: {notebook.get('total_study_count', 0)}회
                                </p>
                                <p style="text-align: center; color: gray; font-size: 14px; margin-bottom: 15px;">
                                    최근 학습: {notebook.get('last_session_title', '-')}
                                </p>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)

                            # 카드 내부 버튼들
                            btn_col1, btn_col2 = st.columns(2)

                            with btn_col1:
                                if st.button(
                                    "열기",
                                    key=f"open_{notebook['notebook_id']}",
                                    use_container_width=True,
                                    type="primary"
                                ):
                                    st.session_state.selected_notebook_id = notebook['notebook_id']
                                    st.session_state.page = "notebook_detail"
                                    st.rerun()

                            with btn_col2:
                                if st.button(
                                    "삭제",
                                    key=f"delete_{notebook['notebook_id']}",
                                    use_container_width=True,
                                    type="secondary"
                                ):
                                    # 삭제 확인
                                    st.session_state.delete_notebook_id = notebook['notebook_id']
                                    st.session_state.show_delete_confirm = True
                                    st.rerun()

                        # 삭제 확인 다이얼로그
                        if (st.session_state.get('show_delete_confirm', False) and
                            st.session_state.get('delete_notebook_id') == notebook['notebook_id']):

                            st.warning(f"⚠️ '{notebook['title']}' 노트북을 정말 삭제하시겠습니까?\n\n모든 학습 데이터가 삭제됩니다!")

                            confirm_col1, confirm_col2 = st.columns(2)
                            with confirm_col1:
                                if st.button("✅ 삭제 확인", key=f"confirm_delete_{notebook['notebook_id']}", use_container_width=True):
                                    try:
                                        with httpx.Client() as client:
                                            response = client.delete(f"{api_url}/notebooks/{notebook['notebook_id']}")

                                        if response.status_code == 200:
                                            st.success("노트북이 삭제되었습니다.")
                                            st.session_state.show_delete_confirm = False
                                            st.session_state.delete_notebook_id = None
                                            st.rerun()
                                        else:
                                            st.error(f"삭제 실패: {response.text}")

                                    except Exception as e:
                                        st.error(f"API 오류: {str(e)}")

                            with confirm_col2:
                                if st.button("❌ 취소", key=f"cancel_delete_{notebook['notebook_id']}", use_container_width=True):
                                    st.session_state.show_delete_confirm = False
                                    st.session_state.delete_notebook_id = None
                                    st.rerun()
    else:
        st.info("아직 노트북이 없습니다. 위에서 새 노트북을 만들어보세요!")
