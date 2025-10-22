import streamlit as st
import httpx
import json


def show():
    """노트북 상세 페이지"""

    notebook_id = st.session_state.selected_notebook_id
    if not notebook_id:
        st.error("노트북이 선택되지 않았습니다.")
        if st.button("대시보드로 돌아가기"):
            st.session_state.page = "dashboard"
            st.rerun()
        return

    # 노트북 정보 조회
    try:
        api_url = st.session_state.api_url
        with httpx.Client() as client:
            response = client.get(f"{api_url}/notebooks/{notebook_id}")

        if response.status_code == 200:
            notebook = response.json()
        else:
            st.error("노트북을 불러올 수 없습니다.")
            return

    except Exception as e:
        st.error(f"API 연결 오류: {str(e)}")
        return

    # 헤더
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← 대시보드"):
            st.session_state.page = "dashboard"
            st.rerun()

    with col2:
        # 성장 이모지 계산
        from .dashboard import calc_growth_emoji
        growth_emoji = calc_growth_emoji(
            notebook.get("total_study_days", 0),
            notebook.get("streak_days", 0)
        )

        st.markdown(
            f"<h2 style='text-align: center;'>{growth_emoji} {notebook['title']}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align: center; color: gray;'>학습 횟수: {notebook.get('total_study_count', 0)}회</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 세션 생성 컨테이너
    with st.expander("📄 새 학습 자료 업로드", expanded=True):
        st.write("**지원 파일:** .txt, .pdf")

        uploaded_file = st.file_uploader(
            "파일을 선택하세요",
            type=["txt", "pdf"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            if st.button("학습 시작하기", use_container_width=True, type="primary"):
                # 진행률 표시 컨테이너
                progress_container = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # multipart/form-data로 파일 업로드 (스트리밍)
                    with httpx.Client(timeout=600.0) as client:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        data = {"notebook_id": notebook_id}

                        with client.stream(
                            "POST",
                            f"{api_url}/sessions:upload-stream",
                            files=files,
                            data=data
                        ) as response:
                            if response.status_code == 200:
                                # Server-Sent Events 파싱
                                for line in response.iter_lines():
                                    if line.startswith("data: "):
                                        data_str = line[6:]  # "data: " 제거
                                        try:
                                            event_data = json.loads(data_str)

                                            # 진행률 업데이트
                                            progress = event_data.get("progress", 0)
                                            message = event_data.get("message", "")
                                            stage = event_data.get("stage", "")

                                            progress_bar.progress(progress)
                                            status_text.text(f"{progress}% - {message}")

                                            # 완료 처리
                                            if stage == "complete":
                                                result = event_data
                                                progress_container.success("✅ 파일이 처리되었습니다!")
                                                status_text.info(
                                                    f"총 {result['total_chunks']}개 청크, "
                                                    f"{result['num_subsessions']}개 서브세션으로 분할되었습니다."
                                                )
                                                st.balloons()
                                                st.rerun()
                                                break

                                            # 에러 처리
                                            elif stage == "error":
                                                error_msg = event_data.get("message", "알 수 없는 오류")
                                                st.error(f"업로드 실패: {error_msg}")
                                                break

                                        except json.JSONDecodeError:
                                            continue
                            else:
                                st.error(f"업로드 실패: {response.text}")

                except Exception as e:
                    st.error(f"업로드 오류: {str(e)}")

    st.markdown("---")

    # 세션 리스트
    sessions = notebook.get("sessions", [])

    if sessions:
        st.subheader("📚 학습 세션")

        # 세션별로 표시 (토글 방식)
        for session in sessions:
            session_id = session["session_id"]
            filename = session["filename"]
            status = session["status"]
            subsessions = session.get("subsessions", [])

            # 세션 상태 아이콘
            status_icon = {
                "uploading": "⏳",
                "processing": "⚙️",
                "indexed": "✅",
                "error": "❌"
            }.get(status, "❓")

            # 세션 토글 키
            toggle_key = f"session_{session_id}_expanded"
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = True  # 초기에는 펼쳐진 상태

            # 세션 헤더 (클릭하면 토글)
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"### {status_icon} {filename}")
                st.caption(f"서브세션: {len(subsessions)}개 | 총 청크: {session.get('total_chunks', 0)}개")

            with col2:
                if st.button(
                    "접기" if st.session_state[toggle_key] else "펼치기",
                    key=f"toggle_{session_id}"
                ):
                    st.session_state[toggle_key] = not st.session_state[toggle_key]
                    st.rerun()

            # 서브세션 리스트 (펼쳐진 상태일 때만)
            if st.session_state[toggle_key] and subsessions:
                for subsession in subsessions:
                    subsession_id = subsession["subsession_id"]
                    index = subsession["index"]
                    title = subsession["title"]
                    proficiency = subsession.get("proficiency", 0)
                    study_count = subsession.get("study_count", 0)

                    # 숙련도 표시 (진행 바)
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"**{index}. {title}**")
                        st.progress(proficiency / 100.0)
                        st.caption(f"숙련도: {proficiency:.0f}% | 학습 횟수: {study_count}회")

                    with col2:
                        if st.button("시작", key=f"start_{subsession_id}"):
                            st.session_state.selected_subsession_id = subsession_id
                            st.session_state.selected_notebook_id = notebook_id
                            st.session_state.selected_session_id = session_id
                            st.session_state.page = "learning_session"
                            st.rerun()

                st.markdown("---")

    else:
        st.info("아직 업로드된 학습 자료가 없습니다. 위에서 파일을 업로드해보세요!")
