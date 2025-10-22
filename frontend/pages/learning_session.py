import streamlit as st
import httpx
from datetime import datetime


def show():
    """학습 세션 페이지 (채팅/퀴즈/요약)"""

    subsession_id = st.session_state.selected_subsession_id
    if not subsession_id:
        st.error("서브세션이 선택되지 않았습니다.")
        if st.button("노트북으로 돌아가기"):
            st.session_state.page = "notebook_detail"
            st.rerun()
        return

    # 세션 상태 초기화 (서브세션별로 독립적으로 관리)
    if "current_subsession_id" not in st.session_state:
        st.session_state.current_subsession_id = None

    if "current_learning_notebook_id" not in st.session_state:
        st.session_state.current_learning_notebook_id = None

    # 노트북 또는 서브세션이 변경되었으면 모든 상태 초기화
    notebook_changed = st.session_state.current_learning_notebook_id != st.session_state.selected_notebook_id
    subsession_changed = st.session_state.current_subsession_id != subsession_id

    if notebook_changed or subsession_changed:
        st.session_state.current_learning_notebook_id = st.session_state.selected_notebook_id
        st.session_state.current_subsession_id = subsession_id
        st.session_state.chat_history = []
        st.session_state.quiz_questions = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_result = None
        st.session_state.summary_data = None
        st.session_state.learning_stage = "chat"  # 학습 단계도 초기화
        st.session_state.waiting_for_response = False  # 대기 상태도 초기화

    # 기존 세션 상태 초기화 (첫 실행 시)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = None

    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}

    if "quiz_result" not in st.session_state:
        st.session_state.quiz_result = None

    if "summary_data" not in st.session_state:
        st.session_state.summary_data = None

    # 헤더
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← 노트북"):
            st.session_state.page = "notebook_detail"
            st.rerun()

    with col2:
        st.markdown("<h2>📙 학습 세션</h2>", unsafe_allow_html=True)

    st.markdown("---")

    # 학습 단계 추적 (채팅 -> 퀴즈 -> 요약)
    if "learning_stage" not in st.session_state:
        st.session_state.learning_stage = "chat"  # chat, quiz, summary

    # 진행 상황 표시
    stages = ["💬 채팅", "📝 퀴즈", "📊 요약"]
    current_idx = {"chat": 0, "quiz": 1, "summary": 2}[st.session_state.learning_stage]

    cols = st.columns(3)
    for i, stage in enumerate(stages):
        with cols[i]:
            if i < current_idx:
                st.markdown(f"**✅ {stage}**")
            elif i == current_idx:
                st.markdown(f"**🔵 {stage}**")
            else:
                st.markdown(f"⚪ {stage}")

    st.markdown("---")

    # 학습 단계별 콘텐츠 표시
    if st.session_state.learning_stage == "chat":
        show_chat_tab(subsession_id)
    elif st.session_state.learning_stage == "quiz":
        show_quiz_tab(subsession_id)
    elif st.session_state.learning_stage == "summary":
        show_summary_tab(subsession_id)


def show_chat_tab(subsession_id: int):
    """채팅 탭"""
    st.subheader("AI 튜터와 대화하며 학습하세요")

    # 사용자 프로필에서 아이콘 가져오기
    user_icon = st.session_state.user_profile.get("icon", "🎓")

    # 첫 진입 시 AI 튜터가 먼저 인사 (채팅 히스토리가 비어있는 경우)
    if len(st.session_state.chat_history) == 0:
        with st.spinner("AI 튜터가 준비 중입니다..."):
            try:
                api_url = st.session_state.api_url
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(
                        f"{api_url}/learn/chat/intro",
                        params={"subsession_id": subsession_id}
                    )

                if response.status_code == 200:
                    intro_data = response.json()
                    full_message = intro_data.get("full_message", "")

                    # 튜터의 첫 인사 추가
                    st.session_state.chat_history.append({
                        "role": "tutor",
                        "content": full_message,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.rerun()

            except Exception as e:
                st.error(f"AI 튜터 로딩 오류: {str(e)}")
                # 기본 인사말 추가
                st.session_state.chat_history.append({
                    "role": "tutor",
                    "content": "안녕하세요! 함께 공부해볼까요? 이 주제에 대해 어느 정도 알고 계신가요?",
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()

    # 채팅 히스토리 표시
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]

            if role == "tutor":
                # 튜터 메시지 (왼쪽)
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.markdown("🤖")
                with col2:
                    st.markdown(
                        f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 15px;'>{content}</div>",
                        unsafe_allow_html=True
                    )

            else:  # user
                # 사용자 메시지 (오른쪽)
                col1, col2, col3 = st.columns([1, 4, 1])
                with col2:
                    st.markdown(
                        f"<div style='background-color: #d4edff; padding: 10px; border-radius: 10px; text-align: right; margin-bottom: 15px;'>{content}</div>",
                        unsafe_allow_html=True
                    )
                with col3:
                    st.markdown(user_icon)

            # 메시지 간 간격 추가
            st.write("")

    st.markdown("---")

    # 사용자 입력
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "메시지를 입력하세요",
            placeholder="질문하거나 대답해보세요...",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("전송", use_container_width=True)

        if submitted and user_input.strip():
            # 사용자 메시지를 즉시 히스토리에 추가하고 화면에 표시
            user_message = user_input.strip()
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })

            # 사용자 입력이 대기 중임을 표시
            st.session_state.waiting_for_response = True
            st.rerun()  # 사용자 메시지를 즉시 화면에 표시

    # API 호출 (사용자 메시지 표시 후)
    if st.session_state.get('waiting_for_response', False):
        st.session_state.waiting_for_response = False

        # 마지막 사용자 메시지 가져오기
        user_messages = [msg for msg in st.session_state.chat_history if msg["role"] == "user"]
        if user_messages:
            last_user_msg = user_messages[-1]["content"]

            with st.spinner("AI 튜터가 답변을 생성하는 중..."):
                try:
                    api_url = st.session_state.api_url

                    # ChatRequest 생성 (role을 OpenAI 호환 형식으로 변환)
                    chat_history_for_api = []
                    for msg in st.session_state.chat_history[-10:]:
                        role = msg["role"]
                        # 'tutor'를 'assistant'로 변환
                        if role == "tutor":
                            role = "assistant"
                        chat_history_for_api.append({
                            "role": role,
                            "content": msg["content"]
                        })

                    request_data = {
                        "subsession_id": subsession_id,
                        "user_msg": last_user_msg,
                        "chat_history": chat_history_for_api
                    }

                    with httpx.Client(timeout=60.0) as client:
                        response = client.post(f"{api_url}/learn/chat", json=request_data)

                    if response.status_code == 200:
                        result = response.json()
                        explanation = result.get("explanation", "")
                        prompt_to_user = result.get("prompt_to_user", "")

                        # 튜터 응답 추가
                        tutor_response = f"{explanation}\n\n**질문:** {prompt_to_user}"
                        st.session_state.chat_history.append({
                            "role": "tutor",
                            "content": tutor_response,
                            "timestamp": datetime.now().isoformat()
                        })

                        st.rerun()

                    else:
                        st.error(f"API 오류: {response.text}")

                except Exception as e:
                    st.error(f"채팅 오류: {str(e)}")

    # 채팅 종료 버튼 (3회 이상 대화 후에만 활성화)
    # 사용자 메시지 수를 세기 (AI의 첫 인사는 제외)
    user_message_count = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])

    if user_message_count >= 3:
        st.markdown("---")
        if st.button("✅ 채팅 종료하고 퀴즈 풀기", type="primary", use_container_width=True):
            st.session_state.learning_stage = "quiz"
            st.rerun()
    elif user_message_count > 0:
        st.markdown("---")
        st.info(f"💬 AI 튜터와 {3 - user_message_count}회 더 대화하면 퀴즈를 풀 수 있습니다!")


def show_quiz_tab(subsession_id: int):
    """퀴즈 탭"""
    st.subheader("학습한 내용을 퀴즈로 확인하세요")

    # 퀴즈 생성
    if st.session_state.quiz_questions is None:
        if st.button("퀴즈 생성", use_container_width=True, type="primary"):
            with st.spinner("퀴즈를 생성하는 중..."):
                try:
                    api_url = st.session_state.api_url

                    request_data = {
                        "subsession_id": subsession_id,
                        "num_questions": 6
                    }

                    with httpx.Client(timeout=60.0) as client:
                        response = client.post(f"{api_url}/learn/quiz:generate", json=request_data)

                    if response.status_code == 200:
                        result = response.json()
                        questions = result["questions"]

                        # 디버그: 받은 데이터 확인
                        st.write(f"DEBUG: Received {len(questions)} questions")
                        for i, q in enumerate(questions):
                            st.write(f"DEBUG Question {i+1}: Has 'options' = {'options' in q}")
                            if 'options' in q:
                                st.write(f"  Number of options: {len(q['options'])}")

                        st.session_state.quiz_questions = questions
                        st.rerun()
                    else:
                        st.error(f"퀴즈 생성 실패: {response.text}")

                except Exception as e:
                    st.error(f"퀴즈 생성 오류: {str(e)}")

    # 퀴즈 문제 표시 (3지선다)
    elif st.session_state.quiz_result is None:
        questions = st.session_state.quiz_questions

        st.write(f"**총 {len(questions)}개 문제**")
        st.markdown("---")

        # 문제 표시 및 답변 선택
        with st.form("quiz_form"):
            for i, q in enumerate(questions):
                st.markdown(f"### 문제 {i+1}")
                st.write(q['question'])
                st.write("")

                # 3지선다 라디오 버튼
                options = q.get('options', ['옵션 1', '옵션 2', '옵션 3'])
                selected = st.radio(
                    f"답을 선택하세요 (문제 {i+1})",
                    options=options,
                    key=f"quiz_{i}",
                    label_visibility="collapsed",
                    index=None  # 초기 선택 없음
                )

                # 선택한 옵션의 인덱스 저장
                if selected is not None:
                    st.session_state.quiz_answers[i] = options.index(selected)
                else:
                    st.session_state.quiz_answers[i] = -1  # 선택 안함

                st.markdown("---")

            submitted = st.form_submit_button("제출", use_container_width=True, type="primary")

            if submitted:
                # 모든 문제에 답했는지 확인
                unanswered = [i+1 for i, ans in st.session_state.quiz_answers.items() if ans == -1]
                if unanswered:
                    st.error(f"아직 답하지 않은 문제가 있습니다: 문제 {', '.join(map(str, unanswered))}")
                else:
                    # 답안 제출
                    try:
                        api_url = st.session_state.api_url

                        # 답변 리스트 생성 (3지선다)
                        answers = [
                            {
                                "question_index": i,
                                "selected_option": st.session_state.quiz_answers.get(i, -1)
                            }
                            for i in range(len(questions))
                        ]

                        request_data = {
                            "subsession_id": subsession_id,
                            "answers": answers,
                            "questions": questions
                        }

                        with httpx.Client(timeout=30.0) as client:
                            response = client.post(f"{api_url}/learn/quiz:submit", json=request_data)

                        if response.status_code == 200:
                            st.session_state.quiz_result = response.json()
                            st.rerun()
                        else:
                            st.error(f"채점 실패: {response.text}")

                    except Exception as e:
                        st.error(f"채점 오류: {str(e)}")

    # 퀴즈 결과 표시
    else:
        result = st.session_state.quiz_result
        score = result["score"]
        total = result["total"]
        percentage = result["percentage"]

        # 점수 표시
        if percentage >= 60:
            st.success(f"🎉 점수: {score}/{total} ({percentage:.1f}%) - 합격!")
        else:
            st.error(f"📊 점수: {score}/{total} ({percentage:.1f}%) - 조금 더 공부가 필요해요")

        st.markdown("---")
        st.subheader("상세 결과")

        # 각 문제별 결과 (3지선다)
        for idx, detail in enumerate(result["details"]):
            question = detail["question"]
            options = detail["options"]
            selected_option = detail["selected_option"]
            correct_answer = detail["correct_answer"]
            is_correct = detail["is_correct"]

            if is_correct:
                st.success(f"✅ **문제 {idx+1}: {question}**")
                st.write(f"✓ 정답: **{options[correct_answer]}**")
            else:
                st.error(f"❌ **문제 {idx+1}: {question}**")
                if selected_option >= 0 and selected_option < len(options):
                    st.write(f"당신의 답: {options[selected_option]}")
                else:
                    st.write(f"당신의 답: (선택하지 않음)")
                st.write(f"✓ 정답: **{options[correct_answer]}**")

            st.markdown("---")

        # 합격/불합격에 따른 버튼
        if percentage >= 60:
            # 합격: 요약으로 이동 버튼
            st.markdown("---")
            if st.button("✅ 요약 학습으로 이동", type="primary", use_container_width=True):
                st.session_state.learning_stage = "summary"
                st.rerun()
        else:
            # 불합격: 다시 풀기 버튼
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 다시 풀기", use_container_width=True):
                    st.session_state.quiz_questions = None
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_result = None
                    st.rerun()
            with col2:
                if st.button("📖 채팅으로 다시 학습", use_container_width=True):
                    st.session_state.learning_stage = "chat"
                    st.rerun()


def show_summary_tab(subsession_id: int):
    """요약 탭"""
    st.subheader("학습 내용 요약")

    # 요약 생성
    if st.session_state.summary_data is None:
        if st.button("요약 생성", use_container_width=True, type="primary"):
            with st.spinner("요약을 생성하는 중..."):
                try:
                    api_url = st.session_state.api_url

                    request_data = {"subsession_id": subsession_id}

                    with httpx.Client(timeout=60.0) as client:
                        response = client.post(f"{api_url}/learn/summary", json=request_data)

                    if response.status_code == 200:
                        st.session_state.summary_data = response.json()
                        st.rerun()
                    else:
                        st.error(f"요약 생성 실패: {response.text}")

                except Exception as e:
                    st.error(f"요약 생성 오류: {str(e)}")

    # 요약 표시
    else:
        summary_data = st.session_state.summary_data

        # 핵심 개념 요약
        st.markdown("### 📝 핵심 개념")
        st.markdown(summary_data.get("summary", ""))

        st.markdown("---")

        # 혼동 포인트
        pitfalls = summary_data.get("pitfalls", [])
        if pitfalls:
            st.markdown("### ⚠️ 혼동하기 쉬운 포인트")
            for pitfall in pitfalls:
                st.warning(pitfall)

        st.markdown("---")

        # 다음 추천 주제
        next_topics = summary_data.get("next_topics", [])
        if next_topics:
            st.markdown("### 🚀 다음 추천 주제")
            for topic in next_topics:
                st.info(topic)

        st.markdown("---")

        # 완료 버튼 (학습 완료 처리)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("노트북으로 돌아가기", use_container_width=True):
                # 학습 완료 처리: 숙련도 +20% 업데이트
                complete_learning_session(subsession_id)

                # 세션 상태 초기화
                reset_learning_session()

                st.session_state.page = "notebook_detail"
                st.rerun()

        with col2:
            if st.button("대시보드로 돌아가기", use_container_width=True):
                # 학습 완료 처리: 숙련도 +20% 업데이트
                complete_learning_session(subsession_id)

                # 세션 상태 초기화
                reset_learning_session()

                st.session_state.page = "dashboard"
                st.rerun()


def reset_learning_session():
    """학습 세션 상태 초기화"""
    st.session_state.chat_history = []
    st.session_state.quiz_questions = None
    st.session_state.quiz_answers = {}
    st.session_state.quiz_result = None
    st.session_state.summary_data = None
    st.session_state.learning_stage = "chat"


def complete_learning_session(subsession_id: int):
    """학습 완료 처리 (숙련도 +20%, 학습 횟수 +1)"""
    try:
        api_url = st.session_state.api_url
        with httpx.Client() as client:
            response = client.post(
                f"{api_url}/subsessions/{subsession_id}/complete",
                json={"proficiency_increase": 20}
            )

        if response.status_code == 200:
            st.success("🎉 학습을 완료했습니다! 숙련도가 20% 증가했습니다.")
        else:
            st.warning("학습 기록 업데이트에 실패했습니다.")

    except Exception as e:
        st.warning(f"학습 기록 업데이트 오류: {str(e)}")
