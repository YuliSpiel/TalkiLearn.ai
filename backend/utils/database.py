import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..models import Notebook, Session, Subsession, UserProfile, StudyCycle


class Database:
    """
    간단한 JSON 기반 데이터베이스 (파일 시스템 사용)
    프로토타입용 - 추후 PostgreSQL/SQLite로 전환 가능
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.notebooks_file = os.path.join(data_dir, "notebooks.json")
        self.profiles_file = os.path.join(data_dir, "user_profiles", "profiles.json")
        self.cycles_file = os.path.join(data_dir, "study_cycles.json")

        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "user_profiles"), exist_ok=True)

        # 파일 초기화
        self._init_file(self.notebooks_file, [])
        self._init_file(self.profiles_file, {})
        self._init_file(self.cycles_file, [])

    def _init_file(self, filepath: str, default_data: Any):
        """파일이 없으면 기본 데이터로 초기화"""
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_data, f, ensure_ascii=False, indent=2, default=str)

    def _load_json(self, filepath: str) -> Any:
        """JSON 파일 로드"""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json(self, filepath: str, data: Any):
        """JSON 파일 저장"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    # ========== UserProfile ==========

    def save_profile(self, profile: UserProfile):
        """프로필 저장"""
        profiles = self._load_json(self.profiles_file)
        profiles[profile.user_id] = profile.model_dump()
        self._save_json(self.profiles_file, profiles)

    def get_profile(self, user_id: str = "default_user") -> Optional[UserProfile]:
        """프로필 조회"""
        profiles = self._load_json(self.profiles_file)
        profile_data = profiles.get(user_id)
        if profile_data:
            return UserProfile(**profile_data)
        return None

    # ========== Notebook ==========

    def create_notebook(self, title: str, user_id: str = "default_user") -> Notebook:
        """노트북 생성"""
        notebooks = self._load_json(self.notebooks_file)

        # 새 ID 생성
        max_id = max([nb.get("notebook_id", 0) for nb in notebooks], default=0)
        new_id = max_id + 1

        notebook = Notebook(
            notebook_id=new_id,
            title=title,
            user_id=user_id
        )

        notebooks.append(notebook.model_dump())
        self._save_json(self.notebooks_file, notebooks)

        return notebook

    def get_all_notebooks(self, user_id: str = "default_user") -> List[Notebook]:
        """모든 노트북 조회"""
        notebooks = self._load_json(self.notebooks_file)
        return [
            Notebook(**nb) for nb in notebooks
            if nb.get("user_id") == user_id
        ]

    def get_notebook(self, notebook_id: int) -> Optional[Notebook]:
        """노트북 조회"""
        notebooks = self._load_json(self.notebooks_file)
        for nb in notebooks:
            if nb.get("notebook_id") == notebook_id:
                return Notebook(**nb)
        return None

    def update_notebook(self, notebook: Notebook):
        """노트북 업데이트"""
        notebooks = self._load_json(self.notebooks_file)

        for i, nb in enumerate(notebooks):
            if nb.get("notebook_id") == notebook.notebook_id:
                notebooks[i] = notebook.model_dump()
                break

        self._save_json(self.notebooks_file, notebooks)

    def delete_notebook(self, notebook_id: int) -> bool:
        """노트북 삭제"""
        notebooks = self._load_json(self.notebooks_file)
        original_count = len(notebooks)
        notebooks = [nb for nb in notebooks if nb.get("notebook_id") != notebook_id]
        self._save_json(self.notebooks_file, notebooks)
        return len(notebooks) < original_count  # 삭제 성공 여부 반환

    # ========== Session ==========

    def add_session_to_notebook(self, notebook_id: int, session: Session):
        """노트북에 세션 추가"""
        notebook = self.get_notebook(notebook_id)
        if notebook:
            notebook.sessions.append(session)
            self.update_notebook(notebook)

    def get_session(self, notebook_id: int, session_id: int) -> Optional[Session]:
        """세션 조회"""
        notebook = self.get_notebook(notebook_id)
        if notebook:
            for session in notebook.sessions:
                if session.session_id == session_id:
                    return session
        return None

    def update_session(self, notebook_id: int, session: Session):
        """세션 업데이트"""
        notebook = self.get_notebook(notebook_id)
        if notebook:
            for i, s in enumerate(notebook.sessions):
                if s.session_id == session.session_id:
                    notebook.sessions[i] = session
                    break
            self.update_notebook(notebook)

    # ========== Subsession ==========

    def get_subsession(
        self,
        notebook_id: int,
        session_id: int,
        subsession_id: int
    ) -> Optional[Subsession]:
        """서브세션 조회"""
        session = self.get_session(notebook_id, session_id)
        if session:
            for subsession in session.subsessions:
                if subsession.subsession_id == subsession_id:
                    return subsession
        return None

    def find_subsession_by_id(self, subsession_id: int) -> Optional[tuple[int, int, Subsession]]:
        """
        서브세션 ID로 검색 (notebook_id, session_id를 모를 때 사용)

        Returns:
            tuple[notebook_id, session_id, subsession] or None
        """
        notebooks = self._load_json(self.notebooks_file)
        for nb_data in notebooks:
            notebook = Notebook(**nb_data)
            for session in notebook.sessions:
                for subsession in session.subsessions:
                    if subsession.subsession_id == subsession_id:
                        return (notebook.notebook_id, session.session_id, subsession)
        return None

    def update_subsession(
        self,
        notebook_id: int,
        session_id: int,
        subsession: Subsession
    ):
        """서브세션 업데이트"""
        notebook = self.get_notebook(notebook_id)
        if notebook:
            for session in notebook.sessions:
                if session.session_id == session_id:
                    for i, sub in enumerate(session.subsessions):
                        if sub.subsession_id == subsession.subsession_id:
                            session.subsessions[i] = subsession
                            break
                    break
            self.update_notebook(notebook)

    # ========== StudyCycle ==========

    def save_study_cycle(self, cycle: StudyCycle):
        """학습 사이클 저장"""
        cycles = self._load_json(self.cycles_file)
        cycles.append(cycle.model_dump())
        self._save_json(self.cycles_file, cycles)

    def get_study_cycles(self, subsession_id: int) -> List[StudyCycle]:
        """서브세션의 학습 사이클 조회"""
        cycles = self._load_json(self.cycles_file)
        return [
            StudyCycle(**c) for c in cycles
            if c.get("subsession_id") == subsession_id
        ]


# 싱글톤 제거 - 매번 새 인스턴스를 생성하여 파일에서 최신 데이터를 읽음
def get_database() -> Database:
    """
    데이터베이스 인스턴스 반환 (싱글톤 아님)

    매 요청마다 새 인스턴스를 생성하여 항상 파일에서 최신 데이터를 읽습니다.
    이렇게 하면 여러 백엔드 프로세스가 실행 중이어도 데이터 일관성이 유지됩니다.
    """
    return Database()
