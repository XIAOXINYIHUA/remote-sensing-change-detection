"""
ProjectState — 项目状态：一个项目一个状态文件（状态目录下的 .project_state.json）

文件内容:
    区域（region / boundary / slug）、数据源、指数、时相、云量等项目设置
    task_history   分析记录
    pages          各页需要跨会话保留的状态，按页面前缀分节（cd / cl / ts / bt），
                   每节记下所属区域 aoi；区域换了，旧区域的页面状态不再载入

分析区域只有一个，以项目为准：任何页面（侧栏、向导）切换区域都写到这里，
各页每次运行从项目同步区域，发现区域变了就清掉本页属于旧区域的数据。

用法:
    from frontend.project_state import ProjectState
    ps = ProjectState.get()
    ps.region = "新余市"
    ProjectState.commit()

    ProjectState.save_page("cd", {"cd_data_ok": True})
    ProjectState.load_page("cd", ["cd_data_ok"])     # 区域没变时返回已保存的值

同一状态目录被多个会话（浏览器标签页）共用：get() 发现文件被别的会话改过就重新载入。
旧版的 .shared_state.json 与各页状态文件（.app_state.json 等）在首次启动时并入新文件，
旧文件不再读写，可以删除。
"""

from __future__ import annotations
import dataclasses
import json
import os


@dataclasses.dataclass
class _ProjectState:
    """共享项目状态 — 所有页面共用同一个实例。"""
    region: str = ""
    boundary: str = ""        # GeoJSON 文件路径
    slug: str = ""
    data_source: str = "sentinel2"
    index_type: str = "NDVI"

    # 时相
    year1: int = 2025
    year2: int = 2026
    month_start: int = 6
    month_end: int = 10
    max_cloud_pct: int = 50

    # 上一次分析的统计/状态
    last_analysis_mode: str = ""  # "change_detection" | "classification" | "timeseries"
    last_analysis_time: str = ""

    # 任务历史
    task_history: list = dataclasses.field(default_factory=list)

    # 场景预览（缓存最近搜索的 scene metadata）
    scenes_year1: list = dataclasses.field(default_factory=list)
    scenes_year2: list = dataclasses.field(default_factory=list)

    # 各页状态：{页面前缀: {"aoi": 所属边界, 会话键: 值, ...}}
    pages: dict = dataclasses.field(default_factory=dict)

    def is_ready(self) -> bool:
        """检查是否有可用的边界和地区。"""
        return bool(self.boundary) and bool(self.region) and os.path.exists(self.boundary)

    def to_dict(self) -> dict:
        return {
            "version": STATE_VERSION,
            "region": self.region,
            "boundary": self.boundary,
            "slug": self.slug,
            "data_source": self.data_source,
            "index_type": self.index_type,
            "year1": self.year1,
            "year2": self.year2,
            "month_start": self.month_start,
            "month_end": self.month_end,
            "max_cloud_pct": self.max_cloud_pct,
            "last_analysis_mode": self.last_analysis_mode,
            "last_analysis_time": self.last_analysis_time,
            "task_history": self.task_history,
            "pages": self.pages,
        }

    @classmethod
    def from_dict(cls, d: dict) -> _ProjectState:
        return cls(
            region=d.get("region", ""),
            boundary=d.get("boundary", ""),
            slug=d.get("slug", ""),
            data_source=d.get("data_source", "sentinel2"),
            index_type=d.get("index_type", "NDVI"),
            year1=d.get("year1", 2025),
            year2=d.get("year2", 2026),
            month_start=d.get("month_start", 6),
            month_end=d.get("month_end", 10),
            max_cloud_pct=d.get("max_cloud_pct", 50),
            last_analysis_mode=d.get("last_analysis_mode", ""),
            last_analysis_time=d.get("last_analysis_time", ""),
            task_history=d.get("task_history", []),
            pages=d.get("pages") or {},
        )


STATE_VERSION = 2
STATE_FILE_NAME = ".project_state.json"
LEGACY_SHARED_FILE = ".shared_state.json"
# 旧版各页状态文件 → 页面前缀
LEGACY_PAGE_FILES = {
    ".app_state.json": "cd",
    ".classify_state.json": "cl",
    ".ts_state.json": "ts",
    ".batch_state.json": "bt",
}

_PROJECT_STATE_FILE = ""


class ProjectState:
    """统一项目状态访问器 — 对 SessionState 的轻量封装。

    import streamlit as st
    ps = ProjectState.get()
    ps.region = "新余市"
    """

    _ST_KEY = "_shared_project_state"
    _STAMP_KEY = "_shared_project_state_stamp"     # 本会话最后一次读 / 写时文件的（修改时间, 大小）

    @classmethod
    def init(cls, project_dir: str = ""):
        """设置状态文件位置（app.py 每次运行时调用）；新文件不存在时并入旧版状态文件。"""
        global _PROJECT_STATE_FILE
        _PROJECT_STATE_FILE = os.path.join(project_dir or ".", STATE_FILE_NAME)
        if not os.path.exists(_PROJECT_STATE_FILE):
            _migrate_legacy_files(project_dir or ".", _PROJECT_STATE_FILE)

    @classmethod
    def get(cls):
        """当前会话的项目状态（首次调用时从文件载入；文件被别的会话改过时重新载入）。"""
        import streamlit as st
        if cls._ST_KEY not in st.session_state:
            st.session_state[cls._ST_KEY] = _ProjectState()
            cls._load_from_file()
        else:
            stamp = _file_stamp()
            seen = st.session_state.get(cls._STAMP_KEY)
            if seen is None:
                st.session_state[cls._STAMP_KEY] = stamp
            elif stamp is not None and stamp != seen:
                cls._load_from_file()
        return st.session_state[cls._ST_KEY]

    @classmethod
    def set_aoi(cls, region: str, boundary: str, slug: str = ""):
        """设置 AOI 并自动保存。"""
        ps = cls.get()
        ps.region = region
        ps.boundary = boundary
        ps.slug = slug or _make_slug(region)
        cls.commit()

    @classmethod
    def commit(cls):
        """将当前状态保存到 session 和文件。"""
        import streamlit as st
        ps = st.session_state.get(cls._ST_KEY)
        if ps is None:
            return
        cls._save_to_file(ps)

    @classmethod
    def _save_to_file(cls, ps: _ProjectState):
        """原子写入：先写 .tmp 再 rename，避免并发读取到不完整状态。"""
        if not _PROJECT_STATE_FILE:
            return
        try:
            tmp_path = _PROJECT_STATE_FILE + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(ps.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, _PROJECT_STATE_FILE)  # 原子替换
            import streamlit as st
            st.session_state[cls._STAMP_KEY] = _file_stamp()
        except Exception:
            pass

    @classmethod
    def _load_from_file(cls):
        """从文件载入，原地更新会话中的对象（页面已持有的 ps 引用仍然有效）。"""
        import streamlit as st
        stamp = _file_stamp()
        st.session_state[cls._STAMP_KEY] = stamp
        if stamp is None:
            return
        data = _read_json(_PROJECT_STATE_FILE)
        if isinstance(data, dict):
            loaded = _ProjectState.from_dict(data)
            st.session_state[cls._ST_KEY].__dict__.update(loaded.__dict__)

    # ---- 页面状态 ----

    @classmethod
    def load_page(cls, prefix: str, keys, aoi_bound: bool = True) -> dict:
        """页面保存在项目状态里的值（keys 中非 None 的）。

        aoi_bound: 该页状态属于某个分析区域，保存时的区域与当前项目区域不同则返回空。
        """
        ps = cls.get()
        section = ps.pages.get(prefix) or {}
        if aoi_bound and not same_path(section.get("aoi"), ps.boundary):
            return {}
        return {k: section[k] for k in keys if section.get(k) is not None}

    @classmethod
    def save_page(cls, prefix: str, values: dict):
        """保存页面状态（整节替换），记下所属区域。"""
        ps = cls.get()
        ps.pages[prefix] = {**values, "aoi": ps.boundary}
        cls.commit()

    @classmethod
    def aoi_changed(cls, prefix: str) -> bool:
        """本会话中该页数据所属的区域与当前项目区域不同（随后记下当前区域）。

        首次进入该页时为 False：已保存的页面状态由 load_page 按区域筛选。
        """
        import streamlit as st
        key = f"_{prefix}_aoi"
        current = cls.get().boundary
        previous = st.session_state.get(key)
        st.session_state[key] = current
        return previous is not None and not same_path(previous, current)

    @classmethod
    def add_task(cls, record: dict):
        """添加一条任务记录到历史。"""
        ps = cls.get()
        if len(ps.task_history) > 50:  # 只保留最近 50 条
            ps.task_history = ps.task_history[-50:]
        record["id"] = record.get("id", f"task_{len(ps.task_history) + 1}")
        import time
        record.setdefault("finished_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        ps.task_history.append(record)
        cls.commit()

    @classmethod
    def clear(cls):
        """清除共享状态。"""
        import streamlit as st
        if cls._ST_KEY in st.session_state:
            del st.session_state[cls._ST_KEY]
        if _PROJECT_STATE_FILE and os.path.exists(_PROJECT_STATE_FILE):
            try:
                os.remove(_PROJECT_STATE_FILE)
            except Exception:
                pass


# ============================================================================
#  任务历史 UI
# ============================================================================


def _make_slug(name: str) -> str:
    """将名称转为文件系统友好的短标识。"""
    import re
    slug_str = re.sub(r"[^\w一-鿿]+", "_", name)
    return slug_str.strip("_")[:30] or "default"


def same_path(a, b) -> bool:
    """两个边界路径是否指同一文件（都为空也算相同）。"""
    if not a or not b:
        return not a and not b
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _file_stamp():
    """状态文件的（修改时间 ns, 大小）；用来发现别的会话写过文件。"""
    if not _PROJECT_STATE_FILE:
        return None
    try:
        st = os.stat(_PROJECT_STATE_FILE)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _migrate_legacy_files(state_dir: str, target: str) -> bool:
    """把旧版 .shared_state.json 与各页状态文件并入 target（旧文件保留不动）。

    旧版各页文件的键有两种写法：cd_data_ok（带页面前缀）与 download_ok（更早的版本，无前缀），
    统一成带前缀的会话键；其中的边界作为该节的 aoi，与项目区域不同的页面状态之后不会载入。

    Returns:
        是否写出了新文件（没有任何旧文件时不写）。
    """
    shared = _read_json(os.path.join(state_dir, LEGACY_SHARED_FILE))
    pages = {}
    for name, prefix in LEGACY_PAGE_FILES.items():
        data = _read_json(os.path.join(state_dir, name))
        if not isinstance(data, dict):
            continue
        data = {k if k.startswith(f"{prefix}_") else f"{prefix}_{k}": v for k, v in data.items()}
        aoi_keys = (f"{prefix}_boundary", f"{prefix}_region", f"{prefix}_slug")
        pages[prefix] = {k: v for k, v in data.items() if k not in aoi_keys}
        pages[prefix]["aoi"] = data.get(f"{prefix}_boundary") or ""
    if not isinstance(shared, dict) and not pages:
        return False

    state = _ProjectState.from_dict(shared if isinstance(shared, dict) else {})
    state.pages = pages
    if not state.boundary:
        # 旧版共享状态没有区域：沿用最近使用的页面区域（变化检测优先）
        for name, prefix in LEGACY_PAGE_FILES.items():
            data = _read_json(os.path.join(state_dir, name)) or {}
            boundary = data.get(f"{prefix}_boundary") or data.get("boundary")
            region = data.get(f"{prefix}_region") or data.get("region")
            if boundary and region:
                state.region, state.boundary = region, boundary
                state.slug = data.get(f"{prefix}_slug") or data.get("slug") or _make_slug(region)
                break
    try:
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, target)
        return True
    except Exception:
        return False
