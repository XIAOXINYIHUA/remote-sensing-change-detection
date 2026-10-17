"""
ProjectState — 统一项目/AOI 状态管理

所有页面通过此模块读取/写入共享的 AOI 和项目设置。
每个页面仍有自己的私有状态 (cd_*, cl_*, ts_*, bt_*)。

用法:
    from frontend.project_state import ProjectState
    ps = ProjectState.get()
    ps.region = "新余市"
    ps.boundary = "新余.geojson"
    ps.commit()
"""

from __future__ import annotations
import dataclasses
import json
import os
from pathlib import Path
from typing import Optional


@dataclasses.dataclass
class TaskRecord:
    """单次分析任务的记录。"""
    id: str = ""
    mode: str = ""            # "change_detection" | "classification" | "timeseries"
    region: str = ""
    data_source: str = ""
    index_type: str = ""
    year1: int = 0
    year2: int = 0
    status: str = "pending"   # pending | running | success | failed
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0
    stats_summary: str = ""   # 简短摘要（如"变化 123 ha"）
    output_files: dict = dataclasses.field(default_factory=dict)
    error: str = ""


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

    def is_ready(self) -> bool:
        """检查是否有可用的边界和地区。"""
        return bool(self.boundary) and bool(self.region) and os.path.exists(self.boundary)

    def to_dict(self) -> dict:
        return {
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
        )


_PROJECT_STATE_FILE = ""


class ProjectState:
    """统一项目状态访问器 — 对 SessionState 的轻量封装。

    import streamlit as st
    ps = ProjectState.get()
    ps.region = "新余市"
    """

    _ST_KEY = "_shared_project_state"

    @classmethod
    def init(cls, project_dir: str = ""):
        """初始化 ProjectState 模块（在 app.py 启动时调用一次）。"""
        global _PROJECT_STATE_FILE
        if project_dir:
            _PROJECT_STATE_FILE = os.path.join(project_dir, ".shared_state.json")
        else:
            _PROJECT_STATE_FILE = ".shared_state.json"

    @classmethod
    def get(cls):
        """获取当前 session 的共享项目状态（懒加载）。"""
        import streamlit as st
        if cls._ST_KEY not in st.session_state:
            st.session_state[cls._ST_KEY] = _ProjectState()
            # 尝试从文件加载
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
        global _PROJECT_STATE_FILE
        if not _PROJECT_STATE_FILE:
            return
        try:
            tmp_path = _PROJECT_STATE_FILE + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(ps.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, _PROJECT_STATE_FILE)  # 原子替换
        except Exception:
            pass

    @classmethod
    def _load_from_file(cls):
        global _PROJECT_STATE_FILE
        import streamlit as st
        if not _PROJECT_STATE_FILE or not os.path.exists(_PROJECT_STATE_FILE):
            return
        try:
            with open(_PROJECT_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            ps = _ProjectState.from_dict(data)
            st.session_state[cls._ST_KEY] = ps
        except Exception:
            pass

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
        global _PROJECT_STATE_FILE
        if _PROJECT_STATE_FILE and os.path.exists(_PROJECT_STATE_FILE):
            try:
                os.remove(_PROJECT_STATE_FILE)
            except Exception:
                pass


# ============================================================================
#  任务历史 UI
# ============================================================================

def render_task_history(max_items: int = 10):
    """在侧边栏或页面中渲染任务历史（只读，不写入任何文件）。"""
    import streamlit as st
    ps = ProjectState.get()
    history = list(ps.task_history[-max_items:])[::-1]  # 最新的在前

    if not history:
        st.caption("暂无分析历史")
        return

    for task in history:
        mode_icon = {
            "change_detection": "🛰️",
            "classification": "🗺️",
            "timeseries": "📈",
        }.get(task.get("mode", ""), "📋")
        status_icon = {
            "success": "✅",
            "failed": "❌",
            "running": "⏳",
            "pending": "⏸️",
        }.get(task.get("status", ""), "❓")

        label = (
            f"{status_icon} {mode_icon} "
            f"{task.get('region', '?')} "
            f"{task.get('index_type', '')} "
            f"{task.get('year1', '')}→{task.get('year2', '')}"
        )
        summary = task.get("stats_summary", "")
        t_id = task.get("id", "")
        duration = task.get("duration_s", 0)

        # 可点击的卡片（未来可跳转结果页）
        card_label = f"{status_icon} **{task.get('region', '?')}**  {task.get('index_type', '')}  {task.get('year1', '')}→{task.get('year2', '')}  ({duration:.0f}s)"
        if summary:
            card_label += f"  — {summary}"
        with st.container():
            st.caption(card_label)


def _make_slug(name: str) -> str:
    """将名称转为文件系统友好的短标识。"""
    import re
    slug_str = re.sub(r"[^\w一-鿿]+", "_", name)
    return slug_str.strip("_")[:30] or "default"


def migrate_from_legacy(session_state) -> bool:
    """从页面各自的 cd_* / cl_* / ts_* 状态迁移到共享 ProjectState。

    如果发现 cd_boundary 已存在且 ProjectState 为空，则迁移。

    Returns:
        bool: 是否发生了迁移
    """
    ps = ProjectState.get()
    if ps.is_ready():
        return False  # 已有共享状态

    migrated = False
    legacy_keys = [
        ("cd_boundary", "cd_region", "cd_slug"),
        ("cl_boundary", "cl_region", "cl_slug"),
        ("ts_boundary", "ts_region", "ts_slug"),
    ]
    for bkey, rkey, skey in legacy_keys:
        boundary = session_state.get(bkey)
        region = session_state.get(rkey)
        if boundary and region:
            ps.region = region
            ps.boundary = boundary
            ps.slug = session_state.get(skey, _make_slug(region))
            migrated = True
            break

    if migrated:
        ProjectState.commit()
    return migrated
