"""Project overview and primary analysis entry points."""

import os
import sys
from pathlib import Path

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)

import folium
import streamlit as st
from streamlit_folium import st_folium

from engine.utils import geodesic_area_km2
from frontend.components.ui import (
    empty_state,
    page_header,
    section_header,
    task_row_heading,
)
from frontend.inventory import imagery_summary
from frontend.project_state import ProjectState
from frontend.utils import DATA_SOURCE_LABELS, load_gj, setup_matplotlib_font

setup_matplotlib_font()

ps = ProjectState.get()


def _project_area_km2() -> float | None:
    if not ps.is_ready():
        return None
    try:
        _, geom = load_gj(ps.boundary)
        return geodesic_area_km2(geom)
    except Exception:
        return None


def _open_analysis(page: str) -> None:
    st.switch_page(page)


area_km2 = _project_area_km2()
imagery = imagery_summary(Path(_base) / "data")
history = list(ps.task_history)
success_count = sum(1 for task in history if task.get("status") == "success")
success_rate = success_count / len(history) * 100 if history else 0

page_header(
    "项目总览",
    "工作区",
    "当前分析范围、数据状态与最近任务。",
    meta={
        "项目": ps.region if ps.is_ready() else "未配置",
        "数据源": DATA_SOURCE_LABELS.get(ps.data_source, ps.data_source),
        "指数": ps.index_type,
    },
    status="区域已就绪" if ps.is_ready() else "等待配置",
    status_tone="success" if ps.is_ready() else "warning",
)

metric_cols = st.columns(4)
metric_cols[0].metric("分析范围", f"{area_km2:,.0f} km²" if area_km2 is not None else "--")
metric_cols[1].metric(
    f"可复用影像 · {imagery['reusable_sets']} 组",
    f"{imagery['reusable_mb']:,.0f} MB",
    help=f"缓存时相 {imagery['snapshots']} 期（各分析共用）· 旧版地物分类下载 {imagery['classify_sets']} 组",
)
metric_cols[2].metric(f"分析任务 · {success_count} 个成功", len(history))
metric_cols[3].metric("任务成功率", f"{success_rate:.0f}%" if history else "--")
if imagery["stale_files"]:
    st.caption(
        f"另有 {imagery['stale_files']} 个旧版下载或过期缓存的影像文件"
        f"（{imagery['stale_mb']:,.0f} MB）不会被分析复用，详见「系统与数据」。"
    )

if ps.is_ready():
    section_header("当前分析区域", "边界预览与常用分析入口", "项目")
    map_col, action_col = st.columns([1.85, 1], gap="large")

    with map_col:
        try:
            gj, geom = load_gj(ps.boundary)
            m = folium.Map(
                location=[geom.centroid.y, geom.centroid.x],
                zoom_start=9,
                control_scale=True,
                tiles="CartoDB positron",
            )
            folium.GeoJson(
                gj,
                name="分析边界",
                style_function=lambda _feature: {
                    "fillColor": "#0F766E",
                    "color": "#0A5D57",
                    "weight": 2,
                    "fillOpacity": 0.12,
                },
            ).add_to(m)
            minx, miny, maxx, maxy = geom.bounds
            m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(24, 24))
            st_folium(
                m,
                width=None,
                height=390,
                key="overview_map",
                returned_objects=[],
            )
        except Exception as exc:
            empty_state("边界预览不可用", str(exc), "AOI")

    with action_col:
        section_header("开始分析", f"{ps.year1} 至 {ps.year2} · {ps.index_type}")
        left, right = st.columns(2)
        with left:
            if st.button(
                "变化检测", icon=":material/compare:",
                width="stretch", key="home_change",
            ):
                _open_analysis("pages/01_变化检测.py")
            if st.button(
                "时间序列", icon=":material/show_chart:",
                width="stretch", key="home_timeseries",
            ):
                _open_analysis("pages/03_时间序列.py")
        with right:
            if st.button(
                "地物分类", icon=":material/category:",
                width="stretch", key="home_classify",
            ):
                _open_analysis("pages/02_地物分类.py")
            if st.button(
                "批量任务", icon=":material/layers:",
                width="stretch", key="home_batch",
            ):
                _open_analysis("pages/04_批量处理.py")

        st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)
        if st.button(
            "新建分析配置",
            type="primary",
            icon=":material/add:",
            width="stretch",
        ):
            st.session_state.wizard_step = 0
            st.session_state.wizard_complete = False
            st.session_state._cd_wizard_active = False
            st.switch_page("pages/01_变化检测.py")

        st.markdown(
            f"""
            <div class="rs-banner rs-banner--info">
              <span class="rs-banner-mark"></span>
              <div><b>{ps.region}</b>
              <p>{Path(ps.boundary).name}<br>{ps.month_start} 至 {ps.month_end} 月 · 云量上限 {ps.max_cloud_pct}%</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    section_header("创建首个分析", "设置边界后即可进入各分析工作台", "项目")
    setup_col, source_col = st.columns([1.35, 1], gap="large")
    with setup_col:
        empty_state(
            "尚未配置分析区域",
            "可选择现有 GeoJSON、上传边界文件或按城市名称检索。",
            "AOI",
        )
        if st.button(
            "设置分析区域",
            type="primary",
            icon=":material/add_location_alt:",
            width="stretch",
        ):
            st.session_state.wizard_step = 0
            st.session_state.wizard_complete = False
            st.switch_page("pages/01_变化检测.py")
    with source_col:
        section_header("可用边界", "项目根目录中的 GeoJSON 文件")
        boundary_files = sorted(Path(_base).glob("*.geojson"))
        if boundary_files:
            for path in boundary_files[:6]:
                with st.container(border=True):
                    st.markdown(f"**{path.stem}**")
                    st.caption(path.name)
        else:
            empty_state("没有本地边界文件", "可在分析向导中上传或搜索。", "GEO")

section_header("最近任务", "最近 8 次分析的执行状态与结果入口", "活动")
if history:
    mode_labels = {
        "change_detection": "变化检测",
        "classification": "地物分类",
        "timeseries": "时间序列",
    }
    target_pages = {
        "change_detection": "pages/01_变化检测.py",
        "classification": "pages/02_地物分类.py",
        "timeseries": "pages/03_时间序列.py",
    }
    status_config = {
        "success": ("已完成", "success"),
        "failed": ("失败", "danger"),
        "running": ("运行中", "info"),
        "pending": ("排队中", "warning"),
    }
    for task in history[-8:][::-1]:
        task_id = task.get("id", "")
        mode = task.get("mode", "")
        status_label, tone = status_config.get(task.get("status", ""), ("未知", "neutral"))
        detail_parts = [
            f"{task.get('index_type', '')} {task.get('year1', '')}-{task.get('year2', '')}".strip(),
            f"{task.get('duration_s', 0):.0f} 秒",
            task.get("stats_summary", ""),
        ]
        with st.container(border=True):
            text_col, open_col, delete_col = st.columns([6, 1, 1])
            with text_col:
                task_row_heading(
                    task.get("region", "未命名区域"),
                    mode_labels.get(mode, "分析任务"),
                    status_label,
                    tone,
                    " · ".join(part for part in detail_parts if part),
                )
            with open_col:
                if st.button(
                    "打开", icon=":material/open_in_new:",
                    key=f"open_{task_id}", width="stretch",
                ):
                    st.switch_page(target_pages.get(mode, "pages/01_变化检测.py"))
            with delete_col:
                if st.button(
                    "删除", icon=":material/delete:",
                    key=f"del_{task_id}", width="stretch",
                ):
                    ps.task_history = [item for item in ps.task_history if item.get("id") != task_id]
                    ProjectState.commit()
                    st.rerun()
else:
    empty_state("暂无分析任务", "任务开始后会在这里显示进度与结果。", "00")
