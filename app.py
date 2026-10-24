"""
遥感分析平台 v3 — 统一入口

分层导航（st.navigation 原生分组，无重复元素）:
  工作区      分析工具      管理
  首页        变化检测      批量任务
  任务中心    地物分类      系统设置
  成果库      时间序列
"""

import os
import sys

_base = os.path.dirname(os.path.abspath(__file__))
os.environ["GDAL_TMPDIR"] = os.path.join(_base, "tmp")
os.environ["TMP"] = os.environ["GDAL_TMPDIR"]
os.environ["TEMP"] = os.environ["GDAL_TMPDIR"]
os.makedirs(os.environ["GDAL_TMPDIR"], exist_ok=True)

import streamlit as st

if _base not in sys.path:
    sys.path.insert(0, _base)

from frontend.utils import inject_global_css, setup_matplotlib_font, state_dir, DATA_SOURCE_LABELS
from frontend.project_state import ProjectState
from frontend.components.jobs import get_runner

setup_matplotlib_font()

st.set_page_config(
    page_title="遥感分析工作台",
    page_icon=":material/satellite_alt:",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": "遥感分析工作台 · Sentinel / Landsat 影像分析",
    },
)

inject_global_css()

ProjectState.init(project_dir=str(state_dir()))
ps = ProjectState.get()

with st.sidebar:
    st.markdown(
        """
        <div class="rs-brand">
          <div class="rs-brand-mark">RS</div>
          <div><b>遥感分析工作台</b><span>REMOTE SENSING OPERATIONS</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 分组导航 — st.navigation 原生支持 sections
pg = st.navigation({
    "工作区": [
        st.Page("pages/00_首页.py", title="项目总览", icon=":material/space_dashboard:", default=True),
        st.Page("pages/06_任务中心.py", title="任务中心", icon=":material/view_list:"),
        st.Page("pages/07_成果库.py", title="成果库", icon=":material/folder_open:"),
    ],
    "分析工具": [
        st.Page("pages/01_变化检测.py", title="变化检测", icon=":material/compare:"),
        st.Page("pages/02_地物分类.py", title="地物分类", icon=":material/category:"),
        st.Page("pages/03_时间序列.py", title="时间序列", icon=":material/show_chart:"),
    ],
    "管理": [
        st.Page("pages/04_批量处理.py", title="批量任务", icon=":material/layers:"),
        st.Page("pages/05_工具箱.py", title="系统与数据", icon=":material/settings:"),
    ],
}, position="sidebar")

# 侧边栏额外信息（导航下方）
with st.sidebar:
    if ps.is_ready():
        recent_html = ""
        if ps.task_history:
            recent = ps.task_history[-1]
            recent_html = (
                f"最近任务 · {recent.get('index_type', '')} · "
                f"{recent.get('duration_s', 0):.0f}s"
            )
        st.markdown(
            f"""
            <div class="rs-sidebar-context">
              <small>当前项目</small>
              <b>{ps.region}</b>
              <span>{DATA_SOURCE_LABELS.get(ps.data_source, ps.data_source)}</span>
              <span>{ps.index_type} · {ps.year1} 至 {ps.year2}</span>
              {f'<span>{recent_html}</span>' if recent_html else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="rs-sidebar-context">
              <small>当前项目</small><b>尚未配置</b>
              <span>从项目总览创建首个分析区域</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(f"本地工作区 · {len(ps.task_history)} 次分析")
    # 首次调用时创建任务执行器：上次关闭时未结束的任务记为「已中断」
    n_active = sum(job.active for job in get_runner().list(limit=50))
    if n_active:
        st.caption(f"后台任务 · {n_active} 个进行中，见任务中心")

pg.run()
