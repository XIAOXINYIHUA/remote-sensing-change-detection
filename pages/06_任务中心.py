"""Unified task history and retry management."""

import hashlib
import os
import sys
import time
from datetime import datetime

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)

import streamlit as st

from frontend.components.ui import empty_state, page_header, section_header, task_row_heading
from frontend.project_state import ProjectState
from frontend.utils import setup_matplotlib_font

setup_matplotlib_font()

ps = ProjectState.get()
history = list(ps.task_history)[::-1]

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

total = len(history)
success = sum(1 for task in history if task.get("status") == "success")
failed = sum(1 for task in history if task.get("status") == "failed")
running = sum(1 for task in history if task.get("status") in {"running", "pending"})

page_header(
    "任务中心",
    "工作区",
    "分析任务的状态、错误与重试记录。",
    meta={"总任务": total, "已完成": success, "失败": failed, "进行中": running},
    status="有任务运行" if running else "队列空闲",
    status_tone="info" if running else "success",
)

metrics = st.columns(4)
metrics[0].metric("任务总数", total)
metrics[1].metric("成功", success)
metrics[2].metric("失败", failed)
metrics[3].metric("成功率", f"{success / total * 100:.0f}%" if total else "--")

if not history:
    section_header("任务列表", "最近任务会按时间倒序显示。", "活动")
    empty_state("暂无分析任务", "从项目总览进入任一分析工作台。", "00")
    st.page_link(
        "pages/00_首页.py", label="返回项目总览",
        icon=":material/arrow_back:", use_container_width=False,
    )
else:
    section_header("任务列表", "按状态和类型筛选最近执行记录。", "活动")
    filter_cols = st.columns([1, 1, 2])
    with filter_cols[0]:
        status_filter = st.selectbox("状态", ["全部", "已完成", "失败", "运行中", "排队中"])
    with filter_cols[1]:
        mode_filter = st.selectbox("类型", ["全部", "变化检测", "地物分类", "时间序列"])

    status_values = {label: key for key, (label, _tone) in status_config.items()}
    mode_values = {label: key for key, label in mode_labels.items()}
    filtered = [
        task for task in history
        if (status_filter == "全部" or task.get("status") == status_values.get(status_filter))
        and (mode_filter == "全部" or task.get("mode") == mode_values.get(mode_filter))
    ]

    if not filtered:
        empty_state("没有匹配的任务", "调整筛选条件后重试。", "00")

    for index, task in enumerate(filtered):
        task_id = task.get("id", f"task_{index}")
        mode = task.get("mode", "")
        status = task.get("status", "")
        status_label, tone = status_config.get(status, ("未知", "neutral"))
        detail = " · ".join(part for part in [
            f"{task.get('index_type', '')} {task.get('year1', '')}-{task.get('year2', '')}".strip(),
            f"{task.get('duration_s', 0):.0f} 秒",
            task.get("stats_summary", ""),
        ] if part)

        with st.container(border=True):
            text_col, open_col, retry_col, delete_col = st.columns([6, 1, 1, 1])
            with text_col:
                task_row_heading(
                    task.get("region", "未命名区域"),
                    mode_labels.get(mode, "分析任务"),
                    status_label,
                    tone,
                    detail,
                )
                if status == "failed" and task.get("error"):
                    st.caption(f"错误：{str(task['error'])[:240]}")
            with open_col:
                if status == "success" and st.button(
                    "打开", icon=":material/open_in_new:",
                    key=f"task_open_{task_id}_{index}", use_container_width=True,
                ):
                    st.switch_page(target_pages.get(mode, "pages/01_变化检测.py"))
            with retry_col:
                if status == "failed" and st.button(
                    "重试", icon=":material/refresh:",
                    key=f"task_retry_{task_id}_{index}", use_container_width=True,
                ):
                    unique_id = hashlib.md5(f"{task_id}_{time.time()}".encode()).hexdigest()[:12]
                    retry_task = dict(task)
                    retry_task.update({
                        "id": f"retry_{unique_id}",
                        "status": "pending",
                        "error": "",
                        "started_at": datetime.now().isoformat(),
                    })
                    ps.task_history.append(retry_task)
                    ProjectState.commit()
                    st.rerun()
            with delete_col:
                if st.button(
                    "删除", icon=":material/delete:",
                    key=f"task_delete_{task_id}_{index}", use_container_width=True,
                ):
                    ps.task_history = [item for item in ps.task_history if item.get("id") != task_id]
                    ProjectState.commit()
                    st.rerun()

    if failed:
        with st.popover("处理失败任务", icon=":material/error:"):
            st.caption(f"当前有 {failed} 个失败任务。以下操作会修改任务记录。")
            clear_col, retry_col = st.columns(2)
            with clear_col:
                if st.button("清除失败记录", icon=":material/delete_sweep:", use_container_width=True):
                    ps.task_history = [item for item in ps.task_history if item.get("status") != "failed"]
                    ProjectState.commit()
                    st.rerun()
            with retry_col:
                if st.button("全部重新排队", icon=":material/refresh:", use_container_width=True):
                    for task in ps.task_history:
                        if task.get("status") == "failed":
                            task.update({
                                "status": "pending", "error": "",
                                "started_at": datetime.now().isoformat(),
                            })
                    ProjectState.commit()
                    st.rerun()
