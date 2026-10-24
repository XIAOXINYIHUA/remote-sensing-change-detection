"""任务中心：后台任务（进度、取消、重试、日志）与已完成的分析记录。"""

import os
import sys
from pathlib import Path

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)

import streamlit as st

from frontend.components import jobs as jobs_ui
from frontend.components.ui import (
    empty_state, page_header, section_header, status_banner, task_row_heading,
)
from frontend.project_state import ProjectState
from frontend.utils import setup_matplotlib_font

setup_matplotlib_font()

S = st.session_state
ps = ProjectState.get()
runner = jobs_ui.get_runner()

JOB_LIMIT = 100             # 列出最近的后台任务数

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
record_status = {
    "success": ("已完成", "success"),
    "failed": ("失败", "danger"),
}
state_filters = {
    "全部": None,
    "进行中": ("queued", "running"),
    "已完成": ("succeeded",),
    "失败": ("failed", "interrupted"),
    "已取消": ("cancelled",),
}


def _same_file(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def _adopt(job, new_id):
    """重试的任务交回原页面跟踪（结果自动载入）：该页当前没有跟踪中的任务、且区域没变。"""
    slot = job.meta.get("slot")
    if not slot or S.get(slot):
        return False
    current = S.get(f"{slot.split('_', 1)[0]}_boundary") or ps.boundary
    boundary = job.meta.get("boundary")
    if not current or not boundary or not _same_file(current, boundary):
        return False
    jobs_ui.track(slot, new_id)
    return True


def _retry(job):
    new_id = runner.retry(job.id)
    if _adopt(job, new_id):
        S.tc_notice = ("success", "已重新提交", "完成后结果自动载入原页面。")
    else:
        S.tc_notice = ("success", "已重新提交", "进度与结果在任务中心查看。")


def _job_row(job):
    name, tone = jobs_ui.STATE_LABELS.get(job.state, (job.state, "neutral"))
    detail = [f"提交于 {job.created_at.replace('T', ' ')}"]
    if job.started_at:
        detail.append(f"{'已用时' if job.active else '用时'} {jobs_ui.fmt_seconds(job.elapsed_s)}")
    failed = job.state in ("failed", "interrupted", "cancelled")

    with st.container(border=True):
        text_col, open_col, act_col, del_col = st.columns([6, 1, 1, 1])
        with text_col:
            task_row_heading(job.title, jobs_ui.KIND_LABELS.get(job.kind, job.kind), name, tone,
                             " · ".join(detail))
            if job.active:
                text = ("正在取消…" if job.cancel_requested
                        else job.message if job.state == "running" else "等待空闲的工作进程")
                st.progress(min(job.progress, 100.0) / 100, text=text)
            elif job.error:
                st.caption(f"原因：{job.error[:300]}")
        with open_col:
            page = job.meta.get("page")
            if page and st.button("打开", icon=":material/open_in_new:", key=f"job_open_{job.id}",
                                  width="stretch"):
                st.switch_page(page)
        with act_col:
            if job.active:
                if st.button("取消", icon=":material/cancel:", key=f"job_cancel_{job.id}",
                             width="stretch", disabled=job.cancel_requested):
                    runner.cancel(job.id)
                    st.rerun()
            elif failed and st.button("重试", icon=":material/refresh:", key=f"job_retry_{job.id}",
                                      width="stretch"):
                _retry(job)
                st.rerun()
        with del_col:
            if not job.active and st.button("删除", icon=":material/delete:",
                                            key=f"job_delete_{job.id}", width="stretch",
                                            help="只删除任务记录，不影响分析输出与影像缓存"):
                runner.delete(job.id)
                st.rerun()
        if failed:
            log = runner.log_tail(job.id)
            if log:
                with st.expander("任务日志"):
                    st.code(log, language=None)


def _render_jobs(state_filter, kind_filter):
    """按筛选条件列出后台任务；返回是否还有进行中的任务（不受筛选影响）。"""
    jobs = runner.list(limit=JOB_LIMIT)
    states = state_filters[state_filter]
    shown = [job for job in jobs
             if (states is None or job.state in states)
             and (kind_filter == "全部" or jobs_ui.KIND_LABELS.get(job.kind) == kind_filter)]
    if not shown:
        empty_state("没有匹配的任务", "调整筛选条件后重试。", "00")
    for job in shown:
        _job_row(job)
    return any(job.active for job in jobs)


@st.fragment(run_every=2.0)
def _live_jobs(state_filter, kind_filter):
    """有任务进行中时每 2 秒刷新列表；全部结束后整页刷新（停止轮询，更新顶部计数）。"""
    if not _render_jobs(state_filter, kind_filter):
        st.rerun()


# =====================================================================
#  页面
# =====================================================================

jobs = runner.list(limit=JOB_LIMIT)
history = list(ps.task_history)[::-1]
active = sum(job.active for job in jobs)
failed = sum(job.state in ("failed", "interrupted") for job in jobs)
succeeded = sum(job.state == "succeeded" for job in jobs)

page_header(
    "任务中心",
    "工作区",
    "后台任务的进度、取消、重试与日志，以及已完成的分析记录。",
    meta={"进行中": active, "已完成": succeeded, "失败": failed, "分析记录": len(history)},
    status="有任务运行" if active else "队列空闲",
    status_tone="info" if active else "success",
)

metrics = st.columns(4)
metrics[0].metric("进行中", active)
metrics[1].metric("已完成", succeeded)
metrics[2].metric("失败 / 中断", failed)
metrics[3].metric("同时运行上限", runner.max_workers)

notice = S.pop("tc_notice", None)
if notice:
    status_banner(*notice)

section_header("后台任务", "影像收集、分析与批量任务在后台进程中执行，切换页面或关闭浏览器不受影响。",
               "执行")
if not jobs:
    empty_state("暂无后台任务", "在分析工作台提交收集影像或运行分析后显示在这里。", "00")
else:
    filter_cols = st.columns([1, 1, 2])
    with filter_cols[0]:
        state_filter = st.selectbox("状态", list(state_filters), key="tc_state_filter")
    with filter_cols[1]:
        kind_filter = st.selectbox("类型", ["全部", *jobs_ui.KIND_LABELS.values()],
                                   key="tc_kind_filter")
    if active:
        _live_jobs(state_filter, kind_filter)
    else:
        _render_jobs(state_filter, kind_filter)

    finished = [job for job in jobs if not job.active]
    if finished:
        with st.popover("清理任务记录", icon=":material/delete_sweep:"):
            st.caption(f"{len(finished)} 个已结束任务的记录（参数、日志、结果文件）。"
                       "分析输出与影像缓存不受影响。")
            if st.button("删除已结束的任务记录", icon=":material/delete_sweep:",
                         width="stretch"):
                for job in finished:
                    runner.delete(job.id)
                st.rerun()

section_header("分析记录", "已完成分析的参数与结果摘要。", "历史")
if not history:
    empty_state("暂无分析记录", "从项目总览进入任一分析工作台。", "00")
    st.page_link(
        "pages/00_首页.py", label="返回项目总览",
        icon=":material/arrow_back:",
    )
else:
    mode_filter = st.selectbox("类型", ["全部", *mode_labels.values()], key="tc_mode_filter")
    mode_values = {label: key for key, label in mode_labels.items()}
    filtered = [task for task in history
                if mode_filter == "全部" or task.get("mode") == mode_values.get(mode_filter)]
    if not filtered:
        empty_state("没有匹配的记录", "调整筛选条件后重试。", "00")

    for index, task in enumerate(filtered):
        task_id = task.get("id", f"task_{index}")
        mode = task.get("mode", "")
        status = task.get("status", "")
        status_label, tone = record_status.get(status, ("未知", "neutral"))
        detail = " · ".join(part for part in [
            f"{task.get('index_type', '')} {task.get('year1', '')}-{task.get('year2', '')}".strip(),
            f"{task.get('duration_s', 0):.0f} 秒",
            task.get("stats_summary", ""),
        ] if part)

        with st.container(border=True):
            text_col, open_col, delete_col = st.columns([7, 1, 1])
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
                    key=f"task_open_{task_id}_{index}", width="stretch",
                ):
                    st.switch_page(target_pages.get(mode, "pages/01_变化检测.py"))
            with delete_col:
                if st.button(
                    "删除", icon=":material/delete:",
                    key=f"task_delete_{task_id}_{index}", width="stretch",
                ):
                    ps.task_history = [item for item in ps.task_history if item.get("id") != task_id]
                    ProjectState.commit()
                    st.rerun()
