"""
后台任务的页面组件：提交、收尾、进度显示

页面用法（slot 为会话键，保存该页当前跟踪的任务 ID）:

    running = settle("cd_run_job", label="变化检测", on_success=_apply, persist=_save_state)
    ...
    if st.button("运行", disabled=running):
        submit("cd_run_job", "cd.run", params, title="...")
    progress("cd_run_job")

settle() 放在页面靠前、读取会话状态之前：任务已结束就把结果写入会话状态并记下提示；
progress() 在任务运行时显示进度与「取消」（st.fragment 定时刷新，只重绘这一块），
发现任务结束即整页刷新，由 settle() 收尾；没有任务时显示上一次的结果提示。
"""

import os

import streamlit as st

from frontend.components.ui import status_banner
from services import jobs

STATE_LABELS = {
    "queued": ("排队中", "warning"),
    "running": ("运行中", "info"),
    "succeeded": ("已完成", "success"),
    "failed": ("失败", "danger"),
    "cancelled": ("已取消", "warning"),
    "interrupted": ("已中断", "danger"),
}

KIND_LABELS = {
    "cd.collect": "变化检测 · 收集影像",
    "cd.run": "变化检测 · 分析",
    "cl.download": "地物分类 · 下载影像",
    "ts.run": "时间序列",
    "batch.task": "批量任务",
}


@st.cache_resource(show_spinner=False)
def _runner(jobs_dir, mode):
    return jobs.get_runner(jobs_dir, mode=mode)


def get_runner():
    """本服务进程的任务执行器（任务目录 RSAP_JOBS_DIR，默认 data/jobs）。

    经 st.cache_resource 保存：开发时修改代码、Streamlit 重新导入 services.jobs 后
    仍用原来的执行器，不会再建一个把正在运行的任务误记为 interrupted。
    """
    return _runner(str(jobs.default_jobs_dir()), os.environ.get("RSAP_JOB_MODE") or "process")


def fmt_seconds(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} 秒"
    if seconds < 3600:
        return f"{seconds // 60} 分 {seconds % 60:02d} 秒"
    return f"{seconds // 3600} 小时 {seconds % 3600 // 60} 分"


def _notice_key(slot):
    return f"{slot}__notice"


def submit(slot, kind, params, *, title, meta=None):
    """提交任务并在 slot 中跟踪；返回任务 ID。"""
    job_id = get_runner().submit(kind, params, title=title, meta={**(meta or {}), "slot": slot})
    track(slot, job_id)
    return job_id


def track(slot, job_id):
    """在 slot 中跟踪已提交的任务（如任务中心重试后交回原页面）。"""
    st.session_state[slot] = job_id
    st.session_state.pop(_notice_key(slot), None)


def settle(slot, *, label, on_success, persist=None):
    """收尾已结束的任务：成功时调用 on_success(结果)，失败 / 取消时记下原因。

    on_success 可返回提示文字，或完整的提示 (tone, 标题, 说明)（如成功但有警告）。

    Returns:
        True 表示 slot 中的任务仍在排队或运行。
    """
    S = st.session_state
    job_id = S.get(slot)
    if not job_id:
        return False
    runner = get_runner()
    job = runner.get(job_id)
    if job is not None and job.active:
        return True

    S[slot] = None
    notice = None
    if job is not None and job.state == "succeeded":
        try:
            out = on_success(runner.result(job_id))
            notice = out if isinstance(out, tuple) else (
                "success", f"{label}完成", out or f"用时 {fmt_seconds(job.elapsed_s)}")
        except Exception as e:
            notice = ("danger", f"{label}的结果无法读取", f"{type(e).__name__}: {e}")
    elif job is not None:
        name, tone = STATE_LABELS.get(job.state, (job.state, "warning"))
        notice = (tone, f"{label}{name}", job.error)
    if notice:
        S[_notice_key(slot)] = notice
    if persist:
        persist()
    return False


def render_job(job, runner=None):
    """进度条 + 说明行（默认的进度显示）。"""
    if job.cancel_requested:
        text = "正在取消…"
    elif job.state == "queued":
        ahead = (runner.queue_position(job.id) - 1) if runner else 0
        text = f"排队中，前面还有 {ahead} 个任务" if ahead > 0 else "排队中，等待空闲的工作进程"
    else:
        text = job.message or STATE_LABELS[job.state][0]
    st.progress(min(job.progress, 100.0) / 100, text=text)
    st.caption(f"{job.title} · 已用时 {fmt_seconds(job.elapsed_s)} · 任务 {job.id}")


def progress(slot, *, render=None, interval=1.0):
    """slot 中的任务在运行：显示进度与取消按钮（每 interval 秒刷新）；否则显示上一次的结果提示。"""
    S = st.session_state
    job_id = S.get(slot)
    if not job_id:
        notice = S.get(_notice_key(slot))
        if notice:
            status_banner(*notice)
        return
    runner = get_runner()

    @st.fragment(run_every=interval)
    def _live():
        job = runner.get(job_id)
        if job is None or not job.active:
            st.rerun()          # 整页刷新，由 settle() 收尾
        (render or render_job)(job, runner)
        if st.button("取消任务", key=f"{slot}__cancel", icon=":material/cancel:",
                     disabled=job.cancel_requested):
            runner.cancel(job_id)

    _live()


def forget(*slots):
    """不再跟踪这些任务（如切换了区域）；任务照常运行，结果可在任务中心查看。"""
    for slot in slots:
        st.session_state[slot] = None
        st.session_state.pop(_notice_key(slot), None)
