"""
遥感分析平台 v3 — 批量处理页面

功能:
  多区域/多年份/多指数批量运行
  任务配置可视化
  运行进度与汇总结果
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)

from pathlib import Path
import streamlit as st
import pandas as pd

from frontend.utils import (
    setup_matplotlib_font, DATA_SOURCE_LABELS, INDEX_HELP,
    slug as _slug,
)
from frontend.components import jobs as jobs_ui
from frontend.project_state import ProjectState
from frontend.components.ui import (
    empty_state,
    page_header,
    section_header,
    status_banner,
    task_row_heading,
    workflow_bar,
)

setup_matplotlib_font()

BATCH_CONFIG_PATH = Path(_base) / "batch" / "batch_config.json"

# =====================================================================
#  Session State
# =====================================================================

S = st.session_state
_INIT = {
    "bt_tasks": [],
    "bt_jobs": [],              # 最近一次「执行全部任务」提交的后台任务 ID（按队列顺序）
    "bt_result_cache": {},      # 任务 ID → 汇总用的结果（已结束的任务只读一次结果文件）
    "bt_defaults_loaded": False,
}
for k, v in _INIT.items():
    if k not in S:
        S[k] = v

# 跟踪的任务写入项目状态的 bt 节：刷新浏览器后仍能看到进度与结果（批量任务各有区域，不随项目区域清空）


def _save_state():
    ProjectState.save_page("bt", {"bt_jobs": S.bt_jobs})


for _k, _v in ProjectState.load_page("bt", ["bt_jobs"], aoi_bound=False).items():
    S[_k] = _v


# =====================================================================
#  任务管理
# =====================================================================

def _load_default_tasks():
    """从 batch_config.json 加载默认任务列表。"""
    if BATCH_CONFIG_PATH.exists():
        with open(BATCH_CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("tasks", [])
    return []


def _save_config():
    """保存当前任务到配置文件。"""
    cfg = {
        "tasks": S.bt_tasks,
        "output_base": "output",
        "parallel": False,
    }
    os.makedirs(str(BATCH_CONFIG_PATH.parent), exist_ok=True)
    with open(BATCH_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _submit_batch():
    """每个任务提交为一个后台任务，按队列顺序执行（同时运行数见 config.JOB_MAX_WORKERS）。"""
    runner = jobs_ui.get_runner()
    S.bt_jobs = []
    for i, task in enumerate(S.bt_tasks):
        name = task.get("id") or f"task-{i + 1}"
        S.bt_jobs.append(runner.submit(
            "batch.task", {"task": task, "base_dir": _base}, title=f"批量任务 · {name}",
            meta={"page": "pages/04_批量处理.py", "task_id": name, "region": task.get("region")}))
    _save_state()
    st.rerun()


def _batch_jobs():
    runner = jobs_ui.get_runner()
    return [job for job in map(runner.get, S.bt_jobs) if job is not None]


def _job_result(job):
    """已结束任务的汇总记录：成功时为任务返回值，否则记下原因（参数错误、无影像、取消等）。"""
    if job.id not in S.bt_result_cache:
        res = jobs_ui.get_runner().result(job.id) if job.state == "succeeded" else None
        if res is None:
            reason = "结果文件缺失" if job.state == "succeeded" else (
                job.error or jobs_ui.STATE_LABELS[job.state][0])
            res = {"task_id": job.meta.get("task_id", job.id), "error": reason, "elapsed": 0}
        S.bt_result_cache[job.id] = res
    return S.bt_result_cache[job.id]


@st.fragment(run_every=1.0)
def _live_batch():
    """运行中的批量任务：总进度 + 逐个任务状态，全部结束后整页刷新出汇总。"""
    jobs = _batch_jobs()
    if not any(job.active for job in jobs):
        st.rerun()
    done = sum(not job.active for job in jobs)
    overall = sum(100.0 if not job.active else job.progress for job in jobs) / len(jobs)
    st.progress(overall / 100, text=f"已结束 {done} / {len(jobs)} 个任务")
    for job in jobs:
        name, tone = jobs_ui.STATE_LABELS[job.state]
        if job.state == "running":
            detail = (f"{job.message} · {job.progress:.0f}% · "
                      f"已用时 {jobs_ui.fmt_seconds(job.elapsed_s)}")
        elif job.state == "queued":
            detail = "等待空闲的工作进程"
        else:
            detail = job.error or f"用时 {jobs_ui.fmt_seconds(job.elapsed_s)}"
        with st.container(border=True):
            task_row_heading(job.meta.get("task_id", job.id), "批量变化检测", name, tone, detail)
    if st.button("取消全部任务", icon=":material/cancel:", key="bt_cancel_all"):
        runner = jobs_ui.get_runner()
        for job in jobs:
            runner.cancel(job.id)


# =====================================================================
#  UI
# =====================================================================

if not S.bt_tasks and not S.bt_defaults_loaded:
    default_tasks = _load_default_tasks()
    if default_tasks:
        S.bt_tasks = default_tasks
    S.bt_defaults_loaded = True

batch_jobs = _batch_jobs()
bt_running = any(job.active for job in batch_jobs)
bt_results = [_job_result(job) for job in batch_jobs if not job.active]
success_results = sum(1 for result in bt_results if not result.get("error"))
page_header(
    "批量任务",
    "自动化",
    "多区域、多时相任务的配置、后台执行与结果对比。",
    meta={
        "任务": len(S.bt_tasks),
        "已执行": len(bt_results),
        "成功": success_results,
    },
    status="运行中" if bt_running else "待执行" if S.bt_tasks else "空队列",
    status_tone="info" if bt_running else "warning" if S.bt_tasks else "neutral",
)
workflow_bar(
    ["配置队列", "检查参数", "后台执行", "汇总结果"],
    active=3 if bt_results and not bt_running else 2 if bt_running else 0,
    completed=[0, 1, 2] if bt_results and not bt_running else [0, 1] if bt_running else [],
)

tab_config, tab_run, tab_summary = st.tabs(["任务配置", "执行队列", "结果汇总"])

# =====================================================================
#  Tab 1 — 任务配置
# =====================================================================

with tab_config:
    section_header("任务队列", "检查任务范围与参数，保存后可重复执行。", "配置")

    # 显示已有任务
    if S.bt_tasks:
        df_tasks = pd.DataFrame(S.bt_tasks)
        display_cols = [c for c in ["id", "region", "year1", "year2", "index_type",
                                      "data_source", "threshold"] if c in df_tasks.columns]
        st.dataframe(df_tasks[display_cols])

        with st.expander("完整配置 JSON"):
            st.json(S.bt_tasks)

        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if st.button("清空队列", icon=":material/delete_sweep:", width="stretch",
                         disabled=bt_running):
                S.bt_tasks = []
                S.bt_jobs = []
                _save_state()
                st.rerun()
        with c2:
            if st.button("保存配置", icon=":material/save:", width="stretch"):
                _save_config()
                status_banner("success", "任务配置已保存", "batch/batch_config.json")
    else:
        empty_state("队列中没有任务", "在下方添加一个或多个分析任务。", "00")

    # 添加新任务
    with st.expander("添加任务", expanded=not bool(S.bt_tasks)):
        boundary_names = [path.name for path in sorted(Path(_base).glob("*.geojson"))]
        with st.form("batch_task_form"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                task_region = st.text_input("区域名", "", placeholder="例如：新余")
                task_boundary = st.selectbox(
                    "边界文件",
                    boundary_names or ["未发现 GeoJSON"],
                    disabled=not boundary_names,
                )
            with col_b:
                task_year1 = st.number_input("前时相年份", 2015, 2027, 2025)
                task_year2 = st.number_input("后时相年份", 2015, 2027, 2026)
                task_ms, task_me = st.select_slider(
                    "月份范围", options=list(range(1, 13)), value=(6, 10),
                    format_func=lambda value: f"{value} 月",
                )
            with col_c:
                task_data_src = st.selectbox(
                    "数据源", ["sentinel2", "landsat8", "landsat9"],
                    format_func=lambda value: DATA_SOURCE_LABELS.get(value, value),
                )
                task_index = st.selectbox(
                    "指数", ["NDVI", "NDWI", "MNDWI", "NBR", "EVI", "SAVI", "NDBI"],
                    format_func=lambda value: f"{value} · {INDEX_HELP.get(value, '')}",
                )
                task_threshold = st.slider("变化阈值", 0.05, 0.50, 0.15, 0.01)

            valid_task = (
                bool(task_region.strip()) and bool(boundary_names)
                and task_year2 > task_year1 and task_me >= task_ms
            )
            submitted = st.form_submit_button(
                "添加到队列", type="primary", icon=":material/add:",
                width="stretch", disabled=not valid_task,
            )

        task_id = f"{_slug(task_region) if task_region else 'region'}_{task_year1}_{task_year2}"
        if submitted:
            new_task = {
                "id": task_id,
                "region": task_region.strip(),
                "boundary": task_boundary,
                "year1": int(task_year1),
                "year2": int(task_year2),
                "month_start": int(task_ms),
                "month_end": int(task_me),
                "data_source": task_data_src,
                "index_type": task_index,
                "threshold": task_threshold,
                "detection_mode": "multi",
                "formats": "geojson,csv",
                "report": True,
            }
            S.bt_tasks.append(new_task)
            st.rerun()


# =====================================================================
#  Tab 2 — 运行
# =====================================================================

with tab_run:
    section_header("执行队列", "任务按列表顺序进入后台队列；可切换页面，任务继续运行。", "运行")

    if not S.bt_tasks:
        empty_state("没有可执行任务", "请先在“任务配置”中添加任务。", "00")
    else:
        status_banner("info", f"队列包含 {len(S.bt_tasks)} 个任务",
                      f"按配置顺序执行，最多同时运行 {jobs_ui.get_runner().max_workers} 个。")
        if st.button("执行全部任务", type="primary", icon=":material/play_arrow:",
                      width="stretch", disabled=bt_running):
            _submit_batch()

        if bt_running:
            _live_batch()
        elif bt_results:
            section_header("执行结果", "最近一次执行的任务状态。")
            for r in bt_results:
                err = r.get("error")
                stats = r.get("stats", {})
                detail = str(err) if err else f"耗时 {r.get('elapsed', 0):.0f} 秒"
                if not err and "changed_total" in stats:
                    detail += f" · 变化 {stats['changed_total']['area_ha']:,.0f} ha"
                elif not err and "changed_area_ha" in stats:
                    detail += f" · 变化 {stats['changed_area_ha']:,.0f} ha"
                with st.container(border=True):
                    task_row_heading(
                        r.get("task_id", "未命名任务"),
                        "批量变化检测",
                        "失败" if err else "已完成",
                        "danger" if err else "success",
                        detail,
                    )


# =====================================================================
#  Tab 3 — 汇总
# =====================================================================

with tab_summary:
    section_header("结果汇总", "跨区域、跨时相任务的指标对比。", "汇总")

    if not bt_results:
        empty_state("暂无批量结果", "执行队列后显示任务对比。", "SUM")
    else:
        rows = []
        for r in bt_results:
            err = r.get("error")
            if err:
                rows.append({
                    "任务ID": r.get("task_id", ""),
                    "结果": "失败",
                    "变化面积(ha)": "-",
                    "占比(%)": "-",
                    "耗时(s)": "-",
                })
            else:
                s = r.get("stats", {})
                if "changed_total" in s:
                    t = s["changed_total"]
                    rows.append({
                        "任务ID": r.get("task_id", ""),
                        "结果": "完成",
                        "变化面积(ha)": f"{t['area_ha']:,.0f}",
                        "占比(%)": f"{t['percent']:.1f}",
                        "耗时(s)": f"{r.get('elapsed', 0):.0f}",
                    })
                elif "changed_area_ha" in s:
                    rows.append({
                        "任务ID": r.get("task_id", ""),
                        "结果": "完成",
                        "变化面积(ha)": f"{s['changed_area_ha']:,.0f}",
                        "占比(%)": "-",
                        "耗时(s)": f"{r.get('elapsed', 0):.0f}",
                    })
                else:
                    rows.append({
                        "任务ID": r.get("task_id", ""),
                        "结果": "完成",
                        "变化面积(ha)": "-",
                        "占比(%)": "-",
                        "耗时(s)": f"{r.get('elapsed', 0):.0f}",
                    })

        if rows:
            st.dataframe(pd.DataFrame(rows))

            total_time = sum(
                r.get("elapsed", 0) for r in bt_results
                if "error" not in r or not r.get("error"))
            metrics = st.columns(3)
            metrics[0].metric("任务总数", len(bt_results))
            metrics[1].metric("成功任务", success_results)
            metrics[2].metric("总耗时", f"{total_time:.0f} 秒")
