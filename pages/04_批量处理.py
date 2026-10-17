"""
遥感分析平台 v3 — 批量处理页面

功能:
  多区域/多年份/多指数批量运行
  任务配置可视化
  运行进度与汇总结果
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)

from pathlib import Path
import streamlit as st
import pandas as pd

import config
from frontend.utils import (
    setup_matplotlib_font, DATA_SOURCE_LABELS, INDEX_HELP,
    slug as _slug,
)
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
    "bt_running": False,
    "bt_progress": {},
    "bt_results": [],
    "bt_current_task": None,
    "bt_defaults_loaded": False,
}
for k, v in _INIT.items():
    if k not in S:
        S[k] = v


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


def _run_batch():
    """按顺序运行所有任务。"""
    S.bt_running = True
    S.bt_results = []
    tasks = S.bt_tasks
    total = len(tasks)

    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, task in enumerate(tasks):
        S.bt_current_task = task.get("id", f"任务{i+1}")
        progress_text.info(f"[{i+1}/{total}] 正在执行 {task.get('id', '')}...")
        progress_bar.progress((i) / max(total, 1))

        try:
            config.REGION_NAME = task["region"]
            config.BOUNDARY_FILE = task["boundary"]
            config.YEAR1 = int(task["year1"])
            config.YEAR2 = int(task["year2"])
            config.MONTH_START = int(task.get("month_start", 6))
            config.MONTH_END = int(task.get("month_end", 10))
            config.DETECTION_MODE = task.get("detection_mode", "multi")
            config.INDEX_TYPE = task.get("index_type", "NDVI")
            config.CHANGE_THRESHOLD = task.get("threshold", 0.15)
            config.SEVERE_THRESHOLD = task.get("severe_threshold", 0.25)
            config.OUTPUT_FORMATS = task.get("formats", "geojson,csv")
            config.GENERATE_REPORT = task.get("report", True)
            config.DATA_SOURCE = task.get("data_source", "sentinel2")

            if task.get("min_area_m2"):
                config.MIN_CHANGE_AREA_M2 = task["min_area_m2"]
                config.MIN_CHANGE_AREA_PX = max(1, int(task["min_area_m2"]) // 100)

            out_dir = os.path.join(
                _base, "output", task.get("id", f"{task['region']}_{task['year1']}_{task['year2']}"))

            from engine.core import run_pipeline_v3

            def _cb(msg, pct):
                pass

            t0 = time.time()
            result = run_pipeline_v3(
                data_source_name=config.DATA_SOURCE,
                data_dir=os.path.join(_base, "data", task.get("id", "default")),
                out_dir=out_dir,
                index_type=config.INDEX_TYPE,
                detection_mode=config.DETECTION_MODE,
                progress_callback=_cb,
            )
            elapsed = round(time.time() - t0, 1)
            result["task_id"] = task.get("id", f"task-{i}")
            result["elapsed"] = elapsed
            S.bt_results.append(result)

        except Exception as e:
            S.bt_results.append({
                "task_id": task.get("id", f"task-{i}"),
                "error": str(e),
                "elapsed": 0,
            })
            st.error(f"{task.get('id', '')}: {e}")

    progress_bar.progress(1.0)
    progress_text.success(f"{total} 个任务已完成")
    S.bt_running = False


# =====================================================================
#  UI
# =====================================================================

if not S.bt_tasks and not S.bt_defaults_loaded:
    default_tasks = _load_default_tasks()
    if default_tasks:
        S.bt_tasks = default_tasks
    S.bt_defaults_loaded = True

success_results = sum(1 for result in S.bt_results if not result.get("error"))
page_header(
    "批量任务",
    "自动化",
    "多区域、多时相任务的配置、顺序执行与结果对比。",
    meta={
        "任务": len(S.bt_tasks),
        "已执行": len(S.bt_results),
        "成功": success_results,
    },
    status="运行中" if S.bt_running else "待执行" if S.bt_tasks else "空队列",
    status_tone="info" if S.bt_running else "warning" if S.bt_tasks else "neutral",
)
workflow_bar(
    ["配置队列", "检查参数", "顺序执行", "汇总结果"],
    active=3 if S.bt_results and not S.bt_running else 2 if S.bt_running else 0,
    completed=[0, 1, 2] if S.bt_results and not S.bt_running else [0, 1] if S.bt_running else [],
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
        st.dataframe(df_tasks[display_cols], use_container_width=True)

        with st.expander("完整配置 JSON"):
            st.json(S.bt_tasks)

        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if st.button("清空队列", icon=":material/delete_sweep:", use_container_width=True):
                S.bt_tasks = []
                S.bt_results = []
                st.rerun()
        with c2:
            if st.button("保存配置", icon=":material/save:", use_container_width=True):
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
                use_container_width=True, disabled=not valid_task,
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
    section_header("执行队列", "任务按列表顺序运行，执行结果实时写入汇总。", "运行")

    if not S.bt_tasks:
        empty_state("没有可执行任务", "请先在“任务配置”中添加任务。", "00")
    else:
        status_banner("info", f"队列包含 {len(S.bt_tasks)} 个任务", "任务将按配置顺序执行。")
        can_run = not S.bt_running
        if st.button("执行全部任务", type="primary", icon=":material/play_arrow:",
                      use_container_width=True, disabled=not can_run):
            _run_batch()

        if S.bt_running:
            status_banner("info", "正在执行", S.bt_current_task or "")

        if S.bt_results:
            section_header("执行结果", "当前会话中的任务状态。")
            for r in S.bt_results:
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

    if not S.bt_results:
        empty_state("暂无批量结果", "执行队列后显示任务对比。", "SUM")
    else:
        rows = []
        for r in S.bt_results:
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            total_time = sum(
                r.get("elapsed", 0) for r in S.bt_results
                if "error" not in r or not r.get("error"))
            metrics = st.columns(3)
            metrics[0].metric("任务总数", len(S.bt_results))
            metrics[1].metric("成功任务", success_results)
            metrics[2].metric("总耗时", f"{total_time:.0f} 秒")
