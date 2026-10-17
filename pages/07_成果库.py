"""Versioned analysis result library."""

import os
import sys
from pathlib import Path

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)

import pandas as pd
import streamlit as st

from frontend.components.ui import empty_state, page_header, section_header, task_row_heading
from frontend.project_state import ProjectState
from frontend.utils import setup_matplotlib_font

setup_matplotlib_font()

ps = ProjectState.get()
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
mime_types = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".csv": "text/csv",
    ".json": "application/json",
    ".geojson": "application/geo+json",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".zip": "application/zip",
}

successful = [task for task in ps.task_history if task.get("status") == "success"]
existing_files = []
for task in successful:
    existing_files.extend(
        Path(path) for path in task.get("output_files", {}).values()
        if path and os.path.exists(path)
    )
total_size_mb = sum(path.stat().st_size for path in existing_files) / 1024 / 1024
regions = sorted({task.get("region", "未命名区域") for task in successful})

page_header(
    "成果库",
    "工作区",
    "已完成分析的版本、摘要与输出文件。",
    meta={
        "成果": len(successful),
        "项目": len(regions),
        "文件": len(existing_files),
        "占用": f"{total_size_mb:,.1f} MB",
    },
    status="成果可用" if successful else "暂无成果",
    status_tone="success" if successful else "neutral",
)

if not successful:
    section_header("分析成果", "完成分析后自动归档。", "归档")
    empty_state("暂无分析成果", "成功完成的任务会在这里生成版本记录。", "00")
    st.page_link(
        "pages/00_首页.py", label="返回项目总览",
        icon=":material/arrow_back:", use_container_width=False,
    )
else:
    # Assign versions chronologically so the newest result receives the highest version.
    version_counter = {}
    version_by_id = {}
    for index, task in enumerate(successful):
        group = (task.get("region", ""), task.get("mode", ""), task.get("index_type", ""))
        version_counter[group] = version_counter.get(group, 0) + 1
        version_by_id[id(task)] = version_counter[group]

    section_header("分析成果", "按项目和分析类型筛选版本记录。", "归档")
    filters = st.columns([1, 1, 2])
    with filters[0]:
        selected_region = st.selectbox("项目", ["全部"] + regions)
    with filters[1]:
        selected_mode = st.selectbox("类型", ["全部"] + list(mode_labels.values()))

    mode_values = {label: key for key, label in mode_labels.items()}
    filtered = [
        task for task in successful[::-1]
        if (selected_region == "全部" or task.get("region") == selected_region)
        and (selected_mode == "全部" or task.get("mode") == mode_values.get(selected_mode))
    ]

    if not filtered:
        empty_state("没有匹配的成果", "调整项目或类型筛选。", "00")

    for index, task in enumerate(filtered):
        task_id = task.get("id", f"result_{index}")
        mode = task.get("mode", "")
        version = version_by_id.get(id(task), 1)
        detail = " · ".join(part for part in [
            f"{task.get('index_type', '')} {task.get('year1', '')}-{task.get('year2', '')}".strip(),
            f"{task.get('duration_s', 0):.0f} 秒",
            task.get("stats_summary", ""),
        ] if part)
        files = task.get("output_files", {})

        with st.container(border=True):
            text_col, open_col, files_col = st.columns([6, 1, 1.25])
            with text_col:
                task_row_heading(
                    task.get("region", "未命名区域"),
                    mode_labels.get(mode, "分析成果"),
                    f"v{version}",
                    "success",
                    detail,
                )
            with open_col:
                if st.button(
                    "打开", icon=":material/open_in_new:",
                    key=f"result_open_{task_id}_{index}", use_container_width=True,
                ):
                    st.switch_page(target_pages.get(mode, "pages/01_变化检测.py"))
            with files_col:
                with st.popover("输出文件", icon=":material/folder_open:", use_container_width=True):
                    available = 0
                    for file_type, file_path in files.items():
                        path = Path(file_path) if file_path else None
                        if path and path.exists():
                            available += 1
                            st.caption(f"{file_type.upper()} · {path.stat().st_size / 1024:,.0f} KB")
                            with path.open("rb") as handle:
                                st.download_button(
                                    path.name,
                                    handle.read(),
                                    path.name,
                                    mime_types.get(path.suffix.lower(), "application/octet-stream"),
                                    key=f"download_{task_id}_{index}_{file_type}",
                                    icon=":material/download:",
                                    use_container_width=True,
                                )
                    if not available:
                        st.caption("输出文件已移动或删除")

    section_header("成果汇总", "当前筛选结果的关键执行信息。", "对比")
    rows = [
        {
            "项目": task.get("region", ""),
            "类型": mode_labels.get(task.get("mode", ""), ""),
            "版本": f"v{version_by_id.get(id(task), 1)}",
            "指数": task.get("index_type", ""),
            "时相": f"{task.get('year1', '')}-{task.get('year2', '')}",
            "耗时": f"{task.get('duration_s', 0):.0f} 秒",
            "摘要": task.get("stats_summary", ""),
        }
        for task in filtered
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
