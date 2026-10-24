"""Versioned analysis result library."""

import os
import sys
import time
from pathlib import Path

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)

import pandas as pd
import streamlit as st

from frontend.components.ui import empty_state, page_header, section_header, task_row_heading
from frontend.inventory import unarchived_outputs
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

def _download_buttons(entries, key_prefix):
    """为 (标签, 路径) 列表逐个渲染下载按钮，返回实际存在的文件数。"""
    available = 0
    for label, raw_path in entries:
        path = Path(raw_path) if raw_path else None
        if not path or not path.exists():
            continue
        available += 1
        st.caption(f"{label.upper()} · {path.stat().st_size / 1024:,.0f} KB")
        with path.open("rb") as handle:
            st.download_button(
                path.name,
                handle.read(),
                path.name,
                mime_types.get(path.suffix.lower(), "application/octet-stream"),
                key=f"{key_prefix}_{label}_{path.name}",
                icon=":material/download:",
                width="stretch",
            )
    return available


successful = [task for task in ps.task_history if task.get("status") == "success"]
archived_paths = [
    path for task in successful for path in task.get("output_files", {}).values() if path
]
existing_files = [Path(path) for path in archived_paths if os.path.exists(path)]
# CLI / 批量任务 / 旧版本生成、没有任务记录的输出也要能在成果库里找到
loose_groups = unarchived_outputs(Path(_base) / "output", archived_paths)
loose_count = sum(len(group["files"]) for group in loose_groups)
total_size_mb = (sum(path.stat().st_size for path in existing_files) / 1024 / 1024
                 + sum(group["size_mb"] for group in loose_groups))
regions = sorted({task.get("region", "未命名区域") for task in successful})
project_names = set(regions) | {group["region"] for group in loose_groups}
has_results = bool(successful or loose_groups)

page_header(
    "成果库",
    "工作区",
    "已完成分析的版本、摘要与输出文件。",
    meta={
        "成果": len(successful),
        "项目": len(project_names),
        "文件": len(existing_files) + loose_count,
        "占用": f"{total_size_mb:,.1f} MB",
    },
    status="成果可用" if has_results else "暂无成果",
    status_tone="success" if has_results else "neutral",
)

if not has_results:
    section_header("分析成果", "完成分析后自动归档。", "归档")
    empty_state("暂无分析成果", "成功完成的任务会在这里生成版本记录。", "00")
    st.page_link(
        "pages/00_首页.py", label="返回项目总览",
        icon=":material/arrow_back:",
    )

if successful:
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
                    key=f"result_open_{task_id}_{index}", width="stretch",
                ):
                    st.switch_page(target_pages.get(mode, "pages/01_变化检测.py"))
            with files_col:
                with st.popover("输出文件", icon=":material/folder_open:", width="stretch"):
                    if not _download_buttons(files.items(), f"download_{task_id}_{index}"):
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
    st.dataframe(pd.DataFrame(rows), hide_index=True)

if loose_groups:
    section_header(
        "未归档的输出",
        "output 目录中没有任务记录的文件，来自命令行、批量任务或旧版本。",
        "文件",
    )
    for group_index, group in enumerate(loose_groups):
        modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(group["modified"]))
        with st.container(border=True):
            text_col, files_col = st.columns([6, 1.6])
            with text_col:
                task_row_heading(
                    group["region"],
                    "未归档输出",
                    f"{len(group['files'])} 个文件",
                    "info",
                    f"{group['size_mb']:,.1f} MB · 最近修改 {modified}",
                )
            with files_col:
                with st.popover("输出文件", icon=":material/folder_open:", width="stretch"):
                    _download_buttons(
                        [(path.suffix.lstrip(".") or "file", path) for path in group["files"]],
                        f"loose_{group_index}",
                    )
