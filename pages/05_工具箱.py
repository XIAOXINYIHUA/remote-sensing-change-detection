"""System, storage and runtime information."""

import os
import platform
import shutil
import sys
from pathlib import Path

_base = Path(__file__).resolve().parent.parent
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))

import pandas as pd
import streamlit as st

from frontend.components.ui import empty_state, page_header, section_header, status_banner
from frontend.utils import setup_matplotlib_font

setup_matplotlib_font()


def _dir_summary(path: Path) -> tuple[int, float]:
    if not path.exists():
        return 0, 0.0
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files) / 1024 / 1024


storage_rows = []
for name in ["data", "output", "tmp", "models", "reports"]:
    count, size_mb = _dir_summary(_base / name)
    storage_rows.append({"目录": name, "文件数": count, "占用": f"{size_mb:,.1f} MB"})

data_files, data_mb = _dir_summary(_base / "data")
output_files, output_mb = _dir_summary(_base / "output")
tmp_files, tmp_mb = _dir_summary(_base / "tmp")
disk = shutil.disk_usage(_base)

page_header(
    "系统与数据",
    "管理",
    "本地存储、运行环境与模块可用性。",
    meta={
        "Python": platform.python_version(),
        "Streamlit": st.__version__,
        "平台": platform.system(),
    },
    status="运行正常",
    status_tone="success",
)

metrics = st.columns(4)
metrics[0].metric(f"影像数据 · {data_files} 个文件", f"{data_mb:,.0f} MB")
metrics[1].metric(f"分析成果 · {output_files} 个文件", f"{output_mb:,.0f} MB")
metrics[2].metric(f"临时文件 · {tmp_files} 个文件", f"{tmp_mb:,.0f} MB")
metrics[3].metric("磁盘可用", f"{disk.free / 1024**3:,.1f} GB")

tab_storage, tab_runtime, tab_capability = st.tabs(["存储概览", "运行环境", "能力状态"])

with tab_storage:
    section_header("目录占用", "项目工作目录的文件数量与磁盘占用。", "本地")
    st.dataframe(pd.DataFrame(storage_rows), use_container_width=True, hide_index=True)

    section_header("最近成果", "output 目录中最近生成的文件。")
    output_root = _base / "output"
    recent_outputs = sorted(
        (path for path in output_root.rglob("*") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:12] if output_root.exists() else []
    if recent_outputs:
        rows = [
            {
                "文件": path.name,
                "项目": path.parent.name,
                "格式": path.suffix.lower().lstrip(".").upper() or "--",
                "大小": f"{path.stat().st_size / 1024:,.0f} KB",
            }
            for path in recent_outputs
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        empty_state("暂无成果文件", "完成分析后会写入 output 目录。", "OUT")

with tab_runtime:
    section_header("运行环境", "解释器、项目路径与核心依赖版本。", "环境")
    status_banner("info", "项目路径", str(_base))
    dependencies = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("rasterio", "rasterio"),
        ("geopandas", "geopandas"),
        ("shapely", "shapely"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("folium", "folium"),
        ("reportlab", "reportlab"),
    ]
    rows = []
    for package, module_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "已安装")
            rows.append({"依赖": package, "版本": version, "状态": "可用"})
        except ImportError:
            rows.append({"依赖": package, "版本": "--", "状态": "缺失"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_capability:
    section_header("模块能力", "当前版本中各业务模块的可用状态。", "能力")
    capabilities = [
        {"模块": "Sentinel-2 数据获取", "状态": "可用", "范围": "光学多光谱 · 10 m"},
        {"模块": "Landsat 8/9 数据获取", "状态": "可用", "范围": "光学多光谱 · 30 m"},
        {"模块": "Sentinel-1 数据获取", "状态": "受限", "范围": "SAR 数据 · 光谱指数不可用"},
        {"模块": "变化检测", "状态": "可用", "范围": "多级、二值与机器学习"},
        {"模块": "地物分类", "状态": "可用", "范围": "阈值、GMM 与主动学习"},
        {"模块": "时间序列", "状态": "可用", "范围": "Mann-Kendall 与 Sen's slope"},
        {"模块": "PDF 报告", "状态": "可用", "范围": "分析结果自动汇编"},
    ]
    st.dataframe(pd.DataFrame(capabilities), use_container_width=True, hide_index=True)
