"""
遥感分析平台 v3 — 时间序列分析页面

功能:
  多年植被/水体指数变化趋势分析
  Mann-Kendall 趋势检验 + Sen's slope
  年际变化可视化
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["GDAL_TMPDIR"] = os.path.join(_base, "tmp")
os.environ["TMP"] = os.environ["GDAL_TMPDIR"]
os.environ["TEMP"] = os.environ["GDAL_TMPDIR"]
os.makedirs(os.environ["GDAL_TMPDIR"], exist_ok=True)

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

if str(Path(_base)) not in sys.path:
    sys.path.insert(0, str(_base))

import config
from services import timeseries as ts_svc
from frontend.utils import (
    setup_matplotlib_font,
    DATA_SOURCE_LABELS,
)
from frontend.project_state import ProjectState
from frontend.components import jobs as jobs_ui
from frontend.components.sidebar import (
    data_source_picker, index_picker, month_range, region_picker, year_series,
)
from frontend.components.ui import (
    empty_state,
    page_header,
    section_header,
    sidebar_header,
    status_banner,
    workflow_bar,
)

setup_matplotlib_font()


# =====================================================================
#  Session State
# =====================================================================

S = st.session_state
JOB_TS = "ts_job"

# 分析区域以项目为准（任何页面的侧栏或向导切换区域都写入项目状态）
ps = ProjectState.get()
S.ts_boundary, S.ts_region, S.ts_slug = ps.boundary or None, ps.region or None, ps.slug or None

_INIT = {
    "ts_done": False,
    "ts_result": None,
    "ts_years": None,
    "ts_stats_summary": None,
    "ts_trend_data": None,
    JOB_TS: None,
}
for k, v in _INIT.items():
    if k not in S:
        S[k] = v

# 区域换了：清掉属于旧区域的趋势结果，不再跟踪旧区域的任务
if ProjectState.aoi_changed("ts"):
    for k, v in _INIT.items():
        S[k] = v
    jobs_ui.forget(JOB_TS)

# 状态持久化：项目状态的 ts 节（正在跟踪的任务）
_PERSIST = [JOB_TS]


def _save_state():
    ProjectState.save_page("ts", {k: S.get(k) for k in _PERSIST})


def _load_state():
    for k, v in ProjectState.load_page("ts", _PERSIST).items():
        S[k] = v


_load_state()


# =====================================================================
#  运行分析
# =====================================================================

def _submit_timeseries(years, idx_type, data_src, ms, me):
    """提交全流程任务（后台执行）：收集影像 → 计算指数 → 趋势分析。"""
    boundary = str(Path(S.ts_boundary).resolve())
    years = [int(y) for y in years]
    # 影像存入 data/cache 的单期快照；data/ 下只放下载用的临时目录
    jobs_ui.submit(JOB_TS, "ts.run", {
        "boundary_file": boundary, "years": years, "index_type": idx_type,
        "data_source": data_src, "month_start": int(ms), "month_end": int(me),
        "max_cloud_pct": ProjectState.get().max_cloud_pct,
        "data_dir": os.path.join(_base, "data"),
    }, title=f"{S.ts_region or '未知'} · {idx_type} 趋势 {years[0]}–{years[-1]}（{len(years)} 期）",
       meta={"page": "pages/03_时间序列.py", "region": S.ts_region, "boundary": boundary})
    _save_state()
    st.rerun()


def _apply_timeseries(res):
    S.ts_years = res["ts_data"]["years"]
    S.ts_trend_data = res["trend"]
    S.ts_stats_summary = res["summary"]
    S.ts_result = res
    S.ts_done = True
    return f"有效时相 {len(S.ts_years)} 期：{'、'.join(map(str, S.ts_years))}"


running = jobs_ui.settle(JOB_TS, label="趋势分析", on_success=_apply_timeseries,
                         persist=_save_state)


# =====================================================================
#  Sidebar 参数面板
# =====================================================================

sidebar_header("时间序列参数", "多年时相、数据源与趋势指数")


region_picker("ts")
st.sidebar.markdown("---")

st.sidebar.subheader("时间范围")
years, start_year, end_year, step = year_series(2020, 2026)
ms, me = month_range(ps.month_start or config.MONTH_START, ps.month_end or config.MONTH_END)
st.sidebar.markdown("---")

st.sidebar.subheader("数据源与指数")
data_src = data_source_picker(ps.data_source)
idx_type = index_picker(data_src, default=ps.index_type,
                        allowed=["NDVI", "NDWI", "MNDWI", "NBR", "EVI", "SAVI"],
                        remember_as="ts_cur_idx")

# =====================================================================
#  主面板
# =====================================================================

region_display = S.ts_region or "未选择"
source_label = DATA_SOURCE_LABELS.get(data_src, data_src)
page_header(
    "时间序列",
    "分析工作台",
    f"{region_display} 的多年指数趋势与显著性分析。",
    meta={
        "区域": region_display,
        "年份": f"{start_year} - {end_year}",
        "步长": f"{step} 年",
        "时相": f"{len(years)} 期",
        "数据": source_label,
        "指数": idx_type,
    },
    status="运行中" if running else "分析完成" if S.ts_result else "等待执行",
    status_tone="info" if running else "success" if S.ts_result else "warning",
)
workflow_bar(
    ["配置序列", "获取影像", "趋势检验", "结果解读"],
    active=2 if running else 3 if S.ts_result else 0,
    completed=[0, 1] if running else [0, 1, 2] if S.ts_result else [],
)

valid_period = len(years) >= ts_svc.MIN_PERIODS and me >= ms
can_run = (
    bool(S.ts_boundary) and Path(S.ts_boundary).exists()
    and not running and valid_period
)
if not valid_period:
    status_banner("warning", "时间序列不足",
                  f"至少需要 {ts_svc.MIN_PERIODS} 个时相（每个像元至少 {ts_svc.MIN_PERIODS} 期有效观测"
                  "才做趋势检验），且结束时间不得早于开始时间。")
btn = st.button(
    f"运行 {idx_type} 趋势分析 · {len(years)} 期",
    type="primary", icon=":material/play_arrow:",
    width="stretch", disabled=not can_run)
if btn and S.ts_boundary:
    _submit_timeseries(years, idx_type, data_src, ms, me)
jobs_ui.progress(JOB_TS)

# =====================================================================
#  结果显示
# =====================================================================

if S.ts_result:
    summary = S.ts_stats_summary
    trend = S.ts_trend_data

    section_header("趋势摘要", "显著改善、退化与整体变化速率。", "结果")

    # 顶部指标
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("改善区域", f"{summary.get('improve_percent', 0):.1f}%")
        with c2:
            st.metric("退化区域", f"{summary.get('degrade_percent', 0):.1f}%")
        with c3:
            sm = summary.get("mean_slope", 0)
            st.metric("平均斜率", f"{sm:.4f}" if sm != 0 else "0")
        with c4:
            st.metric("有效像元", f"{summary.get('total_pixels', 0):,}")

    # — 折线图 —
    with st.expander("年际变化", expanded=True):
        if summary and summary.get("years"):
            fig_l, ax_l = plt.subplots(1, 1, figsize=(12, 5))
            yrs = summary["years"]
            means = [summary["yearly_mean_index"][y] for y in yrs]
            ax_l.plot(yrs, means, "o-", color="#2E86AB", linewidth=2, markersize=8)
            ax_l.fill_between(yrs, means, alpha=0.25, color="#2E86AB")
            ax_l.axhline(y=0, color="#666", linewidth=0.5, linestyle="--")
            ax_l.set_xlabel("年份", fontsize=12)
            ax_l.set_ylabel(f"{idx_type} 均值", fontsize=12)
            ax_l.set_title(f"{region_display} {idx_type} 年际变化趋势", fontsize=14, fontweight="bold")
            ax_l.grid(True, alpha=0.3)
            ax_l.tick_params(colors="#526159")
            ax_l.xaxis.label.set_color("#18221D")
            ax_l.yaxis.label.set_color("#18221D")
            ax_l.title.set_color("#18221D")
            for spine in ax_l.spines.values():
                spine.set_color("#D9E1DA")
            fig_l.patch.set_facecolor("#FFFFFF")
            ax_l.set_facecolor("#FFFFFF")

            # 在图上标注改善/退化
            if summary.get("improve_percent"):
                ax_l.text(
                    0.02, 0.95,
                    f"改善 {summary['improve_percent']:.1f}%  |  退化 {summary['degrade_percent']:.1f}%",
                    transform=ax_l.transAxes, fontsize=11, color="#34443B",
                    bbox=dict(boxstyle="round", facecolor="#EDF2EE", edgecolor="#C5D0C7"))

            st.pyplot(fig_l)
            plt.close()

    # — 趋势分布 —
    with st.expander("趋势构成", expanded=True):
        if summary:
            fig_p, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))

            # 饼图
            sizes = [
                summary.get("improve_percent", 0),
                summary.get("degrade_percent", 0),
                max(0, 100 - summary.get("improve_percent", 0) - summary.get("degrade_percent", 0)),
            ]
            labels_p = ["改善", "退化", "稳定"]
            colors_p = ["#00b450", "#b40000", "#666666"]
            a1.pie(sizes, labels=labels_p, autopct="%1.1f%%",
                   colors=colors_p, startangle=90,
                   textprops={"color": "#18221D", "fontsize": 12})
            a1.set_title("趋势面积占比", fontsize=14, color="#18221D")

            # 年均柱状图
            yrs = summary["years"]
            means = [summary["yearly_mean_index"][y] for y in yrs]
            bars = a2.bar([str(y) for y in yrs], means,
                         color="#2D65A1", edgecolor="#FFFFFF", width=0.6)
            a2.set_xlabel("年份", fontsize=12, color="#18221D")
            a2.set_ylabel(f"{idx_type} 均值", fontsize=12, color="#18221D")
            a2.set_title(f"年均 {idx_type}", fontsize=14, color="#18221D")
            a2.tick_params(colors="#526159")
            a2.grid(True, alpha=0.2, axis="y")
            for spine in a2.spines.values():
                spine.set_color("#D9E1DA")

            fig_p.patch.set_facecolor("#FFFFFF")
            a1.set_facecolor("#FFFFFF")
            a2.set_facecolor("#FFFFFF")
            fig_p.tight_layout()
            st.pyplot(fig_p)
            plt.close()

    # — 趋势空间分布图 —
    with st.expander("空间趋势", expanded=True):
        if trend is not None:
            tt = trend["trend_type"]
            sv = trend.get("significant")
            slope = trend.get("slope")

            fig_m, axes_m = plt.subplots(1, 2, figsize=(18, 8))

            # 图1: 趋势分类
            trend_map = tt.copy()
            h_m, w_m = trend_map.shape
            # 降采样显示
            scale_m = max(1, min(h_m, w_m) // 500)
            if scale_m > 1:
                import cv2
                trend_small = cv2.resize(
                    trend_map.astype(np.float32),
                    (w_m // scale_m, h_m // scale_m),
                    interpolation=cv2.INTER_NEAREST).astype(np.int8)
            else:
                trend_small = trend_map

            cmap_t = ListedColormap(["#b40000", "#666666", "#00b450"])
            im0 = axes_m[0].imshow(trend_small + 1, cmap=cmap_t, interpolation="nearest")
            axes_m[0].set_title(f"{idx_type} 变化趋势 ({start_year}–{end_year})", fontsize=14, color="#18221D")
            axes_m[0].axis("off")
            legend_patches = [
                Patch(color="#00b450", label=f"改善 ({summary.get('improve_percent',0):.1f}%)"),
                Patch(color="#666666", label="稳定"),
                Patch(color="#b40000", label=f"退化 ({summary.get('degrade_percent',0):.1f}%)"),
            ]
            axes_m[0].legend(handles=legend_patches, loc="lower right",
                             fontsize=10, facecolor="#FFFFFF", labelcolor="#18221D")

            # 图2: Sen's slope 空间分布
            if slope is not None:
                slope_valid = slope.copy()
                slope_valid[~trend["significant"]] = np.nan
                s_small = slope_valid[::scale_m, ::scale_m] if scale_m > 1 else slope_valid
                vmax = max(abs(np.nanpercentile(s_small, 95)), 0.001)
                im1 = axes_m[1].imshow(s_small, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
                axes_m[1].set_title("Sen's slope (仅显著区域 p<0.05)", fontsize=14, color="#18221D")
                axes_m[1].axis("off")
                plt.colorbar(im1, ax=axes_m[1], shrink=0.5, label=f"{idx_type}/年")

            fig_m.patch.set_facecolor("#FFFFFF")
            for ax in axes_m:
                ax.set_facecolor("#FFFFFF")
            fig_m.tight_layout()
            st.pyplot(fig_m)
            plt.close()

    # — 详细统计 —
    with st.expander("详细统计"):
        if summary:
            srows = [
                {"指标": "总有效像元", "值": f"{summary['total_pixels']:,}"},
                {"指标": "改善像元", "值": f"{summary['improve_pixels']:,} ({summary['improve_percent']:.1f}%)"},
                {"指标": "退化像元", "值": f"{summary['degrade_pixels']:,} ({summary['degrade_percent']:.1f}%)"},
                {"指标": "稳定像元", "值": f"{summary['stable_pixels']:,}"},
                {"指标": "平均斜率 (Sen's)", "值": f"{summary['mean_slope']:.6f}"},
            ]
            st.table(pd.DataFrame(srows))

            # 各年均值表
            yr_rows = []
            for y in summary["years"]:
                yr_rows.append({"年份": y, f"{idx_type} 均值": f"{summary['yearly_mean_index'].get(y, 0):.4f}"})
            if yr_rows:
                section_header(f"各年 {idx_type} 均值")
                st.dataframe(pd.DataFrame(yr_rows))

elif not running:
    empty_state("暂无趋势结果", "确认侧栏参数并运行分析后显示。", "TS")
