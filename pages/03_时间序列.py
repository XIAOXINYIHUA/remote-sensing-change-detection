"""
遥感分析平台 v3 — 时间序列分析页面

功能:
  多年植被/水体指数变化趋势分析
  Mann-Kendall 趋势检验 + Sen's slope
  年际变化可视化
"""

import os, sys, json, time, warnings
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
from frontend.utils import (
    setup_matplotlib_font,
    detect_region, slug as _slug,
    load_gj, fetch_boundary_osm,
    DATA_SOURCE_LABELS, INDEX_HELP,
    save_app_state, load_app_state,
)
from frontend.project_state import ProjectState
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

# 从 ProjectState 初始化
ps = ProjectState.get()
if ps.is_ready():
    S.setdefault("ts_boundary", ps.boundary)
    S.setdefault("ts_region", ps.region)
    S.setdefault("ts_slug", ps.slug)

_INIT = {
    "ts_boundary": None,
    "ts_region": None,
    "ts_slug": None,
    "ts_running": False,
    "ts_done": False,
    "ts_error": None,
    "ts_result": None,
    "ts_years": None,
    "ts_stats_summary": None,
    "ts_trend_data": None,
}
for k, v in _INIT.items():
    if k not in S:
        S[k] = v

STATE_FILE = Path(_base) / ".ts_state.json"


def _save_state():
    save_app_state(STATE_FILE, {
        "ts_boundary": S.get("ts_boundary"),
        "ts_region": S.get("ts_region"),
        "ts_slug": S.get("ts_slug"),
    })


def _load_state():
    loaded = load_app_state(STATE_FILE, ["ts_boundary", "ts_region", "ts_slug"])
    for k, v in loaded.items():
        S[k] = v


_load_state()


# =====================================================================
#  运行分析
# =====================================================================

def _run_timeseries(boundary, years, idx_type, data_src, ms, me):
    """全流程：收集影像 → 计算指数 → 趋势分析。"""
    S.ts_running = True
    S.ts_done = False
    S.ts_error = None

    prog_bar = st.progress(0)
    status_text = st.empty()

    try:
        from engine.timeseries import TimeSeriesAnalyzer

        status_text.info("初始化分析器...")
        prog_bar.progress(5)

        analyzer = TimeSeriesAnalyzer(boundary, data_source=data_src)
        ts_dir = os.path.join(_base, "data", "timeseries", S.ts_slug or "default")
        os.makedirs(ts_dir, exist_ok=True)

        status_text.info(f"📥 收集 {len(years)} 期 {idx_type} 影像...")
        prog_bar.progress(15)
        ts_data = analyzer.collect_time_series(
            years, index_type=idx_type,
            month_start=ms, month_end=me,
            data_dir=ts_dir)

        S.ts_years = ts_data["years"]
        status_text.success(f"✅ {len(ts_data['years'])} 期影像收集完成")
        prog_bar.progress(50)

        status_text.info("📊 逐像元趋势分析 (Mann-Kendall + Sen's slope)...")
        trend = analyzer.compute_trend(ts_data)
        prog_bar.progress(80)

        status_text.info("📋 生成统计摘要...")
        summary = analyzer.summarize(ts_data, trend)
        prog_bar.progress(95)

        S.ts_trend_data = trend
        S.ts_stats_summary = summary
        S.ts_result = {"ts_data": ts_data, "trend": trend, "summary": summary}
        S.ts_done = True
        prog_bar.progress(100)
        status_text.success("✅ 时间序列分析完成！")

    except Exception as e:
        S.ts_error = str(e)
        st.error(f"❌ 分析失败: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        S.ts_running = False
        time.sleep(1)
        status_text.empty()
        prog_bar.empty()


# =====================================================================
#  Sidebar 参数面板
# =====================================================================

sidebar_header("时间序列参数", "多年时相、数据源与趋势指数")
st.sidebar.subheader("分析区域")

exist = {f.name: str(f) for f in Path(_base).glob("*.geojson")}
cur_name = Path(S.ts_boundary).name if S.ts_boundary else None
if exist:
    sel = st.sidebar.selectbox(
        "边界文件", list(exist.keys()),
        index=list(exist.keys()).index(cur_name) if cur_name in exist else 0)
    chosen = exist[sel]
    if chosen != S.ts_boundary:
        S.ts_boundary = chosen
        r = detect_region(chosen)
        S.ts_region = r
        S.ts_slug = _slug(r)
        S.ts_done = False
        _save_state()
        st.rerun()

sn = st.sidebar.text_input("搜索地区", "", placeholder="城市/区县名")
if st.sidebar.button("获取边界", icon=":material/search:", use_container_width=True,
                      disabled=not sn.strip()):
    with st.sidebar.status(f"搜索 {sn}..."):
        gj, err = fetch_boundary_osm(sn.strip())
        if err:
            st.sidebar.error(err)
        else:
            fp = Path(_base) / f"{_slug(sn)}.geojson"
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(gj, f, ensure_ascii=False)
            S.ts_boundary = str(fp)
            S.ts_region = sn.strip()
            S.ts_slug = _slug(sn)
            S.ts_done = False
            _save_state()
            st.rerun()

if S.ts_boundary and Path(S.ts_boundary).exists():
    try:
        _, geom = load_gj(S.ts_boundary)
        st.sidebar.caption(f"中心: {geom.centroid.y:.3f}, {geom.centroid.x:.3f}")
    except Exception:
        st.sidebar.error("边界解析失败")
else:
    st.sidebar.error("请选择边界文件")

st.sidebar.markdown("---")
st.sidebar.subheader("时间范围")

start_year = st.sidebar.number_input("起始年份", 2015, 2025, 2020, 1)
end_year = st.sidebar.number_input("结束年份", 2016, 2026, 2026, 1)
step = st.sidebar.selectbox("步长", [1, 2, 3], index=0)
years = list(range(start_year, end_year + 1, step))
if years:
    st.sidebar.caption(f"共 {len(years)} 期: {years[0]} 至 {years[-1]}")
else:
    st.sidebar.error("结束年份必须晚于起始年份")

ms = st.sidebar.slider("起始月份", 1, 12, 6)
me = st.sidebar.slider("结束月份", 1, 12, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("数据参数")
data_src = st.sidebar.selectbox(
    "数据源",
    ["sentinel2", "landsat8", "landsat9"],
    format_func=lambda x: DATA_SOURCE_LABELS.get(x, x))
idx_type = st.sidebar.selectbox(
    "遥感指数",
    ["NDVI", "NDWI", "MNDWI", "NBR", "EVI", "SAVI"],
    format_func=lambda x: f"{x} ({INDEX_HELP.get(x, '')})")

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
    status="分析完成" if S.ts_result else "运行中" if S.ts_running else "等待执行",
    status_tone="success" if S.ts_result else "info" if S.ts_running else "warning",
)
workflow_bar(
    ["配置序列", "获取影像", "趋势检验", "结果解读"],
    active=3 if S.ts_result else 2 if S.ts_running else 0,
    completed=[0, 1, 2] if S.ts_result else [0, 1] if S.ts_running else [],
)

valid_period = bool(years) and len(years) >= 3 and end_year >= start_year and me >= ms
can_run = (
    bool(S.ts_boundary) and Path(S.ts_boundary).exists()
    and not S.ts_running and valid_period
)
if not valid_period:
    status_banner("warning", "时间序列不足", "至少需要 3 个有效时相，且结束时间不得早于开始时间。")
btn = st.button(
    f"运行 {idx_type} 趋势分析 · {len(years)} 期",
    type="primary", icon=":material/play_arrow:",
    use_container_width=True, disabled=not can_run)
if btn and S.ts_boundary:
    boundary = load_gj(S.ts_boundary)[1]
    _run_timeseries(boundary, years, idx_type, data_src, ms, me)

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
                axes_m[1].set_title(f"Sen's slope (仅显著区域 p<0.05)", fontsize=14, color="#18221D")
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
                st.dataframe(pd.DataFrame(yr_rows), use_container_width=True)

elif S.ts_running:
    status_banner("info", "分析正在进行", "正在收集时相并计算趋势。")
elif S.ts_error:
    status_banner("danger", "分析失败", str(S.ts_error))
else:
    empty_state("暂无趋势结果", "确认侧栏参数并运行分析后显示。", "TS")
