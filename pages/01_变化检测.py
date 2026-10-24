"""
遥感分析平台 v3 — 变化检测页面

两阶段工作流:
  阶段1: 选择地区 → 自动收集遥感影像（多源/多指数）
  阶段2: 一键运行变化检测分析
"""

import os, sys, io, warnings
warnings.filterwarnings("ignore")

# GDAL 临时文件指向项目盘（避免写 C 盘）
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
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from engine.utils import get_needed_bands
from engine.core import find_cached_snapshot
from engine.params import RunParams
from services import change_detection as cd_svc

from frontend.utils import (
    setup_matplotlib_font,
    slug as _slug,
    load_gj,
    DATA_SOURCE_LABELS, MODE_LABELS,
)
from frontend.project_state import ProjectState
from frontend.components import jobs as jobs_ui
from frontend.components.sidebar import (
    data_source_picker, index_picker, month_range, region_picker, year_pair,
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
#  后台任务：收集影像 / 运行分析
# =====================================================================

JOB_COLLECT, JOB_RUN = "cd_collect_job", "cd_run_job"

STEPS_ANALYSIS = [
    ("📂", "加载波段"),
    ("🧮", "指数计算"),
    ("📊", "差值分析"),
    ("🏷️", "多级分类"),
    ("🔬", "连通域过滤"),
    ("📍", "生成采样点"),
    ("📋", "输出报告"),
    ("🏁", "完成"),
]
# 流水线进度（%）达到这些值时进入下一步骤
_STEP_PCT = (10, 20, 35, 55, 70, 80, 95)


def _render_run(job, runner):
    """分析任务的进度：步骤条 + 进度条。"""
    active = sum(job.progress >= p for p in _STEP_PCT) if job.state == "running" else 0
    parts = []
    for i, (_, label) in enumerate(STEPS_ANALYSIS):
        if i < active:
            parts.append(f"<span style='color:#0a0'>✅{label}</span>")
        elif i == active:
            parts.append(f"<span style='color:#08f;font-weight:bold'>⏳{label}</span>")
        else:
            parts.append(f"<span style='color:#888'>⏹️{label}</span>")
    st.markdown(f"<div style='font-size:13px'>{' &nbsp;→&nbsp; '.join(parts)}</div>",
                unsafe_allow_html=True)
    jobs_ui.render_job(job, runner)


def _base_params(y1, y2, ms, me, data_src, idx_type, **extra):
    """由界面取值构造 RunParams（只读 config 默认值，不修改 config）。

    云量阈值取项目设置，与地物分类 / 时间序列一致，单期缓存才能互相命中。
    """
    extra.setdefault("max_cloud_pct", ProjectState.get().max_cloud_pct)
    return RunParams.from_config(
        region=st.session_state.cd_region or "未知",
        boundary_file=st.session_state.cd_boundary,
        year1=int(y1), year2=int(y2), month_start=int(ms), month_end=int(me),
        data_source=data_src, index_type=idx_type, **extra)


def _region_slug():
    return st.session_state.cd_slug or _slug(st.session_state.cd_region or "未知")


def _boundary_file():
    return str(Path(st.session_state.cd_boundary).resolve())


def _job_meta():
    """任务中心据此显示区域、跳回本页，并判断重试结果能否交回本页。"""
    return {"page": "pages/01_变化检测.py", "region": st.session_state.cd_region,
            "boundary": _boundary_file()}


def _submit_collect(y1, y2, ms, me, data_src, idx_type):
    """提交「收集影像」任务：检索 + 下载并写入快照缓存，分析阶段以相同参数直接命中缓存。"""
    try:
        params = _base_params(y1, y2, ms, me, data_src, idx_type)
    except ValueError as e:
        st.error(f"参数无效: {e}")
        return

    ps = ProjectState.get()
    ps.region, ps.data_source, ps.index_type = params.region, data_src, idx_type
    ps.year1, ps.year2 = int(y1), int(y2)
    ProjectState.commit()

    S.cd_data_ok, S.cd_collected = False, None
    label = DATA_SOURCE_LABELS.get(data_src, data_src)
    jobs_ui.submit(JOB_COLLECT, "cd.collect", {
        "run_params": params.to_dict(),
        "boundary_file": _boundary_file(),
        "data_dir": str(BASE_DIR / "data" / _region_slug()),
    }, title=f"{params.region} · {y1}/{y2} 年 {label} 影像（{idx_type}）", meta=_job_meta())
    _save_state()
    st.rerun()


def _apply_collect(ctx):
    S.cd_collected = {"context": ctx}
    S.cd_data_ok = True
    return f"{ctx['region']} · {ctx['size_mb']:.0f} MB，可进入“运行分析”执行变化检测。"


def _check_cache(y1, y2, ms, me, data_src, idx_type):
    """检查快照缓存是否已有与当前参数（区域/数据源/时相/月份/云量/波段）匹配的数据。

    Returns:
        缓存信息 dict(key, size_mb, tiles)，未命中返回 None。
    """
    bnd = st.session_state.cd_boundary
    if not bnd or not Path(bnd).exists():
        return None
    try:
        params = _base_params(y1, y2, ms, me, data_src, idx_type)
        return find_cached_snapshot(params, load_gj(bnd)[1])
    except Exception:
        return None


def _submit_run(y1, y2, ms, me, dmode, ct, sthr, ma, bf,
                np_pts, md_pts, greport, idx_type, data_src):
    """提交分析任务（参数通过 RunParams 显式传入，不修改全局 config）。"""
    try:
        params = _base_params(
            y1, y2, ms, me, data_src, idx_type,
            detection_mode=dmode, threshold=ct, severe_threshold=sthr,
            min_area_m2=ma, use_bilateral_filter=bf,
            num_sample_points=int(np_pts) if int(np_pts) > 0 else None,
            min_point_distance_m=int(md_pts) if int(md_pts) > 0 else None,
            output_formats=("geojson", "csv"), generate_report=greport)
    except ValueError as e:
        st.error(f"参数无效: {e}")
        return

    slug_name = _region_slug()
    jobs_ui.submit(JOB_RUN, "cd.run", {
        "run_params": params.to_dict(),
        "boundary_file": _boundary_file(),
        "data_dir": str(BASE_DIR / "data" / slug_name),
        "out_dir": str(BASE_DIR / "output" / slug_name),
        "slug": slug_name,
        "source_label": DATA_SOURCE_LABELS.get(data_src, data_src),
        "pdf": bool(S.get("cd_gen_pdf", True)),
    }, title=f"{params.region} · {idx_type} 变化检测 {y1}→{y2}", meta=_job_meta())
    _save_state()
    st.rerun()


def _apply_run(res):
    S.cd_results = res
    S.cd_stats = res.get("stats")
    S.cd_arr, S.cd_meta, S.cd_cands = cd_svc.load_outputs(res)
    S.cd_analysis_done = True
    try:
        ProjectState.add_task(res["task"])
    except Exception:
        pass
    if res.get("unmasked_years"):
        years = "、".join(map(str, res["unmasked_years"]))
        return ("warning", "变化检测完成，部分年份未做云掩膜",
                f"{years} 年缺少质量波段，云区可能被误判为变化。")
    return (f"{res.get('region', '')} · {res.get('index_type', '')} · "
            f"{res.get('year1')}→{res.get('year2')} · "
            f"耗时 {res.get('timing', {}).get('total', 0):.0f}s")


# =====================================================================
#  结果展示
# =====================================================================

def _show_map():
    if st.session_state.cd_arr is None or not st.session_state.cd_boundary:
        return
    try:
        gj, geom = load_gj(st.session_state.cd_boundary)
    except Exception:
        st.error("边界解析失败")
        return

    from frontend.components.map_view import (
        add_boundary_layer, legend_html,
    )

    idx = (st.session_state.cd_results or {}).get("index_type", "NDVI")
    meta = st.session_state.cd_meta

    # 透明度控制
    map_opacity = st.session_state.get("cd_map_opacity", 0.6)
    map_opacity = st.slider(
        "结果图层透明度", 0.0, 1.0, map_opacity, 0.05,
        key="cd_opacity_slider",
        on_change=lambda: st.session_state.__setitem__(
            "cd_map_opacity", st.session_state.cd_opacity_slider))

    m = folium.Map(
        location=[geom.centroid.y, geom.centroid.x],
        zoom_start=11, control_scale=True,
        tiles="CartoDB positron",
    )
    minx, miny, maxx, maxy = geom.bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(20, 20))
    Fullscreen().add_to(m)
    add_boundary_layer(m, gj)

    # 叠加图按结果文件缓存：拖动透明度 / 点击地图不再重算
    from frontend.components.map_view import (
        add_change_result_layer as _add_layer, cached_change_overlays,
    )
    gtif = (st.session_state.cd_results or {}).get("files", {}).get("geotiff", "")
    mtime = os.path.getmtime(gtif) if gtif and os.path.exists(gtif) else id(st.session_state.cd_arr)
    overlays = cached_change_overlays(
        st.session_state, (gtif, mtime, idx), st.session_state.cd_arr, meta, idx)
    _add_layer(m, index_type=idx, opacity=map_opacity, overlays=overlays)

    # 底图切换
    folium.TileLayer("OpenStreetMap", name="街道图", show=False).add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="卫星影像", attr="Esri", show=False
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html(idx)))
    output = st_folium(m, width=None, height=600, key="change_result_map")
    if overlays:
        st.caption("点击地图查看该处像元类别 · 图层包含全部变化像元（"
                   + "、".join(f"{o['name']} {o['pixels']:,}" for o in overlays) + "）")

    # ---- 点击交互：显示像元信息 ----
    clicked = output.get("last_clicked")
    if clicked:
        lat, lng = clicked["lat"], clicked["lng"]
        arr = st.session_state.cd_arr
        # 将经纬度转回像素坐标
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
        x, y = t.transform(lng, lat)
        from rasterio.transform import rowcol
        if meta and meta.get("transform"):
            row, col = rowcol(meta["transform"], x, y)
            row, col = int(row), int(col)
            if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                cls_val = int(arr[row, col])
                from engine.indices import INDEX_CLASS_NAMES
                names = INDEX_CLASS_NAMES.get(idx, INDEX_CLASS_NAMES["NDVI"])
                cls_name = names.get(cls_val, f"未知({cls_val})")
                status_banner(
                    "info", f"{idx} · {cls_name}",
                    f"位置 {lat:.5f}, {lng:.5f} · 像素 ({row}, {col})",
                )

    # 下载按钮
    fs = st.session_state.cd_results.get("files", {}) if st.session_state.cd_results else {}
    if fs:
        cols = st.columns(len(fs))
        for i, (fmt, p) in enumerate(fs.items()):
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    cols[i].download_button(
                        Path(p).name, fh.read(), Path(p).name,
                        icon=":material/download:")


def _show_stats():
    s = st.session_state.cd_stats
    res = st.session_state.cd_results or {}
    idx = res.get("index_type", "NDVI")
    dmode = res.get("detection_mode", "multi")

    if dmode == "binary":
        st.metric("变化面积", f"{s.get('changed_area_ha', 0):,.0f} ha")
        return

    from engine.indices import INDEX_CLASS_NAMES, INDEX_COLORS
    cls_n = INDEX_CLASS_NAMES.get(idx, INDEX_CLASS_NAMES["NDVI"])
    cls_c = INDEX_COLORS.get(idx, INDEX_COLORS["NDVI"])

    rows = []
    for cv in [-3, -2, 1, 2, 3]:
        if s.get(cv):
            rows.append({
                "类别": s[cv]["name"],
                "面积(ha)": f"{s[cv]['area_ha']:,.0f}",
                "占比": f"{s[cv]['percent']:.1f}%",
            })
    if "changed_total" in s:
        t = s["changed_total"]
        rows.append({"类别": "合计", "面积(ha)": f"{t['area_ha']:,.0f}",
                      "占比": f"{t['percent']:.1f}%"})
    st.table(pd.DataFrame(rows))

    cd = [(c, s.get(c, {}).get("area_ha", 0)) for c in [-3, -2, 2, 3]]
    cd = [(n, a) for n, a in cd if a > 0]
    if cd:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        labs = [cls_n[c] for c, _ in cd]
        vals = [a for _, a in cd]
        colors = [cls_c[c] for c, _ in cd]
        a1.bar(labs, vals, color=colors, edgecolor="white")
        a1.set_ylabel("面积 (ha)")
        a1.set_title(f"{idx} 各级变化面积", fontweight="bold")
        a1.tick_params(axis="x", rotation=15)
        a2.pie(vals, labels=labs, autopct="%1.1f%%", colors=colors, startangle=90)
        a2.set_title("变化占比", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        t = s["changed_total"]
        status_banner(
            "info", "变化检测摘要",
            f"{res.get('region', '')} · {res.get('year1')}-{res.get('year2')} · "
            f"{idx} 显著变化 {t['area_ha']:,.0f} ha ({t['percent']:.1f}%)",
        )

        # —— 多期对比视图 ——
        with st.expander("变化分级对比"):
            arr = st.session_state.get("cd_arr")
            if arr is not None:
                h, w = arr.shape
                scale = max(1, min(h, w) // 400)
                small = arr[::scale, ::scale]

                chg = np.isin(small, [-3, -2, 2, 3])
                cls_small = small.copy()
                cls_small[cls_small == 0] = 5  # no-data

                def _rgb(hx):
                    return tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))
                colors_rgb = {cv: _rgb(c) for cv, c in cls_c.items()}
                colors_rgb[5] = (10, 10, 10)
                rgb = np.zeros((*cls_small.shape, 3), dtype=np.uint8)
                for cv, c in colors_rgb.items():
                    rgb[cls_small == cv] = c
                change_overlay = np.full((*cls_small.shape, 4), 0, dtype=np.uint8)
                change_overlay[chg] = [255, 60, 60, 180]

                fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
                axes2[0].imshow(rgb)
                axes2[0].set_title(f"{idx} 分级结果", fontsize=13, color="#18221D")
                axes2[0].axis("off")

                axes2[1].imshow(rgb)
                axes2[1].imshow(change_overlay, interpolation="nearest")
                axes2[1].set_title("变化区域高亮", fontsize=13, color="#18221D")
                axes2[1].axis("off")

                axes2[2].axis("off")
                from matplotlib.patches import Patch as MPatch
                patches_val = [
                    MPatch(color=cls_c[cv], label=(
                        cls_n[cv] if cv == 1 else
                        f"{cls_n[cv]} {s.get(cv, {}).get('area_ha', 0):.0f}ha"))
                    for cv in [-3, -2, 1, 2, 3]
                ] + [MPatch(color="#ff3c3c",
                            label=f"变化合计 {s.get('changed_total', {}).get('area_ha', 0):.0f}ha")]
                axes2[2].legend(handles=patches_val, loc="center", fontsize=11,
                                facecolor="#FFFFFF", labelcolor="#18221D")
                axes2[2].set_title("图例", fontsize=13, color="#18221D")

                fig2.patch.set_facecolor("#FFFFFF")
                for ax in axes2:
                    ax.set_facecolor("#FFFFFF")
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close()


def _show_points():
    if not st.session_state.cd_cands:
        st.warning("无采样点")
        return
    rows = [{
        "序号": f["properties"].get("id", ""),
        "经度": round(f["geometry"]["coordinates"][0], 6),
        "纬度": round(f["geometry"]["coordinates"][1], 6),
        "面积(ha)": f["properties"].get("area_ha", ""),
        "类别": f["properties"].get("class_name", ""),
    } for f in st.session_state.cd_cands]
    df = pd.DataFrame(rows)
    section_header(f"采样点 · {len(rows)}", "候选斑块中心点与类别属性。")
    st.dataframe(df, height=400)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "下载采样点 CSV", buf.getvalue(), "sample_points.csv", "text/csv",
        icon=":material/download:",
    )
    if "类别" in df.columns:
        st.bar_chart(df["类别"].value_counts())


# =====================================================================
#  Session State 初始化
# =====================================================================

S = st.session_state

# 分析区域以项目为准（任何页面的侧栏或向导切换区域都写入项目状态）
ps = ProjectState.get()
S.cd_boundary, S.cd_region, S.cd_slug = ps.boundary or None, ps.region or None, ps.slug or None

_INIT = {
    "cd_data_ok": False,
    "cd_collected": None,
    JOB_COLLECT: None,
    JOB_RUN: None,
    "cd_results": None,
    "cd_arr": None,
    "cd_meta": None,
    "cd_stats": None,
    "cd_cands": None,
    "cd_analysis_done": False,
    "cd_map_opacity": 0.6,
}
for k, v in _INIT.items():
    if k not in S:
        S[k] = v

# 区域换了：清掉属于旧区域的影像状态与结果，不再跟踪旧区域的任务（任务照常运行，结果在任务中心）
if ProjectState.aoi_changed("cd"):
    for k, v in _INIT.items():
        if k != "cd_map_opacity":
            S[k] = v
    jobs_ui.forget(JOB_COLLECT, JOB_RUN)

# ---- 状态持久化：项目状态的 cd 节（含正在跟踪的任务：刷新浏览器后继续显示进度并取回结果）----
_PERSIST = ["cd_data_ok", "cd_collected", JOB_COLLECT, JOB_RUN]

def _save_state():
    ProjectState.save_page("cd", {k: S.get(k) for k in _PERSIST})

def _load_state():
    for k, v in ProjectState.load_page("cd", _PERSIST).items():
        S[k] = v

_load_state()

# 已结束的后台任务：结果写入会话状态
collecting = jobs_ui.settle(JOB_COLLECT, label="影像收集", on_success=_apply_collect,
                            persist=_save_state)
running = jobs_ui.settle(JOB_RUN, label="变化检测", on_success=_apply_run, persist=_save_state)

# =====================================================================
#  Sidebar 参数面板
# =====================================================================

sidebar_header("变化检测参数", "分析范围、时相与检测阈值")


region_picker("cd")
st.sidebar.markdown("---")

st.sidebar.subheader("时间范围")
# 默认值：本会话 ProjectState（向导/上次选择）> config
y1, y2 = year_pair(ps.year1 or config.YEAR1, ps.year2 or config.YEAR2)
ms, me = month_range(ps.month_start or config.MONTH_START, ps.month_end or config.MONTH_END)
period_ok = y1 != y2 and ms <= me
st.sidebar.markdown("---")

st.sidebar.subheader("数据源与指数")
data_src = data_source_picker(ps.data_source)
idx_type = index_picker(data_src, default=ps.index_type or "NDVI", remember_as="cd_cur_idx")
st.sidebar.markdown("---")

# --- 检测参数 ---
with st.sidebar.expander("检测参数", expanded=True):
    # 检查 ML 模型是否可用
    ml_model_path = getattr(config, 'ML_MODEL_PATH', 'models/change_detector.pkl')
    ml_available = ml_model_path and os.path.exists(os.path.join(BASE_DIR, ml_model_path))
    all_modes = ["multi", "binary"]
    if ml_available:
        all_modes.append("ml")
    else:
        all_modes.append("ml_disabled")

    dmode_idx = 0
    cur_dmode = st.session_state.get("cd_cur_dmode", "multi")
    if cur_dmode in all_modes:
        dmode_idx = all_modes.index(cur_dmode)
    elif cur_dmode == "ml" and not ml_available:
        dmode_idx = 0  # fallback to multi

    def _fmt_mode(m):
        if m == "ml_disabled":
            return "机器学习 ⚠️（无模型）"
        return MODE_LABELS.get(m, m)

    dmode = st.selectbox(
        "模式",
        all_modes,
        index=dmode_idx,
        format_func=_fmt_mode,
    )
    if dmode == "ml_disabled":
        st.caption("⚠️ 未检测到 ML 模型文件。请先训练模型或切换为多级/二值模式。")
        dmode = "multi"
    st.session_state.cd_cur_dmode = dmode

    ct = st.slider(
        f"{idx_type} 阈值", 0.05, 0.50, config.CHANGE_THRESHOLD, 0.01,
        help=f"{idx_type} 差值超过此值视为变化",
    )
    sthr = st.slider("强烈阈值", 0.10, 0.50, config.SEVERE_THRESHOLD, 0.01)
    ma = st.number_input("最小斑块 (m²)", 100, 50000, config.MIN_CHANGE_AREA_M2, 100)
    bf = st.checkbox("双边预滤波", config.USE_BILATERAL_FILTER)

with st.sidebar.expander("采样设置"):
    st.caption("每个变化斑块取一个内部代表点，按面积从大到小保留。")
    np_pts = st.number_input("上限 (0=不限)", 0, 100000, 0)
    md_pts = st.number_input("最小间距 (m, 0=不限)", 0, 10000, 0)

with st.sidebar.expander("输出设置"):
    greport = st.checkbox("报告 PNG", True)
    gen_pdf = st.checkbox("自动 PDF 报告", True, key="cd_gen_pdf",
                          help="分析完成后自动生成专业 PDF 报告（需 reportlab）")

# =====================================================================
#  引导式分析向导（无历史分析时显示）
# =====================================================================

wizard_key = "_cd_wizard_active"
if (not S.cd_data_ok and not S.cd_results and not collecting and not running
        and not S.get(wizard_key, False)):
    from frontend.components.wizard import render_wizard
    wizard_done = render_wizard()
    if wizard_done:
        S[wizard_key] = True
        st.rerun()
    # wizard 未完成时隐藏下方内容
    st.stop()

# =====================================================================
#  主面板
# =====================================================================

region_display = S.cd_region or "未选择"
src_label = DATA_SOURCE_LABELS.get(data_src, data_src)
page_header(
    "变化检测",
    "分析工作台",
    f"{region_display} 的两时相遥感指数差异分析。",
    meta={
        "区域": region_display,
        "时相": f"{y1} - {y2}",
        "月份": f"{ms} - {me} 月",
        "数据": src_label,
        "指数": idx_type,
        "阈值": f"±{ct}",
    },
    status=("正在分析" if running else "正在收集影像" if collecting
            else "分析完成" if S.cd_results else "数据已就绪" if S.cd_data_ok else "等待数据"),
    status_tone=("info" if running or collecting
                 else "success" if S.cd_results else "info" if S.cd_data_ok else "warning"),
)

workflow_bar(
    ["准备影像", "运行检测", "空间结果", "统计与采样"],
    active=2 if S.cd_results else 1 if S.cd_data_ok else 0,
    completed=[0, 1] if S.cd_results else [0] if S.cd_data_ok else [],
)

# 任务进度（后台执行：可切换页面或关闭浏览器，回来后继续显示）/ 上一次任务的结果提示
jobs_ui.progress(JOB_COLLECT)
jobs_ui.progress(JOB_RUN, render=_render_run)

tab_dl, tab_run, tab_map, tab_stat, tab_pt = st.tabs(
    ["影像数据", "运行分析", "结果地图", "统计摘要", "采样点"]
)

# =====================================================================
#  Tab 1 — 资料收集
# =====================================================================

with tab_dl:
    section_header("准备遥感影像", "检查本地缓存，按需检索并下载所需波段。", "阶段 01")

    can_collect = (bool(S.cd_boundary) and Path(S.cd_boundary).exists()
                   and period_ok and not collecting)
    band_str = ", ".join(get_needed_bands(idx_type))
    btn_label = f"收集 {y1} / {y2} 影像 · {idx_type}"
    if st.button(btn_label, type="primary", icon=":material/download:", width="stretch",
                  disabled=not can_collect):
        _submit_collect(y1, y2, ms, me, data_src, idx_type)

    if S.cd_data_ok:
        info = S.cd_collected or {}
        ctx = info.get("context", {})
        status_banner("success", "影像数据已就绪", "可进入“运行分析”执行变化检测。")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("地区", ctx.get("region", "-"))
        with c2:
            st.metric("数据量", f"{ctx.get('size_mb', 0):.0f} MB")
        with c3:
            c3.metric("数据源", src_label)
        with c4:
            c4.metric("指数", idx_type)
        if S.cd_boundary:
            try:
                gj, geom = load_gj(S.cd_boundary)
                m = folium.Map(
                    location=[geom.centroid.y, geom.centroid.x],
                    zoom_start=11, control_scale=True,
                )
                folium.GeoJson(gj, style_function=lambda x:
                               {"fillColor": "#3388ff", "color": "#3388ff",
                                "weight": 2, "fillOpacity": 0.1}).add_to(m)
                st_folium(m, width="100%", height=350)
            except Exception:
                pass
    elif collecting:
        status_banner("info", "正在收集影像",
                      "检索、下载与波段裁剪在后台进行，进度见页面上方；可先切换到其他页面。")
    else:
        # 检查缓存
        cached = _check_cache(y1, y2, ms, me, data_src, idx_type)
        if cached:
            status_banner("success", "检测到匹配的本地缓存",
                          f"{cached['key'][:8]}… · {cached['size_mb']:.0f} MB")
        else:
            status_banner("info", "尚未收集影像",
                          f"{src_label} · {idx_type} · 需要 {band_str}")

        if S.cd_boundary and Path(S.cd_boundary).exists():
            try:
                gj, geom = load_gj(S.cd_boundary)
                m = folium.Map(
                    location=[geom.centroid.y, geom.centroid.x],
                    zoom_start=11, control_scale=True,
                )
                folium.GeoJson(gj, style_function=lambda x:
                               {"fillColor": "#3388ff", "color": "#3388ff",
                                "weight": 2, "fillOpacity": 0.1}).add_to(m)
                st.caption("待分析区域")
                st_folium(m, width="100%", height=350)
            except Exception:
                pass

# =====================================================================
#  Tab 2 — 运行分析
# =====================================================================

with tab_run:
    section_header("执行变化检测", "使用当前侧栏参数计算差值、分级与斑块。", "阶段 02")

    # 检查数据是否就绪
    ready = S.cd_data_ok
    if not ready:
        cached = _check_cache(y1, y2, ms, me, data_src, idx_type)
        if cached:
            status_banner("success", "检测到可用缓存", f"{cached['size_mb']:.0f} MB · 可直接运行")
            ready = True
        else:
            status_banner("warning", "影像数据未就绪", "请先在“影像数据”中完成收集。")

    can_run = ready and period_ok and not running and not collecting
    btn_run = st.button(
        f"运行 {idx_type} 变化检测",
        type="primary", icon=":material/play_arrow:",
        width="stretch", disabled=not can_run,
    )
    if btn_run:
        _submit_run(y1, y2, ms, me, dmode, ct, sthr, ma, bf,
                    np_pts, md_pts, greport, idx_type, data_src)

    if S.cd_results:
        section_header("分析结果", "本次执行的核心指标与输出文件。", "完成")
        res = S.cd_results
        a, b, c, d = st.columns(4)
        with a:
            st.metric("采样点", res.get("num_sample_points", 0))
        with b:
            st.metric("变化面积", f"{cd_svc.changed_area_ha(res.get('stats', {})):,.0f} ha")
        with c:
            st.metric("耗时", f"{res.get('timing', {}).get('total', 0):.0f}s")
        with d:
            st.metric("输出文件", len(res.get("files", {})))

        # ---- 实际执行参数（验证 UI 值是否生效） ----
        st.markdown("**实际执行参数**")
        s = res.get("stats", {})
        params_display = {
            "指数": res.get("index_type", idx_type),
            "数据源": res.get("data_source", data_src),
            "变化阈值": s.get("_threshold", ct),
            "强烈阈值": s.get("_severe_threshold", "—"),
            "检测模式": res.get("detection_mode", dmode),
            "像元面积": f"{s.get('_area_per_px_ha', 0.01)*10000:.0f} m²",
            "最小斑块面积": f"{ma:,} m²",
        }
        st.caption(" | ".join(f"{k}: {v}" for k, v in params_display.items()))

        with st.expander("输出文件"):
            for fmt, path in res.get("files", {}).items():
                if os.path.exists(path):
                    sz = os.path.getsize(path) / 1024
                    st.text(f"  {fmt.upper():8s}  {Path(path).name}  ({sz:,.0f} KB)")

            # PDF 报告自动生成后提供下载
            pdf_path = res.get("files", {}).get("pdf")
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as fh:
                    st.download_button(
                        "下载 PDF 报告", fh.read(),
                        Path(pdf_path).name, "application/pdf",
                        icon=":material/download:", width="stretch")

# =====================================================================
#  Tab 3-5 — 结果展示
# =====================================================================

with tab_map:
    if S.cd_arr is not None:
        _show_map()
    elif S.cd_results:
        status_banner("warning", "地图数据不可用")
    else:
        empty_state("暂无结果地图", "完成变化检测后显示空间分布。", "MAP")

with tab_stat:
    if S.cd_stats:
        _show_stats()
    elif S.cd_results:
        status_banner("warning", "统计数据不可用")
    else:
        empty_state("暂无统计摘要", "完成变化检测后显示分类面积与占比。", "SUM")

with tab_pt:
    if S.cd_cands:
        _show_points()
    elif S.cd_results:
        status_banner("warning", "采样点数据不可用")
    else:
        empty_state("暂无采样点", "完成变化检测后显示候选采样位置。", "PTS")
