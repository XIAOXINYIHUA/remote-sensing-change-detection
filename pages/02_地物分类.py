"""
遥感分析平台 v3 — 地物分类页面

三种分类模式:
  1. 快速分类 — NDVI + MNDWI + NDBI 阈值，零参数即时出图
  2. GMM 聚类 — PCA 降维 + GMM，自动发现光谱簇
  3. 聚类标注 — 用户将聚类映射为地物类型
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

# GDAL 临时文件指向项目所在盘
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

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from services import ServiceError
from services import classification as cl_svc

from frontend.utils import (
    setup_matplotlib_font,
    slug as _slug,
    load_gj,
    DATA_SOURCE_LABELS,
)
from frontend.project_state import ProjectState
from frontend.components import jobs as jobs_ui
from frontend.components.sidebar import (
    data_source_picker, month_range, region_picker, single_year,
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

from engine.classifier import CLASS_NAMES, CLASS_COLORS
from frontend.components.classifier_view import render_3d_scatter, render_annotation_patch


# =====================================================================
#  取数与分类（流程在 services/classification.py，这里只管进度显示与会话状态）
# =====================================================================

JOB_DOWNLOAD = "cl_download_job"


def _cloud_pct():
    """云量阈值取项目设置，三个分析模块一致，缓存才能互相命中。"""
    return ProjectState.get().max_cloud_pct


def _progress_ui():
    """进度条 + 状态行；返回 (清除函数, 进度回调)。"""
    prog, status = st.progress(0), st.empty()

    def cb(msg, pct):
        prog.progress(min(int(pct), 100))
        (status.success if pct >= 100 else status.info)(msg)

    def clear():
        status.empty()
        prog.empty()

    return clear, cb


def _show_exception(prefix, e):
    st.error(f"{prefix}: {e}")
    if not isinstance(e, ServiceError):
        import traceback
        st.code(traceback.format_exc())


def _submit_download(year, ms, me, data_src):
    """提交下载任务（后台执行；已缓存的时相直接复用）。"""
    S = st.session_state
    boundary = str(Path(S.cl_boundary).resolve())
    jobs_ui.submit(JOB_DOWNLOAD, "cl.download", {
        "data_source": data_src, "boundary_file": boundary, "year": int(year),
        "month_start": int(ms), "month_end": int(me), "max_cloud_pct": _cloud_pct(),
        "data_dir": str(BASE_DIR / "data"),
    }, title=f"{S.cl_region or '未知'} · {year} 年 {DATA_SOURCE_LABELS.get(data_src, data_src)} 分类影像",
       meta={"page": "pages/02_地物分类.py", "region": S.cl_region, "boundary": boundary})
    _save_state()
    st.rerun()


def _apply_download(info):
    S = st.session_state
    S.cl_data_dir = info["dir"]
    S.cl_download_ok = True
    return f"{info['size_mb']:.0f} MB，{len(info['files'])} 个文件；点击“加载波段数据”开始分类。"


def _scene():
    """当前影像的分类输入：波段、网格元数据、研究区边界、有效像元掩膜。"""
    S = st.session_state
    boundary = None
    if S.cl_boundary and Path(S.cl_boundary).exists():
        try:
            boundary = load_gj(S.cl_boundary)[1]
        except Exception:
            boundary = None
    return {"bands": S.cl_bands, "meta": S.cl_meta, "boundary": boundary,
            "valid": S.get("cl_valid")}


def _run_quick():
    """快速分类，带进度条。"""
    S = st.session_state
    S.cl_running = True
    clear, cb = _progress_ui()
    try:
        out = cl_svc.quick(**_scene(), progress_callback=cb)
        S.cl_class_result = out["result"]
        S.cl_class_rgb = out["rgb"]
        S.cl_class_mode = "quick"
        S.cl_stats = out["stats"]
        time.sleep(0.8)
    except Exception as e:
        _show_exception("分类失败", e)
    finally:
        S.cl_running = False
        clear()


def _run_gmm(n_clusters, cov_type, pca_dims, sample_ratio):
    """GMM 聚类，带进度条。"""
    S = st.session_state
    S.cl_running = True
    clear, cb = _progress_ui()
    try:
        out = cl_svc.gmm(**_scene(), n_clusters=n_clusters, covariance_type=cov_type,
                         pca_dims=pca_dims, sample_ratio=sample_ratio, progress_callback=cb)
        S.cl_class_result = out["labels"]
        S.cl_class_mode = "gmm"
        S.cl_gmm_result = out["gmm"]
        S.cl_pca_3d = out["gmm"]["pca_3d"]
        S.cl_n_clusters = n_clusters
        S.cl_cluster_mapping = out["mapping"]
        S.cl_stats = out["stats"]
        S.cl_al_patches = None      # 主动学习斑块按新的聚类结果重新采样
        time.sleep(0.8)
    except Exception as e:
        _show_exception("GMM 失败", e)
    finally:
        S.cl_running = False
        clear()


def _export_result():
    S = st.session_state
    if S.get("cl_class_result") is None or S.cl_meta is None:
        return
    S.cl_export_files = cl_svc.export(
        S.cl_class_result, S.cl_meta, S.get("cl_stats") or [],
        str(BASE_DIR / "output" / (S.cl_slug or "default")),
        region=S.cl_region or "未知", mode=S.get("cl_class_mode") or "unknown")
    st.success("导出完成！")


# =====================================================================
#  主动学习
# =====================================================================

def _init_active_learning(n_patches):
    """初始化（或重置）主动学习状态。"""
    S = st.session_state
    if S.cl_gmm_result is None:
        return
    al = cl_svc.sample_annotation_patches(S.cl_bands, S.cl_gmm_result, n_patches=n_patches)
    S.cl_al_patches = al["patches"]
    S.cl_al_uncertainties = al["uncertainties"]
    S.cl_al_assignments = {}
    S.cl_al_current_idx = 0
    S.cl_al_refined = None
    S.cl_al_refined_confidence = None
    S.cl_al_accumulated_labels = al["labels"]
    S.cl_al_accumulated_mask = al["mask"]


def _run_al_refine():
    """用已标注的像元训练 RF 并预测全图。"""
    S = st.session_state
    try:
        with st.spinner(f"训练随机森林 ({int(S.cl_al_accumulated_mask.sum())} 个标注像元)..."):
            out = cl_svc.refine(**_scene(), labels=S.cl_al_accumulated_labels,
                                mask=S.cl_al_accumulated_mask)
    except ServiceError as e:
        st.error(str(e))
        return
    S.cl_al_refined = out["result"]
    S.cl_al_refined_confidence = out["confidence"]
    S.cl_class_result = out["result"]
    S.cl_class_mode = "al_refined"
    S.cl_stats = out["stats"]
    st.success("✅ 主动学习完成！")
    st.rerun()


# =====================================================================
#  分类后处理 — 连通域过滤
# =====================================================================

def _apply_cc_filter(min_px):
    """对当前分类结果做连通域过滤，消除小斑块噪声。"""
    S = st.session_state
    if S.cl_class_result is None:
        return
    out = cl_svc.filter_small_patches(S.cl_class_result, S.cl_meta, min_px)
    S.cl_class_result = out["result"]
    S.cl_stats = out["stats"]
    st.success(f"✅ 连通域过滤完成（最小 {min_px} px）")


# =====================================================================
#  Session State
# =====================================================================

S = st.session_state

# 分析区域以项目为准（任何页面的侧栏或向导切换区域都写入项目状态）
ps = ProjectState.get()
S.cl_boundary, S.cl_region, S.cl_slug = ps.boundary or None, ps.region or None, ps.slug or None

_INIT = {
    "cl_download_ok": False,
    JOB_DOWNLOAD: None,
    "cl_data_dir": None,
    "cl_bands": None,
    "cl_meta": None,
    "cl_valid": None,
    "cl_prep_info": None,
    "cl_running": False,
    "cl_class_result": None,
    "cl_class_rgb": None,
    "cl_class_mode": None,
    "cl_gmm_result": None,
    "cl_pca_3d": None,
    "cl_n_clusters": 6,
    "cl_cluster_mapping": None,
    "cl_stats": None,
    "cl_export_files": None,
    # 主动学习
    "cl_al_patches": None,
    "cl_al_uncertainties": None,
    "cl_al_assignments": None,
    "cl_al_current_idx": 0,
    "cl_al_refined": None,
    "cl_al_refined_confidence": None,
    "cl_al_accumulated_labels": None,
    "cl_al_accumulated_mask": None,
}
for k, v in _INIT.items():
    if k not in S:
        S[k] = v

# 区域换了：清掉属于旧区域的影像、分类与标注结果，不再跟踪旧区域的下载任务
if ProjectState.aoi_changed("cl"):
    for k, v in _INIT.items():
        S[k] = v
    jobs_ui.forget(JOB_DOWNLOAD)

# 状态持久化：项目状态的 cl 节
_PERSIST = ["cl_download_ok", "cl_data_dir", JOB_DOWNLOAD]


def _save_state():
    ProjectState.save_page("cl", {k: S.get(k) for k in _PERSIST})


def _load_state():
    for k, v in ProjectState.load_page("cl", _PERSIST).items():
        S[k] = v


_load_state()
downloading = jobs_ui.settle(JOB_DOWNLOAD, label="影像下载", on_success=_apply_download,
                             persist=_save_state)

# =====================================================================
#  Sidebar
# =====================================================================

sidebar_header("地物分类参数", "分析范围、影像与模型设置")


region_picker("cl")
st.sidebar.markdown("---")

st.sidebar.subheader("时间范围")
year = single_year(config.YEAR1)
ms, me = month_range(ps.month_start or config.MONTH_START, ps.month_end or config.MONTH_END)
st.sidebar.markdown("---")

st.sidebar.subheader("数据源")
data_src = data_source_picker(ps.data_source)

with st.sidebar.expander("GMM 与主动学习", expanded=True):
    n_clusters = st.slider("聚类数", 3, 12, 6, 1)
    cov_type = st.selectbox(
        "协方差类型",
        ["diag", "full", "tied", "spherical"],
        index=0,
        help="diag=默认(快)  full=最灵活(慢)  tied=共享  spherical=球形")
    pca_dims = st.slider("PCA 分量数", 2, 6, 3, 1)
    sample_ratio = st.slider("采样比例", 0.01, 0.20, 0.05, 0.01,
                              format="%.2f")
    al_n_patches = st.slider("主动学习斑块数", 10, 100, 50, 5)

# =====================================================================
#  Main
# =====================================================================

region_display = S.cl_region or "未选择"
source_label = DATA_SOURCE_LABELS.get(data_src, data_src)
page_header(
    "地物分类",
    "分析工作台",
    f"{region_display} 的多光谱地物识别与人工精化。",
    meta={
        "区域": region_display,
        "年份": year,
        "月份": f"{ms} - {me} 月",
        "数据": source_label,
        "聚类": f"{n_clusters} 类",
    },
    status=("分类完成" if S.cl_class_result is not None else "影像已加载" if S.cl_bands
            else "正在下载影像" if downloading else "等待影像"),
    status_tone=("success" if S.cl_class_result is not None
                 else "info" if S.cl_bands or downloading else "warning"),
)
workflow_bar(
    ["准备影像", "分类建模", "类别精化", "结果导出"],
    active=3 if S.cl_class_result is not None else 1 if S.cl_bands else 0,
    completed=[0, 1, 2] if S.cl_class_result is not None else [0] if S.cl_bands else [],
)

jobs_ui.progress(JOB_DOWNLOAD)

tab_dl, tab_quick, tab_gmm, tab_label, tab_al, tab_result = st.tabs(
    ["影像数据", "快速分类", "GMM 聚类", "类别映射", "主动学习", "结果与导出"])

# =====================================================================
#  Tab 1 — 下载
# =====================================================================

with tab_dl:
    section_header("准备分类影像", "下载或加载当前区域的多光谱波段。", "阶段 01")
    can_dl = (bool(S.cl_boundary) and Path(S.cl_boundary).exists() and ms <= me
              and not downloading)
    if st.button(
            f"下载 {year} 年影像",
            type="primary", icon=":material/download:", width="stretch",
            disabled=not can_dl):
        _submit_download(year, ms, me, data_src)

    # 先找与当前参数匹配的缓存时相（变化检测 / 时间序列下载过的也算），再兼容旧版下载目录
    cached = None
    if S.cl_boundary and Path(S.cl_boundary).exists():
        try:
            cached = cl_svc.find_cached(
                data_src, load_gj(S.cl_boundary)[1], year, ms, me, _cloud_pct())
        except Exception:
            cached = None
    slug_name = S.cl_slug or _slug(S.cl_region or "default")
    legacy_dir = BASE_DIR / "data" / slug_name / str(year)
    load_dir = Path(cached["dir"]) if cached else legacy_dir if legacy_dir.exists() else None
    if cached:
        status_banner("success", "检测到匹配的本地缓存",
                      f"{cached['key'][:8]}… · {len(cached['files'])} 个文件 · {cached['size_mb']:.0f} MB")
    elif load_dir:
        tifs = sorted(load_dir.glob("*.tif"))
        mb = sum(f.stat().st_size for f in tifs) / 1024 / 1024 if tifs else 0
        status_banner("info", "检测到旧版下载目录",
                      f"data/{slug_name}/{year}/ · {len(tifs)} 个文件 · {mb:.0f} MB")
    if load_dir:
        with st.expander("文件列表"):
            for f in sorted(load_dir.glob("*.tif")):
                sz = f.stat().st_size / 1024
                st.text(f"  {f.name:20s} {sz:>7.0f} KB")
        if st.button("加载波段数据", type="primary", icon=":material/folder_open:", width="stretch"):
            loaded = cl_svc.load_bands(str(load_dir), data_src)
            if loaded:
                S.cl_bands = loaded["bands"]
                S.cl_meta = loaded["meta"]
                S.cl_valid = loaded["valid"]
                S.cl_prep_info = loaded["prep_info"]
                S.cl_data_dir = str(load_dir)
                S.cl_download_ok = True
                shapes = {k: v.shape for k, v in S.cl_bands.items()}
                status_banner("success", f"已加载 {len(S.cl_bands)} 个波段", str(shapes))
                st.rerun()
            else:
                st.error("❌ 未找到匹配的波段文件")

# =====================================================================
#  Tab 2 — 快速分类
# =====================================================================

with tab_quick:
    section_header("快速分类", "NDVI、MNDWI 与 NDBI 联合阈值。", "模型 A")
    if S.cl_bands and not S.cl_running:
        if st.button("运行快速分类", type="primary", icon=":material/play_arrow:", width="stretch"):
            _run_quick()
    elif not S.cl_bands:
        status_banner("warning", "尚未加载影像", "请先在“影像数据”中加载波段。")
    if S.cl_class_result is not None and S.cl_class_mode == "quick":
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(S.cl_class_rgb)
        ax.set_title(f"{S.cl_region or ''} 快速分类", fontsize=14)
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        st.pyplot(fig)
        plt.close()
        if S.cl_stats:
            df = pd.DataFrame(S.cl_stats)[["name", "area_ha", "percent"]]
            df = df.rename(columns={"name": "类型", "area_ha": "面积(ha)", "percent": "占比(%)"})
            st.dataframe(df)

# =====================================================================
#  Tab 3 — GMM 聚类
# =====================================================================

with tab_gmm:
    section_header("GMM 光谱聚类", f"PCA {pca_dims}D · {cov_type} · {n_clusters} 类 · 采样 {sample_ratio*100:.0f}%", "模型 B")
    if S.cl_bands and not S.cl_running:
        if st.button("运行 GMM 聚类", type="primary", icon=":material/play_arrow:", width="stretch"):
            _run_gmm(n_clusters, cov_type, pca_dims, sample_ratio)
    if S.cl_gmm_result:
        gmm = S.cl_gmm_result
        ev = gmm.get("explained_variance", [])
        if ev:
            cols = st.columns(len(ev))
            for i, (c, v) in enumerate(zip(cols, ev)):
                c.metric(f"PC{i+1}", f"{v*100:.1f}%")
        section_header("光谱空间", "主成分三维分布")
        fig = render_3d_scatter(
            gmm["pca_3d"], gmm["labels"], S.cl_n_clusters)
        st.plotly_chart(fig)

# =====================================================================
#  Tab 4 — 聚类标注
# =====================================================================

with tab_label:
    section_header("类别映射", "将无监督光谱簇映射为地物类型。", "精化")
    if S.cl_gmm_result is None:
        empty_state("暂无聚类结果", "请先运行 GMM 光谱聚类。", "GMM")
    else:
        mapping = dict(S.cl_cluster_mapping or {})
        actual_n = len(np.unique(S.cl_gmm_result["labels"]))
        clm = st.columns(min(3, actual_n))
        choices = list(CLASS_NAMES.items())
        for cid in range(actual_n):
            with clm[cid % len(clm)]:
                cur = mapping.get(cid, 1)
                idx = next(
                    (i for i, (v, n) in enumerate(choices) if v == cur), 0)
                mapping[cid] = st.selectbox(
                    f"Cluster {cid}",
                    options=[v for v, n in choices],
                    format_func=lambda v: CLASS_NAMES.get(v, f"类别{v}"),
                    index=idx,
                    key=f"cl_m{cid}")
        S.cl_cluster_mapping = mapping
        if st.button("应用类别映射", type="primary", icon=":material/check:", width="stretch"):
            out = cl_svc.label_clusters(S.cl_gmm_result["labels"], mapping, S.cl_meta)
            S.cl_class_result = out["result"]
            S.cl_class_mode = "gmm_labeled"
            S.cl_stats = out["stats"]
            status_banner("success", "类别映射已应用")

# =====================================================================
#  Tab 5 — 主动学习
# =====================================================================

with tab_al:
    section_header("主动学习", "不确定性采样、人工标注与随机森林精化。", "精化")

    has_gmm = S.cl_gmm_result is not None
    has_bands = S.cl_bands is not None

    if not has_gmm:
        empty_state("暂无聚类结果", "请先运行 GMM 光谱聚类。", "GMM")
    elif not has_bands:
        empty_state("尚未加载影像", "请先在“影像数据”中加载波段。", "IMG")
    else:
        # 初始化或重置
        if S.cl_al_patches is None:
            _init_active_learning(al_n_patches)

        patches = S.cl_al_patches
        uncertainties = S.cl_al_uncertainties
        assignments = S.cl_al_assignments
        current_idx = S.cl_al_current_idx

        if patches is None or len(patches) == 0:
            st.warning("没有采样到不确定斑块")
        else:
            st.markdown(f"**待标注斑块**: {len(patches)} 个  已标注: {len(assignments)}/{len(patches)}")

            # 进度条
            pct = len(assignments) / max(len(patches), 1)
            st.progress(min(pct, 1.0))

            # 显示当前斑块
            if current_idx < len(patches):
                row, col = patches[current_idx]
                patch_img = render_annotation_patch(
                    S.cl_bands, row, col, patch_size=config.AL_PATCH_SIZE)

                col_left, col_right = st.columns([2, 1])
                with col_left:
                    st.image(patch_img, caption=f"斑块 #{current_idx+1}  (行{row}, 列{col})",
                             clamp=True, channels="RGB", width="stretch")

                with col_right:
                    st.metric("不确定性", f"{uncertainties[current_idx]:.3f}")
                    st.markdown("**选择类别:**")
                    cls_val = st.selectbox(
                        "类别", options=[v for v, n in CLASS_NAMES.items()],
                        format_func=lambda v: CLASS_NAMES.get(v, f"类别{v}"),
                        index=0, key="al_class_select")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("标注并继续", type="primary", icon=":material/check:", width="stretch"):
                            # 标注当前斑块的所有像元
                            if S.cl_al_accumulated_labels is not None:
                                cl_svc.annotate_patch(
                                    S.cl_al_accumulated_labels, S.cl_al_accumulated_mask,
                                    row, col, cls_val)
                            assignments[current_idx] = cls_val
                            S.cl_al_assignments = assignments
                            S.cl_al_current_idx = current_idx + 1
                            st.rerun()

                    with col_b:
                        if st.button("跳过", icon=":material/skip_next:", width="stretch"):
                            S.cl_al_current_idx = current_idx + 1
                            st.rerun()
            else:
                status_banner("success", "所有斑块已完成标注")
                if st.button("训练 RF 并预测全图", type="primary", icon=":material/play_arrow:", width="stretch"):
                    _run_al_refine()

            # 底部控制
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("上一个", icon=":material/arrow_back:", width="stretch", disabled=current_idx <= 0):
                    S.cl_al_current_idx = current_idx - 1
                    st.rerun()
            with c2:
                if st.button("下一个", icon=":material/arrow_forward:", width="stretch", disabled=current_idx >= len(patches) - 1):
                    S.cl_al_current_idx = current_idx + 1
                    st.rerun()
            with c3:
                if st.button("重新采样", icon=":material/refresh:", width="stretch"):
                    _init_active_learning(al_n_patches)
                    st.rerun()
            with c4:
                n_annotated = S.cl_al_accumulated_mask.sum() if S.cl_al_accumulated_mask is not None else 0
                st.metric("已标注像元", n_annotated)

            # 显示已标注斑块列表
            if assignments:
                with st.expander("已标注斑块"):
                    rows = []
                    for idx, cls_val in sorted(assignments.items()):
                        rows.append({"斑块#": idx + 1, "类别": CLASS_NAMES.get(cls_val, "未知"),
                                     "不确定性": f"{uncertainties[idx]:.3f}"})
                    st.dataframe(pd.DataFrame(rows))

            # 显示 RF 精化结果
            if S.cl_al_refined is not None:
                st.markdown("---")
                section_header("RF 精化结果", "分类结果与预测置信度")
                refined = S.cl_al_refined
                conf = S.cl_al_refined_confidence
                rgb_r = np.zeros((*refined.shape, 3), dtype=np.uint8)
                for cv, hc in CLASS_COLORS.items():
                    if cv == 0:
                        continue
                    m = refined == cv
                    if m.any():
                        rgb_r[m] = [int(hc[i:i+2], 16) for i in (1, 3, 5)]
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 8))
                a1.imshow(rgb_r)
                a1.set_title("RF 分类结果", fontsize=14)
                a1.axis("off")
                if conf is not None:
                    im = a2.imshow(conf, cmap="viridis", vmin=0.3, vmax=1.0)
                    a2.set_title("预测置信度", fontsize=14)
                    a2.axis("off")
                    plt.colorbar(im, ax=a2, shrink=0.6)
                fig.patch.set_facecolor("#FFFFFF")
                st.pyplot(fig)
                plt.close()

# =====================================================================
#  Tab 6 — 结果
# =====================================================================

with tab_result:
    section_header("分类结果", "空间分类、面积统计与成果导出。", "结果")
    if S.cl_class_result is None:
        empty_state("暂无分类结果", "运行快速分类或完成聚类映射后显示。", "CLS")
    else:
        result = S.cl_class_result
        rgb = np.zeros((*result.shape, 3), dtype=np.uint8)
        for cv, hc in CLASS_COLORS.items():
            if cv == 0:
                continue
            m = result == cv
            if m.any():
                rgb[m] = [int(hc[i:i+2], 16) for i in (1, 3, 5)]
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(rgb)
        ax.set_title(f"{S.cl_region or ''} 分类图", fontsize=14)
        ax.axis("off")
        if S.cl_stats:
            patches = [
                Patch(color=CLASS_COLORS.get(s["class"], "#888"),
                      label=f"{s['name']} {s['area_ha']:.0f}ha")
                for s in S.cl_stats]
            ax.legend(handles=patches, loc="lower right",
                      fontsize=10, facecolor="#FFFFFF", labelcolor="#18221D")
        fig.patch.set_facecolor("#FFFFFF")
        st.pyplot(fig)
        plt.close()
        if S.cl_stats:
            df = pd.DataFrame(S.cl_stats)
            df = df[["name", "area_ha", "percent"]]
            df = df.rename(columns={"name": "类型", "area_ha": "面积(ha)", "percent": "占比(%)"})
            st.dataframe(df)
        section_header("成果导出", "运行后处理并下载标准数据格式。")

        # 分类后处理
        with st.expander("后处理"):
            col_cc, col_cc_go = st.columns([3, 1])
            with col_cc:
                min_px = st.slider("最小斑块 (px)", 10, 500,
                                    config.CLASSIFIER_MIN_PATCH_PX, 10,
                                    key="cc_min_px")
            with col_cc_go:
                if st.button("运行连通域过滤", width="stretch"):
                    _apply_cc_filter(min_px)
                    st.rerun()
        if st.button("导出 GeoTIFF、GeoJSON 与 CSV",
                      type="primary", icon=":material/output:", width="stretch"):
            _export_result()
        if S.cl_export_files:
            cols = st.columns(len(S.cl_export_files))
            for i, (fmt, p) in enumerate(S.cl_export_files.items()):
                if os.path.exists(p):
                    with open(p, "rb") as fh:
                        cols[i].download_button(
                            Path(p).name, fh.read(), Path(p).name,
                            icon=":material/download:")
