"""
遥感分析平台 v3 — 地物分类页面

三种分类模式:
  1. 快速分类 — NDVI + MNDWI + NDBI 阈值，零参数即时出图
  2. GMM 聚类 — PCA 降维 + GMM，自动发现光谱簇
  3. 聚类标注 — 用户将聚类映射为地物类型
"""

import os, sys, json, io, re, time, warnings
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
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
from shapely.geometry import shape
import rasterio
import requests

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from engine.utils import log, get_boundary, get_needed_bands, \
    INDEX_BAND_REQUIREMENTS
from engine.data_sources import get_data_source

from frontend.utils import (
    setup_matplotlib_font,
    detect_region, slug as _slug,
    load_gj, fetch_boundary_osm,
    DATA_SOURCE_LABELS,
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

from engine.classifier import (
    classify_multi_index,
    pca_gmm_classify,
    build_feature_stack,
    classification_statistics,
    uncertainty_sampling,
    train_from_annotations,
    CLASS_NAMES,
    CLASS_COLORS,
)
from frontend.components.classifier_view import render_3d_scatter, render_annotation_patch

DATA_SOURCE_LABELS_CL = {k: v for k, v in DATA_SOURCE_LABELS.items() if k != "sentinel1"}


# =====================================================================
#  数据下载
# =====================================================================

def _download_data(year, ms, me, data_src):
    """下载影像，带进度条。"""
    st.session_state.cl_downloading = True
    prog_bar = st.progress(0)
    status_text = st.empty()

    def set_progress(pct, msg):
        prog_bar.progress(min(pct, 100))
        status_text.info(msg)

    try:
        region = st.session_state.cl_region or "未知"
        slug_name = st.session_state.cl_slug or _slug(region)
        config.REGION_NAME = region
        config.YEAR1 = int(year)
        config.MONTH_START = int(ms)
        config.MONTH_END = int(me)
        config.DATA_SOURCE = data_src
        if st.session_state.cl_boundary:
            config.BOUNDARY_FILE = st.session_state.cl_boundary
        data_dir = str(BASE_DIR / "data" / slug_name / str(year))
        os.makedirs(data_dir, exist_ok=True)
        boundary = load_gj(st.session_state.cl_boundary)[1]
        all_bands = ["blue", "green", "red", "nir", "swir1", "swir2"]
        source = get_data_source(data_src)
        src_label = DATA_SOURCE_LABELS_CL.get(data_src, data_src)

        set_progress(5, f"🔍 搜索 {year} 年 {src_label} 影像...")
        scenes = source.search_scenes(boundary, year, ms, me, config.MAX_CLOUD_PCT)
        if not scenes:
            raise RuntimeError(f"{year} 年未找到可用影像")
        set_progress(15, f"📡 找到 {len(scenes)} 个瓦片")

        set_progress(20, f"📥 下载 {year} ({len(scenes)} 瓦片, 6 波段)...")
        result = source.download_bands(scenes, data_dir, all_bands)
        mb = sum(os.path.getsize(p) for p in result.values()
                 if os.path.exists(p)) / 1024 / 1024
        set_progress(90, f"✅ 完成 ({mb:.0f} MB, {len(result)} 波段)")

        st.session_state.cl_data_dir = data_dir
        st.session_state.cl_download_ok = True
        _save_state()
        set_progress(100, "✅ 下载完成！")

    except Exception as e:
        st.error(f"下载失败: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        st.session_state.cl_downloading = False
        time.sleep(1)
        status_text.empty()
        prog_bar.empty()


def _load_bands(data_dir):
    """从下载目录加载所有波段，匹配实际文件名。

    使用 rasterio.warp.reproject 对齐不同分辨率波段，
    不再使用 cv2.resize()。
    """
    from engine.grid import GridSpec, reproject_to_grid, as_gridspec

    bands = {}
    meta = None
    BAND_FILE_MAP = {
        "blue":  ["B2", "B02"],
        "green": ["B3", "B03"],
        "red":   ["B4", "B04"],
        "nir":   ["B8", "B08", "B5"],
        "swir1": ["B11", "B6"],
        "swir2": ["B12", "B7"],
    }
    data_path = Path(data_dir)
    for bname, prefixes in BAND_FILE_MAP.items():
        found = None
        for prefix in prefixes:
            cands = list(data_path.glob(f"{prefix}_*.tif"))
            if cands:
                found = str(cands[0])
                break
        if not found:
            continue
        with rasterio.open(found) as src:
            arr = src.read(1).astype(np.float32)
            bands[bname] = arr
            if meta is None:
                meta = src.meta.copy()
    if bands:
        shapes = {k: v.shape for k, v in bands.items()}
        if len(set(shapes.values())) > 1:
            # 取分辨率最粗（尺寸最小）的波段作为目标网格
            target_name = min(shapes, key=lambda k: shapes[k][0] * shapes[k][1])
            target_key = target_name
            with rasterio.open(
                str(next(data_path.glob(
                    f"{BAND_FILE_MAP[target_key][0]}_*.tif")))
            ) as src:
                target_grid = GridSpec.from_src(src)
            for k, v in bands.items():
                if v.shape != target_grid.shape:
                    bands[k] = reproject_to_grid(
                        (v, None), target_grid,
                        resampling=rasterio.enums.Resampling.bilinear)
            meta = target_grid.to_meta()
            meta.update(dtype="float32", count=1, compress="lzw")
    return bands, meta


# =====================================================================
#  边界裁剪
# =====================================================================

def _clip_to_boundary(result, rgb=None):
    """将分类结果裁剪到研究区边界内。

    边界外置为 -1（而非 0），避免与 Cluster 0 或有效类别冲突。
    """
    meta = st.session_state.cl_meta
    if meta is None:
        return result, rgb
    bf = st.session_state.cl_boundary
    if not bf or not Path(bf).exists():
        return result, rgb
    try:
        from engine.utils import boundary_pixel_mask
        _, geom = load_gj(bf)
        h, w = result.shape
        mask = boundary_pixel_mask(geom, meta, h, w)
        clipped = result.copy()
        clipped[~mask] = -1  # 用 -1 替代 0 作为 no-data
        clipped_rgb = None
        if rgb is not None and rgb.shape[:2] == result.shape:
            clipped_rgb = rgb.copy()
            clipped_rgb[~mask] = [40, 40, 40]
        return clipped, clipped_rgb
    except Exception as e:
        print(f"  Boundary clip skipped: {e}")
        return result, rgb


# =====================================================================
#  分类
# =====================================================================

def _run_quick():
    """快速分类，带进度条。"""
    st.session_state.cl_running = True
    prog = st.progress(0)
    status = st.empty()
    try:
        status.info("运行多指数阈值分类...")
        prog.progress(30)
        result, rgb = classify_multi_index(st.session_state.cl_bands)
        prog.progress(50)
        status.info("裁剪到研究区边界...")
        result, rgb = _clip_to_boundary(result, rgb)
        st.session_state.cl_class_result = result
        st.session_state.cl_class_rgb = rgb
        st.session_state.cl_class_mode = "quick"
        prog.progress(70)
        status.info("生成统计...")
        st.session_state.cl_stats = classification_statistics(result)
        prog.progress(100)
        status.success("✅ 快速分类完成！")
        time.sleep(0.8)
        status.empty()
        prog.empty()
    except Exception as e:
        st.error(f"分类失败: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        st.session_state.cl_running = False


def _run_gmm():
    """GMM 聚类，带进度条。"""
    st.session_state.cl_running = True
    prog = st.progress(0)
    status = st.empty()
    nc = st.session_state.get("cl_n_clusters", 6)
    ct = st.session_state.get("cl_cov_type", "diag")
    pdims = st.session_state.get("cl_pca_dims", 3)
    sratio = st.session_state.get("cl_sample_ratio", 0.05)
    try:
        status.info("🔄 构建特征张量 (波段+指数+纹理)...")
        prog.progress(20)
        stack = build_feature_stack(st.session_state.cl_bands, add_texture=True)
        prog.progress(35)
        status.info(f"🔄 PCA 降维 ({pdims} 分量, 采样 {sratio*100:.0f}% 像元)...")
        res = pca_gmm_classify(
            stack, n_clusters=nc, sample_ratio=sratio, random_state=42,
            pca_dims=pdims, covariance_type=ct)
        prog.progress(60)
        status.info("🏷️ 裁剪到研究区边界...")
        clipped, _ = _clip_to_boundary(res["labels"])
        res["labels"] = clipped
        st.session_state.cl_class_result = clipped
        st.session_state.cl_class_mode = "gmm"
        st.session_state.cl_gmm_result = res
        st.session_state.cl_pca_3d = res["pca_3d"]
        st.session_state.cl_n_clusters = nc
        mapping = {i: (i % 7) + 1 for i in range(nc)}
        st.session_state.cl_cluster_mapping = mapping
        st.session_state.cl_stats = classification_statistics(
            _apply_mapping(clipped, mapping),
            meta=st.session_state.cl_meta)
        prog.progress(100)
        status.success(f"✅ GMM 聚类完成！{nc} 个光谱簇")
        time.sleep(0.8)
        status.empty()
        prog.empty()
    except Exception as e:
        st.error(f"GMM 失败: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        st.session_state.cl_running = False


def _apply_mapping(labels, mapping):
    result = np.full(labels.shape, -1, dtype=np.int8)
    for cid, cls_val in mapping.items():
        result[labels == cid] = cls_val
    result[result == -1] = 0
    return result


def _export_result():
    result = st.session_state.get("cl_class_result")
    if result is None:
        return
    out_dir = Path(BASE_DIR) / "output" / (st.session_state.cl_slug or "default")
    os.makedirs(str(out_dir), exist_ok=True)
    region = st.session_state.cl_region or "未知"
    mode = st.session_state.get("cl_class_mode", "unknown")
    if st.session_state.cl_meta:
        rmeta = st.session_state.cl_meta.copy()
        rmeta.update(dtype="int8", count=1, compress="lzw")
        tif_path = str(out_dir / f"{region}_{mode}_分类.tif")
        with rasterio.open(tif_path, "w", **rmeta) as dst:
            dst.write(result.astype(np.int8), 1)
        st.session_state.cl_export_files = {"geotiff": tif_path}

    # 矢量导出：优先简化 + 同类融合，避免随机丢弃
    try:
        from rasterio.features import shapes as rshapes
        from shapely.geometry import shape as shp_shape, mapping as shp_mapping
        from shapely.ops import unary_union, transform as shp_transform
        from collections import defaultdict
        from pyproj import Transformer

        mask = result > 0
        if not mask.any():
            return

        raw_shapes = list(rshapes(
            result.astype(np.int8), mask=mask,
            transform=st.session_state.cl_meta["transform"],
            connectivity=8))

        # 准备 WGS84 转换器（GeoJSON 标准要求 EPSG:4326）
        src_crs = st.session_state.cl_meta.get("crs", "EPSG:32650")
        t_to_wgs84 = Transformer.from_crs(
            src_crs, "EPSG:4326", always_xy=True)

        # 按类别分组 -> 融合相邻同类斑块 -> 简化
        by_class = defaultdict(list)
        for g, v in raw_shapes:
            if int(v) > 0 and g["type"] == "Polygon":
                poly = shp_shape(g)
                # 从投影坐标转 WGS84
                poly_wgs = shp_transform(
                    lambda x, y, z=None: t_to_wgs84.transform(x, y), poly)
                if poly_wgs.is_empty:
                    continue
                # 面积判断使用原始投影面积（已是正确单位）
                from engine.grid import GridSpec as _GS
                _gs = _GS.from_meta(st.session_state.cl_meta)
                area_m2_proj = poly.area  # 投影面积，单位 m²
                if area_m2_proj < 100:  # <100 m² 的碎斑忽略
                    continue
                # 简化：去除锯齿
                poly_s = poly_wgs.simplify(0.0001, preserve_topology=True)
                if poly_s.geom_type == "Polygon" and poly_s.area > 0:
                    by_class[int(v)].append(poly_s)
                elif poly_s.geom_type == "MultiPolygon":
                    for part in poly_s.geoms:
                        if part.area > 0:
                            by_class[int(v)].append(part)

        # 同类融合 (dissolve)
        feats = []
        for cls_val, polys in by_class.items():
            if not polys:
                continue
            if len(polys) > 1:
                merged = unary_union(polys)
                if merged.geom_type == "MultiPolygon":
                    for part in merged.geoms:
                        if part.area > 0:
                            feats.append((shp_mapping(part), cls_val))
                else:
                    feats.append((shp_mapping(merged), cls_val))
            else:
                feats.append((shp_mapping(polys[0]), cls_val))

        # 按面积排序
        feats.sort(key=lambda x: -shp_shape(x[0]).area)

        # 若仍然过多，取前 50000 并标记
        total_feats = len(feats)
        if total_feats > 50000:
            feats = feats[:50000]

        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": g,
                 "properties": {"class": int(v), "class_name": CLASS_NAMES.get(int(v), "未知")}}
                for g, v in feats
            ],
        }
        if total_feats > 50000:
            fc["metadata"] = {
                "note": f"原始 {total_feats} 斑块，按面积取最大 50000 个。完整分类结果请下载 GeoTIFF。",
                "total_features": total_feats,
                "exported": 50000,
            }

        gj_path = str(out_dir / f"{region}_{mode}_分类.geojson")
        with open(gj_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False, indent=2)
        st.session_state.cl_export_files["geojson"] = gj_path
    except Exception:
        import traceback
        log(f"  Vector export skipped: {traceback.format_exc()}")
    stats = st.session_state.get("cl_stats", [])
    if stats:
        df = pd.DataFrame(stats)
        csv_path = str(out_dir / f"{region}_{mode}_统计.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        st.session_state.cl_export_files["csv"] = csv_path
    st.success("导出完成！")


# =====================================================================
#  主动学习
# =====================================================================

def _init_active_learning():
    """初始化（或重置）主动学习状态。"""
    gmm_res = st.session_state.cl_gmm_result
    if gmm_res is None:
        return
    from engine.classifier import sample_patches
    patches, uncertainties = sample_patches(
        st.session_state.cl_bands,
        gmm_res["pca_3d"],
        gmm_res["gmm"],
        patch_size=config.AL_PATCH_SIZE,
        n_patches=config.AL_N_PATCHES)

    st.session_state.cl_al_patches = patches
    st.session_state.cl_al_uncertainties = uncertainties
    st.session_state.cl_al_assignments = {}
    st.session_state.cl_al_current_idx = 0
    st.session_state.cl_al_refined = None
    st.session_state.cl_al_refined_confidence = None
    st.session_state.cl_al_accumulated_labels = np.zeros(
        list(st.session_state.cl_bands.values())[0].shape, dtype=np.int8)
    st.session_state.cl_al_accumulated_mask = np.zeros(
        list(st.session_state.cl_bands.values())[0].shape, dtype=bool)


def _run_al_refine():
    """用已标注的像元训练 RF 并预测全图。"""
    labels = st.session_state.cl_al_accumulated_labels
    mask = st.session_state.cl_al_accumulated_mask
    if mask.sum() < 10:
        st.error(f"标注像元太少 ({mask.sum()}/10)，请标注更多斑块")
        return

    stack = build_feature_stack(st.session_state.cl_bands, add_texture=True)
    with st.spinner(f"训练随机森林 ({mask.sum()} 个标注像元)..."):
        result, confidence, model = train_from_annotations(
            stack, labels, mask,
            n_estimators=config.AL_RF_N_ESTIMATORS,
            max_depth=config.AL_RF_MAX_DEPTH)

    # 裁剪到边界
    result, _ = _clip_to_boundary(result)
    st.session_state.cl_al_refined = result
    st.session_state.cl_al_refined_confidence = confidence
    st.session_state.cl_class_result = result
    st.session_state.cl_class_mode = "al_refined"
    st.session_state.cl_stats = classification_statistics(result)
    st.success("✅ 主动学习完成！")
    st.rerun()


# =====================================================================
#  分类后处理 — 连通域过滤
# =====================================================================

def _apply_cc_filter():
    """对当前分类结果做连通域过滤，消除小斑块噪声。"""
    result = st.session_state.cl_class_result
    if result is None:
        return
    from engine.core import _filter_by_connected_components
    min_px = config.CLASSIFIER_MIN_PATCH_PX
    filtered = result.copy()
    for cls_val in range(1, 8):
        mask = (result == cls_val)
        if mask.sum() == 0:
            continue
        clean = _filter_by_connected_components(mask, min_px)
        filtered[mask & ~clean] = 0  # 过滤掉的小斑块置为未分类
    st.session_state.cl_class_result = filtered
    st.session_state.cl_stats = classification_statistics(filtered)
    st.success(f"✅ 连通域过滤完成（最小 {min_px} px）")


# =====================================================================
#  Session State
# =====================================================================

S = st.session_state

# 从 ProjectState 初始化
ps = ProjectState.get()
if ps.is_ready():
    S.setdefault("cl_boundary", ps.boundary)
    S.setdefault("cl_region", ps.region)
    S.setdefault("cl_slug", ps.slug)

_INIT = {
    "cl_boundary": None,
    "cl_region": None,
    "cl_slug": None,
    "cl_downloading": False,
    "cl_download_ok": False,
    "cl_data_dir": None,
    "cl_bands": None,
    "cl_meta": None,
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

STATE_FILE = BASE_DIR / ".classify_state.json"


def _save_state():
    save_app_state(STATE_FILE, {
        "cl_boundary": S.get("cl_boundary"),
        "cl_region": S.get("cl_region"),
        "cl_slug": S.get("cl_slug"),
        "cl_download_ok": S.get("cl_download_ok", False),
        "cl_data_dir": S.get("cl_data_dir"),
    })


def _load_state():
    loaded = load_app_state(STATE_FILE, [
        "cl_boundary", "cl_region", "cl_slug",
        "cl_download_ok", "cl_data_dir"])
    for k, v in loaded.items():
        S[k] = v


_load_state()

# =====================================================================
#  Sidebar
# =====================================================================

sidebar_header("地物分类参数", "分析范围、影像与模型设置")
st.sidebar.subheader("分析区域")

exist = {f.name: str(f) for f in BASE_DIR.glob("*.geojson")}
cur_name = Path(S.cl_boundary).name if S.cl_boundary else None
if exist:
    sel = st.sidebar.selectbox(
        "边界文件", list(exist.keys()),
        index=list(exist.keys()).index(cur_name) if cur_name in exist else 0)
    chosen = exist[sel]
    if chosen != S.cl_boundary:
        S.cl_boundary = chosen
        r = detect_region(chosen)
        S.cl_region = r
        S.cl_slug = _slug(r)
        S.cl_download_ok = False
        ProjectState.set_aoi(r, chosen)
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
            fp = BASE_DIR / f"{_slug(sn)}.geojson"
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(gj, f, ensure_ascii=False)
            S.cl_boundary = str(fp)
            S.cl_region = sn.strip()
            S.cl_slug = _slug(sn)
            S.cl_download_ok = False
            ProjectState.set_aoi(sn.strip(), str(fp))
            _save_state()
            st.rerun()

if S.cl_boundary and Path(S.cl_boundary).exists():
    try:
        _, geom = load_gj(S.cl_boundary)
        st.sidebar.caption(f"中心: {geom.centroid.y:.3f}, {geom.centroid.x:.3f}")
    except Exception:
        st.sidebar.error("边界解析失败")
else:
    st.sidebar.error("请选择边界文件")

st.sidebar.markdown("---")
st.sidebar.subheader("时间范围")
year = st.sidebar.number_input("年份", 2015, 2027, config.YEAR1, 1)
ms = st.sidebar.slider("起始月份", 1, 12, 6)
me = st.sidebar.slider("结束月份", 1, 12, 10)
st.sidebar.markdown("---")
data_src = st.sidebar.selectbox(
    "数据源",
    ["sentinel2", "landsat8", "landsat9"],
    format_func=lambda x: DATA_SOURCE_LABELS_CL.get(x, x))

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
source_label = DATA_SOURCE_LABELS_CL.get(data_src, data_src)
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
    status=("分类完成" if S.cl_class_result is not None else "影像已加载" if S.cl_bands else "等待影像"),
    status_tone=("success" if S.cl_class_result is not None else "info" if S.cl_bands else "warning"),
)
workflow_bar(
    ["准备影像", "分类建模", "类别精化", "结果导出"],
    active=3 if S.cl_class_result is not None else 1 if S.cl_bands else 0,
    completed=[0, 1, 2] if S.cl_class_result is not None else [0] if S.cl_bands else [],
)

tab_dl, tab_quick, tab_gmm, tab_label, tab_al, tab_result = st.tabs(
    ["影像数据", "快速分类", "GMM 聚类", "类别映射", "主动学习", "结果与导出"])

# =====================================================================
#  Tab 1 — 下载
# =====================================================================

with tab_dl:
    section_header("准备分类影像", "下载或加载当前区域的多光谱波段。", "阶段 01")
    can_dl = bool(S.cl_boundary) and Path(S.cl_boundary).exists() and not S.cl_downloading
    if st.button(
            f"下载 {year} 年影像",
            type="primary", icon=":material/download:", use_container_width=True,
            disabled=not can_dl):
        _download_data(year, ms, me, data_src)

    slug_name = S.cl_slug or _slug(S.cl_region or "default")
    dd = BASE_DIR / "data" / slug_name / str(year)
    if dd.exists():
        tifs = sorted(dd.glob("*.tif"))
        mb = sum(f.stat().st_size for f in tifs) / 1024 / 1024 if tifs else 0
        status_banner("success", "检测到本地影像", f"data/{slug_name}/ · {len(tifs)} 个文件 · {mb:.0f} MB")
        with st.expander("文件列表"):
            for f in tifs:
                sz = f.stat().st_size / 1024
                st.text(f"  {f.name:20s} {sz:>7.0f} KB")
        if st.button("加载波段数据", type="primary", icon=":material/folder_open:", use_container_width=True):
            bands, meta = _load_bands(str(dd))
            if bands:
                S.cl_bands = bands
                S.cl_meta = meta
                S.cl_data_dir = str(dd)
                S.cl_download_ok = True
                shapes = {k: v.shape for k, v in bands.items()}
                status_banner("success", f"已加载 {len(bands)} 个波段", str(shapes))
                st.rerun()
            else:
                st.error("❌ 未找到匹配的波段文件")

# =====================================================================
#  Tab 2 — 快速分类
# =====================================================================

with tab_quick:
    section_header("快速分类", "NDVI、MNDWI 与 NDBI 联合阈值。", "模型 A")
    if S.cl_bands and not S.cl_running:
        if st.button("运行快速分类", type="primary", icon=":material/play_arrow:", use_container_width=True):
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
            st.dataframe(df, use_container_width=True)

# =====================================================================
#  Tab 3 — GMM 聚类
# =====================================================================

with tab_gmm:
    section_header("GMM 光谱聚类", f"PCA {pca_dims}D · {cov_type} · {n_clusters} 类 · 采样 {sample_ratio*100:.0f}%", "模型 B")
    if S.cl_bands and not S.cl_running:
        if st.button("运行 GMM 聚类", type="primary", icon=":material/play_arrow:", use_container_width=True):
            S.cl_n_clusters = n_clusters
            S.cl_cov_type = cov_type
            S.cl_pca_dims = pca_dims
            S.cl_sample_ratio = sample_ratio
            _run_gmm()
    if S.cl_gmm_result:
        gmm = S.cl_gmm_result
        ev = gmm.get("explained_variance", [])
        if ev:
            cols = st.columns(len(ev))
            for i, (c, v) in enumerate(zip(cols, ev)):
                c.metric(f"PC{i+1}", f"{v*100:.1f}%")
        section_header("光谱空间", "主成分三维分布")
        fig = render_3d_scatter(
            gmm["pca_3d"], gmm["labels"], n_clusters)
        st.plotly_chart(fig, use_container_width=True)

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
        if st.button("应用类别映射", type="primary", icon=":material/check:", use_container_width=True):
            result = _apply_mapping(S.cl_gmm_result["labels"], mapping)
            S.cl_class_result = result
            S.cl_class_mode = "gmm_labeled"
            S.cl_stats = classification_statistics(result)
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
            _init_active_learning()

        patches = S.cl_al_patches
        uncertainties = S.cl_al_uncertainties
        assignments = S.cl_al_assignments
        current_idx = S.cl_al_current_idx

        if patches is None or len(patches) == 0:
            st.warning("没有采样到不确定斑块")
        else:
            st.markdown(f"**待标注斑块**: {len(patches)} 个  已标注: {len(assignments)}/{al_n_patches}")

            # 进度条
            pct = len(assignments) / max(al_n_patches, 1)
            st.progress(min(pct, 1.0))

            # 显示当前斑块
            if current_idx < len(patches):
                row, col = patches[current_idx]
                patch_img = render_annotation_patch(
                    S.cl_bands, row, col, patch_size=config.AL_PATCH_SIZE)

                col_left, col_right = st.columns([2, 1])
                with col_left:
                    st.image(patch_img, caption=f"斑块 #{current_idx+1}  (行{row}, 列{col})",
                             clamp=True, channels="RGB", use_container_width=True)

                with col_right:
                    st.metric("不确定性", f"{uncertainties[current_idx]:.3f}")
                    st.markdown("**选择类别:**")
                    cls_val = st.selectbox(
                        "类别", options=[v for v, n in CLASS_NAMES.items()],
                        format_func=lambda v: CLASS_NAMES.get(v, f"类别{v}"),
                        index=0, key="al_class_select")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("标注并继续", type="primary", icon=":material/check:", use_container_width=True):
                            # 标注当前斑块的所有像元
                            half = config.AL_PATCH_SIZE // 2
                            if cls_val > 0 and S.cl_al_accumulated_labels is not None:
                                r_slice = slice(row - half, row + half + 1)
                                c_slice = slice(col - half, col + half + 1)
                                S.cl_al_accumulated_labels[r_slice, c_slice] = cls_val
                                S.cl_al_accumulated_mask[r_slice, c_slice] = True
                            assignments[current_idx] = cls_val
                            S.cl_al_assignments = assignments
                            S.cl_al_current_idx = current_idx + 1
                            st.rerun()

                    with col_b:
                        if st.button("跳过", icon=":material/skip_next:", use_container_width=True):
                            S.cl_al_current_idx = current_idx + 1
                            st.rerun()
            else:
                status_banner("success", "所有斑块已完成标注")
                if st.button("训练 RF 并预测全图", type="primary", icon=":material/play_arrow:", use_container_width=True):
                    _run_al_refine()

            # 底部控制
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("上一个", icon=":material/arrow_back:", use_container_width=True, disabled=current_idx <= 0):
                    S.cl_al_current_idx = current_idx - 1
                    st.rerun()
            with c2:
                if st.button("下一个", icon=":material/arrow_forward:", use_container_width=True, disabled=current_idx >= len(patches) - 1):
                    S.cl_al_current_idx = current_idx + 1
                    st.rerun()
            with c3:
                if st.button("重新采样", icon=":material/refresh:", use_container_width=True):
                    _init_active_learning()
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
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

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
            st.dataframe(df, use_container_width=True)
        section_header("成果导出", "运行后处理并下载标准数据格式。")

        # 分类后处理
        with st.expander("后处理"):
            col_cc, col_cc_go = st.columns([3, 1])
            with col_cc:
                min_px = st.slider("最小斑块 (px)", 10, 500,
                                    config.CLASSIFIER_MIN_PATCH_PX, 10,
                                    key="cc_min_px")
            with col_cc_go:
                if st.button("运行连通域过滤", use_container_width=True):
                    config.CLASSIFIER_MIN_PATCH_PX = min_px
                    _apply_cc_filter()
                    st.rerun()
        if st.button("导出 GeoTIFF、GeoJSON 与 CSV",
                      type="primary", icon=":material/output:", use_container_width=True):
            _export_result()
        if S.cl_export_files:
            cols = st.columns(len(S.cl_export_files))
            for i, (fmt, p) in enumerate(S.cl_export_files.items()):
                if os.path.exists(p):
                    with open(p, "rb") as fh:
                        cols[i].download_button(
                            Path(p).name, fh.read(), Path(p).name,
                            icon=":material/download:")
