"""
遥感解译助手 v3 — 核心变化检测引擎

包含:
  - NDVI 计算与差值分析
  - 多级/二值变化检测 + 连通域过滤
  - 通用指数变化检测 (v3)
  - 采样点生成 (GeoJSON/CSV/Shapefile)
  - 解译报告图生成
  - 完整流水线 (run_pipeline / run_pipeline_v3)
"""

import os, sys, json, time, warnings
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Point
from shapely.prepared import prep
from pyproj import Transformer
import cv2
import pandas as pd

import config
from engine.utils import (
    log, get_boundary, boundary_pixel_mask,
    get_needed_bands, INDEX_BAND_REQUIREMENTS,
)
from engine.grid import (
    GridSpec, reproject_to_grid, reproject_bands,
    assert_grid_consistent, as_gridspec,
)

warnings.filterwarnings("ignore")


# ============================================================================
#  NDVI
# ============================================================================

def calc_ndvi(nir, red):
    """计算归一化植被指数 (NDVI)。"""
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denom = nir + red
    denom[denom == 0] = 0.001
    return (nir - red) / denom


# ============================================================================
#  波段加载
# ============================================================================

def _load_bands(data_dir, target_grid=None):
    """加载 B4/B8 波段文件，返回 (r1, n1, r2, n2, meta)。

    如果提供 target_grid，所有波段会被重投影对齐。
    """
    y1, y2 = config.YEAR1, config.YEAR2

    def _find_band(base_name, year):
        candidates = [
            os.path.join(data_dir, f"B4_{year}.tif") if base_name == "B4" else
            os.path.join(data_dir, f"B8_{year}.tif"),
            os.path.join(data_dir, f"RED_{year}.tif") if base_name == "B4" else
            os.path.join(data_dir, f"NIR_{year}.tif"),
        ]
        candidates.append(os.path.join(data_dir, f"{base_name}_{year}.tif"))
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    b4_1 = _find_band("B4", y1)
    b8_1 = _find_band("B8", y1)
    b4_2 = _find_band("B4", y2)
    b8_2 = _find_band("B8", y2)

    missing = []
    if not b4_1: missing.append(f"B4_{y1}")
    if not b8_1: missing.append(f"B8_{y1}")
    if not b4_2: missing.append(f"B4_{y2}")
    if not b8_2: missing.append(f"B8_{y2}")
    if missing:
        raise FileNotFoundError(
            f"Missing band files in {data_dir}: {missing}. "
            f"请先在「资料收集」标签页下载影像数据。")

    t0 = time.time()

    # 确定目标网格
    if target_grid is None:
        with rasterio.open(b4_1) as src:
            target_grid = GridSpec.from_src(src)

    with rasterio.open(b4_1) as src:
        meta = src.meta
        ref_grid = GridSpec.from_src(src)
        if target_grid != ref_grid:
            r1 = reproject_to_grid((src.read(1).astype(np.float32), src.meta), target_grid)
        else:
            r1 = src.read(1).astype(np.float32)
    with rasterio.open(b8_1) as src:
        if target_grid != as_gridspec(src.meta):
            n1 = reproject_to_grid((src.read(1).astype(np.float32), src.meta), target_grid)
        else:
            n1 = src.read(1).astype(np.float32)
    log(f"  Grid: {target_grid.width}x{target_grid.height} px  "
        f"({target_grid.width*10/1000:.0f}x{target_grid.height*10/1000:.0f} km)")

    with rasterio.open(b4_2) as src:
        if target_grid != as_gridspec(src.meta):
            r2 = reproject_to_grid((src.read(1).astype(np.float32), src.meta), target_grid)
        else:
            r2 = src.read(1).astype(np.float32)
    with rasterio.open(b8_2) as src:
        if target_grid != as_gridspec(src.meta):
            n2 = reproject_to_grid((src.read(1).astype(np.float32), src.meta), target_grid)
        else:
            n2 = src.read(1).astype(np.float32)
    log(f"  Loaded bands: {time.time()-t0:.1f}s")
    return r1, n1, r2, n2, target_grid.to_meta()


# ============================================================================
#  连通域过滤
# ============================================================================

def _filter_by_connected_components(binary_mask, min_area_px):
    """连通域分析：向量化按面积过滤斑块。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8)

    if num_labels <= 1:
        log(f"    CC filter: 0 patches -> 0 kept (>{min_area_px} px)")
        return np.zeros_like(binary_mask, dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(num_labels, dtype=bool)
    keep[1:] = areas >= min_area_px
    kept = int(keep.sum())
    log(f"    CC filter: {num_labels-1} patches -> {kept} kept (>{min_area_px} px)")
    return keep[labels]


# ============================================================================
#  自适应阈值估计
# ============================================================================

def estimate_adaptive_threshold(diff, valid_mask, k_change=3.0, k_severe=5.0):
    """根据稳定区域的差异分布自动估计阈值。

    策略: 大部分像元未变化，差值近似服从 N(μ, σ²)。用稳健估计
    σ = 1.4826 * median(|diff - median(diff)|)（对少量真实变化不敏感），
    阈值取 k·σ。

    注意：不能对 |diff| 再取 MAD —— 那只有约 0.4σ，3 倍后仅 1.2σ，
    纯噪声下会有 ~23% 的稳定像元被判为变化。

    Args:
        diff: 差值数组 (y2 - y1)
        valid_mask: 有效像元掩膜
        k_change: 变化阈值 = k_change * σ，默认 3.0
        k_severe: 强烈变化阈值 = k_severe * σ，默认 5.0

    Returns:
        (change_threshold, severe_threshold)
    """
    valid_diff = diff[valid_mask & np.isfinite(diff)]

    if len(valid_diff) < 100:
        # 像素太少时回退到 config
        return config.CHANGE_THRESHOLD, config.SEVERE_THRESHOLD

    med = np.median(valid_diff)
    sigma = 1.4826 * np.median(np.abs(valid_diff - med))

    if sigma < 1e-6:
        return config.CHANGE_THRESHOLD, config.SEVERE_THRESHOLD

    # 限制在合理范围，且强烈阈值严格大于变化阈值
    T = min(max(0.02, k_change * sigma), 0.5)
    S = min(max(0.05, k_severe * sigma), 0.8)
    S = max(S, T + 0.01)

    log(f"  Adaptive threshold: sigma={sigma:.4f}, T={T:.4f}, S={S:.4f}")

    return round(T, 4), round(S, 4)


# ============================================================================
#  变化检测
# ============================================================================

def _classify_multi_level(diff, valid_mask, threshold=None, severe_threshold=None):
    """多级分类: -3严重退化, -2中度退化, 1稳定, 2中度恢复, 3强烈恢复。"""
    T = threshold if threshold is not None else config.CHANGE_THRESHOLD
    S = severe_threshold if severe_threshold is not None else config.SEVERE_THRESHOLD
    result = np.ones_like(diff, dtype=np.int8)
    result[~valid_mask] = 0
    result[diff < -S] = -3
    result[(diff >= -S) & (diff < -T)] = -2
    result[(diff > T) & (diff <= S)] = 2
    result[diff > S] = 3
    return result


def _filter_multi_level(result, area_per_px_ha=0.01, min_area_m2=None):
    """对多级结果的每个变化类别分别连通域过滤。

    Args:
        result: 多级分类结果数组
        area_per_px_ha: 每像元面积 (ha)
        min_area_m2: 全局最小斑块面积 (m²)。若提供，覆盖 config.MIN_AREA_BY_CLASS 所有变化类。
                      若为 None，使用 config.MIN_AREA_BY_CLASS 各变化类独立值。
    """
    area_per_px_m2 = area_per_px_ha * 10000.0
    for cls_val in [-3, -2, 2, 3]:
        if min_area_m2 is not None:
            # 使用全局统一值
            min_area = min_area_m2
        elif cls_val in config.MIN_AREA_BY_CLASS:
            min_area = config.MIN_AREA_BY_CLASS[cls_val]
        else:
            continue
        min_px = max(4, int(round(min_area / max(area_per_px_m2, 1))))
        mask = (result == cls_val)
        if mask.sum() == 0:
            continue
        filtered = _filter_by_connected_components(mask, min_px)
        result[mask & ~filtered] = 1
    return result


def run_change_detection_v2(data_dir, out_dir, boundary_wgs84):
    """v2 变化检测：多级 + 二进制双模式。"""
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    log("Loading and aligning bands...")
    r1, n1, r2, n2, meta = _load_bands(data_dir)
    ref_h, ref_w = meta["height"], meta["width"]

    valid = (r1 > 0) & (n1 > 0) & (r2 > 0) & (n2 > 0)
    ndvi1 = calc_ndvi(n1, r1)
    ndvi2 = calc_ndvi(n2, r2)

    if config.USE_BILATERAL_FILTER:
        log(f"  Bilateral filter (d={config.BILATERAL_D}, "
            f"sigmaColor={config.BILATERAL_SIGMA_COLOR}, "
            f"sigmaSpace={config.BILATERAL_SIGMA_SPACE})...")
        ndvi1 = cv2.bilateralFilter(
            ndvi1, config.BILATERAL_D, config.BILATERAL_SIGMA_COLOR,
            config.BILATERAL_SIGMA_SPACE)
        ndvi2 = cv2.bilateralFilter(
            ndvi2, config.BILATERAL_D, config.BILATERAL_SIGMA_COLOR,
            config.BILATERAL_SIGMA_SPACE)

    ndvi_diff = ndvi2 - ndvi1
    ndvi_diff[~valid] = 0

    log("  Clipping to boundary polygon...")
    bnd_mask = boundary_pixel_mask(boundary_wgs84, meta, ref_h, ref_w)
    ndvi_diff[~bnd_mask] = 0
    valid = valid & bnd_mask

    mode = config.DETECTION_MODE
    log(f"  Detection mode: {mode}")

    if mode == "multi":
        result = _classify_multi_level(ndvi_diff, valid)

        # 基于实际网格计算像元面积
        try:
            gs = GridSpec.from_meta(meta)
            area_px_ha = gs.area_per_pixel_ha
        except Exception:
            area_px_ha = config.AREA_PER_PIXEL_HECTARES
        log(f"  Pixel area: {area_px_ha*10000:.1f} m² ({area_px_ha:.4f} ha)")
        result = _filter_multi_level(result, area_px_ha)

        cls_names = {-3: "严重退化", -2: "中度退化", 1: "稳定",
                      2: "中度恢复", 3: "强烈恢复"}
        stats = {}
        total_valid = valid.sum()
        for cv in [-3, -2, 1, 2, 3]:
            n_px = (result == cv).sum()
            area_ha = n_px * area_px_ha
            pct = n_px * 100 / max(total_valid, 1)
            stats[cv] = {
                "name": cls_names[cv], "pixels": int(n_px),
                "area_ha": round(area_ha, 2), "percent": round(pct, 2)
            }
        changed_px = sum((result == c).sum() for c in [-3, -2, 2, 3])
        stats["changed_total"] = {
            "pixels": int(changed_px),
            "area_ha": round(changed_px * area_px_ha, 2),
            "percent": round(changed_px * 100 / max(total_valid, 1), 2)
        }
        stats["valid_pixels"] = int(total_valid)
        stats["total_pixels"] = ref_h * ref_w
        stats["_threshold"] = config.CHANGE_THRESHOLD
        stats["_area_per_px_ha"] = area_px_ha
    else:
        change_mask = np.abs(ndvi_diff) > config.CHANGE_THRESHOLD
        k = np.ones((config.MORPH_KERNEL_SIZE,) * 2, np.uint8)
        change_mask = cv2.morphologyEx(
            change_mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
        if config.MIN_CHANGE_AREA_PX > config.MORPH_KERNEL_SIZE ** 2:
            change_mask = _filter_by_connected_components(
                change_mask, config.MIN_CHANGE_AREA_PX)
        result = np.zeros((ref_h, ref_w), dtype=np.uint8)
        result[change_mask] = 1
        chg_px = change_mask.sum()
        stats = {
            "mode": "binary", "changed_pixels": int(chg_px),
            "changed_area_ha": round(chg_px * config.AREA_PER_PIXEL_HECTARES, 2),
            "changed_percent": round(chg_px * 100 / max(valid.sum(), 1), 2),
            "valid_pixels": int(valid.sum()), "total_pixels": ref_h * ref_w
        }

    log(f"  Total time: {time.time()-t_start:.1f}s")
    return result, meta, stats


# ============================================================================
#  通用指数变化检测 (v3)
# ============================================================================

def run_change_detection_generic(bands_y1, bands_y2, meta, boundary_wgs84,
                                  index_type="NDVI", detection_mode="multi",
                                  threshold=None, severe_threshold=None,
                                  adaptive_threshold=False, min_area_m2=None,
                                  valid_mask=None):
    """v3 通用指数变化检测：支持 NDVI/NDWI/NBR/EVI/SAVI/MNDWI。

    阈值优先级:
        adaptive_threshold=True > 显式传入参数 > INDEX_THRESHOLDS > config

    valid_mask: 预处理得到的有效像元（含云掩膜）；None 时退回「所有波段 > 0」。
    """
    from engine.indices import compute_index, INDEX_THRESHOLDS, INDEX_CLASS_NAMES

    ref_h, ref_w = list(bands_y1.values())[0].shape[:2]

    # 断言波段网格一致
    try:
        if meta:
            grids = [as_gridspec(meta)]
            names = ["meta"]
            for bname, arr in bands_y1.items():
                names.append(f"y1_{bname}")
            for bname, arr in bands_y2.items():
                names.append(f"y2_{bname}")
            assert_grid_consistent(grids, names)
    except Exception:
        pass  # 非致命 — 元数据可能不完整

    if valid_mask is not None:
        valid = valid_mask.copy()
    else:
        valid = np.ones((ref_h, ref_w), dtype=bool)
        for arr in list(bands_y1.values()) + list(bands_y2.values()):
            valid &= (arr > 0)

    idx1 = compute_index(index_type, bands_y1)
    idx2 = compute_index(index_type, bands_y2)
    diff = idx2 - idx1
    valid &= np.isfinite(diff)
    diff[~valid] = 0

    bnd_mask = boundary_pixel_mask(boundary_wgs84, meta, ref_h, ref_w)
    diff[~bnd_mask] = 0
    valid = valid & bnd_mask

    # 阈值优先级：自适应 > 显式 > INDEX_THRESHOLDS > config
    reg = INDEX_THRESHOLDS.get(index_type, {})

    if adaptive_threshold:
        T, S = estimate_adaptive_threshold(diff, valid)
        log(f"  Adaptive thresholds: change={T}, severe={S}")
    else:
        if threshold is not None:
            T = threshold
        elif "change" in reg:
            T = reg["change"]
        else:
            T = config.CHANGE_THRESHOLD

        if severe_threshold is not None:
            S = severe_threshold
        elif "severe" in reg:
            S = reg["severe"]
        else:
            S = config.SEVERE_THRESHOLD

    cls_names = INDEX_CLASS_NAMES.get(index_type, {
        -3: "严重退化", -2: "中度退化", 1: "稳定", 2: "中度恢复", 3: "强烈恢复"
    })

    # 从实际网格计算像元面积
    try:
        gs = as_gridspec(meta)
        area_per_px_ha = gs.area_per_pixel_ha
    except Exception:
        area_per_px_ha = config.AREA_PER_PIXEL_HECTARES

    if detection_mode == "multi":
        result = _classify_multi_level(diff, valid, T, S)
        result = _filter_multi_level(result, area_per_px_ha, min_area_m2=min_area_m2)
        stats = {}
        total_valid = valid.sum()
        for cv in [-3, -2, 1, 2, 3]:
            n_px = (result == cv).sum()
            area_ha = n_px * area_per_px_ha
            pct = n_px * 100 / max(total_valid, 1)
            stats[cv] = {
                "name": cls_names[cv], "pixels": int(n_px),
                "area_ha": round(area_ha, 2), "percent": round(pct, 2)
            }
        changed_px = sum((result == c).sum() for c in [-3, -2, 2, 3])
        stats["changed_total"] = {
            "pixels": int(changed_px),
            "area_ha": round(changed_px * area_per_px_ha, 2),
            "percent": round(changed_px * 100 / max(total_valid, 1), 2)
        }
        stats["valid_pixels"] = int(total_valid)
        stats["total_pixels"] = ref_h * ref_w
        stats["index_type"] = index_type
        stats["_threshold"] = T
        stats["_severe_threshold"] = S
        stats["_area_per_px_ha"] = area_per_px_ha
    else:
        change_mask = np.abs(diff) > T
        k = np.ones((config.MORPH_KERNEL_SIZE,) * 2, np.uint8)
        change_mask = cv2.morphologyEx(
            change_mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
        # 二值模式连通域过滤
        area_m2 = min_area_m2 or getattr(config, 'MIN_CHANGE_AREA_M2', 5000)
        min_px_bin = max(4, int(round(area_m2 / max(area_per_px_ha * 10000, 1))))
        change_mask = _filter_by_connected_components(
            change_mask.astype(bool), min_px_bin)
        result = np.zeros((ref_h, ref_w), dtype=np.uint8)
        result[change_mask] = 1
        chg_px = change_mask.sum()
        stats = {
            "mode": "binary", "changed_pixels": int(chg_px),
            "changed_area_ha": round(chg_px * area_per_px_ha, 2),
            "changed_percent": round(chg_px * 100 / max(valid.sum(), 1), 2),
            "valid_pixels": int(valid.sum()), "total_pixels": ref_h * ref_w,
            "index_type": index_type,
            "_threshold": T,
            "_area_per_px_ha": area_per_px_ha,
        }

    return result, meta, stats


# ============================================================================
#  采样点生成
# ============================================================================

def _get_change_mask_multi(result):
    return (result == -3) | (result == -2) | (result == 2) | (result == 3)


def _extract_contours(change_mask):
    contours, _ = cv2.findContours(
        change_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _get_class_for_point(result, col, row):
    if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
        return int(result[row, col])
    return 0


def generate_sample_points_v2(result, boundary_wgs84, meta, out_dir):
    """v2 采样点生成：斑块质心提取 + 向量化坐标变换。"""
    mode = config.DETECTION_MODE
    change_mask = _get_change_mask_multi(result) if mode == "multi" else result.astype(bool)

    chg_px = change_mask.sum()
    log(f"\nSampling change patch centroids...")
    log(f"  Change pixels: {chg_px:,}")
    if chg_px == 0:
        log("  No change!")
        return None

    t0 = time.time()
    contours = _extract_contours(change_mask)
    log(f"  Patches: {len(contours)} ({time.time()-t0:.1f}s)")
    if not contours:
        return None

    t_crs = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True)
    min_area = max(config.MIN_CHANGE_AREA_PX, 4)
    cls_names = {-3: "严重退化", -2: "中度退化", 1: "稳定",
                  2: "中度恢复", 3: "强烈恢复"}

    contour_data = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        contour_data.append({
            'area': area,
            'cx': M["m10"] / M["m00"],
            'cy': M["m01"] / M["m00"]
        })

    if not contour_data:
        log("  No valid patches after area filter!")
        return None

    cxs = np.array([d['cx'] for d in contour_data])
    cys = np.array([d['cy'] for d in contour_data])
    xs, ys = rasterio.transform.xy(meta["transform"], cys, cxs)
    lons, lats = t_crs.transform(xs, ys)

    boundary_prepared = prep(boundary_wgs84)
    candidates = []
    for i, d in enumerate(contour_data):
        pt = Point(lons[i], lats[i])
        if not boundary_prepared.contains(pt):
            continue
        col, row = int(round(d['cx'])), int(round(d['cy']))
        cv = _get_class_for_point(result, col, row)
        candidates.append({
            "lon": lons[i], "lat": lats[i],
            "col": col, "row": row,
            "area_px": int(round(d['area'])),
            "area_ha": round(d['area'] * config.AREA_PER_PIXEL_HECTARES, 4),
            "class": cv, "class_name": cls_names.get(cv, "未知")
        })

    if not candidates:
        log("  No valid patches inside boundary!")
        return None

    candidates.sort(key=lambda x: -x["area_px"])

    if config.NUM_SAMPLE_POINTS is not None:
        candidates = candidates[:config.NUM_SAMPLE_POINTS]

    if config.MIN_POINT_DISTANCE_M is not None:
        min_dist_deg = config.MIN_POINT_DISTANCE_M / 111000.0
        filtered = []
        for p in candidates:
            too_close = any(
                ((p["lon"] - q["lon"])**2 + (p["lat"] - q["lat"])**2)**0.5 < min_dist_deg
                for q in filtered
            )
            if not too_close:
                filtered.append(p)
        candidates = filtered
        log(f"  After distance filter ({config.MIN_POINT_DISTANCE_M}m): {len(candidates)}")

    log(f"  Valid patches: {len(candidates)}")

    formats = [f.strip() for f in config.OUTPUT_FORMATS.split(",")]
    out_paths = {}
    for fmt in formats:
        if fmt == "geojson":
            p = _export_geojson(candidates, out_dir)
        elif fmt == "csv":
            p = _export_csv(candidates, out_dir)
        elif fmt == "shp":
            p = _export_shapefile(candidates, meta["crs"], out_dir)
        else:
            log(f"  Unknown format: {fmt}")
            continue
        if p:
            out_paths[fmt] = p

    return out_paths, candidates


def _export_geojson(candidates, out_dir):
    features = []
    for i, p in enumerate(candidates, 1):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            "properties": {
                "id": i, "row": p["row"], "col": p["col"],
                "area_px": p["area_px"], "area_ha": p["area_ha"],
                "class": p["class"], "class_name": p["class_name"]
            }
        })
    fc = {
        "type": "FeatureCollection", "features": features,
        "metadata": {
            "description": f"变化检测采样点 ({config.YEAR1} vs {config.YEAR2})",
            "total_points": len(features), "region": config.REGION_NAME
        }
    }
    path = os.path.join(out_dir, config.SAMPLE_OUTPUT)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    log(f"  GeoJSON: {path} ({len(features)} points)")
    return path


def _export_csv(candidates, out_dir):
    rows = []
    for i, p in enumerate(candidates, 1):
        rows.append({
            "id": i, "lon": p["lon"], "lat": p["lat"],
            "row": p["row"], "col": p["col"],
            "area_px": p["area_px"], "area_ha": p["area_ha"],
            "class": p["class"], "class_name": p["class_name"]
        })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "sample_points.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"  CSV: {path} ({len(rows)} rows)")
    return path


def _export_shapefile(candidates, crs, out_dir):
    try:
        import fiona
        from fiona.crs import from_epsg
    except ImportError:
        log("  Shapefile export requires fiona. Install: pip install fiona")
        return None
    epsg_code = int(crs.to_epsg()) if hasattr(crs, "to_epsg") else 4326
    schema = {
        "geometry": "Point",
        "properties": {
            "id": "int", "area_px": "int", "area_ha": "float",
            "class": "int", "class_name": "str"
        }
    }
    path = os.path.join(out_dir, "sample_points.shp")
    with fiona.open(path, "w", driver="ESRI Shapefile",
                    schema=schema, crs=from_epsg(epsg_code),
                    encoding="utf-8") as dst:
        for i, p in enumerate(candidates, 1):
            dst.write({
                "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                "properties": {
                    "id": i, "area_px": p["area_px"], "area_ha": p["area_ha"],
                    "class": p["class"], "class_name": p["class_name"]
                }
            })
    log(f"  Shapefile: {path} ({len(candidates)} points)")
    return path


# ============================================================================
#  输出工具
# ============================================================================

def write_cog(arr, meta, path, dtype=None, compress="lzw"):
    """写入 Cloud Optimized GeoTIFF (COG)。

    Args:
        arr: numpy 数组
        meta: rasterio 元数据
        path: 输出路径
        dtype: 数据类型（默认自动推断）
        compress: 压缩方式

    Returns:
        str: 输出路径
    """
    if dtype is None:
        dtype = arr.dtype.name if arr.dtype.name.startswith(("float", "int")) else "float32"

    profile = meta.copy()
    profile.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="IF_SAFER",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.dtype(dtype)), 1)
        # 添加 COG 标签（让 GDAL 识别为 COG）
        dst.update_tags(ns="IMAGE_STRUCTURE", LAYOUT="COG")

    # 验证
    try:
        with rasterio.open(path) as src:
            assert src.width == profile["width"]
            assert src.height == profile["height"]
    except Exception:
        pass  # 非致命

    log(f"  COG: {path} ({arr.shape[0]}x{arr.shape[1]}, {dtype})")
    return path


def write_cog_multi(
    band_dict: dict[str, np.ndarray],
    meta: dict,
    path_prefix: str,
) -> dict[str, str]:
    """将多个波段分别写入 COG。

    Args:
        band_dict: {name: array}
        meta: 参考元数据
        path_prefix: 路径前缀（不含扩展名）

    Returns:
        {name: file_path}
    """
    import os
    paths = {}
    os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
    for name, arr in band_dict.items():
        p = f"{path_prefix}_{name}.tif"
        write_cog(arr, meta, p)
        paths[name] = p
    return paths

def generate_report_v2(result, meta, stats, boundary_wgs84, out_dir):
    """生成解译报告 PNG。"""
    if config.DETECTION_MODE == "multi":
        return _gen_report_multi(result, meta, stats, out_dir)
    return _gen_report_binary(result, meta, stats, out_dir)


def _gen_report_multi(result, meta, stats, out_dir):
    ref_h, ref_w = result.shape
    colors = {-3: (180, 0, 0), -2: (255, 100, 0), 1: (40, 40, 40),
               2: (0, 180, 80), 3: (0, 80, 0)}
    rgb = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
    for cv, c in colors.items():
        rgb[result == cv] = c
    rgb[result == 0] = (10, 10, 10)

    cls_names = {-3: "严重退化", -2: "中度退化", 1: "稳定",
                  2: "中度恢复", 3: "强烈恢复"}
    cls_colors = {-3: "#b40000", -2: "#ff6400", 1: "#282828",
                   2: "#00b450", 3: "#005000"}

    fig, (ax_map, ax_stats) = plt.subplots(1, 2, figsize=(32, 16),
                                            gridspec_kw={"width_ratios": [2, 1]})
    ax_map.imshow(rgb)
    ax_map.set_title(
        f"{config.REGION_NAME}  植被变化分级图  ({config.YEAR1} vs {config.YEAR2})",
        fontsize=18, fontweight="bold")
    ax_map.axis("off")

    patches = []
    for cv in [-3, -2, 2, 3, 1]:
        s = stats.get(cv, {})
        label = f"{cls_names[cv]}  {s.get('area_ha', 0):.0f}ha ({s.get('percent', 0):.1f}%)"
        patches.append(Patch(color=cls_colors[cv], label=label))
    total = stats.get("changed_total", {})
    patches.append(Patch(color="#ffffff",
                         label=f"变化合计: {total.get('area_ha', 0):.0f}ha"))
    ax_map.legend(handles=patches, loc="lower right", fontsize=13,
                  framealpha=0.9, facecolor="#111111", labelcolor="white")

    cd = [(c, stats.get(c, {}).get("area_ha", 0)) for c in [-3, -2, 2, 3]]
    cd = [(n, a) for n, a in cd if a > 0]
    if cd:
        labels = [cls_names[c] for c, _ in cd]
        sizes = [a for _, a in cd]
        pie_colors = [cls_colors[c] for c, _ in cd]
        wedges, texts, autotexts = ax_stats.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=pie_colors, startangle=90,
            textprops={"fontsize": 14, "color": "white"})
        for t in autotexts:
            t.set_color("white")
            t.set_fontweight("bold")
        ax_stats.set_title("变化面积占比", fontsize=16, fontweight="bold", color="white")

    info = (
        f"区域: {config.REGION_NAME}\n"
        f"时相: {config.YEAR1} -> {config.YEAR2}\n"
        f"NDVI阈值: +/-{config.CHANGE_THRESHOLD}  强烈阈值: +/-{config.SEVERE_THRESHOLD}\n"
        f"有效像元: {stats.get('valid_pixels', 0):,}\n"
        f"总变化面积: {total.get('area_ha', 0):.0f} ha  ({total.get('percent', 0):.1f}%)"
    )
    ax_stats.text(0.5, -0.15, info, transform=ax_stats.transAxes,
                  fontsize=13, ha="center", va="top", color="#cccccc",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#222222",
                            edgecolor="#444444"))
    fig.patch.set_facecolor("#1a1a1a")
    ax_stats.set_facecolor("#1a1a1a")

    path = os.path.join(out_dir, f"{config.REGION_NAME}_变化检测报告.png")
    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()
    log(f"  Report: {path}")
    return path


def _gen_report_binary(result, meta, stats, out_dir):
    chg_px = stats.get("changed_pixels", (result == 1).sum())
    ref_h, ref_w = result.shape
    rgb = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
    rgb[(result == 1)] = [255, 60, 60]

    fig, ax = plt.subplots(1, 1, figsize=(24, 24))
    ax.imshow(rgb)
    ax.set_title(
        f"{config.REGION_NAME}  NDVI显著变化 ({config.YEAR1} vs {config.YEAR2})",
        fontsize=18, fontweight="bold")
    ax.axis("off")
    ax.legend(handles=[
        Patch(color="#ff3c3c",
              label=f"变化 {chg_px*config.AREA_PER_PIXEL_HECTARES:.0f} ha"),
    ], loc="lower right", fontsize=14)
    path = os.path.join(out_dir, "lichuan_change.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"  PNG: {path}")
    return path


# ============================================================================
#  流水线 (Pipeline)
# ============================================================================

def run_pipeline(data_dir=None, out_dir=None, boundary=None,
                 progress_callback=None):
    """v2 完整流水线：下载 -> 检测 -> 采样 -> 报告。"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = data_dir or os.path.join(base, "data")
    out_dir = out_dir or os.path.join(base, config.OUTPUT_DIR)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if boundary is None:
        boundary = get_boundary()

    results = {
        "boundary_file": config.BOUNDARY_FILE,
        "region": config.REGION_NAME,
        "year1": config.YEAR1, "year2": config.YEAR2,
        "detection_mode": config.DETECTION_MODE,
        "output_dir": out_dir,
        "files": {}, "stats": {}, "timing": {}
    }

    def cb(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)
        log(msg)

    t_total = time.time()

    # Step 1: 下载
    y1, y2 = config.YEAR1, config.YEAR2
    b4_1 = os.path.join(data_dir, f"B4_{y1}.tif")
    b4_2 = os.path.join(data_dir, f"B4_{y2}.tif")
    need_dl = not (os.path.exists(b4_1) and os.path.exists(b4_2))

    if need_dl:
        cb("下载影像中...", 5)
        for old_f in os.listdir(data_dir):
            if old_f.endswith(".tif"):
                os.remove(os.path.join(data_dir, old_f))
        for year in [y1, y2]:
            try:
                from engine.data_sources import get_data_source
                src = get_data_source("sentinel2")
                scenes = src.search_scenes(
                    boundary, year, config.MONTH_START,
                    config.MONTH_END, config.MAX_CLOUD_PCT)
                bands = src.download_bands(scenes, data_dir, ['red', 'nir'])
                if 'red' not in bands:
                    raise RuntimeError(f"{year} download failed")
            except Exception:
                import main as _m
                b4, b8 = _m.download_and_mosaic(boundary, year, data_dir)
                if b4 is None:
                    results["error"] = f"{year} download failed"
                    return results
                cb(f"  {year} done (legacy)", 15 if year == y1 else 30)
    else:
        cb("使用缓存影像数据", 30)

    # Step 2: 变化检测
    cb("运行变化检测...", 35)
    t_cd = time.time()
    result, meta, stats = run_change_detection_v2(data_dir, out_dir, boundary)
    results["timing"]["change_detection"] = round(time.time() - t_cd, 1)
    results["stats"] = stats

    if config.DETECTION_MODE == "multi":
        rmeta = meta.copy()
        rmeta.update(dtype="int8", count=1, compress="lzw")
        out_tif = os.path.join(out_dir, f"{config.REGION_NAME}_变化检测.tif")
        with rasterio.open(out_tif, "w", **rmeta) as dst:
            dst.write(result.astype(np.int8), 1)
    else:
        rmeta = meta.copy()
        rmeta.update(dtype="uint8", count=1, compress="lzw")
        out_tif = os.path.join(out_dir, "lichuan_change.tif")
        with rasterio.open(out_tif, "w", **rmeta) as dst:
            dst.write(result.astype(np.uint8), 1)
    results["files"]["geotiff"] = out_tif
    cb("变化检测完成", 60)

    # Step 3: 采样点
    cb("生成采样点...", 65)
    t_sp = time.time()
    sp_result = generate_sample_points_v2(result, boundary, meta, out_dir)
    results["timing"]["sampling"] = round(time.time() - t_sp, 1)
    if sp_result:
        out_paths, candidates = sp_result
        results["files"].update(out_paths)
        results["num_sample_points"] = len(candidates)
    else:
        results["num_sample_points"] = 0
    cb("采样点生成完成", 80)

    # Step 4: 报告
    if config.GENERATE_REPORT:
        cb("生成解译报告...", 85)
        rp = generate_report_v2(result, meta, stats, boundary, out_dir)
        if rp:
            results["files"]["report"] = rp

    results["timing"]["total"] = round(time.time() - t_total, 1)
    cb(f"全部完成! 耗时 {results['timing']['total']}s", 100)
    return results


def _cache_root():
    """快照缓存根目录（测试中可替换）。"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "cache")


def _find_quality_file(directory, year):
    """在目录中查找某年的质量波段文件 (SCL / QA_PIXEL)。"""
    for prefix in ("SCL", "QA_PIXEL"):
        path = os.path.join(directory, f"{prefix}_{year}.tif")
        if os.path.exists(path):
            return path
    return None


def _load_aligned_bands(paths_y1, paths_y2, quality_y1, quality_y2, data_source):
    """加载两期波段并对齐到统一网格，再在该网格上做定标 + 云掩膜。

    云掩膜必须在对齐之后做：S2 的 SCL 为 20 m，B4/B8 为 10 m，
    MNDWI/NBR 等还混用 10 m 与 20 m 波段。

    Args:
        paths_y1/paths_y2: {波段名: GeoTIFF 路径}（原始 DN）
        quality_y1/quality_y2: 质量波段路径，None=不做云掩膜
        data_source: 数据源名称

    Returns:
        (ay1, ay2, valid, meta, prep_info)，无可用波段时返回 None。
        ay1/ay2 为反射率，valid 为两期共同有效像元。
    """
    from engine.preprocessing import preprocess_arrays, quality_valid_on_grid

    target_grid, meta = None, None
    for path in list(paths_y1.values()) + list(paths_y2.values()):
        if path and os.path.exists(path):
            with rasterio.open(path) as src:
                target_grid = GridSpec.from_src(src)
                meta = src.meta
            break
    if target_grid is None:
        return None
    log(f"  Target grid: {target_grid}")

    def _read_aligned(path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if GridSpec.from_src(src) != target_grid:
                arr = reproject_to_grid((arr, src.meta), target_grid)
        return arr

    aligned, valid, prep_info = [], None, {}
    for key, paths, qpath in (("y1", paths_y1, quality_y1),
                              ("y2", paths_y2, quality_y2)):
        raw = {bn: _read_aligned(p) for bn, p in paths.items()
               if p and os.path.exists(p)}
        qvalid = None
        if qpath and os.path.exists(qpath):
            qvalid = quality_valid_on_grid(qpath, data_source, target_grid)
        bands, v = preprocess_arrays(raw, data_source, qvalid)

        has_data = np.logical_and.reduce([a > 0 for a in raw.values()])
        cloud_pct = None
        if qvalid is not None:
            cloud_pct = round(
                float((has_data & ~qvalid).sum()) * 100 / max(int(has_data.sum()), 1), 2)
            log(f"  Cloud mask {key}: {cloud_pct}% of data pixels masked")
        prep_info[key] = {"cloud_mask": qvalid is not None, "cloud_pct": cloud_pct}

        aligned.append(bands)
        valid = v if valid is None else (valid & v)

    return aligned[0], aligned[1], valid, meta, prep_info


def run_pipeline_v3(data_source_name="sentinel2", data_dir=None, out_dir=None,
                     boundary=None, bands=None, index_type=None,
                     detection_mode=None, progress_callback=None,
                     threshold=None, severe_threshold=None,
                     adaptive_threshold=False, min_area_m2=None):
    """v3 多源多指数流水线。

    Args:
        data_source_name: 数据源 ("sentinel2"/"landsat8"/"landsat9"/"sentinel1")
        data_dir: 数据缓存目录
        out_dir: 输出目录
        boundary: Shapely 边界几何
        bands: 所需波段列表 (None=自动推断)
        index_type: 指数类型 ("NDVI"/"NDWI"/etc)
        detection_mode: 检测模式 ("multi"/"binary")
        progress_callback: 进度回调
        threshold: 变化阈值 (None=使用config或注册表)
        severe_threshold: 强烈变化阈值 (None=使用config或注册表)
        adaptive_threshold: 是否自适应阈值
        min_area_m2: 最小斑块面积 m² (None=使用config)
    """
    from engine.data_sources import get_data_source

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = data_dir or os.path.join(base, "data")
    out_dir = out_dir or os.path.join(base, config.OUTPUT_DIR)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if boundary is None:
        boundary = get_boundary()

    idx = (index_type or getattr(config, 'INDEX_TYPE', 'NDVI')).upper()
    dmode = detection_mode or config.DETECTION_MODE

    def cb(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)
        log(msg)

    t_total = time.time()
    results = {
        "region": config.REGION_NAME, "year1": config.YEAR1,
        "year2": config.YEAR2, "data_source": data_source_name,
        "index_type": idx, "detection_mode": dmode,
        "output_dir": out_dir, "files": {}, "stats": {}, "timing": {}
    }

    source = get_data_source(data_source_name)
    needed = bands or get_needed_bands(idx)
    log(f"Data source: {source.name}, Index: {idx}, Bands: {needed}")

    from engine.preprocessing import PREPROCESSING_VERSION

    # =========================================================================
    #  检查已有缓存，避免重复下载
    #  缓存只存原始 DN + 质量波段，预处理每次在内存中执行
    # =========================================================================
    cache_key = None
    aoi_hash = None
    q1_path = q2_path = None
    try:
        from engine.cache import SnapshotCache, compute_aoi_hash
        sc = SnapshotCache(cache_root=_cache_root())
        aoi_hash = compute_aoi_hash(boundary)
        cached_key = sc.find(
            aoi_hash, data_source_name, idx,
            config.YEAR1, config.YEAR2,
            config.MONTH_START, config.MONTH_END,
            config.MAX_CLOUD_PCT, needed,
            preprocessing_version=PREPROCESSING_VERSION)
        if cached_key:
            cache_key = cached_key
            cached_dir = sc.get_band_dir(cached_key)
            log(f"  Cache hit: {cached_key}")
            # 检查缓存中的文件是否齐全
            by1, by2 = {}, {}
            for bn in needed:
                fname = getattr(source, 'FILE_NAME_MAP', {}).get(bn, bn.upper())
                y1_path = os.path.join(cached_dir, f"{fname}_{config.YEAR1}.tif")
                y2_path = os.path.join(cached_dir, f"{fname}_{config.YEAR2}.tif")
                if os.path.exists(y1_path):
                    by1[bn] = y1_path
                if os.path.exists(y2_path):
                    by2[bn] = y2_path
            if len(by1) == len(needed) and len(by2) == len(needed):
                cb(f"✅ 使用缓存数据 ({cached_key[:8]}...)", 25)
                q1_path = _find_quality_file(cached_dir, config.YEAR1)
                q2_path = _find_quality_file(cached_dir, config.YEAR2)
                source_pending_dl = False
            else:
                by1, by2 = {}, {}
                source_pending_dl = True
        else:
            source_pending_dl = True
    except Exception as e:
        log(f"  Cache check skipped: {e}")
        source_pending_dl = True

    if source_pending_dl:
        cb(f"搜索 {config.YEAR1} 年影像...", 10)
        s1 = source.search_scenes(
            boundary, config.YEAR1, config.MONTH_START,
            config.MONTH_END, config.MAX_CLOUD_PCT)
        cb(f"搜索 {config.YEAR2} 年影像...", 20)
        s2 = source.search_scenes(
            boundary, config.YEAR2, config.MONTH_START,
            config.MONTH_END, config.MAX_CLOUD_PCT)

        if not s1 or not s2:
            results["error"] = "No suitable scenes found"
            return results

        cb(f"下载 {config.YEAR1} 年波段...", 30)
        by1 = source.download_bands(s1, data_dir, needed)
        cb(f"下载 {config.YEAR2} 年波段...", 45)
        by2 = source.download_bands(s2, data_dir, needed)

        # --- 下载质量波段 (SCL / QA_PIXEL)，云掩膜在对齐后的网格上执行 ---
        q1 = source.download_quality_bands(s1, data_dir) if hasattr(source, "download_quality_bands") else {}
        q2 = source.download_quality_bands(s2, data_dir) if hasattr(source, "download_quality_bands") else {}
        q1_path = next(iter(q1.values()), None)
        q2_path = next(iter(q2.values()), None)

        # 存入缓存（原始 DN，不写回预处理结果，避免重复定标）
        if aoi_hash and cache_key is None:
            try:
                from engine.cache import SnapshotCache, snapshot_key
                sc = SnapshotCache(cache_root=_cache_root())
                new_key = snapshot_key(
                    aoi_hash, data_source_name, idx,
                    config.YEAR1, config.YEAR2,
                    config.MONTH_START, config.MONTH_END,
                    config.MAX_CLOUD_PCT, needed, PREPROCESSING_VERSION)
                snap_dict = {
                    "region": config.REGION_NAME, "aoi_hash": aoi_hash,
                    "data_source": data_source_name, "index_type": idx,
                    "year1": config.YEAR1, "year2": config.YEAR2,
                    "month_start": config.MONTH_START,
                    "month_end": config.MONTH_END,
                    "max_cloud_pct": config.MAX_CLOUD_PCT,
                    "bands": list(needed),
                    "tile_ids": [],
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                sc.put_snapshot(new_key, snap_dict)
                cache_dir = sc.get_band_dir(new_key)
                import shutil
                for bif in list(by1.values()) + list(by2.values()):
                    if os.path.exists(bif):
                        shutil.copy2(bif, cache_dir)
                # 也缓存质量波段
                for qif in list(q1.values()) + list(q2.values()):
                    if os.path.exists(qif):
                        shutil.copy2(qif, cache_dir)
                sc.mark_complete(new_key)
                log(f"  Cached: {new_key}")
            except Exception as e:
                log(f"  Cache write skipped: {e}")

    cb("加载波段数据...", 55)
    loaded = _load_aligned_bands(by1, by2, q1_path, q2_path, data_source_name)
    if loaded is None:
        results["error"] = "Failed to determine target grid"
        return results
    ay1, ay2, valid, meta, prep_info = loaded
    results["preprocessing"] = prep_info
    for key in ("y1", "y2"):
        if not prep_info[key]["cloud_mask"]:
            log(f"  WARNING: {key} 缺少质量波段，未做云掩膜")

    cb("运行变化检测...", 65)
    t_cd = time.time()

    if dmode == "ml":
        from engine.ml_detector import run_ml_change_detection
        result, stats = run_ml_change_detection(
            ay1, ay2, valid,
            method=getattr(config, 'ML_MODEL', 'random_forest'),
            model_path=getattr(config, 'ML_MODEL_PATH', None))
    else:
        result, meta, stats = run_change_detection_generic(
            ay1, ay2, meta, boundary,
            index_type=idx, detection_mode=dmode,
            threshold=threshold, severe_threshold=severe_threshold,
            adaptive_threshold=adaptive_threshold, min_area_m2=min_area_m2,
            valid_mask=valid)

    results["timing"]["change_detection"] = round(time.time() - t_cd, 1)
    results["stats"] = stats

    rmeta = meta.copy()
    rmeta.update(dtype="int8", count=1, compress="lzw")
    out_tif = os.path.join(out_dir, f"{config.REGION_NAME}_{idx}_变化检测.tif")
    write_cog(result.astype(np.int8), rmeta, out_tif)
    results["files"]["geotiff"] = out_tif
    cb("变化检测完成", 75)

    cb("生成采样点...", 80)
    sp = generate_sample_points_v2(result, boundary, meta, out_dir)
    if sp:
        out_paths, candidates = sp
        results["files"].update(out_paths)
        results["num_sample_points"] = len(candidates)
    else:
        results["num_sample_points"] = 0

    if config.GENERATE_REPORT:
        cb("生成报告...", 90)
        rp = generate_report_v2(result, meta, stats, boundary, out_dir)
        if rp:
            results["files"]["report"] = rp

    results["timing"]["total"] = round(time.time() - t_total, 1)
    cb(f"完成! 耗时 {results['timing']['total']}s", 100)
    return results
