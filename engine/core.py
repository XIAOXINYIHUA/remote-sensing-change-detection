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

import os, json, time, shutil, tempfile, warnings
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pyproj import Transformer
import cv2
import pandas as pd

import config
from engine.utils import (
    log, get_boundary, boundary_pixel_mask, get_needed_bands,
)
from engine.grid import GridSpec, reproject_to_grid, as_gridspec

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


def _filter_multi_level(result, area_per_px_ha=0.01, min_area_m2=None,
                        min_area_by_class=None):
    """对多级结果的每个变化类别分别连通域过滤。

    Args:
        result: 多级分类结果数组
        area_per_px_ha: 每像元面积 (ha)
        min_area_m2: 全局最小斑块面积 (m²)。若提供，覆盖各变化类的独立值。
        min_area_by_class: {类别: 最小面积 m²}，None=config.MIN_AREA_BY_CLASS
    """
    by_class = config.MIN_AREA_BY_CLASS if min_area_by_class is None else min_area_by_class
    by_class = {int(k): v for k, v in by_class.items()}
    area_per_px_m2 = area_per_px_ha * 10000.0
    for cls_val in [-3, -2, 2, 3]:
        if min_area_m2 is not None:
            # 使用全局统一值
            min_area = min_area_m2
        elif cls_val in by_class:
            min_area = by_class[cls_val]
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
                                  valid_mask=None, min_area_by_class=None,
                                  binary_min_area_m2=None, morph_kernel_size=None,
                                  bilateral_filter=False):
    """v3 通用指数变化检测：支持 NDVI/NDWI/MNDWI/NBR/EVI/SAVI/NDBI。

    阈值优先级:
        adaptive_threshold=True > 显式传入参数 > INDEX_THRESHOLDS > config

    valid_mask: 预处理得到的有效像元（含云掩膜）；None 时退回「所有波段 > 0」。
    min_area_by_class / binary_min_area_m2 / morph_kernel_size: None=config 默认值。
    bilateral_filter: 差值前对两期指数做双边滤波降噪。
    """
    from engine.indices import compute_index, INDEX_THRESHOLDS, INDEX_CLASS_NAMES

    ref_h, ref_w = list(bands_y1.values())[0].shape[:2]

    # 所有波段必须已在 meta 描述的同一网格上
    expected = (meta["height"], meta["width"])
    for label, bands in (("y1", bands_y1), ("y2", bands_y2)):
        for bname, arr in bands.items():
            if arr.shape[:2] != expected:
                raise ValueError(
                    f"Band {label}_{bname} shape {arr.shape[:2]} != grid {expected}; "
                    f"align bands with reproject_to_grid() first")

    if valid_mask is not None:
        valid = valid_mask.copy()
    else:
        valid = np.ones((ref_h, ref_w), dtype=bool)
        for arr in list(bands_y1.values()) + list(bands_y2.values()):
            valid &= (arr > 0)

    idx1 = compute_index(index_type, bands_y1)
    idx2 = compute_index(index_type, bands_y2)
    if bilateral_filter:
        idx1 = cv2.bilateralFilter(idx1.astype(np.float32), config.BILATERAL_D,
                                   config.BILATERAL_SIGMA_COLOR, config.BILATERAL_SIGMA_SPACE)
        idx2 = cv2.bilateralFilter(idx2.astype(np.float32), config.BILATERAL_D,
                                   config.BILATERAL_SIGMA_COLOR, config.BILATERAL_SIGMA_SPACE)
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
        result = _filter_multi_level(result, area_per_px_ha, min_area_m2=min_area_m2,
                                     min_area_by_class=min_area_by_class)
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
        ksize = morph_kernel_size or config.MORPH_KERNEL_SIZE
        k = np.ones((ksize, ksize), np.uint8)
        change_mask = cv2.morphologyEx(
            change_mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
        # 二值模式连通域过滤
        area_m2 = min_area_m2 or binary_min_area_m2 or config.MIN_CHANGE_AREA_M2
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
            "_severe_threshold": S,
            "_area_per_px_ha": area_per_px_ha,
        }

    return result, meta, stats


# ============================================================================
#  采样点生成
# ============================================================================

SAMPLE_BASENAME = "sample_points"
MULTI_CHANGE_CLASSES = (-3, -2, 2, 3)


def _patch_points(result, change_classes):
    """每个变化类别分别做连通域，返回每个斑块的内部代表点。

    相邻的不同类别不会被合并成一个斑块；代表点取质心，质心落在斑块外
    （凹形/环形斑块）时改取斑块内离质心最近的像元，保证类别取值正确。

    Returns:
        list[(row, col, area_px, class)]
    """
    h, w = result.shape
    points = []
    for cv in change_classes:
        mask = (result == cv).astype(np.uint8)
        if not mask.any():
            continue
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            continue
        ids = np.arange(1, n)
        cx = np.clip(np.rint(cents[1:, 0]).astype(int), 0, w - 1)
        cy = np.clip(np.rint(cents[1:, 1]).astype(int), 0, h - 1)
        for j in np.nonzero(labels[cy, cx] != ids)[0]:
            lab = ids[j]
            x0, y0, bw, bh = stats[lab, :4]
            rr, cc = np.nonzero(labels[y0:y0 + bh, x0:x0 + bw] == lab)
            t = np.argmin((rr + y0 - cents[lab, 1]) ** 2 + (cc + x0 - cents[lab, 0]) ** 2)
            cy[j], cx[j] = rr[t] + y0, cc[t] + x0
        areas = stats[1:, cv2.CC_STAT_AREA]
        points.extend(zip(cy.tolist(), cx.tolist(), areas.tolist(), [cv] * len(ids)))
    return points


def _thin_by_distance(xs, ys, min_dist):
    """按输入顺序贪心保留点，使任意两点间距 >= min_dist（网格哈希，O(n)）。"""
    cell = float(min_dist)
    buckets = {}
    keep = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        gx, gy = int(x // cell), int(y // cell)
        near = (j for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                for j in buckets.get((gx + dx, gy + dy), ()))
        if any((x - xs[j]) ** 2 + (y - ys[j]) ** 2 < cell * cell for j in near):
            continue
        keep.append(i)
        buckets.setdefault((gx, gy), []).append(i)
    return keep


def generate_sample_points(result, boundary_wgs84, meta, out_dir, params, stats=None):
    """斑块代表点采样 + 多格式导出。

    Args:
        result: 变化检测结果（多级 -3..3 或二值 0/1）
        boundary_wgs84: WGS84 边界
        meta: 结果网格元数据
        out_dir: 输出目录
        params: RunParams
        stats: 检测统计（用于取实际像元面积与类别名）

    Returns:
        (out_paths, candidates)；无变化斑块时返回 None。
    """
    import shapely

    binary = params.detection_mode == "binary"
    classes = (1,) if binary else MULTI_CHANGE_CLASSES
    names = ({1: "变化"} if binary else params.change_class_names)

    log("\nSampling change patches...")
    points = _patch_points(result, classes)
    log(f"  Patches: {len(points)}")
    if not points:
        return None

    area_px_ha = (stats or {}).get("_area_per_px_ha") or GridSpec.from_meta(meta).area_per_pixel_ha

    # 面积降序：大斑块优先保留
    points.sort(key=lambda p: -p[2])
    rows = np.array([p[0] for p in points])
    cols = np.array([p[1] for p in points])
    xs, ys = rasterio.transform.xy(meta["transform"], rows, cols)
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    order = np.arange(len(points))
    if params.min_point_distance_m:
        # 结果网格为投影坐标 (m)，直接按米计算距离
        order = np.array(_thin_by_distance(xs, ys, params.min_point_distance_m), dtype=int)
        log(f"  After distance filter ({params.min_point_distance_m} m): {len(order)}")
    if params.num_sample_points:
        order = order[:params.num_sample_points]

    t_crs = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True)
    lons, lats = t_crs.transform(xs[order], ys[order])
    inside = shapely.contains_xy(boundary_wgs84, lons, lats)

    candidates = []
    for k, i in enumerate(order):
        if not inside[k]:
            continue
        row, col, area_px, cv = points[i]
        candidates.append({
            "lon": float(lons[k]), "lat": float(lats[k]),
            "col": int(col), "row": int(row),
            "area_px": int(area_px),
            "area_ha": round(area_px * area_px_ha, 4),
            "class": int(cv), "class_name": names.get(cv, "未知"),
        })
    log(f"  Sample points: {len(candidates)}")
    if not candidates:
        return None

    exporters = {
        "geojson": lambda: _export_geojson(candidates, out_dir, params),
        "csv": lambda: _export_csv(candidates, out_dir),
        "shp": lambda: _export_shapefile(candidates, out_dir),
    }
    out_paths = {}
    for fmt in params.output_formats:
        p = exporters[fmt]()
        if p:
            out_paths[fmt] = p
    return out_paths, candidates


def generate_sample_points_v2(result, boundary_wgs84, meta, out_dir):
    """[v2 兼容] 以 config 默认参数采样。"""
    from engine.params import RunParams
    return generate_sample_points(result, boundary_wgs84, meta, out_dir,
                                  RunParams.from_config())


def _export_geojson(candidates, out_dir, params):
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
            "description": (f"{params.index_type} 变化检测采样点 "
                            f"({params.year1} vs {params.year2})"),
            "total_points": len(features), "region": params.region,
        }
    }
    path = os.path.join(out_dir, f"{SAMPLE_BASENAME}.geojson")
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
    path = os.path.join(out_dir, f"{SAMPLE_BASENAME}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"  CSV: {path} ({len(rows)} rows)")
    return path


def _export_shapefile(candidates, out_dir):
    """导出 Shapefile。坐标为经纬度，因此声明 EPSG:4326。"""
    try:
        import fiona
        from fiona.crs import CRS as FionaCRS
    except ImportError:
        log("  Shapefile export requires fiona. Install: pip install fiona")
        return None
    schema = {
        "geometry": "Point",
        "properties": {
            "id": "int", "area_px": "int", "area_ha": "float",
            "class": "int", "class_name": "str"
        }
    }
    path = os.path.join(out_dir, f"{SAMPLE_BASENAME}.shp")
    with fiona.open(path, "w", driver="ESRI Shapefile",
                    schema=schema, crs=FionaCRS.from_epsg(4326),
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
    """写入 Cloud Optimized GeoTIFF（GDAL COG 驱动：内部分块 + 概览）。

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
        dtype = arr.dtype.name if arr.dtype.name.startswith(("float", "int", "uint")) else "float32"

    profile = {k: v for k, v in meta.items()
               if k in ("crs", "transform", "width", "height", "nodata")}
    profile.update(driver="COG", dtype=dtype, count=1, compress=compress,
                   blocksize=256, overview_resampling="nearest", bigtiff="IF_SAFER")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.dtype(dtype)), 1)

    log(f"  COG: {path} ({arr.shape[0]}x{arr.shape[1]}, {dtype})")
    return path


def generate_report(result, stats, out_dir, params):
    """生成解译报告 PNG（类别名、配色、阈值随指数与实际参数变化）。"""
    if params.detection_mode == "binary":
        return _gen_report_binary(result, stats, out_dir, params)
    return _gen_report_multi(result, stats, out_dir, params)


def generate_report_v2(result, meta, stats, boundary_wgs84, out_dir):
    """[v2 兼容] 以 config 默认参数生成报告。"""
    from engine.params import RunParams
    return generate_report(result, stats, out_dir, RunParams.from_config())


def _hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def _gen_report_multi(result, stats, out_dir, params):
    from engine.indices import INDEX_COLORS
    idx = params.index_type
    cls_names = params.change_class_names
    cls_colors = INDEX_COLORS.get(idx, INDEX_COLORS["NDVI"])

    rgb = np.full(result.shape + (3,), 10, dtype=np.uint8)   # 无数据
    for cv, c in cls_colors.items():
        rgb[result == cv] = _hex_to_rgb(c)

    fig, (ax_map, ax_stats) = plt.subplots(1, 2, figsize=(24, 12),
                                            gridspec_kw={"width_ratios": [2, 1]})
    ax_map.imshow(rgb)
    ax_map.set_title(
        f"{params.region}  {idx} 变化分级图  ({params.year1} vs {params.year2})",
        fontsize=18, fontweight="bold", color="white")
    ax_map.axis("off")

    total = stats.get("changed_total", {})
    patches = []
    for cv in [-3, -2, 2, 3, 1]:
        s = stats.get(cv, {})
        label = f"{cls_names[cv]}  {s.get('area_ha', 0):.0f}ha ({s.get('percent', 0):.1f}%)"
        patches.append(Patch(color=cls_colors[cv], label=label))
    patches.append(Patch(color="#ffffff",
                         label=f"变化合计: {total.get('area_ha', 0):.0f}ha"))
    ax_map.legend(handles=patches, loc="lower right", fontsize=13,
                  framealpha=0.9, facecolor="#111111", labelcolor="white")

    cd = [(c, stats.get(c, {}).get("area_ha", 0)) for c in MULTI_CHANGE_CLASSES]
    cd = [(c, a) for c, a in cd if a > 0]
    if cd:
        _, _, autotexts = ax_stats.pie(
            [a for _, a in cd], labels=[cls_names[c] for c, _ in cd],
            autopct="%1.1f%%", colors=[cls_colors[c] for c, _ in cd], startangle=90,
            textprops={"fontsize": 14, "color": "white"})
        for t in autotexts:
            t.set_fontweight("bold")
        ax_stats.set_title("变化面积占比", fontsize=16, fontweight="bold", color="white")
    else:
        ax_stats.axis("off")

    T, S = stats.get("_threshold"), stats.get("_severe_threshold")
    info = (
        f"区域: {params.region}\n"
        f"时相: {params.year1} -> {params.year2}（{params.month_start}-{params.month_end} 月）\n"
        f"数据源: {params.data_source}   指数: {idx}\n"
        f"{idx} 阈值: ±{T}  强烈阈值: ±{S}\n"
        f"有效像元: {stats.get('valid_pixels', 0):,}\n"
        f"总变化面积: {total.get('area_ha', 0):.0f} ha  ({total.get('percent', 0):.1f}%)"
    )
    ax_stats.text(0.5, -0.15, info, transform=ax_stats.transAxes,
                  fontsize=13, ha="center", va="top", color="#cccccc",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#222222",
                            edgecolor="#444444"))
    fig.patch.set_facecolor("#1a1a1a")
    ax_stats.set_facecolor("#1a1a1a")

    path = os.path.join(out_dir, f"{params.region}_{idx}_变化检测报告.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    log(f"  Report: {path}")
    return path


def _gen_report_binary(result, stats, out_dir, params):
    idx = params.index_type
    rgb = np.zeros(result.shape + (3,), dtype=np.uint8)
    rgb[result == 1] = [255, 60, 60]

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.imshow(rgb)
    ax.set_title(
        f"{params.region}  {idx} 显著变化 ({params.year1} vs {params.year2}，"
        f"阈值 ±{stats.get('_threshold')})",
        fontsize=18, fontweight="bold")
    ax.axis("off")
    ax.legend(handles=[
        Patch(color="#ff3c3c",
              label=f"变化 {stats.get('changed_area_ha', 0):.0f} ha "
                    f"({stats.get('changed_percent', 0):.1f}%)"),
    ], loc="lower right", fontsize=14)
    path = os.path.join(out_dir, f"{params.region}_{idx}_变化检测报告.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Report: {path}")
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
    """快照缓存根目录：环境变量 RSAP_CACHE_DIR，默认 <项目>/data/cache。

    用环境变量而不只是在测试里替换函数，是为了让后台任务的工作进程也指向同一处。
    """
    if os.environ.get("RSAP_CACHE_DIR"):
        return os.environ["RSAP_CACHE_DIR"]
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
    from engine.preprocessing import load_period

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

    aligned, valid, prep_info = [], None, {}
    for key, paths, qpath in (("y1", paths_y1, quality_y1),
                              ("y2", paths_y2, quality_y2)):
        bands, v, info = load_period(paths, qpath, data_source, target_grid)
        if info["cloud_mask"]:
            log(f"  Cloud mask {key}: {info['cloud_pct']}% of data pixels masked")
        prep_info[key] = info
        aligned.append(bands)
        valid = v if valid is None else (valid & v)

    return aligned[0], aligned[1], valid, meta, prep_info


def _period_files(source, directory, year, needed):
    """单期目录中各波段文件路径（文件名 <前缀>_<年份>.tif）。"""
    return {bn: os.path.join(directory, f"{source.FILE_NAME_MAP.get(bn, bn.upper())}_{year}.tif")
            for bn in needed}


def _open_period_cache():
    """返回 (SnapshotCache, 预处理版本)；缓存不可用时 (None, 版本)。"""
    from engine.cache import SnapshotCache
    from engine.preprocessing import PREPROCESSING_VERSION
    try:
        return SnapshotCache(cache_root=_cache_root()), PREPROCESSING_VERSION
    except Exception as e:
        log(f"  Cache unavailable: {e}")
        return None, PREPROCESSING_VERSION


def _cached_period(sc, version, source, data_source, aoi_hash, year,
                   month_start, month_end, max_cloud_pct, needed):
    """命中则返回 dict(paths, quality, key, dir, tiles)，否则 None。"""
    if sc is None:
        return None
    key = sc.find_period(aoi_hash, data_source, year, month_start, month_end,
                         max_cloud_pct, needed, version)
    if not key:
        return None
    cdir = sc.get_band_dir(key)
    paths = _period_files(source, cdir, year, needed)
    if not all(os.path.exists(f) for f in paths.values()):
        log(f"  Cache entry {key} incomplete, re-downloading")
        return None
    snap = sc.get_snapshot(key) or {}
    return dict(paths=paths, quality=_find_quality_file(cdir, year), key=key,
                dir=cdir, tiles=len(snap.get("tile_ids", [])))


def _store_period(sc, version, data_source, aoi_hash, year, month_start, month_end,
                  max_cloud_pct, needed, paths, quality, scenes):
    """把下载好的一期写入缓存，返回缓存中的 dict(paths, quality, key, dir)；失败返回 None。"""
    from engine.cache import PERIOD_KIND, period_key
    if sc is None or not aoi_hash:
        return None
    try:
        key = period_key(aoi_hash, data_source, year, month_start, month_end,
                         max_cloud_pct, needed, version)
        sc.put_snapshot(key, {
            "kind": PERIOD_KIND, "aoi_hash": aoi_hash, "data_source": data_source,
            "year": int(year), "month_start": int(month_start), "month_end": int(month_end),
            "max_cloud_pct": max_cloud_pct, "bands": sorted(needed),
            "tile_ids": [s["item"].id for s in scenes],
            "preprocessing_version": version,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        cdir = sc.get_band_dir(key)
        for f in list(paths.values()) + ([quality] if quality else []):
            shutil.copy2(f, cdir)
        sc.mark_complete(key)
    except Exception as e:
        log(f"  Cache write skipped ({year}): {e}")
        return None
    log(f"  Cached {year}: {key}")
    return dict(key=key, dir=cdir,
                paths={bn: os.path.join(cdir, os.path.basename(f)) for bn, f in paths.items()},
                quality=os.path.join(cdir, os.path.basename(quality)) if quality else None)


def _acquire_periods(source, data_source, boundary, years, month_start, month_end,
                     max_cloud_pct, needed, work_dir, cb, require_all=True, pct=(10, 50)):
    """逐期取得原始 DN 波段与质量波段，变化检测 / 地物分类 / 时间序列共用。

    每期先查单期快照缓存（波段为所需的超集即可复用）；未命中的期先全部检索，
    require_all 时任一期无影像就在下载前报错，否则跳过该期；再下载到本次运行
    独占的 work_dir 并写入缓存，并发运行不会互相覆盖同名文件。

    Returns:
        {年份: dict(paths, quality, key, dir, tiles)}；key 为 None 表示写缓存失败、
        文件仍在 work_dir 中（调用方须在 work_dir 清理前读取）。
        require_all 且失败时返回错误信息字符串。
    """
    from engine.cache import compute_aoi_hash

    lo, hi = pct
    sc, version = _open_period_cache()
    aoi_hash = compute_aoi_hash(boundary)
    if aoi_hash == "unknown_aoi":       # 无法哈希的边界不能共用缓存，否则不同 AOI 会互相命中
        sc = None
    args = (month_start, month_end, max_cloud_pct)

    got, todo = {}, []
    for year in years:
        hit = _cached_period(sc, version, source, data_source, aoi_hash, year, *args, needed)
        if hit:
            got[year] = hit
            cb(f"✅ 使用缓存数据 {year} ({hit['key'][:8]}...)", lo)
        else:
            todo.append(year)

    step = (hi - lo) / max(2 * len(todo), 1)
    scenes = {}
    for i, year in enumerate(todo):
        cb(f"搜索 {year} 年影像...", lo + step * i)
        try:
            scenes[year] = source.search_scenes(boundary, year, *args)
        except Exception as e:
            if require_all:
                raise
            log(f"  {year} search failed: {e}")
            scenes[year] = []
    empty = [str(y) for y in todo if not scenes[y]]
    if empty and require_all:
        return f"未找到符合条件的影像（{', '.join(empty)} 年，云量 < {max_cloud_pct}%）"
    for year in empty:
        log(f"  No scenes for {year}, skipping")

    for i, year in enumerate(y for y in todo if scenes[y]):
        cb(f"下载 {year} 年波段...", lo + step * (len(todo) + i))
        year_dir = os.path.join(work_dir, str(year))
        try:
            paths = source.download_bands(scenes[year], year_dir, needed, boundary)
            # 质量波段 (SCL / QA_PIXEL)：云掩膜在对齐后的网格上执行
            quality = next(iter(source.download_quality_bands(
                scenes[year], year_dir, boundary).values()), None)
        except Exception as e:
            if require_all:
                raise
            log(f"  {year} skipped: {e}")
            continue
        lost = [f"{bn}_{year}" for bn in needed if bn not in paths]
        if lost:
            if require_all:
                return f"波段下载失败: {lost}"
            log(f"  {year}: 波段下载失败 {lost}, skipping")
            continue
        stored = _store_period(sc, version, data_source, aoi_hash, year, *args,
                               needed, paths, quality, scenes[year])
        got[year] = dict(stored or dict(paths=paths, quality=quality, key=None, dir=year_dir),
                         tiles=len(scenes[year]))
    return got


def _acquire_bands(source, boundary, params, needed, work_dir, cb):
    """变化检测：取得两期波段（命中缓存的期不再下载）。

    Returns:
        dict(by1, by2, q1, q2, cache_key)；失败时返回错误信息字符串。
        cache_key 为两期快照键以 "+" 连接，任一期未写入缓存时为 None。
    """
    p = params
    got = _acquire_periods(source, p.data_source, boundary, (p.year1, p.year2),
                           p.month_start, p.month_end, p.max_cloud_pct,
                           needed, work_dir, cb)
    if isinstance(got, str):
        return got
    a, b = got[p.year1], got[p.year2]
    return dict(by1=a["paths"], by2=b["paths"], q1=a["quality"], q2=b["quality"],
                cache_key=f"{a['key']}+{b['key']}" if a["key"] and b["key"] else None)


def _dir_size_mb(directory):
    return sum(os.path.getsize(os.path.join(directory, f))
               for f in os.listdir(directory) if f.endswith(".tif")) / 1024 / 1024


def find_cached_period(data_source, boundary, year, month_start, month_end,
                       max_cloud_pct, bands):
    """查找单期快照缓存（波段为 bands 的超集即可）。

    Returns:
        dict(key, dir, size_mb, tiles, files)；未命中返回 None。
    """
    from engine.cache import compute_aoi_hash
    from engine.data_sources import get_data_source

    sc, version = _open_period_cache()
    try:
        aoi_hash = compute_aoi_hash(boundary)
        if aoi_hash == "unknown_aoi":
            return None
        hit = _cached_period(sc, version, get_data_source(data_source), data_source,
                             aoi_hash, year, month_start, month_end,
                             max_cloud_pct, list(bands))
    except Exception as e:
        log(f"  Cache lookup failed: {e}")
        return None
    if not hit:
        return None
    files = sorted(f for f in os.listdir(hit["dir"]) if f.endswith(".tif"))
    return {"key": hit["key"], "dir": hit["dir"], "tiles": hit["tiles"],
            "size_mb": round(_dir_size_mb(hit["dir"]), 1), "files": files}


def find_cached_snapshot(params, boundary, bands=None):
    """查找变化检测两期是否都已缓存（与本次参数匹配）。

    Returns:
        dict(key, size_mb, tiles)，key 为两期快照键以 "+" 连接；任一期未命中返回 None。
    """
    p = params
    needed = list(bands or get_needed_bands(p.index_type))
    hits = [find_cached_period(p.data_source, boundary, year, p.month_start, p.month_end,
                               p.max_cloud_pct, needed) for year in (p.year1, p.year2)]
    if not all(hits):
        return None
    return {"key": "+".join(h["key"] for h in hits),
            "size_mb": round(sum(h["size_mb"] for h in hits), 1),
            "tiles": sum(h["tiles"] for h in hits)}


def prepare_period(data_source, boundary, year, month_start, month_end, max_cloud_pct,
                   bands, data_dir=None, progress_callback=None):
    """地物分类：取得一期影像（命中缓存直接返回，否则检索 + 下载 + 写入缓存）。

    Returns:
        dict(key, dir, size_mb, tiles, files)；失败时 dict(error=...)
    """
    from engine.data_sources import get_data_source

    def cb(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)
        log(msg)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = data_dir or os.path.join(base, "data")
    os.makedirs(data_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dl_", dir=data_dir,
                                     ignore_cleanup_errors=True) as work_dir:
        got = _acquire_periods(get_data_source(data_source), data_source, boundary, (year,),
                               month_start, month_end, max_cloud_pct, list(bands),
                               work_dir, cb, pct=(5, 90))
    if isinstance(got, str):
        return {"error": got}
    if not got[year]["key"]:
        return {"error": "影像已下载但写入缓存失败，请检查 data/cache 目录权限与磁盘空间"}
    info = find_cached_period(data_source, boundary, year, month_start, month_end,
                              max_cloud_pct, bands)
    cb("影像准备完成", 100)
    return info or {"error": "缓存写入后未能读取，请重试"}


def prepare_data(params, boundary, data_dir=None, bands=None, progress_callback=None):
    """只做取数（检索 + 下载 + 写入快照缓存），供界面「资料收集」阶段调用。

    之后 run_pipeline_v3 以相同参数运行时会直接命中缓存。

    Returns:
        dict(key, size_mb, tiles, bands)；失败时 dict(error=...)
    """
    from engine.data_sources import get_data_source

    def cb(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)
        log(msg)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = data_dir or os.path.join(base, "data")
    os.makedirs(data_dir, exist_ok=True)
    needed = list(bands or get_needed_bands(params.index_type))
    source = get_data_source(params.data_source)

    with tempfile.TemporaryDirectory(prefix="dl_", dir=data_dir,
                                     ignore_cleanup_errors=True) as work_dir:
        acq = _acquire_bands(source, boundary, params, needed, work_dir, cb)
    if isinstance(acq, str):
        return {"error": acq}
    if not acq["cache_key"]:
        return {"error": "影像已下载但写入缓存失败，请检查 data/cache 目录权限与磁盘空间"}
    info = find_cached_snapshot(params, boundary, needed) or {"key": acq["cache_key"]}
    info["bands"] = needed
    cb("资料收集完成", 100)
    return info


def run_pipeline_v3(data_source_name=None, data_dir=None, out_dir=None,
                     boundary=None, bands=None, index_type=None,
                     detection_mode=None, progress_callback=None,
                     threshold=None, severe_threshold=None,
                     adaptive_threshold=None, min_area_m2=None, params=None):
    """v3 多源多指数流水线：取数 -> 预处理 -> 检测 -> 采样 -> 报告。

    全部运行参数来自 params (RunParams)，流水线不读写全局 config 的运行参数，
    可在多个 Streamlit 会话 / 线程中并发调用。

    Args:
        params: RunParams；None 时由 config 默认值 + 下列关键字参数构造（兼容旧调用）
        data_dir: 数据目录（下载临时目录与缓存位于其下）
        out_dir: 输出目录
        boundary: WGS84 Shapely 边界（None=读取 params.boundary_file）
        bands: 所需波段列表（None=按指数推断）
        progress_callback: 进度回调 (msg, pct)
        data_source_name/index_type/detection_mode/threshold/severe_threshold/
        adaptive_threshold/min_area_m2: 仅在 params=None 时生效

    Returns:
        dict: 结果摘要（files/stats/timing/params，失败时含 error）
    """
    from engine.data_sources import get_data_source
    from engine.params import RunParams

    if params is None:
        params = RunParams.from_config(
            data_source=data_source_name, index_type=index_type,
            detection_mode=detection_mode, threshold=threshold,
            severe_threshold=severe_threshold,
            adaptive_threshold=adaptive_threshold, min_area_m2=min_area_m2)
    p = params

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = data_dir or os.path.join(base, "data")
    out_dir = out_dir or os.path.join(base, config.OUTPUT_DIR)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if boundary is None:
        boundary = get_boundary(p.boundary_file)

    idx, dmode = p.index_type, p.detection_mode

    def cb(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)
        log(msg)

    t_total = time.time()
    results = {
        "region": p.region, "year1": p.year1, "year2": p.year2,
        "data_source": p.data_source, "index_type": idx, "detection_mode": dmode,
        "params": p.to_dict(),
        "output_dir": out_dir, "files": {}, "stats": {}, "timing": {}
    }

    source = get_data_source(p.data_source)
    needed = list(bands or get_needed_bands(idx))
    log(f"Data source: {source.name}, Index: {idx}, Bands: {needed}")

    with tempfile.TemporaryDirectory(prefix="run_", dir=data_dir,
                                     ignore_cleanup_errors=True) as work_dir:
        acq = _acquire_bands(source, boundary, p, needed, work_dir, cb)
        if isinstance(acq, str):
            results["error"] = acq
            return results
        results["cache_key"] = acq["cache_key"]

        cb("加载波段数据...", 55)
        loaded = _load_aligned_bands(acq["by1"], acq["by2"], acq["q1"], acq["q2"],
                                     p.data_source)
    if loaded is None:
        results["error"] = "无法确定分析网格（波段文件缺失）"
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
        valid = valid & boundary_pixel_mask(boundary, meta, meta["height"], meta["width"])
        result, stats = run_ml_change_detection(
            ay1, ay2, valid, method=p.ml_model, model_path=p.ml_model_path,
            area_per_px_ha=GridSpec.from_meta(meta).area_per_pixel_ha,
            class_names=p.change_class_names)
    else:
        result, meta, stats = run_change_detection_generic(
            ay1, ay2, meta, boundary,
            index_type=idx, detection_mode=dmode,
            threshold=p.threshold, severe_threshold=p.severe_threshold,
            adaptive_threshold=p.adaptive_threshold, min_area_m2=p.min_area_m2,
            valid_mask=valid, min_area_by_class=p.min_area_by_class,
            binary_min_area_m2=p.binary_min_area_m2,
            morph_kernel_size=p.morph_kernel_size,
            bilateral_filter=p.use_bilateral_filter)

    results["timing"]["change_detection"] = round(time.time() - t_cd, 1)
    results["stats"] = stats

    rmeta = meta.copy()
    rmeta.update(nodata=None)
    out_tif = os.path.join(out_dir, f"{p.region}_{idx}_变化检测.tif")
    write_cog(result.astype(np.int8), rmeta, out_tif)
    results["files"]["geotiff"] = out_tif
    cb("变化检测完成", 75)

    cb("生成采样点...", 80)
    sp = generate_sample_points(result, boundary, meta, out_dir, p, stats)
    if sp:
        out_paths, candidates = sp
        results["files"].update(out_paths)
        results["num_sample_points"] = len(candidates)
    else:
        results["num_sample_points"] = 0

    if p.generate_report:
        cb("生成报告...", 90)
        rp = generate_report(result, stats, out_dir, p)
        if rp:
            results["files"]["report"] = rp

    results["timing"]["total"] = round(time.time() - t_total, 1)
    cb(f"完成! 耗时 {results['timing']['total']}s", 100)
    return results
