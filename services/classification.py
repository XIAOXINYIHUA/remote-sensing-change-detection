"""
地物分类服务：取数 → 加载波段 → 快速分类 / GMM 聚类 / 主动学习 → 后处理 → 导出

分类结果为 int8 数组：边界外与云 / 无数据像元为 -1，0 为未分类，其余见 CLASS_NAMES。
面积统计一律按 meta 的实际像元大小计算（Sentinel-2 10 m、Landsat 30 m）。
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
from engine.classifier import (
    CLASS_NAMES,
    build_feature_stack,
    classification_statistics,
    classify_multi_index,
    pca_gmm_classify,
    sample_patches,
    train_from_annotations,
)
from engine.core import find_cached_period, prepare_period
from engine.utils import log
from services import ServiceError, report

# 分类需要的 6 个波段；经单期快照缓存取数，与变化检测 / 时间序列共用
CLASSIFY_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]

# 主动学习至少需要的标注像元数
MIN_ANNOTATED_PX = 10

# 矢量导出的最大斑块数（按面积取最大的）
MAX_EXPORT_FEATURES = 50000


# =====================================================================
#  取数与加载
# =====================================================================

def find_cached(data_source, boundary, year, month_start, month_end, max_cloud_pct):
    """与当前参数匹配的已缓存时相（变化检测 / 时间序列下载过的也算），未命中返回 None。"""
    return find_cached_period(data_source, boundary, int(year), int(month_start), int(month_end),
                              max_cloud_pct, CLASSIFY_BANDS)


def download(data_source, boundary, year, month_start, month_end, max_cloud_pct, data_dir,
             progress_callback=None):
    """取得一期分类影像（已缓存的时相直接复用）。

    Returns:
        缓存条目信息 dict(key, dir, files, size_mb, ...)
    Raises:
        ServiceError: 无影像、下载或写入缓存失败
    """
    info = prepare_period(data_source, boundary, int(year), int(month_start), int(month_end),
                          max_cloud_pct, CLASSIFY_BANDS, data_dir=data_dir,
                          progress_callback=progress_callback)
    if "error" in info:
        raise ServiceError(info["error"])
    return info


def load_bands(data_dir, data_source):
    """加载目录中的分类波段：对齐到同一网格 + 反射率定标 + 云掩膜。

    目录可以是快照缓存条目（snapshot.json）或旧版下载目录（source.json），
    以其中记录的数据源为准。

    Returns:
        dict(bands, meta, valid, prep_info, data_source)；未找到波段时 None。
        valid 为有效像元掩膜（云 / 阴影 / 无数据为 False）。
    """
    import rasterio

    from engine.data_sources import get_data_source
    from engine.grid import GridSpec
    from engine.preprocessing import load_period

    data_path = Path(data_dir)
    for marker in (data_path / "snapshot.json", data_path / "source.json"):
        if marker.exists():
            with open(marker, encoding="utf-8") as f:
                data_source = json.load(f).get("data_source", data_source)
            break

    fmap = get_data_source(data_source).FILE_NAME_MAP
    paths = {}
    for bname in CLASSIFY_BANDS:
        cands = sorted(data_path.glob(f"{fmap.get(bname, bname.upper())}_*.tif"))
        if cands:
            paths[bname] = str(cands[0])
    if not paths:
        return None

    quality = next((str(f) for prefix in ("SCL", "QA_PIXEL")
                    for f in sorted(data_path.glob(f"{prefix}_*.tif"))), None)
    with rasterio.open(next(iter(paths.values()))) as src:
        target_grid = GridSpec.from_src(src)
    bands, valid, info = load_period(paths, quality, data_source, target_grid)
    meta = target_grid.to_meta()
    meta.update(dtype="float32", count=1, compress="lzw")
    return {"bands": bands, "meta": meta, "valid": valid, "prep_info": info,
            "data_source": data_source}


def clip_to_boundary(result, meta, boundary=None, valid=None, rgb=None):
    """把分类结果裁剪到研究区边界内，并剔除云 / 无数据像元。

    边界外与无效像元置为 -1（而非 0），避免与 Cluster 0 或有效类别冲突；
    rgb 中对应像元置为深灰。

    Returns:
        (result, rgb)；rgb 为 None 或尺寸不符时返回的 rgb 为 None。
    """
    if meta is None:
        return result, rgb
    try:
        from engine.utils import boundary_pixel_mask
        h, w = result.shape
        mask = (boundary_pixel_mask(boundary, meta, h, w) if boundary is not None
                else np.ones((h, w), dtype=bool))
        if valid is not None and valid.shape == mask.shape:
            mask &= valid
        clipped = result.copy()
        clipped[~mask] = -1
        clipped_rgb = None
        if rgb is not None and rgb.shape[:2] == result.shape:
            clipped_rgb = rgb.copy()
            clipped_rgb[~mask] = [40, 40, 40]
        return clipped, clipped_rgb
    except Exception as e:
        log(f"  Boundary clip skipped: {e}")
        return result, rgb


# =====================================================================
#  分类
# =====================================================================

def quick(bands, meta, *, boundary=None, valid=None, progress_callback=None):
    """NDVI + MNDWI + NDBI 联合阈值分类。

    Returns:
        dict(result, rgb, stats)
    """
    report(progress_callback, "运行多指数阈值分类...", 30)
    result, rgb = classify_multi_index(bands)
    report(progress_callback, "裁剪到研究区边界...", 50)
    result, rgb = clip_to_boundary(result, meta, boundary, valid, rgb)
    report(progress_callback, "生成统计...", 70)
    stats = classification_statistics(result, meta=meta)
    report(progress_callback, "✅ 快速分类完成！", 100)
    return {"result": result, "rgb": rgb, "stats": stats}


def default_mapping(n_clusters):
    """聚类 → 地物类别的初始映射（轮流分配 1–7 类）。"""
    return {i: (i % 7) + 1 for i in range(n_clusters)}


def apply_mapping(labels, mapping):
    """按 {聚类号: 类别} 把聚类标签转为地物类别；未映射与无数据像元记为 0（未分类）。"""
    result = np.full(labels.shape, -1, dtype=np.int8)
    for cid, cls_val in mapping.items():
        result[labels == cid] = cls_val
    result[result == -1] = 0
    return result


def gmm(bands, meta, *, n_clusters=6, covariance_type="diag", pca_dims=3, sample_ratio=0.05,
        boundary=None, valid=None, progress_callback=None):
    """特征张量（波段 + 指数 + 纹理）→ PCA 降维 → GMM 聚类。

    Returns:
        dict(gmm, labels, mapping, stats)：gmm 为 pca_gmm_classify 的结果
        （其中 labels 已裁剪到边界）；stats 按初始映射统计。
    """
    report(progress_callback, "🔄 构建特征张量 (波段+指数+纹理)...", 20)
    stack = build_feature_stack(bands, add_texture=True)
    report(progress_callback,
           f"🔄 PCA 降维 ({pca_dims} 分量, 采样 {sample_ratio * 100:.0f}% 像元)...", 35)
    res = pca_gmm_classify(stack, n_clusters=n_clusters, sample_ratio=sample_ratio,
                           random_state=42, pca_dims=pca_dims, covariance_type=covariance_type)
    report(progress_callback, "🏷️ 裁剪到研究区边界...", 60)
    labels, _ = clip_to_boundary(res["labels"], meta, boundary, valid)
    res["labels"] = labels
    mapping = default_mapping(n_clusters)
    stats = classification_statistics(apply_mapping(labels, mapping), meta=meta)
    report(progress_callback, f"✅ GMM 聚类完成！{n_clusters} 个光谱簇", 100)
    return {"gmm": res, "labels": labels, "mapping": mapping, "stats": stats}


def label_clusters(labels, mapping, meta):
    """应用聚类 → 类别映射。Returns dict(result, stats)。"""
    result = apply_mapping(labels, mapping)
    return {"result": result, "stats": classification_statistics(result, meta=meta)}


# =====================================================================
#  主动学习
# =====================================================================

def sample_annotation_patches(bands, gmm_result, n_patches=None, patch_size=None):
    """在 GMM 不确定性最高处采样不重叠斑块，并准备空的标注数组。

    Returns:
        dict(patches, uncertainties, labels, mask)：patches 为斑块中心 [(row, col)]；
        labels (int8) / mask (bool) 为与影像同尺寸的累计标注。
    """
    patches, uncertainties = sample_patches(
        bands, gmm_result["pca_3d"], gmm_result["gmm"],
        patch_size=patch_size or config.AL_PATCH_SIZE,
        n_patches=n_patches or config.AL_N_PATCHES)
    shape = next(iter(bands.values())).shape
    return {"patches": patches, "uncertainties": uncertainties,
            "labels": np.zeros(shape, dtype=np.int8), "mask": np.zeros(shape, dtype=bool)}


def annotate_patch(labels, mask, row, col, class_value, patch_size=None):
    """把以 (row, col) 为中心的斑块标为 class_value（原地修改）；0（未分类）不计为标注。"""
    if class_value <= 0:
        return
    half = (patch_size or config.AL_PATCH_SIZE) // 2
    rows = slice(max(row - half, 0), row + half + 1)
    cols = slice(max(col - half, 0), col + half + 1)
    labels[rows, cols] = class_value
    mask[rows, cols] = True


def refine(bands, meta, labels, mask, *, boundary=None, valid=None, progress_callback=None):
    """用已标注像元训练随机森林并预测全图。

    Returns:
        dict(result, confidence, stats)
    Raises:
        ServiceError: 标注像元少于 MIN_ANNOTATED_PX
    """
    n = int(mask.sum())
    if n < MIN_ANNOTATED_PX:
        raise ServiceError(f"标注像元太少 ({n}/{MIN_ANNOTATED_PX})，请标注更多斑块")
    report(progress_callback, f"训练随机森林 ({n} 个标注像元)...", 10)
    stack = build_feature_stack(bands, add_texture=True)
    result, confidence, _ = train_from_annotations(
        stack, labels, mask,
        n_estimators=config.AL_RF_N_ESTIMATORS,
        max_depth=config.AL_RF_MAX_DEPTH)
    result, _ = clip_to_boundary(result, meta, boundary, valid)
    report(progress_callback, "✅ 主动学习完成！", 100)
    return {"result": result, "confidence": confidence,
            "stats": classification_statistics(result, meta=meta)}


# =====================================================================
#  后处理与导出
# =====================================================================

def filter_small_patches(result, meta, min_px):
    """逐类别做连通域过滤，小于 min_px 的斑块置为未分类。Returns dict(result, stats)。"""
    from engine.core import _filter_by_connected_components

    filtered = result.copy()
    for cls_val in CLASS_NAMES:
        if cls_val <= 0:
            continue
        mask = result == cls_val
        if not mask.any():
            continue
        clean = _filter_by_connected_components(mask, int(min_px))
        filtered[mask & ~clean] = 0
    return {"result": filtered, "stats": classification_statistics(filtered, meta=meta)}


def export(result, meta, stats, out_dir, *, region, mode):
    """导出 GeoTIFF（int8）、GeoJSON（WGS84，同类融合）与统计 CSV。

    Returns:
        {"geotiff": path, "geojson": path, "csv": path}（矢量导出失败或无类别时不含 geojson）
    """
    import rasterio

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{region}_{mode}")
    files = {}

    rmeta = meta.copy()
    rmeta.update(dtype="int8", count=1, compress="lzw")
    files["geotiff"] = f"{stem}_分类.tif"
    with rasterio.open(files["geotiff"], "w", **rmeta) as dst:
        dst.write(result.astype(np.int8), 1)

    try:
        fc = _vectorize(result, meta)
        if fc:
            files["geojson"] = f"{stem}_分类.geojson"
            with open(files["geojson"], "w", encoding="utf-8") as f:
                json.dump(fc, f, ensure_ascii=False, indent=2)
    except Exception:
        import traceback
        log(f"  Vector export skipped: {traceback.format_exc()}")

    if stats:
        import pandas as pd
        files["csv"] = f"{stem}_统计.csv"
        pd.DataFrame(stats).to_csv(files["csv"], index=False, encoding="utf-8-sig")
    return files


def _vectorize(result, meta, max_features=MAX_EXPORT_FEATURES):
    """分类栅格 → WGS84 FeatureCollection：同类相邻斑块融合、简化，按面积取最大的若干个。

    Returns:
        FeatureCollection dict；没有已分类像元时 None。
    """
    from pyproj import Transformer
    from rasterio.features import shapes as rshapes
    from shapely.geometry import mapping as shp_mapping
    from shapely.geometry import shape as shp_shape
    from shapely.ops import transform as shp_transform
    from shapely.ops import unary_union

    mask = result > 0
    if not mask.any():
        return None

    raw_shapes = rshapes(result.astype(np.int8), mask=mask,
                         transform=meta["transform"], connectivity=8)
    # GeoJSON 标准要求 EPSG:4326
    to_wgs84 = Transformer.from_crs(meta.get("crs", "EPSG:32650"), "EPSG:4326", always_xy=True)

    by_class = defaultdict(list)
    for g, v in raw_shapes:
        if int(v) <= 0 or g["type"] != "Polygon":
            continue
        poly = shp_shape(g)
        if poly.area < 100:      # 投影面积 < 100 m² 的碎斑忽略
            continue
        poly_wgs = shp_transform(lambda x, y, z=None: to_wgs84.transform(x, y), poly)
        if poly_wgs.is_empty:
            continue
        poly_s = poly_wgs.simplify(0.0001, preserve_topology=True)   # 去锯齿
        parts = poly_s.geoms if poly_s.geom_type == "MultiPolygon" else [poly_s]
        by_class[int(v)].extend(p for p in parts if p.geom_type == "Polygon" and p.area > 0)

    feats = []
    for cls_val, polys in by_class.items():
        merged = unary_union(polys) if len(polys) > 1 else polys[0]
        parts = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]
        feats.extend((part, cls_val) for part in parts if part.area > 0)
    feats.sort(key=lambda x: -x[0].area)

    total = len(feats)
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": shp_mapping(g),
             "properties": {"class": v, "class_name": CLASS_NAMES.get(v, "未知")}}
            for g, v in feats[:max_features]
        ],
    }
    if total > max_features:
        fc["metadata"] = {
            "note": f"原始 {total} 斑块，按面积取最大 {max_features} 个。完整分类结果请下载 GeoTIFF。",
            "total_features": total,
            "exported": max_features,
        }
    return fc
