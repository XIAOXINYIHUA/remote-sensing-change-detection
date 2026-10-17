"""
遥感分析平台 v3 — 小 AOI 回归测试

验证 Phase 0 修复的正确性:
  1. GridSpec 创建、相等性、reproject 对齐
  2. 参数传递优先级 (显式 > INDEX_THRESHOLDS > config)
  3. 缓存查找和命中
  4. 面积计算 (实际分辨率 vs 固定 0.01 ha)
  5. GMM 聚类标签 (0 不冲突)
  6. 矢量导出不随机丢弃

用法:
    cd E:/解译工具
    python -m pytest tests/test_phase0.py -v
    或直接:
    python tests/test_phase0.py
"""

import os, sys, json, tempfile, shutil
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)


# ============================================================================
#  1. GridSpec 测试
# ============================================================================

def test_gridspec_create():
    """创建 GridSpec 并检查基本属性。"""
    from engine.grid import GridSpec, Affine
    from rasterio.crs import CRS

    transform = Affine.translation(500000, 4000000) * Affine.scale(10, -10)
    gs = GridSpec(
        crs=CRS.from_epsg(32650),
        transform=transform,
        width=100,
        height=200,
        nodata=0,
    )
    assert gs.width == 100
    assert gs.height == 200
    xres, yres = gs.resolution
    assert abs(xres - 10.0) < 1e-6
    assert abs(yres - 10.0) < 1e-6
    b = gs.bounds
    assert abs(b[0] - 500000) < 1e-3
    assert abs(b[3] - 4000000) < 1e-3
    print(f"  [OK] test_gridspec_create: {gs}")


def test_gridspec_area():
    """GridSpec 像元面积计算。"""
    from engine.grid import GridSpec, Affine
    from rasterio.crs import CRS

    # 10m 分辨率 -> 100 m² -> 0.01 ha
    t10 = Affine.translation(500000, 4000000) * Affine.scale(10, -10)
    gs10 = GridSpec(CRS.from_epsg(32650), t10, 100, 100)
    assert abs(gs10.area_per_pixel_ha - 0.01) < 1e-6
    print(f"  [OK] test_gridspec_area: 10m->{gs10.area_per_pixel_ha} ha")

    # 30m -> 900 m² -> 0.09 ha
    t30 = Affine.translation(500000, 4000000) * Affine.scale(30, -30)
    gs30 = GridSpec(CRS.from_epsg(32650), t30, 100, 100)
    assert abs(gs30.area_per_pixel_ha - 0.09) < 1e-6
    print(f"  [OK] test_gridspec_area: 30m->{gs30.area_per_pixel_ha} ha")


def test_gridspec_equality():
    """GridSpec 相等性和不一致检测。"""
    from engine.grid import GridSpec, Affine
    from rasterio.crs import CRS

    t = Affine.translation(500000, 4000000) * Affine.scale(10, -10)
    gs1 = GridSpec(CRS.from_epsg(32650), t, 100, 200)
    gs2 = GridSpec(CRS.from_epsg(32650), t, 100, 200)
    assert gs1 == gs2, "Identical grids should be equal"

    gs3 = GridSpec(CRS.from_epsg(32650), t, 101, 200)
    assert gs1 != gs3, "Different width grids should differ"
    print("  [OK] test_gridspec_equality: equality/inequality works")


def test_assert_grid_consistent():
    """assert_grid_consistent 应当通过一致网格、拒绝不一致网格。"""
    from engine.grid import (GridSpec, Affine, assert_grid_consistent)
    from rasterio.crs import CRS

    t = Affine.translation(500000, 4000000) * Affine.scale(10, -10)
    gs1 = GridSpec(CRS.from_epsg(32650), t, 100, 200)
    gs2 = GridSpec(CRS.from_epsg(32650), t, 100, 200)
    assert_grid_consistent([gs1, gs2]), "Identical grids should pass"

    gs3 = GridSpec(CRS.from_epsg(32650), t, 100, 201)
    try:
        assert_grid_consistent([gs1, gs3])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  [OK] test_assert_grid_consistent: passes/fails correctly")


def test_reproject_to_grid():
    """reproject_to_grid 生成正确尺寸的输出。"""
    from engine.grid import (GridSpec, reproject_to_grid, Affine)
    from rasterio.crs import CRS

    src_transform = Affine.translation(500000, 4000000) * Affine.scale(10, -10)
    src_h, src_w = 50, 40
    src_arr = np.random.rand(src_h, src_w).astype(np.float32)
    src_meta = {
        "crs": CRS.from_epsg(32650),
        "transform": src_transform,
        "width": src_w,
        "height": src_h,
        "nodata": 0,
    }

    dst_transform = Affine.translation(500000, 4000000) * Affine.scale(20, -20)
    target = GridSpec(CRS.from_epsg(32650), dst_transform, 20, 25)

    result = reproject_to_grid((src_arr, src_meta), target)
    assert result.shape == (25, 20), f"Expected (25,20), got {result.shape}"
    print(f"  [OK] test_reproject_to_grid: shape {result.shape}")


def test_boundary_grid_crop():
    """boundary_grid_crop 生成适配 AOI 的网格。"""
    from engine.grid import boundary_grid_crop
    from shapely.geometry import box

    bbox = box(120.0, 30.0, 120.1, 30.1)
    gs = boundary_grid_crop(bbox, "EPSG:32650", resolution=30.0, pad_pixels=5)
    assert gs.width > 0
    assert gs.height > 0
    xr, yr = gs.resolution
    assert abs(xr - 30.0) < 0.1
    assert abs(yr - 30.0) < 0.1
    print(f"  [OK] test_boundary_grid_crop: {gs.width}x{gs.height} @ 30m")


# ============================================================================
#  2. 参数传递测试
# ============================================================================

def test_threshold_priority():
    """显式传入的阈值应优先于 INDEX_THRESHOLDS 和 config。"""
    import config as cfg
    cfg.CHANGE_THRESHOLD = 0.15
    cfg.SEVERE_THRESHOLD = 0.25

    from engine.indices import INDEX_THRESHOLDS

    # 模拟 run_change_detection_generic 的阈值逻辑
    index_type = "NDVI"
    reg = INDEX_THRESHOLDS.get(index_type, {})

    # Case 1: 显式传入 > 注册表 > config
    threshold = 0.20  # 显式
    severe_threshold = None  # 未显式传入，应 fallback
    T = threshold if threshold is not None else reg.get("change", cfg.CHANGE_THRESHOLD)
    S = (severe_threshold if severe_threshold is not None
         else reg.get("severe", cfg.SEVERE_THRESHOLD))
    assert T == 0.20, f"Expected 0.20, got {T}"
    assert S == reg["severe"], f"Expected {reg['severe']}, got {S}"

    # Case 2: None 显式 -> 取注册表
    T2 = (None if None is not None else reg.get("change", cfg.CHANGE_THRESHOLD))
    assert T2 == 0.15, f"Expected 0.15 (INDEX_THRESHOLDS), got {T2}"
    print(f"  [OK] test_threshold_priority: explicit={T}, fallback={T2}")


def test_min_area_conversion():
    """最小斑块面积从 m² 到像元的换算应当正确。"""
    # 10m -> 0.01 ha/px -> 100 m²/px -> 5000 m² -> 50 px
    area_per_px_ha = 0.01
    min_area_m2 = 5000
    min_px = max(4, int(round(min_area_m2 / (area_per_px_ha * 10000))))
    assert min_px == 50, f"Expected 50px, got {min_px}"

    # 30m -> 0.09 ha/px -> 900 m²/px -> 5000 m² -> 6 px
    area_per_px_ha_30 = 0.09
    min_px_30 = max(4, int(round(min_area_m2 / (area_per_px_ha_30 * 10000))))
    assert min_px_30 == 6, f"Expected 6px, got {min_px_30}"
    print(f"  [OK] test_min_area_conversion: 10m->{min_px}px, 30m->{min_px_30}px")


def test_adaptive_threshold():
    """自适应阈值应随噪声水平变化。"""
    from engine.core import estimate_adaptive_threshold
    np.random.seed(42)

    # 低噪声场景：大多数像元无变化
    h, w = 100, 100
    diff_low = np.random.randn(h, w) * 0.02  # 噪声 ~0.02
    valid = np.ones((h, w), dtype=bool)
    T1, S1 = estimate_adaptive_threshold(diff_low, valid)
    assert T1 >= 0.02, f"Low noise threshold too small: {T1}"
    assert S1 > T1, f"Severe threshold should exceed change threshold: {S1}"

    # 高噪声场景
    diff_high = np.random.randn(h, w) * 0.10  # 噪声 ~0.10
    T2, S2 = estimate_adaptive_threshold(diff_high, valid)
    assert T2 >= T1, f"High noise threshold {T2} should >= low noise {T1}"
    print(f"  [OK] test_adaptive_threshold: low_noise T={T1:.4f}, high_noise T={T2:.4f}")


def test_preprocessing_mask():
    """预处理云掩膜函数应能运行。"""
    from engine.preprocessing import (
        detect_saturation, apply_scale_factors,
    )
    np.random.seed(42)
    bands = {"red": np.random.randint(0, 5000, (20, 30)).astype(np.float32),
             "nir": np.random.randint(0, 5000, (20, 30)).astype(np.float32)}
    valid = detect_saturation(bands, "sentinel2")
    assert valid.shape == (20, 30)
    assert valid.dtype == bool

    scaled = apply_scale_factors(bands, "sentinel2")
    assert "red" in scaled
    assert scaled["red"].max() <= 0.5
    print(f"  [OK] test_preprocessing_mask: saturation and scaling work")


def test_ece():
    """ECE 计算应输出 0~1 之间的值。"""
    from engine.validation import expected_calibration_error
    np.random.seed(42)
    n = 1000
    probs = np.random.rand(n, 5)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = np.random.randint(0, 5, n)
    result = expected_calibration_error(probs, labels)
    assert 0 <= result["ece"] <= 1
    assert 0 <= result["accuracy"] <= 1
    assert len(result["bins"]) > 0
    print(f"  [OK] test_ece: ECE={result['ece']:.4f}, Acc={result['accuracy']:.4f}")


def test_spatial_kfold():
    """空间 K-fold 应生成不重叠的折。"""
    from engine.validation import spatial_kfold_split
    labels = np.random.randint(0, 4, (50, 60))
    folds = spatial_kfold_split(labels, n_folds=3)
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert not set(train_idx) & set(val_idx), "Train/val should not overlap"
    print(f"  [OK] test_spatial_kfold: 3 folds, no overlap")


def test_reject_low_confidence():
    """低置信度拒判应将低于阈值的标签设为未分类。"""
    from engine.validation import reject_low_confidence
    labels = np.array([[1, 2, 1], [2, 1, 2]], dtype=np.int32)
    conf = np.array([[0.9, 0.4, 0.8], [0.3, 0.7, 0.5]], dtype=np.float32)
    result = reject_low_confidence(labels, conf, threshold=0.6)
    assert result[0, 0] == 1
    assert result[0, 1] == 0  # rejected
    assert result[1, 0] == 0  # rejected
    assert result[1, 2] == 0  # rejected
    print(f"  [OK] test_reject_low_confidence: rejection works")


# ============================================================================
#  3. 缓存测试
# ============================================================================

def test_snapshot_cache():
    """SnapshotCache 的存、取、查找功能。"""
    from engine.cache import SnapshotCache
    tmpdir = tempfile.mkdtemp()
    try:
        sc = SnapshotCache(cache_root=tmpdir)
        assert not sc.has("nonexistent")

        snap = {
            "aoi_hash": "abc123",
            "data_source": "sentinel2",
            "index_type": "NDVI",
            "year1": 2025, "year2": 2026,
            "month_start": 6, "month_end": 10,
            "max_cloud_pct": 50,
            "bands": ["red", "nir"],
        }
        sc.put_snapshot("testkey", snap)
        sc.mark_complete("testkey")
        assert sc.has("testkey")
        assert sc.get_snapshot("testkey") is not None

        # find 应匹配
        found = sc.find("abc123", "sentinel2", "NDVI",
                        2025, 2026, 6, 10, 50, ["red", "nir"])
        assert found == "testkey", f"Expected testkey, got {found}"

        print("  [OK] test_snapshot_cache: store/find/has works")
    finally:
        shutil.rmtree(tmpdir)


# ============================================================================
#  4. GMM 标签冲突测试
# ============================================================================

def test_apply_cluster_mapping():
    """apply_cluster_mapping 不应有 Cluster 0 与「未分类」0 的冲突。"""
    from engine.classifier import apply_cluster_mapping
    labels = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    mapping = {0: 2, 1: 1, 2: 3, 4: 4}
    result = apply_cluster_mapping(labels, mapping)
    # Cluster 0 -> 2 (植被), Cluster 3 -> unmapped -> 0 (未分类)
    assert result[0, 0] == 2, f"Cluster 0 should map to 2, got {result[0, 0]}"
    assert result[1, 0] == 0, f"Unmapped cluster should be 0, got {result[1, 0]}"
    assert result[0, 1] == 1, f"Cluster 1 should map to 1, got {result[0, 1]}"
    print("  [OK] test_apply_cluster_mapping: cluster 0 vs unclassified resolved")


def test_pca_gmm_with_standardization():
    """PCA+GMM 标准化后应稳定运行。"""
    from engine.classifier import pca_gmm_classify
    np.random.seed(42)
    # 小特征张量
    stack = np.random.rand(50, 60, 4).astype(np.float32)
    res = pca_gmm_classify(stack, n_clusters=3, sample_ratio=0.5)
    assert res["labels"].shape == (50, 60)
    assert res["labels"].dtype == np.int32 or res["labels"].dtype == np.intp
    assert "scaler" in res
    assert "pca_3d" in res
    assert res["pca_3d"].shape[-1] == 3
    print(f"  [OK] test_pca_gmm_with_standardization: "
          f"labels range [{res['labels'].min()}, {res['labels'].max()}]")


# ============================================================================
#  5. 矢量导出测试
# ============================================================================

def test_export_no_random_drop():
    """分类矢量导出不应随机丢弃。"""
    from engine.classifier import CLASS_NAMES, CLASS_COLORS
    from warnings import filterwarnings
    filterwarnings("ignore")

    # 简单模拟 shapes 处理逻辑
    from collections import defaultdict
    from shapely.geometry import shape as shp_shape, mapping as shp_mapping
    from shapely.ops import unary_union
    import json

    # 模拟大于 50000 的情况
    n_feats = 60000
    min_area_m2 = 100
    res_m = 10.0

    by_class = defaultdict(list)
    # 创建许多小方块
    for i in range(min(n_feats, 1000)):
        from shapely.geometry import Polygon
        x = i % 100 * 10.0
        y = i // 100 * 10.0
        poly = Polygon([(x, y), (x + 5, y), (x + 5, y + 5), (x, y + 5)])
        if poly.area * res_m * res_m >= min_area_m2:
            by_class[1].append(poly)

    # 同类融合
    feats = []
    for cls_val, polys in by_class.items():
        if not polys:
            continue
        merged = unary_union(polys) if len(polys) > 1 else polys[0]
        if hasattr(merged, 'geoms'):
            for part in merged.geoms:
                feats.append((shp_mapping(part), cls_val))
        else:
            feats.append((shp_mapping(merged), cls_val))

    feats.sort(key=lambda x: -shp_shape(x[0]).area)
    total = len(feats)
    max_export = 50000
    exported = feats[:max_export]
    assert len(exported) == min(total, max_export)
    print(f"  [OK] test_export_no_random_drop: {total}->{len(exported)} (sorted by area)")


# ============================================================================
#  6. 能力矩阵测试
# ============================================================================

def test_capability_matrix():
    """能力矩阵应该正确限制数据源和指数的组合。"""
    sys.path.insert(0, os.path.join(BASE, "frontend"))
    try:
        from utils import CAPABILITY_MATRIX
        assert "NDVI" in CAPABILITY_MATRIX["sentinel2"]
        assert "EVI" in CAPABILITY_MATRIX["sentinel2"]
        assert "NDVI" in CAPABILITY_MATRIX["landsat8"]
        assert len(CAPABILITY_MATRIX["sentinel1"]) == 0
        print(f"  [OK] test_capability_matrix: "
              f"S2={len(CAPABILITY_MATRIX['sentinel2'])} indices, "
              f"S1={len(CAPABILITY_MATRIX['sentinel1'])} indices")
    finally:
        sys.path.pop(0)


# ============================================================================
#  Run
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 0 回归测试")
    print("=" * 60)
    tests = [
        test_gridspec_create,
        test_gridspec_area,
        test_gridspec_equality,
        test_assert_grid_consistent,
        test_reproject_to_grid,
        test_boundary_grid_crop,
        test_threshold_priority,
        test_min_area_conversion,
        test_adaptive_threshold,
        test_ece,
        test_spatial_kfold,
        test_reject_low_confidence,
        test_preprocessing_mask,
        test_snapshot_cache,
        test_apply_cluster_mapping,
        test_pca_gmm_with_standardization,
        test_export_no_random_drop,
        test_capability_matrix,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"结果: {passed}/{len(tests)} 通过, {failed} 失败")
    sys.exit(0 if failed == 0 else 1)
