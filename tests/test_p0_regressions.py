"""
P0 正确性修复的回归测试（全部离线，不访问网络）

  1. 云掩膜：SCL 20 m 与 10 m 波段对齐后生效；缓存命中路径也执行预处理；
     Landsat QA_PIXEL 返回有效路径
  2. Sentinel-2 处理基线 04.00 的 +1000 DN 偏移被归一化
  3. 自适应阈值在纯噪声下误报率 < 1%
  4. 批量运行器不再因 copy.copy(config) 崩溃，任务结束后还原 config
  5. 主动学习斑块落在高不确定性区域

用法:
    python -m pytest tests/test_p0_regressions.py -v
"""

import os, sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import config

CRS = "EPSG:32649"
X0, Y0 = 500000.0, 2800000.0


def _write_tif(path, arr, res, crs=CRS, nodata=0):
    """写单波段 GeoTIFF，左上角固定在 (X0, Y0)。"""
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype, crs=crs, nodata=nodata,
        transform=Affine(res, 0, X0, 0, -res, Y0),
    ) as dst:
        dst.write(arr, 1)
    return str(path)


def _boundary_for_grid(width_m, height_m, pad=50.0):
    """覆盖测试网格的 WGS84 边界多边形。"""
    from pyproj import Transformer
    from shapely.geometry import box
    t = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    x1, y1 = t.transform(X0 - pad, Y0 - height_m - pad)
    x2, y2 = t.transform(X0 + width_m + pad, Y0 + pad)
    return box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


# ============================================================================
#  1. 云掩膜
# ============================================================================

def test_cloud_mask_applied_across_10m_20m_grids(tmp_path):
    """SCL(20 m) 必须在对齐到 10 m 网格后生效；混合分辨率波段不能让预处理崩溃。"""
    from engine.core import _load_aligned_bands

    red = np.full((20, 20), 500, np.uint16)
    nir = np.full((20, 20), 3000, np.uint16)
    swir = np.full((10, 10), 1500, np.uint16)          # 20 m 波段
    scl = np.full((10, 10), 4, np.uint8)               # 4 = 植被
    scl[:5, :5] = 9                                    # 高云 -> 10 m 网格左上 10x10

    paths = {
        "red": _write_tif(tmp_path / "B4.tif", red, 10),
        "nir": _write_tif(tmp_path / "B8.tif", nir, 10),
        "swir1": _write_tif(tmp_path / "B11.tif", swir, 20),
    }
    scl_path = _write_tif(tmp_path / "SCL.tif", scl, 20)

    ay1, ay2, valid, meta, prep = _load_aligned_bands(
        paths, paths, scl_path, scl_path, "sentinel2")

    assert valid.shape == (20, 20)
    assert not valid[:10, :10].any(), "云区应被掩膜"
    assert valid[10:, 10:].all()
    assert prep["y1"]["cloud_mask"] and prep["y2"]["cloud_mask"]
    assert prep["y1"]["cloud_pct"] == pytest.approx(25.0)
    # 已定标为反射率，且 20 m 波段已对齐到 10 m
    assert ay1["red"][15, 15] == pytest.approx(0.05)
    assert ay1["swir1"].shape == (20, 20)
    assert (ay1["red"][:10, :10] == 0).all()


@pytest.mark.parametrize("with_scl", [True, False])
def test_pipeline_cache_hit_still_masks_clouds(tmp_path, monkeypatch, with_scl):
    """缓存命中时也必须做云掩膜（原先 UI 主路径会跳过整个预处理）。

    后时相左上 20x20 px 被云覆盖：有 SCL 时不应检出变化；
    无 SCL 时（对照组）云区会被误判为严重退化，并在结果中标明未做云掩膜。
    """
    import engine.core as core
    import engine.data_sources as ds
    from engine.cache import SnapshotCache, compute_aoi_hash, snapshot_key
    from engine.preprocessing import PREPROCESSING_VERSION

    y1, y2 = 2021, 2023
    for k, v in {"YEAR1": y1, "YEAR2": y2, "MONTH_START": 6, "MONTH_END": 10,
                 "MAX_CLOUD_PCT": 50, "REGION_NAME": "测试区",
                 "DETECTION_MODE": "multi", "GENERATE_REPORT": False}.items():
        monkeypatch.setattr(config, k, v)

    n = 40
    boundary = _boundary_for_grid(n * 10, n * 10)
    aoi_hash = compute_aoi_hash(boundary)
    bands = ["red", "nir"]

    cache_root = tmp_path / "cache"
    sc = SnapshotCache(cache_root=str(cache_root))
    key = snapshot_key(aoi_hash, "sentinel2", "NDVI", y1, y2, 6, 10, 50,
                       bands, PREPROCESSING_VERSION)
    sc.put_snapshot(key, {
        "aoi_hash": aoi_hash, "data_source": "sentinel2", "index_type": "NDVI",
        "year1": y1, "year2": y2, "month_start": 6, "month_end": 10,
        "max_cloud_pct": 50, "bands": bands,
        "preprocessing_version": PREPROCESSING_VERSION,
    })
    d = sc.get_band_dir(key)

    red = np.full((n, n), 500, np.uint16)
    nir = np.full((n, n), 3000, np.uint16)
    red2, nir2 = red.copy(), nir.copy()
    red2[:20, :20], nir2[:20, :20] = 3000, 3200        # 亮云
    _write_tif(os.path.join(d, f"B4_{y1}.tif"), red, 10)
    _write_tif(os.path.join(d, f"B8_{y1}.tif"), nir, 10)
    _write_tif(os.path.join(d, f"B4_{y2}.tif"), red2, 10)
    _write_tif(os.path.join(d, f"B8_{y2}.tif"), nir2, 10)
    if with_scl:
        scl1 = np.full((n // 2, n // 2), 4, np.uint8)
        scl2 = scl1.copy()
        scl2[:10, :10] = 9
        _write_tif(os.path.join(d, f"SCL_{y1}.tif"), scl1, 20)
        _write_tif(os.path.join(d, f"SCL_{y2}.tif"), scl2, 20)
    sc.mark_complete(key)

    class OfflineSource:
        name = "sentinel2"
        FILE_NAME_MAP = {"red": "B4", "nir": "B8"}

        def search_scenes(self, *a, **k):
            raise AssertionError("缓存命中时不应访问网络")

        download_bands = download_quality_bands = search_scenes

    monkeypatch.setattr(ds, "get_data_source", lambda name: OfflineSource())
    monkeypatch.setattr(core, "_cache_root", lambda: str(cache_root))

    res = core.run_pipeline_v3(
        data_source_name="sentinel2", data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"), boundary=boundary,
        index_type="NDVI", detection_mode="multi")

    assert "error" not in res, res.get("error")
    changed = res["stats"]["changed_total"]["pixels"]
    if with_scl:
        assert res["preprocessing"]["y2"]["cloud_mask"] is True
        assert changed == 0, f"云区被误判为变化: {changed} px"
    else:
        assert res["preprocessing"]["y2"]["cloud_mask"] is False
        assert changed == 400


def test_old_cache_versions_are_not_reused(tmp_path):
    """旧版缓存（v3.1 存的是写回的浮点反射率）不能被新预处理复用。"""
    from engine.cache import SnapshotCache, snapshot_key
    from engine.preprocessing import PREPROCESSING_VERSION

    sc = SnapshotCache(cache_root=str(tmp_path))
    args = ("aoi", "sentinel2", "NDVI", 2024, 2025, 6, 10, 50, ["red", "nir"])
    sc.put_snapshot("old", dict(zip(
        ["aoi_hash", "data_source", "index_type", "year1", "year2",
         "month_start", "month_end", "max_cloud_pct", "bands"], args),
        preprocessing_version="v3.1"))
    sc.mark_complete("old")

    assert sc.find(*args, preprocessing_version=PREPROCESSING_VERSION) is None
    # 参数不同 -> 目录名不同，不再互相覆盖
    assert snapshot_key(*args, PREPROCESSING_VERSION) != \
        snapshot_key(*args[:5], 7, *args[6:], PREPROCESSING_VERSION)
    assert snapshot_key(*args, "v3.1") != snapshot_key(*args, PREPROCESSING_VERSION)


@pytest.mark.parametrize("n_tiles", [1, 2])
def test_landsat_qa_pixel_path_exists(tmp_path, n_tiles):
    """QA_PIXEL 下载后返回的路径必须存在（原先返回已被移走/删除的临时文件）。"""
    from engine.data_sources import _download_qa_pixel_generic

    def fake_download(item, band_name, out_path):
        arr = np.full((10, 10), 21824, np.uint16)       # Landsat 晴空值
        with rasterio.open(
            out_path, "w", driver="GTiff", height=10, width=10, count=1,
            dtype="uint16", crs=CRS, nodata=1,
            transform=Affine(30, 0, X0 + item.offset, 0, -30, Y0),
        ) as dst:
            dst.write(arr, 1)

    scenes = [{"tile": f"t{i}",
               "item": SimpleNamespace(datetime=datetime(2023, 7, 1), offset=i * 300)}
              for i in range(n_tiles)]
    out = _download_qa_pixel_generic(scenes, str(tmp_path), fake_download)

    assert out["qa_pixel"] == os.path.join(str(tmp_path), "QA_PIXEL_2023.tif")
    assert os.path.exists(out["qa_pixel"])


# ============================================================================
#  2. Sentinel-2 基线 04.00 偏移
# ============================================================================

def _s2_item(dt, baseline=None, href=None):
    props = {"s2:processing_baseline": baseline} if baseline else {}
    assets = {"B04": SimpleNamespace(href=href), "SCL": SimpleNamespace(href=href)}
    return SimpleNamespace(id="S2_TEST", datetime=dt, properties=props, assets=assets)


def test_s2_boa_offset_detection():
    from engine.preprocessing import s2_boa_offset
    assert s2_boa_offset(_s2_item(datetime(2023, 7, 1), "04.00")) == 1000
    assert s2_boa_offset(_s2_item(datetime(2023, 7, 1), "02.14")) == 0
    # 重处理的历史数据（基线 05.xx）同样带偏移
    assert s2_boa_offset(_s2_item(datetime(2019, 7, 1), "05.10")) == 1000
    # 缺少基线属性时按日期判断
    assert s2_boa_offset(_s2_item(datetime(2022, 1, 24))) == 0
    assert s2_boa_offset(_s2_item(datetime(2022, 1, 25))) == 1000


def test_harmonized_ndvi_matches_across_baselines(tmp_path):
    """同一地表在新旧基线下下载后 NDVI 必须一致（原先 0.71 vs 0.45）。"""
    from engine.data_sources import Sentinel2DataSource
    from engine.indices import ndvi
    from engine.preprocessing import harmonize_s2_dn

    # 反射率 red=0.05 nir=0.30；旧基线 DN=refl*1e4，新基线再 +1000
    old = {"red": np.array([[500, 0]], np.uint16), "nir": np.array([[3000, 0]], np.uint16)}
    new = {k: np.where(v > 0, v + 1000, 0).astype(np.uint16) for k, v in old.items()}

    harmonized = {k: harmonize_s2_dn(v, 1000) for k, v in new.items()}
    assert harmonized["red"].dtype == np.uint16
    np.testing.assert_array_equal(harmonized["red"], old["red"])   # 0 仍为无数据
    assert ndvi(harmonized["nir"], harmonized["red"])[0, 0] == \
        pytest.approx(ndvi(old["nir"], old["red"])[0, 0])

    # 暗像元（DN<=1000）钳制为 1，不被当成无数据
    np.testing.assert_array_equal(
        harmonize_s2_dn(np.array([0, 900, 1000, 1001], np.uint16), 1000), [0, 1, 1, 1])

    # 下载函数逐景归一化；SCL 分类层不受影响
    src = Sentinel2DataSource()
    href = _write_tif(tmp_path / "remote.tif", new["red"], 10)
    item = _s2_item(datetime(2023, 7, 1), "04.00", href=href)
    out = src._download_single_band(item, "red", str(tmp_path / "red.tif"))
    with rasterio.open(out) as f:
        np.testing.assert_array_equal(f.read(1), old["red"])
    out = src._download_single_band(item, "scl", str(tmp_path / "scl.tif"))
    with rasterio.open(out) as f:
        np.testing.assert_array_equal(f.read(1), new["red"])


# ============================================================================
#  3. 自适应阈值
# ============================================================================

def test_adaptive_threshold_false_positive_rate():
    """纯噪声下误报率应 < 1%；10% 像元真实变化时阈值仍稳定在 3σ 附近。"""
    from engine.core import estimate_adaptive_threshold
    rng = np.random.default_rng(0)
    sigma = 0.05
    diff = rng.normal(0, sigma, (400, 500)).astype(np.float32)
    valid = np.ones(diff.shape, bool)

    T, S = estimate_adaptive_threshold(diff, valid)
    assert T == pytest.approx(3 * sigma, rel=0.05)
    assert S > T
    assert (np.abs(diff) > T).mean() < 0.01        # 原实现 ~23%

    diff_chg = diff.copy()
    diff_chg[:40] -= 0.4                            # 10% 像元退化
    T2, _ = estimate_adaptive_threshold(diff_chg, valid)
    assert T2 == pytest.approx(3 * sigma, rel=0.15)
    assert (diff_chg[:40] < -T2).mean() > 0.95


# ============================================================================
#  4. 批量运行器
# ============================================================================

def _make_runner(tmp_path, task):
    import json
    from batch.batch_runner import BatchRunner
    cfg = tmp_path / "batch.json"
    cfg.write_text(json.dumps({"tasks": [task], "output_base": str(tmp_path / "out")}),
                   encoding="utf-8")
    return BatchRunner(str(cfg))


def test_batch_runner_task_runs_and_restores_config(tmp_path, monkeypatch):
    """原先 copy.copy(config) 抛 TypeError，所有任务失败；任务参数也要真正生效。"""
    import engine.core as core
    seen = {}

    def fake_pipeline(**kwargs):
        seen.update(kwargs, region=config.REGION_NAME, years=(config.YEAR1, config.YEAR2),
                    n_points=config.NUM_SAMPLE_POINTS)
        return {"stats": {}, "files": {}}

    monkeypatch.setattr(core, "run_pipeline_v3", fake_pipeline)
    before = {k: getattr(config, k) for k in ("REGION_NAME", "YEAR1", "YEAR2",
                                              "NUM_SAMPLE_POINTS", "BOUNDARY_FILE")}
    task = {"id": "t1", "region": "测试区", "boundary": "aoi.geojson",
            "year1": 2019, "year2": 2024, "threshold": 0.2, "sample_points": 500,
            "retry_on_fail": False}

    r = _make_runner(tmp_path, task)._run_single_task(task, 1, 1)

    assert r["status"] == "success", r.get("error")
    assert seen["region"] == "测试区" and seen["years"] == (2019, 2024)
    assert seen["threshold"] == 0.2 and seen["n_points"] == 500
    assert {k: getattr(config, k) for k in before} == before, "任务结束后应还原 config"


def test_batch_runner_restores_config_on_failure(tmp_path, monkeypatch):
    import engine.core as core

    def boom(**kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(core, "run_pipeline_v3", boom)
    region_before = config.REGION_NAME
    task = {"id": "t2", "region": "失败区", "boundary": "aoi.geojson",
            "retry_on_fail": False}

    r = _make_runner(tmp_path, task)._run_single_task(task, 1, 1)

    assert r["status"] == "failed" and "download failed" in r["error"]
    assert config.REGION_NAME == region_before


# ============================================================================
#  5. 主动学习斑块采样
# ============================================================================

@pytest.mark.parametrize("pca_dims", [3, 2])
def test_sample_patches_target_uncertain_region(pca_dims):
    """斑块中心应落在不确定区域（原先固定在左上角，不确定度为 0）。"""
    from sklearn.mixture import GaussianMixture
    from engine.classifier import sample_patches

    rng = np.random.default_rng(0)
    h = w = 200
    gmm = GaussianMixture(2, random_state=0).fit(
        np.r_[rng.normal(-3, 1, (500, pca_dims)), rng.normal(3, 1, (500, pca_dims))])
    pca = np.full((h, w, pca_dims), -3.0, np.float32)    # 确定属于簇 A
    pca[140:170, 140:170] = 0.0                          # 两簇中间 -> 高不确定

    patches, unc = sample_patches({"nir": np.zeros((h, w))}, pca, gmm,
                                  patch_size=8, n_patches=5)

    assert len(patches) == 5
    for r, c in patches:
        assert 140 <= r < 170 and 140 <= c < 170, f"斑块 {(r, c)} 不在不确定区域"
    assert min(unc) > 0.9
