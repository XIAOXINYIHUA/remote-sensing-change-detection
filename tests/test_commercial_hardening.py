"""
v3.2 商业化加固的回归测试（全部离线：「远程」影像用本地 GeoTIFF 充当 asset href）

  1. 分析网格：确定性、对齐传感器像元格网、UTM 带选择
  2. 场景选择按真实足迹计算覆盖率
  3. 窗口读取：跨 UTM 带镶嵌（先到先得）+ S2 基线归一化
  4. Mann-Kendall / Sen 斜率与 scipy 一致；时间序列云掩膜 + 缺测
  5. 采样点：内部代表点、类别不合并、按米稀疏、真实像元面积、Shapefile 坐标系
  6. 报告随指数变化
  7. RunParams 校验；流水线不修改 config、可并发；资料收集与分析共享缓存

用法:
    python -m pytest tests/test_commercial_hardening.py -v
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from pyproj import Transformer
from shapely.geometry import box, mapping

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import config
from engine.grid import GridSpec


def _write(path, arr, transform, crs, nodata=0):
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype, crs=crs, nodata=nodata, transform=transform,
    ) as dst:
        dst.write(arr, 1)
    return str(path)


def _wgs84_box_of_grid(grid, shrink_m=0.0):
    """网格范围（可内缩）对应的 WGS84 外包矩形。"""
    left, bottom, right, top = grid.bounds
    t = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    xs, ys = t.transform([left + shrink_m, right - shrink_m, left + shrink_m, right - shrink_m],
                         [bottom + shrink_m, bottom + shrink_m, top - shrink_m, top - shrink_m])
    return box(min(xs), min(ys), max(xs), max(ys))


def _config_snapshot():
    return {k: getattr(config, k) for k in dir(config) if k.isupper()}


def _s2_item(item_id, dt, assets, baseline=None, geometry=None):
    props = {"s2:processing_baseline": baseline} if baseline else {}
    return SimpleNamespace(
        id=item_id, datetime=dt, properties=props, geometry=geometry,
        assets={k: SimpleNamespace(href=v) for k, v in assets.items()})


# ============================================================================
#  1. 分析网格
# ============================================================================

def test_aoi_grid_is_deterministic_and_snapped():
    from engine.grid import aoi_grid, utm_crs_for

    aoi = box(110.30, 25.40, 110.35, 25.445)
    g1, g2 = aoi_grid(aoi, 10), aoi_grid(aoi, 10)
    assert g1 == g2
    assert g1.crs.to_epsg() == 32649
    assert g1.transform.c % 10 == 0 and g1.transform.f % 10 == 0

    gl = aoi_grid(aoi, 30, origin_offset=15)
    assert gl.transform.c % 30 == 15 and gl.transform.f % 30 == 15
    assert gl.resolution == (30.0, 30.0)

    # 网格完整覆盖 AOI（含外扩）
    t = Transformer.from_crs("EPSG:4326", g1.crs, always_xy=True)
    x0, y0 = t.transform(110.30, 25.40)
    x1, y1 = t.transform(110.35, 25.445)
    left, bottom, right, top = g1.bounds
    assert left < min(x0, x1) and right > max(x0, x1)
    assert bottom < min(y0, y1) and top > max(y0, y1)

    assert utm_crs_for(box(-47.1, -15.9, -47.0, -15.8)).to_epsg() == 32723   # 南半球
    assert utm_crs_for(box(113.98, 25.0, 114.02, 25.03)).to_epsg() == 32650  # 质心在带边界


# ============================================================================
#  2. 场景选择
# ============================================================================

def test_select_scenes_uses_footprint_not_bbox():
    """同一瓦片中 bbox 覆盖 AOI 但真实足迹不覆盖的景，不能因 bbox 被选中。"""
    from engine.data_sources import Sentinel2DataSource

    aoi = box(110.30, 25.40, 110.35, 25.45)
    full = mapping(box(110.0, 25.0, 111.0, 26.0))
    edge = mapping(box(110.0, 25.0, 110.31, 26.0))          # 轨道边缘：只覆盖 AOI 西侧 20%
    items = [
        SimpleNamespace(id="A_edge_clear", datetime=datetime(2023, 7, 1), geometry=edge,
                        bbox=(110.0, 25.0, 111.0, 26.0),
                        properties={"s2:mgrs_tile": "49RDJ", "eo:cloud_cover": 0}),
        SimpleNamespace(id="A_full_cloudy", datetime=datetime(2023, 7, 6), geometry=full,
                        bbox=(110.0, 25.0, 111.0, 26.0),
                        properties={"s2:mgrs_tile": "49RDJ", "eo:cloud_cover": 30}),
        SimpleNamespace(id="B_far", datetime=datetime(2023, 7, 1),
                        geometry=mapping(box(112.0, 25.0, 113.0, 26.0)), bbox=None,
                        properties={"s2:mgrs_tile": "49REJ", "eo:cloud_cover": 0}),
    ]
    scenes = Sentinel2DataSource().select_scenes(aoi, items)

    assert [s["item"].id for s in scenes] == ["A_full_cloudy"]   # 不覆盖 AOI 的瓦片被丢弃
    assert scenes[0]["coverage"] == pytest.approx(1.0)
    assert scenes[0]["aoi"] is aoi


# ============================================================================
#  3. 窗口读取 / 跨带镶嵌 / 基线归一化
# ============================================================================

def test_download_bands_mosaics_across_utm_zones(tmp_path):
    """AOI 横跨 49/50 带：两景各在本带投影，读取后在同一网格上无缝镶嵌；
    新基线景 (+1000 DN) 被归一化；重叠区按场景顺序先到先得。"""
    from engine.data_sources import Sentinel2DataSource

    aoi = box(113.98, 25.00, 114.02, 25.03)

    def tile(name, crs, lon0, lon1, dn):
        t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = t.transform([lon0, lon1, lon0, lon1], [24.99, 24.99, 25.04, 25.04])
        x0, y1 = np.floor(min(xs) / 10) * 10, np.ceil(max(ys) / 10) * 10
        w = int(np.ceil((max(xs) - x0) / 10))
        h = int(np.ceil((y1 - min(ys)) / 10))
        return _write(tmp_path / f"{name}.tif", np.full((h, w), dn, np.uint16),
                      Affine(10, 0, x0, 0, -10, y1), crs)

    west = tile("west", "EPSG:32649", 113.97, 114.005, 1500)    # 新基线: 1500-1000 = 500
    east = tile("east", "EPSG:32650", 113.995, 114.03, 600)     # 旧基线: 600
    scenes = [
        {"tile": "49R", "item": _s2_item("W", datetime(2023, 7, 1), {"B04": west}, "05.10"),
         "aoi": aoi},
        {"tile": "50R", "item": _s2_item("E", datetime(2023, 7, 3), {"B04": east}, "02.14"),
         "aoi": aoi},
    ]
    src = Sentinel2DataSource()
    out = src.download_bands(scenes, str(tmp_path / "out"), ["red"])

    assert out["red"].endswith("B4_2023.tif")
    with rasterio.open(out["red"]) as f:
        red = f.read(1)
        assert GridSpec.from_src(f) == src.analysis_grid(aoi)
        assert f.crs.to_epsg() == 32650
    assert set(np.unique(red)) <= {500, 600}, "无数据缝隙或未归一化"
    w = red.shape[1]
    assert (red[:, : w // 4] == 500).all()          # 仅西景覆盖
    assert (red[:, -w // 4:] == 600).all()          # 仅东景覆盖
    assert (red[:, w // 2 - 2: w // 2 + 2] == 500).all()   # 重叠区取第一景


# ============================================================================
#  4. 时间序列
# ============================================================================

@pytest.mark.parametrize("n", [5, 8, 10])
def test_sen_slope_mk_matches_scipy(n):
    from scipy import stats
    from engine.timeseries import sen_slope_mk

    rng = np.random.default_rng(n)
    years = np.arange(2010, 2010 + n)
    stacked = rng.normal(0, 1, (n, 6, 7)).astype(np.float32)
    stacked += (0.3 * (years - years[0]))[:, None, None] * rng.uniform(0, 1, (1, 6, 7))
    stacked[1, 0, 0] = np.nan                       # 缺测像元
    stacked[:, 5, 6] = np.nan                       # 全缺测像元

    slope, p, n_obs = sen_slope_mk(stacked, years, chunk_rows=2)

    for r in range(6):
        for c in range(7):
            y = stacked[:, r, c]
            ok = np.isfinite(y)
            if ok.sum() < 4:
                assert np.isnan(slope[r, c]) and n_obs[r, c] == ok.sum()
                continue
            ref = stats.theilslopes(y[ok], years[ok])
            assert slope[r, c] == pytest.approx(ref.slope, rel=1e-4, abs=1e-5)
            tau = stats.kendalltau(years[ok], y[ok], method="exact")
            assert p[r, c] == pytest.approx(tau.pvalue, rel=1e-3, abs=1e-6)


def test_mk_pvalue_normal_approximation_for_long_series():
    from scipy import stats
    from engine.timeseries import mk_pvalue

    n = 15
    S = 40
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (S - 1) / np.sqrt(var)
    assert mk_pvalue(np.array([S]), n)[0] == pytest.approx(2 * stats.norm.sf(z), rel=1e-6)
    assert mk_pvalue(np.array([0]), n)[0] == pytest.approx(1.0)


def test_time_series_masks_clouds_and_harmonizes(tmp_path, monkeypatch):
    """6 期：西半部 NDVI 每年 +0.05，东半部不变；2020 年西北角有云；
    2022 起为新基线 (+1000 DN)。期望：云区该年为 NaN 且仍得到显著上升趋势，斜率 ≈ 0.05。"""
    import engine.data_sources as ds
    from engine.data_sources import Sentinel2DataSource
    from engine.timeseries import TimeSeriesAnalyzer

    aoi = box(110.30, 25.40, 110.31, 25.41)
    grid = Sentinel2DataSource().analysis_grid(aoi)
    H, W = grid.shape
    years = list(range(2018, 2024))
    nir = 3000.0
    files = {}
    for y in years:
        ndvi = np.full((H, W), 0.5)
        ndvi[:, : W // 2] = 0.3 + 0.05 * (y - 2018)
        red = np.rint(nir * (1 - ndvi) / (1 + ndvi))
        off = 1000 if y >= 2022 else 0
        scl = np.full(((H + 1) // 2, (W + 1) // 2), 4, np.uint8)
        if y == 2020:
            scl[: H // 4, : W // 4] = 9
        files[y] = {
            "B04": _write(tmp_path / f"r{y}.tif", (red + off).astype(np.uint16), grid.transform, grid.crs),
            "B08": _write(tmp_path / f"n{y}.tif", np.full((H, W), nir + off, np.uint16),
                          grid.transform, grid.crs),
            "SCL": _write(tmp_path / f"s{y}.tif", scl,
                          grid.transform * Affine.scale(2), grid.crs),
        }

    class FakeS2(Sentinel2DataSource):
        def search_scenes(self, boundary, year, *a, **k):
            item = _s2_item(f"S2_{year}", datetime(year, 7, 1), files[year],
                            "05.10" if year >= 2022 else "02.14")
            return [{"tile": "49R", "item": item, "coverage": 1.0, "cloud": 0, "aoi": boundary}]

    monkeypatch.setattr(ds, "get_data_source", lambda name: FakeS2())

    an = TimeSeriesAnalyzer(aoi, "sentinel2")
    ts = an.collect_time_series(years, "NDVI", data_dir=str(tmp_path / "ts"))
    trend = an.compute_trend(ts)

    assert ts["years"] == years and ts["stacked"].shape == (6, H, W)
    assert ts["preprocessing"][2020]["cloud_mask"] and ts["preprocessing"][2020]["cloud_pct"] > 0
    inside = ts["valid_mask"]
    r, c = H // 8 + 1, W // 8 + 1                    # 云区内、AOI 内的像元
    assert inside[r, c] and np.isnan(ts["index_maps"][2020][r, c])
    assert np.isfinite(ts["index_maps"][2021][r, c])

    west = inside.copy()
    west[:, W // 2 - 1:] = False
    east = inside.copy()
    east[:, : W // 2 + 1] = False
    assert (trend["trend_type"][west] == 1).all()
    np.testing.assert_allclose(trend["slope"][west], 0.05, atol=2e-3)
    assert (trend["trend_type"][east] == 0).all()
    np.testing.assert_allclose(trend["slope"][east], 0.0, atol=1e-6)


# ============================================================================
#  5. 采样点
# ============================================================================

def _sampling_case():
    gs = GridSpec(crs=rasterio.crs.CRS.from_epsg(32649),
                  transform=Affine(30, 0, 500000, 0, -30, 2800020), width=100, height=100)
    res = np.ones((100, 100), np.int8)
    res[20:40, 20:40] = -3                          # 环形斑块，质心落在洞里
    res[24:36, 24:36] = 1
    res[60:65, 10:20] = 2                           # 相邻的两个不同类别斑块
    res[60:65, 20:30] = 3
    res[80:82, 80:85] = -2                          # 10 px 小斑块
    return res, gs.to_meta(), _wgs84_box_of_grid(gs, shrink_m=-100)


def _params(**kw):
    from engine.params import RunParams
    base = dict(region="测试区", year1=2020, year2=2024, output_formats=("geojson",),
                generate_report=False)
    base.update(kw)
    return RunParams(**base)


def test_sample_points_interior_classes_and_area(tmp_path):
    from engine.core import generate_sample_points

    res, meta, bnd = _sampling_case()
    paths, pts = generate_sample_points(res, bnd, meta, str(tmp_path),
                                        _params(output_formats=("geojson", "csv", "shp")))

    assert len(pts) == 4
    for p in pts:                                    # 代表点必须落在本类斑块内
        assert res[p["row"], p["col"]] == p["class"]
    by_cls = {p["class"]: p for p in pts}
    assert set(by_cls) == {-3, -2, 2, 3}             # 相邻不同类别不合并
    assert pts[0]["class"] == -3 and pts[0]["area_px"] == 400 - 144   # 面积降序
    assert by_cls[-2]["area_ha"] == pytest.approx(10 * 0.09)          # 30 m 像元
    assert by_cls[2]["class_name"] == "中度恢复"

    import fiona
    with fiona.open(paths["shp"]) as f:
        assert f.crs.to_epsg() == 4326
        assert len(f) == 4
    assert os.path.exists(paths["csv"]) and os.path.exists(paths["geojson"])


def test_sample_points_distance_in_metres_and_limit(tmp_path):
    from engine.core import generate_sample_points

    res, meta, bnd = _sampling_case()
    # 2、3 类斑块代表点相距 300 m：400 m 间距下保留先出现的一个
    _, pts = generate_sample_points(res, bnd, meta, str(tmp_path), _params(min_point_distance_m=400))
    assert sorted(p["class"] for p in pts) == [-3, -2, 2]
    _, pts = generate_sample_points(res, bnd, meta, str(tmp_path), _params(min_point_distance_m=200))
    assert len(pts) == 4

    _, pts = generate_sample_points(res, bnd, meta, str(tmp_path), _params(num_sample_points=2))
    assert [p["class"] for p in pts] == [-3, 2]

    # 边界外的斑块不导出：边界只覆盖网格左上 2100 m × 2100 m（-2 类斑块在第 80 行之后）
    from shapely.geometry import Polygon
    t = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
    corners = [(500000, 2800020), (502100, 2800020), (502100, 2797920), (500000, 2797920)]
    small_bnd = Polygon([t.transform(x, y) for x, y in corners])
    _, pts = generate_sample_points(res, small_bnd, meta, str(tmp_path), _params())
    assert {p["class"] for p in pts} == {-3, 2, 3}


def test_sample_points_binary_mode(tmp_path):
    from engine.core import generate_sample_points

    res, meta, bnd = _sampling_case()
    binary = np.isin(res, [-3, -2, 2, 3]).astype(np.uint8)
    _, pts = generate_sample_points(binary, bnd, meta, str(tmp_path),
                                    _params(detection_mode="binary"))
    assert len(pts) == 3                             # 相邻的 2/3 类在二值图中是同一斑块
    assert all(p["class"] == 1 and p["class_name"] == "变化" for p in pts)


# ============================================================================
#  6. 报告
# ============================================================================

def test_report_uses_index_names_and_thresholds(tmp_path, monkeypatch):
    import engine.core as core

    labels = []
    real_patch = core.Patch
    monkeypatch.setattr(core, "Patch", lambda **kw: (labels.append(kw["label"]), real_patch(**kw))[1])

    res, _, _ = _sampling_case()
    stats = {cv: {"area_ha": 10.0, "percent": 1.0} for cv in (-3, -2, 1, 2, 3)}
    stats.update(changed_total={"area_ha": 40.0, "percent": 4.0}, valid_pixels=10000,
                 _threshold=0.1, _severe_threshold=0.2)

    path = core.generate_report(res, stats, str(tmp_path), _params(index_type="NDWI", region="甲区"))
    assert os.path.basename(path) == "甲区_NDWI_变化检测报告.png" and os.path.exists(path)
    assert any("明显减少" in s for s in labels) and not any("退化" in s for s in labels)

    path = core.generate_report((res != 1).astype(np.uint8),
                                {"changed_area_ha": 5, "changed_percent": 1, "_threshold": 0.1},
                                str(tmp_path), _params(detection_mode="binary", region="甲区"))
    assert os.path.basename(path) == "甲区_NDVI_变化检测报告.png"


def test_ndbi_registry_complete():
    from engine.indices import INDEX_THRESHOLDS, INDEX_CLASS_NAMES, INDEX_COLORS, INDEX_FUNCTIONS
    for idx in INDEX_FUNCTIONS:
        assert idx in INDEX_THRESHOLDS and idx in INDEX_CLASS_NAMES and idx in INDEX_COLORS, idx
    assert "建筑" in INDEX_CLASS_NAMES["NDBI"][3]


# ============================================================================
#  7. RunParams 与流水线
# ============================================================================

@pytest.mark.parametrize("kw, msg", [
    (dict(year2=2020), "年份不能相同"),
    (dict(threshold=0.3, severe_threshold=0.2), "强烈变化阈值"),
    (dict(output_formats="geojson,kml"), "未知输出格式"),
    (dict(month_start=9, month_end=6), "起始月份"),
    (dict(data_source="modis"), "未知数据源"),
    (dict(num_sample_points=0), "采样数量"),
])
def test_run_params_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _params(**kw)


def test_run_params_from_config_is_read_only():
    from engine.params import RunParams
    before = _config_snapshot()
    p = RunParams.from_config(region="乙区", threshold=None, output_formats="csv, shp",
                              index_type="ndwi")
    assert p.region == "乙区" and p.threshold is None
    assert p.output_formats == ("csv", "shp") and p.index_type == "NDWI"
    assert p.year1 == config.YEAR1                   # 未指定 -> config 默认
    assert _config_snapshot() == before
    d = p.to_dict()
    assert d["output_formats"] == ["csv", "shp"] and all(isinstance(k, str) for k in d["min_area_by_class"])


def _scene_files(tmp_path, year, red_dn, nir_dn, n=40):
    """本地「远程」影像：n×n 10 m，左上角 (500000, 2800000)。"""
    tr = Affine(10, 0, 500000, 0, -10, 2800000)
    red = np.full((n, n), red_dn, np.uint16)
    nir = np.full((n, n), nir_dn, np.uint16)
    if year % 2 == 0:
        red[: n // 2, : n // 2] = 2500               # 后时相左上角退化
    return {"B04": _write(tmp_path / f"red{year}.tif", red, tr, "EPSG:32649"),
            "B08": _write(tmp_path / f"nir{year}.tif", nir, tr, "EPSG:32649"),
            "SCL": _write(tmp_path / f"scl{year}.tif", np.full((n // 2, n // 2), 4, np.uint8),
                          Affine(20, 0, 500000, 0, -20, 2800000), "EPSG:32649")}


@pytest.fixture
def offline_world(tmp_path, monkeypatch):
    """离线数据源 + 独立缓存目录；search_calls 记录联网检索次数。"""
    import engine.core as core
    import engine.data_sources as ds
    from engine.data_sources import Sentinel2DataSource

    files = {y: _scene_files(tmp_path, y, 500, 3000) for y in (2021, 2022)}
    calls = []

    class FakeS2(Sentinel2DataSource):
        def search_scenes(self, boundary, year, *a, **k):
            calls.append(year)
            item = _s2_item(f"S2_{year}", datetime(year, 7, 1), files[year], "02.14")
            return [{"tile": "49R", "item": item, "coverage": 1.0, "cloud": 0, "aoi": boundary}]

    monkeypatch.setattr(ds, "get_data_source", lambda name: FakeS2())
    monkeypatch.setattr(core, "_cache_root", lambda: str(tmp_path / "cache"))
    gs = GridSpec(crs=rasterio.crs.CRS.from_epsg(32649),
                  transform=Affine(10, 0, 500000, 0, -10, 2800000), width=40, height=40)
    return SimpleNamespace(boundary=_wgs84_box_of_grid(gs, shrink_m=20), calls=calls, tmp=tmp_path)


def test_prepare_data_then_pipeline_hits_cache(offline_world):
    from engine.core import prepare_data, find_cached_snapshot, run_pipeline_v3

    w = offline_world
    p = _params(year1=2021, year2=2022)
    assert find_cached_snapshot(p, w.boundary) is None

    info = prepare_data(p, w.boundary, str(w.tmp / "data"))
    assert "error" not in info, info
    assert info["tiles"] == 2
    import engine.core as core
    from engine.cache import SnapshotCache
    sc = SnapshotCache(core._cache_root())
    cached = [sorted(f for f in os.listdir(sc.get_band_dir(key)) if f.endswith(".tif"))
              for key in info["key"].split("+")]                # 每期一个缓存条目
    assert cached == [["B4_2021.tif", "B8_2021.tif", "SCL_2021.tif"],
                      ["B4_2022.tif", "B8_2022.tif", "SCL_2022.tif"]]
    assert find_cached_snapshot(p, w.boundary)["key"] == info["key"]
    assert w.calls == [2021, 2022]
    # 下载用的临时目录已清理
    assert not [d for d in os.listdir(w.tmp / "data") if d.startswith("dl_")]

    res = run_pipeline_v3(params=p, boundary=w.boundary, data_dir=str(w.tmp / "data"),
                          out_dir=str(w.tmp / "out"))
    assert "error" not in res, res.get("error")
    assert w.calls == [2021, 2022], "分析阶段应命中缓存，不再检索"
    assert res["cache_key"] == info["key"]
    assert res["stats"][-3]["pixels"] > 0 and res["params"]["region"] == "测试区"
    assert os.path.basename(res["files"]["geotiff"]) == "测试区_NDVI_变化检测.tif"

    # 其他参数（月份）不同 -> 不复用缓存
    assert find_cached_snapshot(_params(year1=2021, year2=2022, month_end=9), w.boundary) is None


def test_pipeline_concurrent_runs_keep_their_own_params(offline_world):
    """两个会话并发运行不同参数：结果各自对应自己的参数，全局 config 不变。"""
    from engine.core import prepare_data, run_pipeline_v3

    w = offline_world
    base = dict(year1=2021, year2=2022)
    assert "error" not in prepare_data(_params(**base), w.boundary, str(w.tmp / "data"))
    before = _config_snapshot()

    runs = {
        "甲": _params(region="甲", threshold=0.10, severe_threshold=0.20, **base),
        "乙": _params(region="乙", threshold=0.30, severe_threshold=0.60, min_area_m2=100, **base),
    }

    def go(name):
        return run_pipeline_v3(params=runs[name], boundary=w.boundary,
                               data_dir=str(w.tmp / "data"), out_dir=str(w.tmp / f"out_{name}"))

    with ThreadPoolExecutor(2) as ex:
        out = dict(zip(runs, ex.map(go, runs)))

    for name, res in out.items():
        assert "error" not in res, res.get("error")
        assert res["stats"]["_threshold"] == runs[name].threshold
        assert res["region"] == name
        assert os.path.basename(res["files"]["geotiff"]).startswith(f"{name}_")
    assert _config_snapshot() == before


def test_cli_maps_arguments_to_run_params(tmp_path, monkeypatch, capsys):
    import main

    seen = {}

    def fake_pipeline(params=None, **kw):
        seen.update(kw, params=params)
        return {"stats": {}, "timing": {}, "num_sample_points": 0}

    monkeypatch.setattr(main, "run_pipeline_v3", fake_pipeline)
    rc = main.main(["--boundary", "examples/sample_aoi.geojson", "--region", "示例区",
                    "--year1", "2023", "--year2", "2024", "--months", "5-9",
                    "--index", "ndbi", "--mode", "binary", "--adaptive",
                    "--formats", "geojson,shp", "--out", str(tmp_path)])
    p = seen["params"]
    assert rc == 0
    assert (p.region, p.year1, p.year2, p.month_start, p.month_end) == ("示例区", 2023, 2024, 5, 9)
    assert p.index_type == "NDBI" and p.detection_mode == "binary" and p.adaptive_threshold
    assert p.output_formats == ("geojson", "shp") and seen["out_dir"] == str(tmp_path)
    assert p.boundary_file == "examples/sample_aoi.geojson"

    assert main.main(["--year1", "2024", "--year2", "2024"]) == 2       # 参数错误
    monkeypatch.setattr(main, "run_pipeline_v3", lambda **kw: {"error": "no scenes"})
    assert main.main([]) == 1                                            # 流水线失败
    assert "no scenes" in capsys.readouterr().err
