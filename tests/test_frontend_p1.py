"""
前端 P1 回归测试（全部离线）

  1. 面积：椭球面积，不再用「度² × 111.32²」（25°N 高估约 11%）
  2. 结果地图：每类一张墨卡托叠加图，全部变化像元都画出来（旧实现超过 3 万斑块随机丢弃），
     渲染结果可缓存
  3. 首页 / 系统页 / 成果库：区分可复用快照与旧版影像；成果库列出没有任务记录的输出
  4. 向导与工作台不再提供无法运行的 SAR 选项

用法:
    python -m pytest tests/test_frontend_p1.py -v
"""

import base64
import json
import os
import sys
import time

import cv2
import numpy as np
import pytest
from pyproj import Transformer
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.ops import transform as shp_transform

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from engine.utils import geodesic_area_km2

SAMPLE_AOI = os.path.join(BASE, "examples", "sample_aoi.geojson")
UTM49 = "EPSG:32649"  # 示例区域（灵川县，110.3°E）所在 UTM 带


def _sample_geom():
    with open(SAMPLE_AOI, encoding="utf-8") as f:
        return shape(json.load(f)["features"][0]["geometry"])


def _utm_area_km2(geom):
    t = Transformer.from_crs("EPSG:4326", UTM49, always_xy=True)
    return shp_transform(t.transform, geom).area / 1e6


# ---------------------------------------------------------------------------
#  1. 面积
# ---------------------------------------------------------------------------

def test_geodesic_area_matches_projected_area():
    geom = _sample_geom()
    expected = _utm_area_km2(geom)
    assert geodesic_area_km2(geom) == pytest.approx(expected, rel=0.005)
    # 旧公式在 25°N 高估一成以上
    assert geom.area * 111.32 ** 2 > expected * 1.10


def test_geodesic_area_handles_holes_and_multipolygons():
    outer = box(110.30, 25.40, 110.40, 25.50)
    hole = box(110.33, 25.43, 110.36, 25.46)
    holed = Polygon(outer.exterior.coords, [hole.exterior.coords])
    assert geodesic_area_km2(holed) == pytest.approx(_utm_area_km2(holed), rel=0.005)
    assert geodesic_area_km2(holed) < geodesic_area_km2(outer)

    other = box(110.50, 25.40, 110.55, 25.45)
    multi = MultiPolygon([outer, other])
    assert geodesic_area_km2(multi) == pytest.approx(
        geodesic_area_km2(outer) + geodesic_area_km2(other), rel=1e-9)


# ---------------------------------------------------------------------------
#  2. 结果地图叠加图
# ---------------------------------------------------------------------------

_TF = Affine(10, 0, 600000, 0, -10, 2810000)  # UTM 49N 10 m 网格，约 25.4°N
_META = {"crs": UTM49, "transform": _TF}


def _decode(layer):
    png = base64.b64decode(layer["image"].split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)


def _covered_fraction(layer, arr):
    """原始栅格中该类每个像元中心，落在叠加图上的位置是否着色。"""
    img = _decode(layer)
    dh, dw = img.shape[:2]
    (south, west), (north, east) = layer["bounds"]
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = to_merc.transform(west, south)
    x1, y1 = to_merc.transform(east, north)
    rows, cols = np.nonzero(arr == layer["value"])
    xs, ys = _TF * (cols + 0.5, rows + 0.5)
    mx, my = Transformer.from_crs(UTM49, "EPSG:3857", always_xy=True).transform(xs, ys)
    c = np.floor((np.asarray(mx) - x0) / ((x1 - x0) / dw)).astype(int)
    r = np.floor((y1 - np.asarray(my)) / ((y1 - y0) / dh)).astype(int)
    inside = (r >= 0) & (r < dh) & (c >= 0) & (c < dw)
    hit = inside & (img[r.clip(0, dh - 1), c.clip(0, dw - 1), 3] > 0)
    return hit.mean(), img


def _lattice(h=600, w=800, step=3, value=2):
    """互不相连的单像元斑块网格：(h/3)·(w/3) 个，超过旧实现 3 万的上限。"""
    arr = np.zeros((h, w), np.int8)
    arr[::step, ::step] = value
    return arr


def test_change_overlays_draw_every_change_pixel():
    from frontend.components.map_view import change_overlays

    arr = _lattice()
    n_patches = int((arr == 2).sum())
    assert n_patches > 30000

    (layer,) = change_overlays(arr, _META, "NDVI")
    assert layer["pixels"] == n_patches
    fraction, img = _covered_fraction(layer, arr)
    assert fraction == 1.0
    # 未缩小时形状不外扩：着色面积与变化像元面积（每个 100 m²）一致
    (south, west), (north, east) = layer["bounds"]
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = to_merc.transform(west, south)
    x1, y1 = to_merc.transform(east, north)
    scale = np.cos(np.radians((south + north) / 2)) ** 2   # 墨卡托面积 → 地面面积
    px_m2 = (x1 - x0) / img.shape[1] * (y1 - y0) / img.shape[0] * scale
    colored_m2 = (img[..., 3] > 0).sum() * px_m2
    assert colored_m2 == pytest.approx(n_patches * 100, rel=0.1)


def test_change_overlays_keep_small_patches_when_downsampled():
    from frontend.components.map_view import change_overlays

    arr = np.zeros((3000, 2500), np.int8)
    rng = np.random.default_rng(0)
    arr.flat[rng.choice(arr.size, 2000, replace=False)] = -3

    (layer,) = change_overlays(arr, _META, "NDVI", max_side=512)
    img = _decode(layer)
    assert max(img.shape[:2]) <= 512
    assert _covered_fraction(layer, arr)[0] == 1.0


def test_change_overlays_use_index_class_names_and_colors():
    from engine.indices import INDEX_CLASS_NAMES, INDEX_COLORS
    from frontend.components.map_view import change_overlays

    arr = np.zeros((200, 200), np.int8)
    for i, value in enumerate((-3, -2, 2, 3)):
        arr[10 + 40 * i: 30 + 40 * i, 20:60] = value
    layers = change_overlays(arr, _META, "NDBI")
    assert [layer["value"] for layer in layers] == [-3, -2, 2, 3]
    for layer in layers:
        assert layer["name"] == INDEX_CLASS_NAMES["NDBI"][layer["value"]]
        assert layer["color"] == INDEX_COLORS["NDBI"][layer["value"]]
        img = _decode(layer)
        b, g, r, _ = img[img[..., 3] > 0][0]
        assert f"#{r:02x}{g:02x}{b:02x}".lower() == layer["color"].lower()
        assert layer["pixels"] == 800


def test_cached_change_overlays_render_once_per_result(monkeypatch):
    from frontend.components import map_view

    calls = []
    monkeypatch.setattr(map_view, "change_overlays",
                        lambda result, meta, index_type: calls.append(index_type) or ["layer"])
    store = {}
    arr = _lattice()
    first = map_view.cached_change_overlays(store, ("a.tif", 1.0, "NDVI"), arr, _META, "NDVI")
    again = map_view.cached_change_overlays(store, ("a.tif", 1.0, "NDVI"), arr, _META, "NDVI")
    assert first is again and calls == ["NDVI"]
    map_view.cached_change_overlays(store, ("a.tif", 2.0, "NDVI"), arr, _META, "NDVI")
    assert calls == ["NDVI", "NDVI"]


def test_result_layer_is_image_overlays_not_sampled_polygons():
    import folium
    from frontend.components.map_view import add_change_result_layer, change_overlays

    arr = np.zeros((200, 200), np.int8)
    arr[10:30, 10:30] = -2
    arr[100:120, 100:120] = 3
    m = folium.Map(location=[25.4, 111.0], zoom_start=12)
    overlays = change_overlays(arr, _META, "NDVI")
    used = add_change_result_layer(m, index_type="NDVI", opacity=0.35, overlays=overlays)
    html = m.get_root().render()
    assert used is overlays
    assert html.count("L.imageOverlay(") == 2
    assert '"opacity": 0.35' in html
    assert "geo_json" not in html


# ---------------------------------------------------------------------------
#  3. 本地影像 / 成果盘点
# ---------------------------------------------------------------------------

def _touch(path, size=1024, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _cache_entry(root, key, version, complete, n_tifs):
    entry = root / "cache" / key
    entry.mkdir(parents=True)
    (entry / "snapshot.json").write_text(
        json.dumps({"kind": "period", "preprocessing_version": version}), encoding="utf-8")
    if complete:
        (entry / ".collected").write_text("1")
    for i in range(n_tifs):
        _touch(entry / f"B{i}_2025.tif", size=1024 * 1024)


def test_imagery_summary_separates_reusable_snapshots_from_legacy(tmp_path):
    from engine.preprocessing import PREPROCESSING_VERSION
    from frontend.inventory import imagery_summary

    data = tmp_path / "data"
    _touch(data / "新余" / "B4_2025.tif", size=2 * 1024 * 1024)   # 旧版下载：无质量波段
    _touch(data / "新余" / "B8_2025.tif", size=2 * 1024 * 1024)
    _cache_entry(data, "fresh", PREPROCESSING_VERSION, True, 3)    # 可复用
    _cache_entry(data, "old", "v3.0", True, 1)                     # 预处理版本过期
    _cache_entry(data, "partial", PREPROCESSING_VERSION, False, 1)  # 收集未完成

    s = imagery_summary(data)
    assert (s["snapshots"], s["snapshot_files"], s["stale_files"]) == (1, 3, 4)
    assert s["snapshot_mb"] == pytest.approx(3.0)
    assert s["stale_mb"] == pytest.approx(6.0)
    assert (s["classify_sets"], s["reusable_sets"]) == (0, 1)
    assert s["reusable_mb"] == pytest.approx(3.0)

    empty = imagery_summary(tmp_path / "missing")
    assert set(empty.values()) == {0}


def test_imagery_summary_counts_classification_downloads_as_reusable(tmp_path):
    from frontend.inventory import imagery_summary

    data = tmp_path / "data"
    # 地物分类页下载成功后写入 source.json，之后可直接「加载波段数据」
    for year in (2024, 2025):
        year_dir = data / "新余" / str(year)
        for band in ("B02", "B03", "B04", "B08", "B11", "B12", "SCL"):
            _touch(year_dir / f"{band}_{year}.tif", size=1024 * 1024)
        (year_dir / "source.json").write_text(
            json.dumps({"data_source": "sentinel2", "year": year}), encoding="utf-8")
    # 下载中断（没写 source.json）、旧版平铺目录：仍不可复用
    _touch(data / "新余" / "2026" / "B04_2026.tif", size=1024 * 1024)
    _touch(data / "新余" / "B4_2025.tif", size=1024 * 1024)

    s = imagery_summary(data)
    assert (s["classify_sets"], s["classify_files"]) == (2, 14)
    assert s["classify_mb"] == pytest.approx(14.0)
    assert (s["snapshots"], s["reusable_sets"]) == (0, 2)
    assert s["reusable_mb"] == pytest.approx(14.0)
    assert (s["stale_files"], s["stale_mb"]) == (2, pytest.approx(2.0))


def test_unarchived_outputs_lists_files_without_task_records(tmp_path):
    from frontend.inventory import unarchived_outputs

    out = tmp_path / "output"
    now = time.time()
    _touch(out / "新余" / "新余_变化检测.tif", mtime=now - 100)
    _touch(out / "新余" / "sample_points.csv", mtime=now - 50)
    kept = [_touch(out / "栖霞区" / "sample_points.geojson"),
            _touch(out / "栖霞区" / "sample_points.shp")]
    for ext in (".dbf", ".shx", ".prj"):        # 已归档 .shp 的附属文件
        _touch(out / "栖霞区" / f"sample_points{ext}")
    _touch(out / "栖霞区" / "旧报告.png", mtime=now - 500)
    _touch(out / "summary.png", mtime=now - 1000)
    _touch(out / "新余" / ".DS_Store")
    _touch(out / "新余" / "partial.tmp")

    groups = unarchived_outputs(out, [str(p) for p in kept] + ["", None])
    assert [g["region"] for g in groups] == ["新余", "栖霞区", "output"]
    assert [p.name for p in groups[0]["files"]] == ["sample_points.csv", "新余_变化检测.tif"]
    assert [p.name for p in groups[1]["files"]] == ["旧报告.png"]
    assert groups[0]["size_mb"] == pytest.approx(2 / 1024)

    assert unarchived_outputs(tmp_path / "missing", []) == []


# ---------------------------------------------------------------------------
#  4. 分析目标 / 数据源不再有死路
# ---------------------------------------------------------------------------

def test_every_wizard_target_is_runnable():
    from frontend.components.wizard import (
        PLANNED_PROBLEM_TYPES, PROBLEM_PAGES, PROBLEM_TYPES, usable_indices,
    )

    assert "SAR 洪水与湿度" in PLANNED_PROBLEM_TYPES
    assert not set(PLANNED_PROBLEM_TYPES) & set(PROBLEM_TYPES)
    assert set(PROBLEM_PAGES) == set(PROBLEM_TYPES)
    for name, problem in PROBLEM_TYPES.items():
        assert os.path.exists(os.path.join(BASE, PROBLEM_PAGES[name]))
        for source in problem["available_sources"]:
            assert usable_indices(name, source), (name, source)
        rec = problem["recommended"]
        assert rec["index_type"] in usable_indices(name, rec["data_source"])
    assert usable_indices("水体变化", "sentinel1") == []


def test_analysis_sources_only_offer_sources_with_indices():
    from frontend.utils import ANALYSIS_SOURCES, CAPABILITY_MATRIX

    assert "sentinel1" not in ANALYSIS_SOURCES
    assert ANALYSIS_SOURCES[0] == "sentinel2"
    assert all(CAPABILITY_MATRIX[source] for source in ANALYSIS_SOURCES)


def _wizard_app():
    from frontend.components.wizard import render_wizard

    render_wizard()


@pytest.fixture
def wizard_test(monkeypatch):
    """渲染向导（不读写项目根目录的共享状态文件）。"""
    from streamlit.testing.v1 import AppTest
    import frontend.project_state as project_state

    monkeypatch.setattr(project_state, "_PROJECT_STATE_FILE", "")

    def make(step, **state):
        at = AppTest.from_function(_wizard_app, default_timeout=60)
        at.session_state["_shared_project_state"] = project_state._ProjectState(
            region="示例区", boundary=SAMPLE_AOI, slug="sample", **state)
        at.session_state["wizard_step"] = step
        return at.run()

    return make


def test_wizard_hides_sar_and_never_blocks_next(wizard_test):
    from frontend.components.wizard import PROBLEM_TYPES

    at = wizard_test(1, data_source="sentinel1")
    assert not at.exception
    radio = at.radio(key="wizard_problem_selector")
    assert list(radio.options) == list(PROBLEM_TYPES)
    assert any("SAR 洪水与湿度" in c.value for c in at.caption)
    for option in PROBLEM_TYPES:
        at = radio.set_value(option).run()
        assert not at.exception
        assert not at.button(key="wizard_target_next").disabled, option
        radio = at.radio(key="wizard_problem_selector")


def test_wizard_confirm_step_shows_geodesic_area(wizard_test):
    at = wizard_test(2)
    assert not at.exception
    table = at.dataframe[0].value
    area = dict(zip(table["项目"], table["配置"]))["面积"]
    assert area == f"{geodesic_area_km2(_sample_geom()):,.1f} km²"
