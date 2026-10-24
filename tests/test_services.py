"""
服务层测试（全部离线）：services/ 下的取数与分析流程不依赖 streamlit，可被页面与后台任务共用

  - 变化检测：收集 → 运行（命中收集阶段的缓存）→ 任务记录 / PDF 参数 → 读取结果；无影像时报错
  - 地物分类：取数 + 加载波段、快速分类 / GMM / 主动学习、按实际像元面积统计、后处理、导出
  - 时间序列：进度单调、时相不足时报错
  - 批量任务：单个任务的输出目录与参数校验

用法:
    python -m pytest tests/test_services.py -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from services import ServiceError
from services import batch as batch_svc
from services import change_detection as cd_svc
from services import classification as cl_svc
from services import timeseries as ts_svc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cd_params(**kw):
    from engine.params import RunParams
    base = dict(region="测试区", year1=2021, year2=2022, output_formats=("geojson",),
                generate_report=False)
    base.update(kw)
    return RunParams(**base)


def test_services_do_not_import_streamlit():
    code = ("import sys; import services.change_detection, services.classification, "
            "services.timeseries, services.batch; sys.exit(int('streamlit' in sys.modules))")
    r = subprocess.run([sys.executable, "-c", code], cwd=BASE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr or "services 不应导入 streamlit"


# =====================================================================
#  变化检测
# =====================================================================

def test_change_detection_collect_then_run(world):
    params = _cd_params()
    progress = []
    ctx = cd_svc.collect(params, world.aoi, world.data,
                         progress_callback=lambda msg, pct: progress.append(pct))
    assert set(ctx["bands"]) == {"red", "nir"} and "+" in ctx["cache_key"]
    assert (ctx["region"], ctx["year1"], ctx["year2"]) == ("测试区", 2021, 2022)
    assert progress and len(world.calls.download) == 2

    res = cd_svc.run(params, world.aoi, world.data, str(world.tmp / "out"), slug="test", pdf=False)
    assert len(world.calls.download) == 2, "分析阶段应直接命中收集阶段写入的缓存"
    assert res["unmasked_years"] == []
    task = res["task"]
    assert task["id"] == "cd_test_2021_2022_NDVI" and task["status"] == "success"
    assert task["stats_summary"].startswith("变化 ") and task["started_at"]
    assert task["output_files"] == res["files"] and "pdf" not in res["files"]
    assert cd_svc.changed_area_ha(res["stats"]) > 0

    arr, meta, cands = cd_svc.load_outputs(res)
    assert arr.shape == (meta["height"], meta["width"])
    assert (arr < 0).any(), "2021 → 2022 红光突增，应检出退化"
    assert len(cands) == res["num_sample_points"] > 0


def test_change_detection_pdf_receives_area_and_points(world, monkeypatch):
    import engine.report_generator as rg
    from engine.utils import geodesic_area_km2

    seen = {}

    def fake_pdf(**kw):
        seen.update(kw)
        path = Path(kw["output_dir"]) / "report.pdf"
        path.write_bytes(b"%PDF-1.4")
        return str(path)

    monkeypatch.setattr(rg, "generate_report_pdf", fake_pdf)
    res = cd_svc.run(_cd_params(), world.aoi, world.data, str(world.tmp / "out"), slug="test",
                     source_label="Sentinel-2 L2A (10m)")
    assert res["files"]["pdf"] == res["task"]["output_files"]["pdf"]
    assert seen["boundary_info"]["area_km2"] == f"{geodesic_area_km2(world.aoi):,.1f}"
    assert seen["metadata"]["data_source"] == "Sentinel-2 L2A (10m)"
    assert seen["sample_points"] and {"lon", "lat", "class_name"} <= set(seen["sample_points"][0])


def test_change_detection_pdf_failure_does_not_fail_analysis(world, monkeypatch):
    import engine.report_generator as rg

    def broken(**kw):
        raise RuntimeError("no reportlab")

    monkeypatch.setattr(rg, "generate_report_pdf", broken)
    res = cd_svc.run(_cd_params(), world.aoi, world.data, str(world.tmp / "out"), slug="test")
    assert "pdf" not in res["files"] and res["task"]["status"] == "success"


def test_change_detection_reports_missing_year(world):
    with pytest.raises(ServiceError, match="2024"):
        cd_svc.collect(_cd_params(year2=2024), world.aoi, world.data)
    with pytest.raises(ServiceError, match="2024"):
        cd_svc.run(_cd_params(year2=2024), world.aoi, world.data, str(world.tmp / "out"),
                   slug="test")
    assert world.calls.download == []


# =====================================================================
#  地物分类
# =====================================================================

# 三条横带：水体 / 植被 / 建设用地（反射率）
_STRIPES = {
    "blue": (0.08, 0.03, 0.15), "green": (0.10, 0.06, 0.17), "red": (0.06, 0.04, 0.20),
    "nir": (0.03, 0.40, 0.25), "swir1": (0.02, 0.20, 0.30), "swir2": (0.01, 0.10, 0.28),
}


def _scene(res_m=10, size=48):
    rng = np.random.default_rng(0)
    rows = np.repeat([0, 1, 2], size // 3)
    bands = {
        name: (np.array(vals, np.float32)[rows][:, None]
               + rng.normal(0, 0.005, (size, size))).astype(np.float32)
        for name, vals in _STRIPES.items()
    }
    from engine.grid import GridSpec
    grid = GridSpec(CRS.from_epsg(32649), Affine(res_m, 0, 500000, 0, -res_m, 2810000), size, size)
    valid = np.ones((size, size), bool)
    valid[:4, :4] = False                       # 一块云
    return bands, grid.to_meta(), valid


@pytest.mark.parametrize("res_m", [10, 30])
def test_quick_classification_uses_real_pixel_area(res_m):
    bands, meta, valid = _scene(res_m)
    out = cl_svc.quick(bands, meta, valid=valid)
    result = out["result"]
    assert (result[:4, :4] == -1).all(), "云区为无数据"
    assert len({s["class"] for s in out["stats"]}) >= 3
    for s in out["stats"]:
        assert s["area_ha"] == pytest.approx(s["pixels"] * res_m * res_m / 10000)


def test_gmm_active_learning_and_post_processing():
    bands, meta, valid = _scene()
    out = cl_svc.gmm(bands, meta, n_clusters=3, pca_dims=3, sample_ratio=0.2, valid=valid)
    labels = out["labels"]
    assert (labels[:4, :4] == -1).all() and set(np.unique(labels[valid])) <= {0, 1, 2}
    assert out["mapping"] == {0: 1, 1: 2, 2: 3}
    mapped = cl_svc.label_clusters(labels, {0: 1, 1: 1, 2: 1}, meta)
    assert (mapped["result"][valid] == 1).all() and (mapped["result"][~valid] == 0).all()

    al = cl_svc.sample_annotation_patches(bands, out["gmm"], n_patches=4, patch_size=8)
    assert 0 < len(al["patches"]) <= 4 and not al["mask"].any()
    with pytest.raises(ServiceError, match="标注像元太少"):
        cl_svc.refine(bands, meta, al["labels"], al["mask"], valid=valid)

    (r0, c0), (r1, c1) = al["patches"][:2]
    cl_svc.annotate_patch(al["labels"], al["mask"], r0, c0, 0, patch_size=8)
    assert not al["mask"].any(), "未分类（0）不计为标注"
    cl_svc.annotate_patch(al["labels"], al["mask"], r0, c0, 1, patch_size=8)
    cl_svc.annotate_patch(al["labels"], al["mask"], r1, c1, 3, patch_size=8)
    assert al["mask"].sum() == 2 * 9 * 9
    refined = cl_svc.refine(bands, meta, al["labels"], al["mask"], valid=valid)
    assert refined["result"].shape == labels.shape and (refined["result"][:4, :4] == -1).all()
    assert refined["confidence"].shape == labels.shape and refined["stats"]

    noisy = np.zeros((48, 48), np.int8)
    noisy[0:2, 0:2] = 1
    noisy[10:30, 10:30] = 2
    filtered = cl_svc.filter_small_patches(noisy, meta, 10)["result"]
    assert (filtered[0:2, 0:2] == 0).all() and (filtered[10:30, 10:30] == 2).all()


def test_classification_download_load_and_export(world):
    info = cl_svc.download("sentinel2", world.aoi, 2021, 6, 10, 50, world.data)
    assert cl_svc.find_cached("sentinel2", world.aoi, 2021, 6, 10, 50)["key"] == info["key"]
    loaded = cl_svc.load_bands(info["dir"], "landsat8")
    assert loaded["data_source"] == "sentinel2", "以缓存条目记录的数据源为准"
    assert set(loaded["bands"]) == set(cl_svc.CLASSIFY_BANDS) and loaded["valid"].all()
    (world.tmp / "empty").mkdir()
    assert cl_svc.load_bands(str(world.tmp / "empty"), "sentinel2") is None
    with pytest.raises(ServiceError):
        cl_svc.download("sentinel2", world.aoi, 2024, 6, 10, 50, world.data)

    bands, meta, _ = _scene()
    out = cl_svc.quick(bands, meta)
    files = cl_svc.export(out["result"], meta, out["stats"], str(world.tmp / "out"),
                          region="测试区", mode="quick")
    assert set(files) == {"geotiff", "geojson", "csv"}
    with rasterio.open(files["geotiff"]) as src:
        assert src.dtypes[0] == "int8" and (src.read(1) == out["result"]).all()
    with open(files["geojson"], encoding="utf-8") as f:
        fc = json.load(f)
    assert {feat["properties"]["class"] for feat in fc["features"]} == \
        {s["class"] for s in out["stats"]}
    lon, lat = fc["features"][0]["geometry"]["coordinates"][0][0]
    assert 110 < lon < 112 and 25 < lat < 26, "GeoJSON 为 WGS84 经纬度"

    empty = cl_svc.export(np.full((48, 48), -1, np.int8), meta, [], str(world.tmp / "out"),
                          region="测试区", mode="empty")
    assert set(empty) == {"geotiff"}


# =====================================================================
#  时间序列
# =====================================================================

def test_timeseries_run(world):
    progress = []
    res = ts_svc.run(world.aoi, list(range(2019, 2025)), "NDVI", "sentinel2", 6, 10, 50,
                     world.data, progress_callback=lambda msg, pct: progress.append(pct))
    assert res["ts_data"]["years"] == [2019, 2020, 2021, 2022]
    assert res["summary"]["years"] == [2019, 2020, 2021, 2022]
    valid = res["ts_data"]["valid_mask"]
    assert valid.any() and np.nanmean(res["trend"]["slope"][valid]) < 0, "NDVI 逐年下降"
    assert progress == sorted(progress) and progress[0] == 5 and progress[-1] == 100


def test_timeseries_needs_enough_periods(world):
    with pytest.raises(ServiceError, match="至少需要"):
        ts_svc.run(world.aoi, [2020, 2021, 2022], "NDVI", "sentinel2", 6, 10, 50, world.data)
    assert world.calls.search == [], "时相不足时不检索"
    # 2023 网络出错、2024 无影像 → 只有 2 期
    with pytest.raises(ServiceError, match="只取得 2 期"):
        ts_svc.run(world.aoi, [2021, 2022, 2023, 2024], "NDVI", "sentinel2", 6, 10, 50,
                   world.data)


# =====================================================================
#  批量任务
# =====================================================================

def test_batch_run_task(world):
    from shapely.geometry import mapping

    (world.tmp / "aoi.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"name": "测试区"},
                      "geometry": mapping(world.aoi)}],
    }), encoding="utf-8")
    task = {"id": "t1", "region": "测试区", "boundary": "aoi.geojson",
            "year1": 2021, "year2": 2022, "formats": "geojson", "report": False}
    res = batch_svc.run_task(task, base_dir=str(world.tmp))
    assert res["task_id"] == "t1" and res["elapsed"] >= 0
    assert Path(res["files"]["geotiff"]).parent == world.tmp / "output" / "t1"

    with pytest.raises(ValueError):
        batch_svc.run_task({**task, "year2": 2021}, base_dir=str(world.tmp))
    assert batch_svc.task_id({"region": "甲", "year1": 2020, "year2": 2021}) == "甲_2020_2021"
