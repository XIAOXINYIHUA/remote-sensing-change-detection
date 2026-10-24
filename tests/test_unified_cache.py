"""
统一取数回归测试（全部离线）：变化检测 / 地物分类 / 时间序列共用单期快照缓存

  1. 地物分类下载的 6 波段时相，变化检测（只需 red/nir）直接复用，只下载缺的那一期
  2. 时间序列复用已缓存的年份；无影像或检索失败的年份跳过，不中断
  3. 云量阈值不同不互相命中
  4. 变化检测任一期无影像时，在下载前就报错

用法:
    python -m pytest tests/test_unified_cache.py -v
"""

import os

import numpy as np

# world 夹具（2019–2022 的假 Sentinel-2 影像）见 conftest.py
CLASSIFY_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]


def _cd_params(**kw):
    from engine.params import RunParams
    base = dict(region="测试区", year1=2021, year2=2022, output_formats=("geojson",),
                generate_report=False)
    base.update(kw)
    return RunParams(**base)


def test_classification_download_is_reused_by_change_detection(world):
    from engine.core import find_cached_period, prepare_period, run_pipeline_v3

    info = prepare_period("sentinel2", world.aoi, 2021, 6, 10, 50, CLASSIFY_BANDS,
                          data_dir=world.data)
    assert "error" not in info, info
    assert info["files"] == ["B11_2021.tif", "B12_2021.tif", "B2_2021.tif", "B3_2021.tif",
                             "B4_2021.tif", "B8_2021.tif", "SCL_2021.tif"]
    assert find_cached_period("sentinel2", world.aoi, 2021, 6, 10, 50, ["red", "nir"])["key"] \
        == info["key"], "波段子集也应命中"

    res = run_pipeline_v3(params=_cd_params(), boundary=world.aoi, data_dir=world.data,
                          out_dir=str(world.tmp / "out"))
    assert "error" not in res, res.get("error")
    assert world.calls.search == [2021, 2022], "2021 已由地物分类缓存，只检索 2022"
    assert world.calls.download == [(2021, sorted(CLASSIFY_BANDS)), (2022, ["nir", "red"])]
    assert res["cache_key"].split("+")[0] == info["key"]
    assert res["preprocessing"]["y1"]["cloud_mask"] and res["preprocessing"]["y2"]["cloud_mask"]
    # 2021 → 2022 红光突增：用的确实是两期各自的数据
    assert res["stats"]["changed_total"]["pixels"] > 0
    assert not [d for d in os.listdir(world.data) if d.startswith(("dl_", "run_"))]


def test_time_series_reuses_cache_and_skips_unavailable_years(world):
    from engine.core import find_cached_snapshot, prepare_data
    from engine.timeseries import TimeSeriesAnalyzer

    assert "error" not in prepare_data(_cd_params(year1=2020, year2=2021), world.aoi, world.data)
    assert world.calls.search == [2020, 2021]

    analyzer = TimeSeriesAnalyzer(world.aoi, "sentinel2")
    progress = []
    ts = analyzer.collect_time_series(
        range(2019, 2025), "NDVI", data_dir=world.data,
        progress_callback=lambda msg, pct: progress.append(pct))
    assert ts["years"] == [2019, 2020, 2021, 2022]
    # 2020/2021 来自变化检测的缓存；2023 网络出错、2024 无影像 → 跳过
    assert world.calls.search[2:] == [2019, 2022, 2023, 2024]
    assert [year for year, _ in world.calls.download] == [2020, 2021, 2019, 2022]
    assert progress == sorted(progress) and 5 <= progress[0] and progress[-1] <= 80
    mean_ndvi = [float(np.nanmean(ts["index_maps"][y])) for y in ts["years"]]
    assert mean_ndvi == sorted(mean_ndvi, reverse=True)      # 红光逐年升高

    # 再跑一次：有影像的年份全部命中缓存，只重试失败 / 无影像的年份
    analyzer.collect_time_series(range(2019, 2025), "NDVI", data_dir=world.data)
    assert world.calls.search[6:] == [2023, 2024]
    assert len(world.calls.download) == 4
    # 时间序列下载的年份，变化检测也能直接用
    assert find_cached_snapshot(_cd_params(year1=2019, year2=2022), world.aoi) is not None


def test_cloud_threshold_is_part_of_the_cache_key(world):
    from engine.core import find_cached_period, prepare_period

    assert "error" not in prepare_period("sentinel2", world.aoi, 2021, 6, 10, 50,
                                         ["red", "nir"], data_dir=world.data)
    assert find_cached_period("sentinel2", world.aoi, 2021, 6, 10, 30, ["red", "nir"]) is None
    assert find_cached_period("sentinel2", world.aoi, 2021, 6, 10, 50, ["red", "nir"])
    assert find_cached_period("landsat8", world.aoi, 2021, 6, 10, 50, ["red", "nir"]) is None
    assert find_cached_period("sentinel2", world.aoi, 2021, 6, 9, 50, ["red", "nir"]) is None


def test_change_detection_fails_before_downloading_when_a_year_is_missing(world):
    from engine.core import find_cached_snapshot, prepare_data

    info = prepare_data(_cd_params(year1=2021, year2=2024), world.aoi, world.data)
    assert "2024" in info["error"] and "2021" not in info["error"]
    assert world.calls.download == [], "检索阶段就应报错，不浪费下载"
    assert find_cached_snapshot(_cd_params(year1=2021, year2=2024), world.aoi) is None
