"""全部测试共用：

  _isolate_user_data  （自动）把快照缓存与界面状态文件隔离到临时目录。
                      快照缓存在变化检测 / 地物分类 / 时间序列之间共用，任何测试漏掉隔离都会把
                      假影像写进用户的 data/cache，并被真实分析命中。测试内再次 monkeypatch 会覆盖这里的设置。
  world               离线的假 Sentinel-2 数据源（2019–2022 每年一景）。
"""

import os
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

S2_ASSETS = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08",
             "swir1": "B11", "swir2": "B12"}


@pytest.fixture(autouse=True)
def _isolate_user_data(tmp_path_factory, monkeypatch):
    import engine.core as core

    root = tmp_path_factory.mktemp("isolated")
    monkeypatch.setattr(core, "_cache_root", lambda: str(root / "cache"))
    # 环境变量对后台任务的工作进程同样生效
    monkeypatch.setenv("RSAP_CACHE_DIR", str(root / "cache"))
    monkeypatch.setenv("RSAP_STATE_DIR", str(root / "state"))
    monkeypatch.setenv("RSAP_JOBS_DIR", str(root / "jobs"))
    yield
    # 页面测试经 get_runner() 建的执行器：停掉调度线程与残留任务
    from services import jobs
    jobs.shutdown_all()


@pytest.fixture
def world(tmp_path, monkeypatch):
    """2019–2022 每年一景 S2（6 波段 + SCL），2023 检索时网络出错；记录检索与下载。

    aoi 约 1 km × 1 km；红光逐年升高（NDVI 逐年下降），2022 年突增 → 2021→2022 有明显退化。
    """
    from rasterio.transform import Affine
    from shapely.geometry import box

    import engine.data_sources as ds
    from engine.data_sources import Sentinel2DataSource
    from test_commercial_hardening import _s2_item, _write

    aoi = box(110.30, 25.40, 110.31, 25.41)
    grid = Sentinel2DataSource().analysis_grid(aoi)
    h, w = grid.shape
    files = {}
    for year in range(2019, 2023):
        red = 2500 if year == 2022 else 500 + 100 * (year - 2019)
        assets = {
            key: _write(tmp_path / f"{key}_{year}.tif",
                        np.full((h, w), 3000 if band == "nir" else red, np.uint16),
                        grid.transform, grid.crs)
            for band, key in S2_ASSETS.items()
        }
        assets["SCL"] = _write(tmp_path / f"SCL_{year}.tif",
                               np.full(((h + 1) // 2, (w + 1) // 2), 4, np.uint8),
                               grid.transform * Affine.scale(2), grid.crs)
        files[year] = assets
    calls = SimpleNamespace(search=[], download=[])

    class FakeS2(Sentinel2DataSource):
        def search_scenes(self, boundary, year, *a, **k):
            calls.search.append(year)
            if year == 2023:
                raise ConnectionError("network down")
            if year not in files:
                return []
            item = _s2_item(f"S2_{year}", datetime(year, 7, 1), files[year], "02.14")
            return [{"tile": "49R", "item": item, "coverage": 1.0, "cloud": 0, "aoi": boundary}]

        def download_bands(self, scenes, out_dir, bands=None, boundary=None):
            calls.download.append((scenes[0]["item"].datetime.year, sorted(bands)))
            return super().download_bands(scenes, out_dir, bands, boundary)

    monkeypatch.setattr(ds, "get_data_source", lambda name: FakeS2())
    return SimpleNamespace(aoi=aoi, grid=grid, calls=calls, data=str(tmp_path / "data"),
                           tmp=tmp_path)
