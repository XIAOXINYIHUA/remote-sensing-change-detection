"""
遥感解译助手 v3 — 多源遥感数据接口

支持: Sentinel-2 L2A / Landsat 8/9 Collection 2 L2 / Sentinel-1 GRD（实验性）

下载策略:
  - 只读取 AOI 范围：通过 WarpedVRT 对远端 COG 做窗口读取，不再下载整景瓦片
  - 直接重投影到由 AOI 决定的统一分析网格（grid.aoi_grid），
    同一 AOI 的所有年份、所有波段落在完全相同的网格上
  - 跨 UTM 带的瓦片先变换到同一网格再镶嵌（按覆盖率优先，先到先得）
"""

import os
import calendar

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from shapely.geometry import shape, box as sbox

from engine.grid import aoi_grid
from engine.preprocessing import harmonize_s2_dn, s2_boa_offset
from engine.utils import log

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _last_day_of_month(year: int, month: int) -> int:
    """返回指定年月的最后一天（处理2月、大小月）。"""
    return calendar.monthrange(year, month)[1]


def _open_catalog():
    try:
        import pystac_client
        import planetary_computer
    except ImportError as e:
        raise ImportError(
            "需要 pystac-client 和 planetary-computer：pip install -r requirements.txt"
        ) from e
    return pystac_client.Client.open(
        PC_STAC_URL, modifier=planetary_computer.sign_inplace)


def footprint_coverage(boundary, item) -> float:
    """AOI 被该景实际数据足迹覆盖的比例。

    使用 item.geometry（真实足迹）而非 item.bbox：轨道边缘的瓦片
    bbox 内常有大片无数据区域，用 bbox 会高估覆盖率。
    """
    if boundary.area <= 0:
        return 0.0
    geom = getattr(item, "geometry", None)
    footprint = shape(geom) if geom else sbox(*item.bbox)
    return boundary.intersection(footprint).area / boundary.area


class StacDataSource:
    """Planetary Computer STAC 数据源基类。

    子类只需声明集合、波段映射与网格参数；搜索、窗口读取、
    镶嵌与写盘逻辑共用。
    """

    name = ""
    collection = ""
    # 逻辑波段名 -> 候选 asset key（按顺序取第一个存在的）
    BAND_MAP: dict = {}
    # 逻辑波段名 -> 输出文件名前缀
    FILE_NAME_MAP: dict = {}
    # (逻辑波段名, 文件前缀, 网格外填充值)；None=无质量波段
    QUALITY_BAND = None
    RESOLUTION = 10.0
    GRID_ORIGIN_OFFSET = 0.0
    MIN_COVERAGE = 0.01

    def __init__(self):
        self._catalog = None

    # ------------------------------------------------------------------ 搜索

    def _get_catalog(self):
        if self._catalog is None:
            self._catalog = _open_catalog()
        return self._catalog


    def _query(self, max_cloud_pct):
        return {"eo:cloud_cover": {"lt": max_cloud_pct}}

    def _tile_key(self, item):
        return item.id

    def _cloud(self, item):
        return item.properties.get("eo:cloud_cover", 0)

    def search_scenes(self, boundary, year, month_start=6, month_end=10,
                      max_cloud_pct=50):
        """搜索并为每个瓦片选出「覆盖率 × 云量」得分最高的一景。

        返回的每个场景都带上 "aoi"，供 download_bands 确定分析网格。
        """
        date_range = (f"{year}-{month_start:02d}-01/"
                      f"{year}-{month_end:02d}-{_last_day_of_month(year, month_end):02d}")
        items = list(self._get_catalog().search(
            collections=[self.collection], bbox=boundary.bounds,
            datetime=date_range, query=self._query(max_cloud_pct),
            max_items=200).items())
        return self.select_scenes(boundary, items)

    def select_scenes(self, boundary, items):
        """按瓦片分组择优（与网络无关，便于测试）。"""
        groups = {}
        for item in items:
            groups.setdefault(self._tile_key(item), []).append(item)

        best = []
        for tile, its in groups.items():
            scored = []
            for it in its:
                cov = footprint_coverage(boundary, it)
                scored.append((cov * 100 - self._cloud(it) * 0.3, cov, it))
            _, cov, item = max(scored, key=lambda x: x[0])
            if cov < self.MIN_COVERAGE:
                continue
            best.append({"tile": tile, "item": item, "coverage": cov,
                         "cloud": self._cloud(item), "aoi": boundary})
        best.sort(key=lambda s: -s["coverage"])
        return best

    # ------------------------------------------------------------------ 读取

    def analysis_grid(self, boundary):
        return aoi_grid(boundary, self.RESOLUTION, origin_offset=self.GRID_ORIGIN_OFFSET)

    def _asset_href(self, item, band_name):
        for key in self.BAND_MAP.get(band_name, (band_name,)):
            asset = item.assets.get(key)
            if asset is not None:
                return asset.href
        raise ValueError(f"Band '{band_name}' not found in item {item.id}")

    def _postprocess(self, item, band_name, arr):
        """逐景后处理钩子（如 S2 基线偏移归一化）。"""
        return arr

    def _read_to_grid(self, item, band_name, grid, resampling, nodata):
        """把一景的一个波段窗口读取并变换到分析网格上。"""
        with rasterio.open(self._asset_href(item, band_name)) as src:
            src_nodata = src.nodata if src.nodata is not None else nodata
            with WarpedVRT(src, crs=grid.crs, transform=grid.transform,
                           width=grid.width, height=grid.height,
                           resampling=resampling,
                           src_nodata=src_nodata, nodata=nodata) as vrt:
                arr = vrt.read(1)
        return self._postprocess(item, band_name, arr)

    def _mosaic(self, scenes, band_name, grid, resampling, nodata):
        """按场景顺序（覆盖率降序）先到先得镶嵌；全部失败返回 None。"""
        out = None
        for s in scenes:
            try:
                arr = self._read_to_grid(s["item"], band_name, grid, resampling, nodata)
            except Exception as e:
                log(f"    Warning: {band_name}/{s['tile']}: {e}")
                continue
            if out is None:
                out = np.full(grid.shape, nodata, dtype=arr.dtype)
            fill = (out == nodata) & (arr != nodata)
            out[fill] = arr[fill]
        return out

    def _write(self, arr, grid, path, nodata):
        meta = grid.to_meta(dtype=arr.dtype.name)
        meta.update(nodata=nodata, tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(path, "w", **meta) as dst:
            dst.write(arr, 1)
        return path

    def _grid_for(self, scenes, boundary):
        boundary = boundary or (scenes[0].get("aoi") if scenes else None)
        if boundary is None:
            raise ValueError("scenes 缺少 AOI：请使用 search_scenes() 的结果或传入 boundary")
        return self.analysis_grid(boundary)

    def download_bands(self, scenes, out_dir, bands=None, boundary=None):
        """下载 AOI 范围内的波段到统一分析网格。

        Returns:
            {逻辑波段名: GeoTIFF 路径}，文件名 <前缀>_<年份>.tif
        """
        bands = bands or ["red", "nir"]
        if not scenes:
            return {}
        os.makedirs(out_dir, exist_ok=True)
        grid = self._grid_for(scenes, boundary)
        year = scenes[0]["item"].datetime.year
        result = {}
        for band_name in bands:
            arr = self._mosaic(scenes, band_name, grid, Resampling.bilinear, 0)
            if arr is None:
                continue
            fname = self.FILE_NAME_MAP.get(band_name, band_name.upper())
            result[band_name] = self._write(
                arr, grid, os.path.join(out_dir, f"{fname}_{year}.tif"), 0)
        return result

    def download_quality_bands(self, scenes, out_dir, boundary=None) -> dict[str, str]:
        """下载质量波段（最近邻重采样到同一分析网格）。"""
        if not self.QUALITY_BAND or not scenes:
            return {}
        band_name, prefix, fill = self.QUALITY_BAND
        os.makedirs(out_dir, exist_ok=True)
        grid = self._grid_for(scenes, boundary)
        arr = self._mosaic(scenes, band_name, grid, Resampling.nearest, fill)
        if arr is None:
            return {}
        year = scenes[0]["item"].datetime.year
        return {band_name: self._write(
            arr, grid, os.path.join(out_dir, f"{prefix}_{year}.tif"), fill)}


class Sentinel2DataSource(StacDataSource):
    """Sentinel-2 L2A via Planetary Computer STAC."""

    name = "sentinel2"
    collection = "sentinel-2-l2a"

    BAND_MAP = {
        "red": ("B04",), "nir": ("B08",), "green": ("B03",), "blue": ("B02",),
        "swir1": ("B11",), "swir2": ("B12",),
        "rededge1": ("B05",), "rededge2": ("B06",), "rededge3": ("B07",),
        "scl": ("SCL",),
    }

    # 输出文件名映射 (匹配 _load_bands 的 B4/B8 约定)
    FILE_NAME_MAP = {
        "red": "B4", "nir": "B8", "green": "B3", "blue": "B2",
        "swir1": "B11", "swir2": "B12",
    }

    QUALITY_BAND = ("scl", "SCL", 0)
    RESOLUTION = 10.0

    def _tile_key(self, item):
        return item.properties.get("s2:mgrs_tile") or item.id.split("_")[-2]

    def _postprocess(self, item, band_name, arr):
        if band_name == "scl":
            return arr
        # 逐景归一化基线 04.00 偏移（镶嵌前做，混合日期的瓦片也正确）
        return harmonize_s2_dn(arr, s2_boa_offset(item))


class LandsatDataSource(StacDataSource):
    """Landsat 8/9 Collection 2 Level-2 via Planetary Computer STAC."""

    name = "landsat"
    collection = "landsat-c2-l2"

    # STAC 通用名优先，兼容旧版 Bn 键
    BAND_MAP = {
        "red": ("red", "B4"), "nir": ("nir08", "B5"), "green": ("green", "B3"),
        "blue": ("blue", "B2"), "swir1": ("swir16", "B6"), "swir2": ("swir22", "B7"),
        "qa_pixel": ("qa_pixel", "QA_PIXEL"),
    }

    FILE_NAME_MAP = {
        "red": "B4", "nir": "B5", "green": "B3", "blue": "B2",
        "swir1": "B6", "swir2": "B7",
    }

    # QA_PIXEL bit 0 = Fill，网格外填 1
    QUALITY_BAND = ("qa_pixel", "QA_PIXEL", 1)
    RESOLUTION = 30.0
    GRID_ORIGIN_OFFSET = 15.0  # C2 UTM 角点坐标 ≡ 15 (mod 30)

    def __init__(self, satellite="landsat-8"):
        super().__init__()
        self.satellite = satellite
        self.name = satellite.replace("-", "")

    def _query(self, max_cloud_pct):
        return {"eo:cloud_cover": {"lt": max_cloud_pct},
                "platform": {"in": [self.satellite]}}

    def _tile_key(self, item):
        p = item.properties
        if "landsat:wrs_path" in p:
            return f"{p['landsat:wrs_path']}_{p.get('landsat:wrs_row', '')}"
        return item.id.split("_")[2]  # LC08_L2SP_<path><row>_...

    def _cloud(self, item):
        return item.properties.get("eo:cloud_cover", 50)


class Sentinel1DataSource(StacDataSource):
    """Sentinel-1 GRD C-SAR via Planetary Computer STAC（实验性，暂无光学指数）。"""

    name = "sentinel1"
    collection = "sentinel-1-grd"

    BAND_MAP = {"vv": ("vv", "VV"), "vh": ("vh", "VH")}
    FILE_NAME_MAP = {"vv": "VV", "vh": "VH"}

    def _query(self, max_cloud_pct):
        return {"sar:polarizations": {"contains": "VV"}}

    def _tile_key(self, item):
        return item.properties.get("sat:orbit_state", "unknown")

    def _cloud(self, item):
        return 0

    def download_bands(self, scenes, out_dir, bands=None, boundary=None):
        return super().download_bands(scenes, out_dir, bands or ["vv", "vh"], boundary)


def get_data_source(name):
    """Factory: return data source instance by name.

    Args:
        name: "sentinel2" | "landsat8" | "landsat9" | "sentinel1"
    """
    sources = {
        "sentinel2": Sentinel2DataSource,
        "landsat8": lambda: LandsatDataSource("landsat-8"),
        "landsat9": lambda: LandsatDataSource("landsat-9"),
        "sentinel1": Sentinel1DataSource,
    }
    if name not in sources:
        valid = list(sources.keys())
        raise ValueError(f"Unknown data source: {name}. Available: {valid}")
    return sources[name]()
