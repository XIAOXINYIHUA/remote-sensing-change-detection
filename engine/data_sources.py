"""
遥感解译助手 v3 — 多源遥感数据接口

支持: Sentinel-2 L2A / Landsat 8/9 / Sentinel-1 GRD
"""

import os, sys
import numpy as np
import rasterio
from rasterio.merge import merge
from shapely.geometry import box as sbox

from engine.preprocessing import harmonize_s2_dn, s2_boa_offset

import calendar


def _last_day_of_month(year: int, month: int) -> int:
    """返回指定年月的最后一天（处理2月、大小月）。"""
    return calendar.monthrange(year, month)[1]


class Sentinel2DataSource:
    """Sentinel-2 L2A via Planetary Computer STAC."""

    name = "sentinel2"

    BAND_MAP = {
        "red": "B04", "nir": "B08", "green": "B03", "blue": "B02",
        "swir1": "B11", "swir2": "B12",
        "rededge1": "B05", "rededge2": "B06", "rededge3": "B07",
        "scl": "SCL",
    }

    # 输出文件名映射 (匹配 _load_bands 的 B4/B8 约定)
    FILE_NAME_MAP = {
        "red": "B4", "nir": "B8", "green": "B3", "blue": "B2",
        "swir1": "B11", "swir2": "B12",
    }

    def __init__(self):
        self._catalog = None

    def _get_catalog(self):
        if self._catalog is None:
            try:
                import pystac_client
                import planetary_computer
            except ImportError:
                import subprocess as sp
                sp.run([sys.executable, "-m", "pip", "install",
                        "pystac-client", "planetary-computer", "--quiet"],
                       capture_output=True)
                import pystac_client
                import planetary_computer
            self._catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace)
        return self._catalog

    def get_band_aliases(self):
        return self.BAND_MAP.copy()

    def search_scenes(self, boundary, year, month_start=6, month_end=10,
                      max_cloud_pct=50):
        catalog = self._get_catalog()
        date_fmt = f"{year}-{month_start:02d}-01/{year}-{month_end:02d}-{_last_day_of_month(year, month_end):02d}"
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=boundary.bounds,
            datetime=date_fmt,
            query={"eo:cloud_cover": {"lt": max_cloud_pct}},
            max_items=200)
        items = list(search.items())

        tile_groups = {}
        for item in items:
            tile = item.id.split("_")[-2]
            tile_groups.setdefault(tile, []).append(item)

        best_per_tile = []
        for tile, its in tile_groups.items():
            best_item = None
            best_score = -999.0
            best_cov = 0.0
            best_cc = 100.0
            for item in its:
                cc = item.properties["eo:cloud_cover"]
                ib = sbox(*item.bbox)
                cov = (boundary.intersection(ib).area / boundary.area
                       if boundary.area > 0 else 0)
                score = cov * 100 - cc * 0.3
                if score > best_score:
                    best_score = score
                    best_item = item
                    best_cov = cov
                    best_cc = cc
            if best_item is None or best_cov < 0.01:
                continue
            best_per_tile.append({
                "tile": tile, "item": best_item,
                "coverage": best_cov, "cloud": best_cc
            })

        best_per_tile.sort(key=lambda x: -x["coverage"])
        return best_per_tile

    def _download_single_band(self, item, band_name, out_path):
        asset_key = self.BAND_MAP.get(band_name, band_name)
        asset = item.assets.get(asset_key)
        if not asset:
            raise ValueError(
                f"Band '{band_name}' not found in item {item.id}")
        with rasterio.open(asset.href) as src:
            arr = src.read()
            meta = src.meta.copy()
            meta.update(
                height=arr.shape[1], width=arr.shape[2],
                driver="GTiff", compress="lzw")
        if band_name != "scl":
            # 逐景归一化基线 04.00 偏移（镶嵌前做，混合日期的瓦片也正确）
            arr = harmonize_s2_dn(arr, s2_boa_offset(item))
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(arr)
        return out_path

    def download_bands(self, scenes, out_dir, bands=None):
        if bands is None:
            bands = ["red", "nir"]
        os.makedirs(out_dir, exist_ok=True)
        result = {}
        for band_name in bands:
            sources = []
            for s in scenes:
                item = s["item"]
                tile = s["tile"]
                tmp_path = os.path.join(
                    out_dir, f"tmp_{band_name}_{tile}.tif")
                try:
                    self._download_single_band(item, band_name, tmp_path)
                    sources.append(tmp_path)
                except Exception as e:
                    print(f"    Warning: {band_name}/{tile}: {e}")
                    continue
            if not sources:
                continue
            year = item.datetime.year if hasattr(item, "datetime") else \
                item.properties.get("datetime", "0000")[:4]
            fname = self.FILE_NAME_MAP.get(band_name, band_name.upper())
            out_path = os.path.join(out_dir, f"{fname}_{year}.tif")
            if len(sources) == 1:
                import shutil
                shutil.move(sources[0], out_path)
            else:
                srcs = [rasterio.open(p) for p in sources]
                try:
                    mosaic_arr, mosaic_tr = merge(srcs)
                    meta_s = srcs[0].meta.copy()
                    meta_s.update(
                        height=mosaic_arr.shape[1],
                        width=mosaic_arr.shape[2],
                        transform=mosaic_tr)
                    with rasterio.open(out_path, "w", **meta_s) as dst:
                        dst.write(mosaic_arr)
                finally:
                    for s in srcs:
                        s.close()
                for p in sources:
                    if os.path.exists(p) and p != out_path:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            result[band_name] = out_path
        return result

    def download_quality_bands(self, scenes, out_dir) -> dict[str, str]:
        """下载 Sentinel-2 SCL (Scene Classification Layer) 质量波段。"""
        return _download_scl_generic(scenes, out_dir, self._download_single_band)


class LandsatDataSource:
    """Landsat 8/9 Collection 2 Level-2 via Planetary Computer STAC."""

    name = "landsat"

    BAND_MAP = {
        "red": "B4", "nir": "B5", "green": "B3", "blue": "B2",
        "swir1": "B6", "swir2": "B7",
        "qa_pixel": "QA_PIXEL",
    }

    FILE_NAME_MAP = {
        "red": "B4", "nir": "B5", "green": "B3", "blue": "B2",
        "swir1": "B6", "swir2": "B7",
    }

    def __init__(self, satellite="landsat-8"):
        self.satellite = satellite
        self.name = satellite.replace("-", "")
        self._catalog = None

    def _get_catalog(self):
        if self._catalog is None:
            try:
                import pystac_client
                import planetary_computer
            except ImportError:
                import subprocess as sp
                sp.run([sys.executable, "-m", "pip", "install",
                        "pystac-client", "planetary-computer", "--quiet"],
                       capture_output=True)
                import pystac_client
                import planetary_computer
            self._catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace)
        return self._catalog

    def get_band_aliases(self):
        return self.BAND_MAP.copy()

    def search_scenes(self, boundary, year, month_start=6, month_end=10,
                      max_cloud_pct=50):
        catalog = self._get_catalog()
        date_fmt = f"{year}-{month_start:02d}-01/{year}-{month_end:02d}-{_last_day_of_month(year, month_end):02d}"
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=boundary.bounds,
            datetime=date_fmt,
            query={
                "eo:cloud_cover": {"lt": max_cloud_pct},
                "platform": {"in": [self.satellite]},
            },
            max_items=200)
        items = list(search.items())

        tile_groups = {}
        for item in items:
            wrs_path = item.properties.get(
                "landsat:wrs_path", item.id.split("_")[2][:3])
            wrs_row = item.properties.get(
                "landsat:wrs_row", item.id.split("_")[2][3:])
            tile = f"{wrs_path}_{wrs_row}"
            tile_groups.setdefault(tile, []).append(item)

        best_per_tile = []
        for tile, its in tile_groups.items():
            best_item = None
            best_score = -999.0
            best_cov = 0.0
            best_cc = 100.0
            for item in its:
                cc = item.properties.get("eo:cloud_cover", 50)
                ib = sbox(*item.bbox)
                cov = (boundary.intersection(ib).area / boundary.area
                       if boundary.area > 0 else 0)
                score = cov * 100 - cc * 0.3
                if score > best_score:
                    best_score = score
                    best_item = item
                    best_cov = cov
                    best_cc = cc
            if best_item is None or best_cov < 0.01:
                continue
            best_per_tile.append({
                "tile": tile, "item": best_item,
                "coverage": best_cov, "cloud": best_cc
            })

        best_per_tile.sort(key=lambda x: -x["coverage"])
        return best_per_tile

    def _download_single_band(self, item, band_name, out_path):
        asset_key = self.BAND_MAP.get(band_name, band_name)
        asset = item.assets.get(asset_key)
        if not asset:
            raise ValueError(
                f"Band '{band_name}' not found in item {item.id}")
        with rasterio.open(asset.href) as src:
            arr = src.read()
            meta = src.meta.copy()
            meta.update(
                height=arr.shape[1], width=arr.shape[2],
                driver="GTiff", compress="lzw")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(arr)
        return out_path

    def download_bands(self, scenes, out_dir, bands=None):
        if bands is None:
            bands = ["red", "nir"]
        os.makedirs(out_dir, exist_ok=True)
        result = {}
        for band_name in bands:
            sources = []
            for s in scenes:
                item = s["item"]
                tile = s["tile"]
                tmp_path = os.path.join(
                    out_dir, f"tmp_{band_name}_{tile}.tif")
                try:
                    self._download_single_band(item, band_name, tmp_path)
                    sources.append(tmp_path)
                except Exception as e:
                    print(f"    Warning: {band_name}/{tile}: {e}")
                    continue
            if not sources:
                continue
            year = scenes[0]["item"].datetime.year if scenes else 0
            fname = self.FILE_NAME_MAP.get(band_name, band_name.upper())
            out_path = os.path.join(out_dir, f"{fname}_{year}.tif")
            if len(sources) == 1:
                import shutil
                shutil.move(sources[0], out_path)
            else:
                srcs = [rasterio.open(p) for p in sources]
                try:
                    mosaic_arr, mosaic_tr = merge(srcs)
                    meta_s = srcs[0].meta.copy()
                    meta_s.update(
                        height=mosaic_arr.shape[1],
                        width=mosaic_arr.shape[2],
                        transform=mosaic_tr)
                    with rasterio.open(out_path, "w", **meta_s) as dst:
                        dst.write(mosaic_arr)
                finally:
                    for s in srcs:
                        s.close()
                for p in sources:
                    if os.path.exists(p) and p != out_path:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            result[band_name] = out_path
        return result

    def download_quality_bands(self, scenes, out_dir) -> dict[str, str]:
        """下载 Landsat QA_PIXEL 质量波段。"""
        return _download_qa_pixel_generic(scenes, out_dir, self._download_single_band)


class Sentinel1DataSource:
    """Sentinel-1 GRD C-SAR via Planetary Computer STAC."""

    name = "sentinel1"

    FILE_NAME_MAP = {"vv": "VV", "vh": "VH"}

    def __init__(self):
        self._catalog = None

    def _get_catalog(self):
        if self._catalog is None:
            try:
                import pystac_client
                import planetary_computer
            except ImportError:
                import subprocess as sp
                sp.run([sys.executable, "-m", "pip", "install",
                        "pystac-client", "planetary-computer", "--quiet"],
                       capture_output=True)
                import pystac_client
                import planetary_computer
            self._catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace)
        return self._catalog

    def get_band_aliases(self):
        return {"vv": "vv", "vh": "vh"}

    def search_scenes(self, boundary, year, month_start=6, month_end=10,
                      max_cloud_pct=50):
        catalog = self._get_catalog()
        date_fmt = f"{year}-{month_start:02d}-01/{year}-{month_end:02d}-{_last_day_of_month(year, month_end):02d}"
        search = catalog.search(
            collections=["sentinel-1-grd"],
            bbox=boundary.bounds,
            datetime=date_fmt,
            query={"sar:polarizations": {"contains": "VV"}},
            max_items=200)
        items = list(search.items())

        orbit_groups = {}
        for item in items:
            orbit = item.properties.get("sat:orbit_state", "unknown")
            orbit_groups.setdefault(orbit, []).append(item)

        best_per_orbit = []
        for orbit, its in orbit_groups.items():
            best_item = None
            best_cov = 0.0
            for item in its:
                ib = sbox(*item.bbox)
                cov = (boundary.intersection(ib).area / boundary.area
                       if boundary.area > 0 else 0)
                if cov > best_cov:
                    best_cov = cov
                    best_item = item
            if best_item and best_cov > 0.01:
                best_per_orbit.append({
                    "tile": orbit, "item": best_item,
                    "coverage": best_cov, "cloud": 0
                })

        best_per_orbit.sort(key=lambda x: -x["coverage"])
        return best_per_orbit

    def _download_single_band(self, item, band_name, out_path):
        asset = item.assets.get(band_name.upper())
        if not asset:
            raise ValueError(
                f"Polarization '{band_name}' not found in item {item.id}")
        with rasterio.open(asset.href) as src:
            arr = src.read()
            meta = src.meta.copy()
            meta.update(
                height=arr.shape[1], width=arr.shape[2],
                driver="GTiff", compress="lzw")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(arr)
        return out_path

    def download_bands(self, scenes, out_dir, bands=None):
        if bands is None:
            bands = ["vv", "vh"]
        os.makedirs(out_dir, exist_ok=True)
        result = {}
        for band_name in bands:
            sources = []
            for s in scenes:
                item = s["item"]
                tile = s["tile"]
                tmp_path = os.path.join(
                    out_dir, f"tmp_{band_name}_{tile}.tif")
                try:
                    self._download_single_band(item, band_name, tmp_path)
                    sources.append(tmp_path)
                except Exception as e:
                    print(f"    Warning: {band_name}/{tile}: {e}")
                    continue
            if not sources:
                continue
            year = scenes[0]["item"].datetime.year if scenes else 0
            fname = self.FILE_NAME_MAP.get(band_name, band_name.upper())
            out_path = os.path.join(out_dir, f"{fname}_{year}.tif")
            if len(sources) == 1:
                import shutil
                shutil.move(sources[0], out_path)
            else:
                srcs = [rasterio.open(p) for p in sources]
                try:
                    mosaic_arr, mosaic_tr = merge(srcs)
                    meta_s = srcs[0].meta.copy()
                    meta_s.update(
                        height=mosaic_arr.shape[1],
                        width=mosaic_arr.shape[2],
                        transform=mosaic_tr)
                    with rasterio.open(out_path, "w", **meta_s) as dst:
                        dst.write(mosaic_arr)
                finally:
                    for s in srcs:
                        s.close()
                for p in sources:
                    if os.path.exists(p) and p != out_path:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            result[band_name] = out_path
        return result

    def download_quality_bands(self, scenes, out_dir) -> dict[str, str]:
        """Sentinel-1 无质量波段。"""
        return {}


# ============================================================================
#  通用质量波段下载辅助函数
# ============================================================================

def _download_scl_generic(scenes, out_dir, download_fn) -> dict[str, str]:
    """下载 Sentinel-2 SCL (Scene Classification Layer) 质量波段。

    Args:
        scenes: 场景列表
        out_dir: 输出目录
        download_fn: _download_single_band 方法引用

    Returns:
        {"scl": scl_path} 或空字典
    """
    scl_sources = []
    for s in scenes:
        item = s["item"]
        tile = s["tile"]
        tmp_path = os.path.join(out_dir, f"tmp_SCL_{tile}.tif")
        try:
            download_fn(item, "scl", tmp_path)
            scl_sources.append((tmp_path, tile))
        except Exception as e:
            print(f"    Warning: SCL/{tile}: {e}")
            continue
    if not scl_sources:
        return {}
    year = scenes[0]["item"].datetime.year if scenes else 0
    out_path = os.path.join(out_dir, f"SCL_{year}.tif")
    if len(scl_sources) == 1:
        import shutil
        shutil.move(scl_sources[0][0], out_path)
    else:
        srcs = [rasterio.open(p) for p, _ in scl_sources]
        try:
            mosaic_arr, mosaic_tr = merge(srcs)
            meta_s = srcs[0].meta.copy()
            meta_s.update(height=mosaic_arr.shape[1],
                          width=mosaic_arr.shape[2],
                          transform=mosaic_tr)
            with rasterio.open(out_path, "w", **meta_s) as dst:
                dst.write(mosaic_arr)
        finally:
            for s in srcs:
                s.close()
        for p, _ in scl_sources:
            if os.path.exists(p) and p != out_path:
                try:
                    os.remove(p)
                except OSError:
                    pass
    return {"scl": out_path}


def _download_qa_pixel_generic(scenes, out_dir, download_fn) -> dict[str, str]:
    """下载 Landsat QA_PIXEL 质量波段。

    Returns:
        {"qa_pixel": qa_path} 或空字典
    """
    qa_sources = []
    for s in scenes:
        item = s["item"]
        tile = s["tile"]
        tmp_path = os.path.join(out_dir, f"tmp_QA_PIXEL_{tile}.tif")
        try:
            download_fn(item, "qa_pixel", tmp_path)
            qa_sources.append((tmp_path, tile))
        except Exception as e:
            print(f"    Warning: QA_PIXEL/{tile}: {e}")
            continue
    if not qa_sources:
        return {}
    year = scenes[0]["item"].datetime.year if scenes else 0
    out_path = os.path.join(out_dir, f"QA_PIXEL_{year}.tif")
    if len(qa_sources) == 1:
        import shutil
        shutil.move(qa_sources[0][0], out_path)
    else:
        srcs = [rasterio.open(p) for p, _ in qa_sources]
        try:
            mosaic_arr, mosaic_tr = merge(srcs)
            meta_s = srcs[0].meta.copy()
            meta_s.update(height=mosaic_arr.shape[1],
                          width=mosaic_arr.shape[2],
                          transform=mosaic_tr)
            with rasterio.open(out_path, "w", **meta_s) as dst:
                dst.write(mosaic_arr)
        finally:
            for s in srcs:
                s.close()
        for p, _ in qa_sources:
            if os.path.exists(p) and p != out_path:
                try:
                    os.remove(p)
                except OSError:
                    pass
    return {"qa_pixel": out_path}


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
