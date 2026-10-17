"""
GridSpec — 统一网格配准核心模块

所有波段/指数图必须通过 reproject_to_grid() 对齐到目标网格。
连续反射率用双线性插值，分类/掩膜用最近邻。

用法:
    grid = GridSpec.from_meta(src.meta)
    aligned = reproject_to_grid(source_arr, source_meta, grid)
"""

from __future__ import annotations
import dataclasses
import typing as T

import numpy as np
import rasterio
from rasterio import warp
from rasterio.crs import CRS
from rasterio.transform import Affine
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform


@dataclasses.dataclass(frozen=True)
class GridSpec:
    """统一网格规格 — 所有引用该网格的数据必须与此完全一致。"""
    crs: CRS
    transform: Affine
    width: int
    height: int
    nodata: float = 0.0

    @property
    def resolution(self) -> tuple[float, float]:
        """像元大小 (x_res, y_res)，单位与 CRS 一致"""
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """边界 (left, bottom, right, top)"""
        return rasterio.transform.array_bounds(self.height, self.width, self.transform)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def area_per_pixel_ha(self) -> float:
        """单像元面积 (公顷) — 基于 CRS 直接计算"""
        xres, yres = self.resolution
        area_m2 = xres * yres
        return area_m2 / 10000.0

    @classmethod
    def from_meta(cls, meta: dict) -> GridSpec:
        """从 rasterio 元数据创建"""
        return cls(
            crs=CRS(meta["crs"]),
            transform=meta["transform"],
            width=meta["width"],
            height=meta["height"],
            nodata=meta.get("nodata", 0),
        )

    @classmethod
    def from_src(cls, src: rasterio.DatasetReader) -> GridSpec:
        return cls.from_meta(src.meta)

    def to_meta(self, dtype="float32", count=1, compress="lzw") -> dict:
        """转为 rasterio 写入元数据"""
        return {
            "driver": "GTiff",
            "dtype": dtype,
            "count": count,
            "crs": self.crs,
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
            "compress": compress,
            "nodata": self.nodata,
        }

    def contains(self, other: GridSpec) -> bool:
        """检查此网格是否包含 other 的边界"""
        ob = other.bounds
        sb = self.bounds
        return (sb[0] - 0.5 <= ob[0] and sb[1] - 0.5 <= ob[1]
                and sb[2] + 0.5 >= ob[2] and sb[3] + 0.5 >= ob[3])

    def __repr__(self) -> str:
        xr, yr = self.resolution
        b = self.bounds
        return (f"GridSpec({self.width}x{self.height}, "
                f"{xr:.2g}x{yr:.2g}m/px, "
                f"CRS={self.crs.to_epsg() or self.crs}, "
                f"bbox=[{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}])")


# 类型别名（必须定义在 GridSpec 之后）
GridLike = T.Union[dict, np.ndarray, GridSpec, rasterio.DatasetReader]


# ============================================================================
#  网格配准工具
# ============================================================================

def reproject_to_grid(
    source: NDArray | str | rasterio.DatasetReader,
    target_grid: GridSpec,
    resampling: int = warp.Resampling.bilinear,
    nodata: float | None = None,
) -> NDArray:
    """将栅格重投影到目标网格。

    Args:
        source: 源数据 — numpy 数组+meta 元组、文件路径、已打开 DatasetReader
        target_grid: 目标网格
        resampling: rasterio.warp.Resampling 插值方法
        nodata: 目标 nodata 值（默认使用 source 的或 0）

    Returns:
        NDArray: (H, W) 对齐后的数组
    """
    from engine.utils import log as _log

    need_close = False
    if isinstance(source, str):
        src = rasterio.open(source)
        need_close = True
    elif isinstance(source, rasterio.DatasetReader):
        src = source
    elif isinstance(source, tuple):
        arr, meta = source
        if arr.ndim == 3:
            arr = arr[0]
        _aligned = np.empty((target_grid.height, target_grid.width), dtype=np.float32)
        warp.reproject(
            source=arr,
            destination=_aligned,
            src_transform=meta["transform"],
            src_crs=meta["crs"],
            dst_transform=target_grid.transform,
            dst_crs=target_grid.crs,
            dst_nodata=nodata if nodata is not None else meta.get("nodata", 0),
            resampling=resampling,
        )
        return _aligned
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    try:
        arr = src.read(1)
        meta = src.meta
        aligned = reproject_to_grid((arr, meta), target_grid, resampling, nodata)
        return aligned
    finally:
        if need_close:
            src.close()


def reproject_bands(
    band_dict: dict[str, NDArray | str | tuple],
    target_grid: GridSpec,
    resampling: int = warp.Resampling.bilinear,
) -> dict[str, NDArray]:
    """将波段字典全部配准到目标网格。

    Args:
        band_dict: {band_name: array} 或 {band_name: (array, meta)}
                   或 {band_name: file_path}
        target_grid: 目标网格
        resampling: 插值方法

    Returns:
        {band_name: NDArray(H, W)} 全部对齐
    """
    aligned = {}
    for name, src in band_dict.items():
        aligned[name] = reproject_to_grid(src, target_grid, resampling)
    return aligned


# ============================================================================
#  网格一致性断言
# ============================================================================

GridLike = T.Union[dict, np.ndarray, GridSpec, rasterio.DatasetReader]


def as_gridspec(g: GridLike) -> GridSpec:
    """将多种输入转为 GridSpec"""
    if isinstance(g, GridSpec):
        return g
    if isinstance(g, dict):
        return GridSpec.from_meta(g)
    if isinstance(g, np.ndarray):
        raise TypeError("Cannot infer GridSpec from raw array — pass meta or file path")
    if isinstance(g, rasterio.DatasetReader):
        return GridSpec.from_src(g)
    raise TypeError(f"Cannot convert {type(g)} to GridSpec")


def assert_grid_consistent(
    grids: list[GridLike],
    names: list[str] | None = None,
) -> GridSpec:
    """断言所有输入属于同一网格，返回参考 GridSpec

    Raises:
        ValueError: 网格不一致时
    """
    ref: GridSpec | None = None
    for i, g in enumerate(grids):
        gs = as_gridspec(g)
        if ref is None:
            ref = gs
        else:
            ok = (
                ref.crs == gs.crs
                and ref.transform.almost_equals(gs.transform, precision=6)
                and ref.width == gs.width
                and ref.height == gs.height
            )
            if not ok:
                name = f" [{names[i]}]" if names and i < len(names) else ""
                raise ValueError(
                    f"Grid mismatch at index {i}{name}:\n"
                    f"  reference: {ref}\n"
                    f"  actual:    {gs}"
                )
    return ref


def boundary_grid_crop(
    boundary_wgs84,
    target_crs: CRS,
    resolution: float = 10.0,
    pad_pixels: int = 5,
) -> GridSpec:
    """生成适配 AOI 边界的裁剪网格。

    用于「选择地区后自动确定分析网格」。

    Args:
        boundary_wgs84: WGS84 Shapely 几何
        target_crs: 目标投影
        resolution: 像元大小 (m)
        pad_pixels: 边界外扩像元数

    Returns:
        GridSpec: 刚好覆盖 AOI 的网格
    """
    from rasterio.warp import transform_bounds
    t = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    bnd_proj = shp_transform(lambda x, y: t.transform(x, y), boundary_wgs84)
    minx, miny, maxx, maxy = bnd_proj.bounds

    pad = pad_pixels * resolution
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    width = max(1, int(round((maxx - minx) / resolution)))
    height = max(1, int(round((maxy - miny) / resolution)))
    transform = Affine.translation(minx, maxy) * Affine.scale(resolution, -resolution)

    return GridSpec(
        crs=target_crs,
        transform=transform,
        width=width,
        height=height,
        nodata=0.0,
    )


def compute_pixel_area_ha(crs: CRS, lat: float | None = None) -> float:
    """估算像元面积（公顷），处理经纬度坐标系。

    Args:
        crs: 坐标参考系
        lat: 纬度（仅 WGS84 需要）

    Returns:
        单像元公顷数
    """
    if crs.is_geographic:
        if lat is not None:
            from math import cos, radians
            scale = cos(radians(abs(lat)))
            return (0.000277 * scale) ** 2 * 100 / 0.01  # ~1.07 ha at equator
        return 1.07  # 粗略估计
    else:
        return 1.0  # 投影坐标系 1 m²


# ============================================================================
#  便捷函数：从边界+时相创建完整分析网格
# ============================================================================

def ensure_consistent_grid(bands_dicts: list[dict]) -> GridSpec:
    """从多个波段字典中确保网格一致并返回参考网格。

    Args:
        bands_dicts: [{band_name: file_path}, ...] 每期影像

    Returns:
        统一的 GridSpec

    Raises:
        ValueError: 无有效数据
        RuntimeError: 网格不一致且无法自动纠正
    """
    ref_grid: GridSpec | None = None
    for bd in bands_dicts:
        for bname, fpath in bd.items():
            if not isinstance(fpath, str):
                continue
            if not fpath.endswith(".tif") or not os.path.exists(fpath):
                continue
            try:
                with rasterio.open(fpath) as src:
                    gs = GridSpec.from_src(src)
                    if ref_grid is None:
                        ref_grid = gs
                    else:
                        assert_grid_consistent([ref_grid, gs], [str(ref_grid), f"{bname}@{fpath}"])
            except (rasterio.errors.RasterioIOError, ValueError):
                continue

    if ref_grid is None:
        raise ValueError("No valid raster found to determine grid")
    return ref_grid


# 模块级日志函数
import os
from engine.utils import log as _log
