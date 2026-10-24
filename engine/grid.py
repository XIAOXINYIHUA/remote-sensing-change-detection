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
from numpy.typing import NDArray
import rasterio
from rasterio import warp
from rasterio.crs import CRS
from rasterio.transform import Affine
from pyproj import Transformer
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


def utm_crs_for(boundary_wgs84) -> CRS:
    """返回 AOI 质心所在的 UTM 带 (WGS84 / UTM)。"""
    c = boundary_wgs84.centroid
    zone = min(60, max(1, int((c.x + 180) // 6) + 1))
    return CRS.from_epsg((32600 if c.y >= 0 else 32700) + zone)


def aoi_grid(
    boundary_wgs84,
    resolution: float,
    origin_offset: float = 0.0,
    crs: CRS | None = None,
    pad_pixels: int = 2,
) -> GridSpec:
    """由 AOI 确定的统一分析网格。

    同一 AOI + 分辨率总是得到完全相同的网格（与选中的瓦片无关），
    因此不同年份、不同波段无需再次重投影即可逐像元比较。
    网格角点吸附到 origin_offset + k*resolution，与传感器原生像元格网对齐
    （Sentinel-2: 0，Landsat C2: 15 m），同带瓦片读取时不做插值。

    Args:
        boundary_wgs84: WGS84 Shapely 几何
        resolution: 像元大小 (m)
        origin_offset: 格网原点偏移 (m)
        crs: 目标投影，None=AOI 质心所在 UTM 带
        pad_pixels: 边界外扩像元数
    """
    import math
    crs = crs or utm_crs_for(boundary_wgs84)
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    bnd_proj = shp_transform(lambda x, y: t.transform(x, y), boundary_wgs84)
    minx, miny, maxx, maxy = bnd_proj.bounds

    pad = pad_pixels * resolution
    snap_down = lambda v: math.floor((v - origin_offset) / resolution) * resolution + origin_offset
    snap_up = lambda v: math.ceil((v - origin_offset) / resolution) * resolution + origin_offset
    minx, miny = snap_down(minx - pad), snap_down(miny - pad)
    maxx, maxy = snap_up(maxx + pad), snap_up(maxy + pad)

    return GridSpec(
        crs=crs,
        transform=Affine(resolution, 0.0, minx, 0.0, -resolution, maxy),
        width=max(1, int(round((maxx - minx) / resolution))),
        height=max(1, int(round((maxy - miny) / resolution))),
        nodata=0.0,
    )
