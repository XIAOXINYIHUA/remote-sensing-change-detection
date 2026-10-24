"""
遥感解译助手 v3 — 共用工具函数

包含:
  - 中文字体自动配置
  - 统一日志输出
  - 边界文件加载
  - 边界像素掩膜生成
  - 指数-波段需求映射
"""

import sys
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from rasterio.features import geometry_mask

warnings.filterwarnings("ignore")


# ============================================================================
#  中文字体配置
# ============================================================================

def setup_matplotlib_font():
    """自动检测并配置 matplotlib 中文字体，解决报告图片乱码。

    按优先级搜索 Windows / macOS / Linux 常见中文字体，
    找到后设置全局 rcParams。

    Returns:
        str: 匹配到的字体名称，未找到时返回 None。
    """
    candidates = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
        'Noto Sans CJK SC', 'Noto Sans SC', 'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei', 'AR PL UMing CN', 'AR PL UKai CN',
        'Source Han Sans SC', 'PingFang SC', 'Heiti SC', 'STHeiti',
        'STSong', 'STKaiti', 'STFangsong',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            log(f"  Matplotlib font: {font}")
            return font

    # 回退：按关键词匹配
    for f in fm.fontManager.ttflist:
        name_lower = f.name.lower()
        if any(kw in name_lower for kw in
               ['cjk', 'chinese', 'cn', 'sc', 'han', 'hei',
                'ming', 'song', 'kai', 'fang']):
            plt.rcParams['font.sans-serif'] = [f.name, 'DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            log(f"  Matplotlib font (fallback): {f.name}")
            return f.name

    log("  WARNING: No Chinese font found, report may show garbled text")
    return None


# ============================================================================
#  日志工具
# ============================================================================

def log(msg, flush=True):
    """统一日志输出，处理 Windows 控制台 GBK 编码问题。

    Args:
        msg: 日志消息字符串。
        flush: 是否立即刷新输出缓冲。
    """
    try:
        print(msg, flush=flush)
    except UnicodeEncodeError:
        safe = msg.encode(
            sys.stdout.encoding or 'utf-8', errors='replace'
        ).decode(sys.stdout.encoding or 'utf-8', errors='replace')
        print(safe, flush=flush)


# ============================================================================
#  边界工具
# ============================================================================

def get_boundary(boundary_file=None):
    """加载 GeoJSON 边界文件，返回 Shapely 几何对象。

    支持多要素 FeatureCollection（Union 所有要素）。
    """
    import config
    from shapely.ops import unary_union
    path = boundary_file or config.BOUNDARY_FILE
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    if gj.get("type") == "FeatureCollection":
        geoms = []
        for feat in gj["features"]:
            try:
                g = shape(feat["geometry"])
                if g and not g.is_empty:
                    geoms.append(g)
            except Exception:
                continue
        if not geoms:
            raise ValueError("FeatureCollection 中无有效几何")
        return unary_union(geoms) if len(geoms) > 1 else geoms[0]
    return shape(gj)


def boundary_pixel_mask(boundary_wgs84, meta, height, width):
    """生成边界范围内的像素掩膜（True = 在边界内）。

    将 WGS84 边界变换到影像投影坐标系，使用 rasterio 的
    geometry_mask 生成 bool 数组。

    Args:
        boundary_wgs84: WGS84 坐标系下的 Shapely 几何对象。
        meta: rasterio 元数据字典（含 crs 和 transform）。
        height: 输出网格行数。
        width: 输出网格列数。

    Returns:
        np.ndarray[bool]: 形状 (height, width)，True 表示像素在边界内。
    """
    t = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    boundary_proj = shp_transform(lambda x, y: t.transform(x, y), boundary_wgs84)
    mask_arr = geometry_mask(
        [boundary_proj.__geo_interface__],
        transform=meta["transform"],
        invert=True,
        out_shape=(height, width),
    )
    return mask_arr


def geodesic_area_km2(geom_wgs84):
    """WGS84 几何在椭球面上的真实面积 (km²)，支持多面与内环。

    不能用「度² × 111.32²」：经度方向 1° 的长度随 cos(纬度) 缩短，
    该近似在 25°N 就会高估约 11%。
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    polygons = getattr(geom_wgs84, "geoms", [geom_wgs84])
    total = 0.0
    for poly in polygons:
        if poly.geom_type != "Polygon":
            total += geodesic_area_km2(poly) * 1e6 if hasattr(poly, "geoms") else 0.0
            continue
        total += abs(geod.polygon_area_perimeter(*poly.exterior.xy)[0])
        total -= sum(abs(geod.polygon_area_perimeter(*ring.xy)[0])
                     for ring in poly.interiors)
    return total / 1e6


# ============================================================================
#  指数-波段需求映射
# ============================================================================

# 每个指数需要的波段集合
INDEX_BAND_REQUIREMENTS = {
    "NDVI":  {"red", "nir"},
    "NDWI":  {"green", "nir"},
    "MNDWI": {"green", "swir1"},
    "NBR":   {"nir", "swir2"},
    "EVI":   {"blue", "red", "nir"},
    "SAVI":  {"red", "nir"},
    "NDBI":  {"swir1", "nir"},
    "ALL":   {"blue", "green", "red", "nir", "swir1", "swir2"},
}


def get_needed_bands(index_type):
    """根据遥感指数类型返回需要的波段名称列表。

    Args:
        index_type: 指数类型字符串（如 "NDVI", "NDWI", "NBR" 等，大小写不敏感）。

    Returns:
        list[str]: 需要的波段名列表（如 ['red', 'nir']）。

    Raises:
        ValueError: 未知的指数类型。
    """
    key = index_type.upper()
    if key not in INDEX_BAND_REQUIREMENTS:
        valid = list(INDEX_BAND_REQUIREMENTS.keys())
        raise ValueError(f"Unknown index type: {index_type}. Available: {valid}")
    return list(INDEX_BAND_REQUIREMENTS[key])


# 在模块加载时自动配置字体
setup_matplotlib_font()
