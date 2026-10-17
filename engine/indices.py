"""
遥感解译助手 v3 — 遥感指数计算模块

支持指数:
  NDVI  - 归一化植被指数 (植被覆盖变化)
  NDWI  - 归一化水体指数 (水体面积变化)
  MNDWI - 改进归一化水体指数 (建筑区水体提取)
  NBR   - 归一化火烧指数 (火烧迹地检测)
  EVI   - 增强型植被指数 (高植被区，抗大气)
  SAVI  - 土壤调节植被指数 (低植被覆盖区)
"""

import numpy as np


# ============================================================================
#  安全除法
# ============================================================================

def _safe_divide(numerator, denominator, fallback=0.001):
    """安全除法，避免除零错误。"""
    denom = denominator.copy()
    denom[denom == 0] = fallback
    return numerator / denom


# ============================================================================
#  指数计算函数
# ============================================================================

def ndvi(nir, red):
    """归一化植被指数 (Normalized Difference Vegetation Index)。

    公式: (NIR - Red) / (NIR + Red)
    范围: [-1, 1]，植被 > 0.2
    """
    n = nir.astype(np.float32)
    r = red.astype(np.float32)
    return _safe_divide(n - r, n + r)


def ndwi(green, nir):
    """归一化水体指数 (Normalized Difference Water Index)。

    公式: (Green - NIR) / (Green + NIR)
    范围: [-1, 1]，水体 > 0
    """
    g = green.astype(np.float32)
    n = nir.astype(np.float32)
    return _safe_divide(g - n, g + n)


def mndwi(green, swir1):
    """改进归一化水体指数 (Modified NDWI)。

    使用 SWIR1 代替 NIR，能更好区分建筑和水体。
    公式: (Green - SWIR1) / (Green + SWIR1)
    范围: [-1, 1]，水体 > 0
    """
    g = green.astype(np.float32)
    s = swir1.astype(np.float32)
    return _safe_divide(g - s, g + s)


def nbr(nir, swir2):
    """归一化火烧指数 (Normalized Burn Ratio)。

    公式: (NIR - SWIR2) / (NIR + SWIR2)
    范围: [-1, 1]，火烧迹地 < 0.1
    """
    n = nir.astype(np.float32)
    s = swir2.astype(np.float32)
    return _safe_divide(n - s, n + s)


def evi(nir, red, blue):
    """增强型植被指数 (Enhanced Vegetation Index)。

    公式: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    范围: [-1, 1]，抗大气干扰，适用于高植被覆盖区。
    """
    n = nir.astype(np.float32)
    r = red.astype(np.float32)
    b = blue.astype(np.float32)
    return _safe_divide(2.5 * (n - r), n + 6.0 * r - 7.5 * b + 1.0)


def savi(nir, red, L=0.5):
    """土壤调节植被指数 (Soil-Adjusted Vegetation Index)。

    公式: (NIR - Red) * (1 + L) / (NIR + Red + L)
    参数:
        L: 土壤调节因子 (0-1)。
           L=0.5 通用，L=0.25 高覆盖，L=1 低覆盖。
    """
    n = nir.astype(np.float32)
    r = red.astype(np.float32)
    return _safe_divide((n - r) * (1.0 + L), n + r + L)


def ndbi(swir1, nir):
    """归一化建筑指数 (Normalized Difference Built-up Index)。

    公式: (SWIR1 - NIR) / (SWIR1 + NIR)
    范围: [-1, 1]，建筑 > 0（SWIR1 反射强于 NIR），
          植被 < 0（NIR 反射强于 SWIR1），水体 < 0。
    用途: 与 NDVI + MNDWI 构成三维特征空间，区分建筑/裸土。
    """
    s = swir1.astype(np.float32)
    n = nir.astype(np.float32)
    return _safe_divide(s - n, s + n)


# ============================================================================
#  指数注册表
# ============================================================================

# 指数名 -> {函数, 参数名列表}
INDEX_FUNCTIONS = {
    "NDVI":  {"func": ndvi,  "params": ["nir", "red"]},
    "NDWI":  {"func": ndwi,  "params": ["green", "nir"]},
    "MNDWI": {"func": mndwi, "params": ["green", "swir1"]},
    "NBR":   {"func": nbr,   "params": ["nir", "swir2"]},
    "EVI":   {"func": evi,   "params": ["nir", "red", "blue"]},
    "SAVI":  {"func": savi,  "params": ["nir", "red"]},
    "NDBI":  {"func": ndbi,  "params": ["swir1", "nir"]},
}

# 各指数的变化检测阈值
INDEX_THRESHOLDS = {
    "NDVI":  {"change": 0.15, "severe": 0.25},
    "NDWI":  {"change": 0.10, "severe": 0.20},
    "MNDWI": {"change": 0.10, "severe": 0.20},
    "NBR":   {"change": 0.15, "severe": 0.30},
    "EVI":   {"change": 0.10, "severe": 0.20},
    "SAVI":  {"change": 0.12, "severe": 0.22},
}

# 各指数的变化类别名称
INDEX_CLASS_NAMES = {
    "NDVI":  {-3: "严重退化", -2: "中度退化", 1: "稳定", 2: "中度恢复", 3: "强烈恢复"},
    "NDWI":  {-3: "明显减少", -2: "中度减少", 1: "稳定", 2: "中度增加", 3: "明显增加"},
    "MNDWI": {-3: "明显减少", -2: "中度减少", 1: "稳定", 2: "中度增加", 3: "明显增加"},
    "NBR":   {-3: "严重火烧", -2: "中度火烧", 1: "稳定", 2: "中度恢复", 3: "完全恢复"},
    "EVI":   {-3: "严重退化", -2: "中度退化", 1: "稳定", 2: "中度恢复", 3: "强烈恢复"},
    "SAVI":  {-3: "严重退化", -2: "中度退化", 1: "稳定", 2: "中度恢复", 3: "强烈恢复"},
}

# 各指数的分类颜色
INDEX_COLORS = {
    "NDVI":  {-3: "#b40000", -2: "#ff6400", 1: "#282828", 2: "#00b450", 3: "#005000"},
    "NDWI":  {-3: "#8B4513", -2: "#CD853F", 1: "#282828", 2: "#4169E1", 3: "#0000CD"},
    "MNDWI": {-3: "#8B4513", -2: "#CD853F", 1: "#282828", 2: "#4169E1", 3: "#0000CD"},
    "NBR":   {-3: "#8B0000", -2: "#FF4500", 1: "#282828", 2: "#90EE90", 3: "#006400"},
    "EVI":   {-3: "#b40000", -2: "#ff6400", 1: "#282828", 2: "#00b450", 3: "#005000"},
    "SAVI":  {-3: "#b40000", -2: "#ff6400", 1: "#282828", 2: "#00b450", 3: "#005000"},
}


# ============================================================================
#  统一调度入口
# ============================================================================

def compute_index(index_type, bands, **kwargs):
    """统一指数计算入口。

    Args:
        index_type: 指数类型 (NDVI/NDWI/MNDWI/NBR/EVI/SAVI)，大小写不敏感。
        bands: {波段名: numpy数组}，例如 {'nir': arr, 'red': arr}。
        **kwargs: 额外参数（如 SAVI 的 L）。

    Returns:
        np.ndarray: 指数值数组。

    Raises:
        ValueError: 未知指数类型。
        KeyError: 缺少所需波段。
    """
    index_type = index_type.upper()
    if index_type not in INDEX_FUNCTIONS:
        valid = list(INDEX_FUNCTIONS.keys())
        raise ValueError(f"Unknown index: {index_type}. Available: {valid}")

    entry = INDEX_FUNCTIONS[index_type]
    args = [bands[p] for p in entry["params"]]
    return entry["func"](*args, **kwargs)


def compute_all_indices(bands):
    """计算所有可用指数（用于多指数对比分析）。

    Args:
        bands: {波段名: numpy数组} 字典。

    Returns:
        dict[str, np.ndarray]: {指数名: 指数值}。
    """
    results = {}
    for name in INDEX_FUNCTIONS:
        try:
            results[name] = compute_index(name, bands)
        except KeyError as e:
            from engine.utils import log
            log(f"  Warning: {name} skipped, missing band {e}")
    return results
