"""
预处理底座 — 所有数据源共享的预处理流水线

功能:
  - Sentinel-2 L2A SCL 云/云影/雪掩膜
  - Sentinel-2 比例因子 /10000
  - Landsat QA_PIXEL 和 QA_RADSAT
  - 传感器饱和检测
  - 无云中位数合成
  - Sentinel-2 处理基线 04.00 偏移归一化
  - 统一入口 preprocess_arrays()（输入必须已对齐到同一网格）

用法:
    from engine.preprocessing import preprocess_arrays, quality_valid_on_grid
    qvalid = quality_valid_on_grid(scl_path, "sentinel2", target_grid)
    bands, mask = preprocess_arrays(aligned_bands, "sentinel2", qvalid)
"""

from __future__ import annotations
import os
from datetime import date
import numpy as np
import rasterio
from rasterio.warp import Resampling
from typing import Optional


# 预处理逻辑版本 — 变化后旧缓存自动失效
# v3.2: 下载时归一化 S2 基线偏移；云掩膜在对齐后的网格上执行
PREPROCESSING_VERSION = "v3.2"


# ============================================================================
#  Sentinel-2 处理基线 04.00 偏移
# ============================================================================

# 2022-01-25 起 ESA 基线 04.00 的 L2A 产品 DN 带 +1000 偏移 (BOA_ADD_OFFSET=-1000)
S2_BASELINE_04_CUTOFF = date(2022, 1, 25)
S2_BOA_ADD_OFFSET = 1000


def s2_boa_offset(item) -> int:
    """返回该 S2 L2A 景需扣除的 DN 偏移（基线 >= 04.00 为 1000，否则 0）。

    优先读 STAC 的 s2:processing_baseline（重处理的历史数据也带偏移），
    缺失时按成像日期判断。
    """
    baseline = (getattr(item, "properties", None) or {}).get("s2:processing_baseline")
    if baseline:
        try:
            return S2_BOA_ADD_OFFSET if float(baseline) >= 4.0 else 0
        except (TypeError, ValueError):
            pass
    dt = getattr(item, "datetime", None)
    if dt is None:
        return 0
    return S2_BOA_ADD_OFFSET if dt.date() >= S2_BASELINE_04_CUTOFF else 0


def harmonize_s2_dn(arr: np.ndarray, offset: int) -> np.ndarray:
    """把新基线 DN 平移回旧基线尺度：DN - offset。

    0 仍表示无数据；扣除后 <= 0 的暗像元钳制为 1（旧基线的最小有效值），
    避免被当成无数据丢掉。
    """
    if not offset:
        return arr
    out = arr.astype(np.int32) - offset
    np.maximum(out, 1, out=out)
    out[arr == 0] = 0
    return out.astype(arr.dtype)


# ============================================================================
#  Sentinel-2 SCL 掩膜
# ============================================================================


def scl_valid_mask(
    scl: np.ndarray,
    mask_clouds: bool = True,
    mask_shadow: bool = True,
    mask_snow: bool = True,
    mask_saturated: bool = True,
) -> np.ndarray:
    """从 SCL 数组生成质量掩膜（True=有效）。SCL=0 无数据总是无效。"""
    bad = {0}
    if mask_clouds:
        bad |= {7, 8, 9}  # 低/中/高云
    if mask_shadow:
        bad |= {3}  # 云阴影
        bad |= {2}  # 暗区阴影
    if mask_snow:
        bad |= {11}  # 雪/冰
    if mask_saturated:
        bad |= {1}  # 饱和/有缺陷

    valid = np.ones(scl.shape, dtype=bool)
    for v in bad:
        valid &= (scl != v)

    return valid


# ============================================================================
#  比例因子
# ============================================================================

def apply_scale_factors(
    bands: dict[str, np.ndarray],
    data_source: str,
    mask: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """应用传感器比例因子，将 DN 转换为反射率。

    Sentinel-2 L2A: BOA 反射率 = DN / 10000
    Landsat 8/9 C2 L2: 反射率 = DN * 0.0000275 - 0.2

    Args:
        bands: {名称: 数组}
        data_source: "sentinel2" | "landsat8" | "landsat9"
        mask: 可选，只处理掩膜内的像元

    Returns:
        缩放后的波段字典
    """
    result = {}
    for name, arr in bands.items():
        arr_f = arr.astype(np.float32)
        if data_source in ("sentinel2",):
            arr_f = arr_f / 10000.0
        elif data_source in ("landsat8", "landsat9"):
            arr_f = arr_f * 0.0000275 - 0.2
        # 裁剪到合理范围
        arr_f = np.clip(arr_f, -0.2, 1.0)
        result[name] = arr_f
    return result


# ============================================================================
#  Landsat QA_PIXEL 掩膜
# ============================================================================


def landsat_qa_valid_mask(
    qa: np.ndarray,
    mask_cloud: bool = True,
    mask_shadow: bool = True,
    mask_snow: bool = True,
    mask_cirrus: bool = True,
) -> np.ndarray:
    """从 QA_PIXEL 数组生成质量掩膜（True=有效）。"""
    qa = qa.astype(np.uint16)
    bit = lambda x: (qa >> x) & 1

    bad = np.zeros(qa.shape, dtype=bool)
    if mask_cloud:
        bad |= bit(3).astype(bool) | bit(1).astype(bool)
    if mask_shadow:
        bad |= bit(4).astype(bool)
    if mask_snow:
        bad |= bit(5).astype(bool)
    if mask_cirrus:
        bad |= bit(2).astype(bool)
    # Bit 0 = Fill (无数据)
    bad |= bit(0).astype(bool)

    return ~bad


# ============================================================================
#  传感器饱和检测
# ============================================================================

def detect_saturation(
    bands: dict[str, np.ndarray],
    data_source: str,
) -> np.ndarray:
    """检测传感器饱和像元。

    Sentinel-2 L2A: DN=0 或 DN=10000 通常表示饱和/超出范围
    Landsat C2 L2: 结合 QA_RADSAT

    Returns:
        bool 数组: True=有效（未饱和）
    """
    if not bands:
        return np.ones((1, 1), dtype=bool)

    ref = list(bands.values())[0]
    valid = np.ones(ref.shape, dtype=bool)

    if data_source in ("sentinel2",):
        # S2: DN=0 或接近 10000 可能饱和
        for arr in bands.values():
            arr_f = arr.astype(np.float32)
            valid &= (arr_f > 0) & (arr_f < 9999)

    elif data_source in ("landsat8", "landsat9"):
        # Landsat: DN=0 或 DN=65535 无效
        for arr in bands.values():
            valid &= (arr > 0) & (arr < 65535)

    return valid


# ============================================================================
#  统一预处理入口
# ============================================================================

def quality_valid_on_grid(quality_path: str, data_source: str, target_grid) -> np.ndarray:
    """把质量波段 (S2 SCL / Landsat QA_PIXEL) 最近邻重采样到目标网格并生成掩膜。

    S2 的 SCL 为 20 m，而 B4/B8 为 10 m，必须先对齐再与波段掩膜相与。

    Returns:
        bool 数组 (H, W): True=有效像元
    """
    from engine.grid import GridSpec, reproject_to_grid

    is_s2 = data_source == "sentinel2"
    if not is_s2 and data_source not in ("landsat8", "landsat9"):
        raise ValueError(f"No quality band defined for {data_source}")

    # 网格外填充为「无数据」：SCL=0，QA_PIXEL=1 (Fill 位)
    fill = 0 if is_s2 else 1
    with rasterio.open(quality_path) as src:
        arr = src.read(1)
        if GridSpec.from_src(src) != target_grid:
            arr = reproject_to_grid((arr, src.meta), target_grid,
                                    resampling=Resampling.nearest, nodata=fill)
    if is_s2:
        return scl_valid_mask(arr.astype(np.uint8))
    return landsat_qa_valid_mask(arr.astype(np.uint16))


def preprocess_arrays(
    bands: dict[str, np.ndarray],
    data_source: str = "sentinel2",
    quality_valid: Optional[np.ndarray] = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """统一预处理入口：饱和检测 + 云掩膜 + 比例因子。

    Args:
        bands: {波段名: 原始 DN 数组}，必须已对齐到同一网格
        data_source: 数据源名称
        quality_valid: quality_valid_on_grid() 的结果（None=不做云掩膜）

    Returns:
        (band_arrays, valid_mask)
        band_arrays: {波段名: 反射率数组}，无效像元填 0
        valid_mask: bool 数组，True=有效像元
    """
    if not bands:
        raise ValueError("No bands loaded")
    shapes = {arr.shape for arr in bands.values()}
    if len(shapes) != 1:
        raise ValueError(f"Bands must share one grid, got shapes {shapes}")

    valid = detect_saturation(bands, data_source)
    if quality_valid is not None:
        if quality_valid.shape != valid.shape:
            raise ValueError(
                f"Quality mask {quality_valid.shape} != band grid {valid.shape}")
        valid &= quality_valid

    bands = apply_scale_factors(bands, data_source)
    for name in bands:
        bands[name][~valid] = 0.0
    return bands, valid


def load_period(
    band_paths: dict[str, str],
    quality_path: Optional[str],
    data_source: str,
    target_grid,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """读取一期影像到目标网格并完成预处理（定标 + 云掩膜）。

    已在分析网格上的文件直接读取；其余按双线性重投影。

    Returns:
        (bands, valid, info)
        bands: {波段名: 反射率}，无效像元填 0
        valid: bool 数组，True=有效像元
        info: {"cloud_mask": 是否做了云掩膜, "cloud_pct": 有数据像元中被云掩膜的比例(%)}
    """
    from engine.grid import GridSpec, reproject_to_grid

    raw = {}
    for name, path in band_paths.items():
        if not path or not os.path.exists(path):
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if GridSpec.from_src(src) != target_grid:
                arr = reproject_to_grid((arr, src.meta), target_grid)
        raw[name] = arr

    qvalid = None
    if quality_path and os.path.exists(quality_path):
        qvalid = quality_valid_on_grid(quality_path, data_source, target_grid)
    bands, valid = preprocess_arrays(raw, data_source, qvalid)

    cloud_pct = None
    if qvalid is not None:
        has_data = np.logical_and.reduce([a > 0 for a in raw.values()])
        cloud_pct = round(
            float((has_data & ~qvalid).sum()) * 100 / max(int(has_data.sum()), 1), 2)
    return bands, valid, {"cloud_mask": qvalid is not None, "cloud_pct": cloud_pct}
