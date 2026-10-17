"""
预处理底座 — 所有数据源共享的预处理流水线

功能:
  - Sentinel-2 L2A SCL 云/云影/雪掩膜
  - Sentinel-2 比例因子 /10000
  - Landsat QA_PIXEL 和 QA_RADSAT
  - 传感器饱和检测
  - 无云中位数合成
  - 统一入口 preprocess_bands()

用法:
    from engine.preprocessing import preprocess_bands
    bands, mask = preprocess_bands(band_paths, data_source="sentinel2")
"""

from __future__ import annotations
import numpy as np
import rasterio
from typing import Optional


# ============================================================================
#  Sentinel-2 SCL 掩膜
# ============================================================================

# SCL (Scene Classification Layer) 值定义
SCL_VALUES = {
    0: "无数据", 1: "饱和/有缺陷", 2: "暗区阴影",
    3: "云阴影", 4: "植被", 5: "裸土",
    6: "水体", 7: "低云 (<=50%)", 8: "中云 (<=50%)",
    9: "高云 (>50%)", 10: "卷云", 11: "雪/冰",
}

# 需要屏蔽的 SCL 值（云、云影、雪、饱和）
SCL_CLOUD_MASK = {1, 3, 7, 8, 9, 10, 11}
# 高质量像元（植被、裸土、水体）
SCL_CLEAR = {4, 5, 6}


def apply_scl_mask(
    scl_path: str,
    mask_clouds: bool = True,
    mask_shadow: bool = True,
    mask_snow: bool = True,
    mask_saturated: bool = True,
) -> np.ndarray:
    """从 SCL 文件生成质量掩膜。

    Args:
        scl_path: SCL GeoTIFF 路径
        mask_clouds: 屏蔽低/中/高云
        mask_shadow: 屏蔽云阴影
        mask_snow: 屏蔽雪/冰
        mask_saturated: 屏蔽饱和/有缺陷

    Returns:
        bool 数组: True=有效像元
    """
    with rasterio.open(scl_path) as src:
        scl = src.read(1)

    bad = set()
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

def apply_landsat_qa(
    qa_path: str,
    mask_cloud: bool = True,
    mask_shadow: bool = True,
    mask_snow: bool = True,
    mask_cirrus: bool = True,
) -> np.ndarray:
    """从 Landsat QA_PIXEL 生成质量掩膜。

    Landsat Collection 2 QA_PIXEL 位标志:
      Bit 0: Fill
      Bit 1: Dilated Cloud
      Bit 2: Cirrus
      Bit 3: Cloud
      Bit 4: Cloud Shadow
      Bit 5: Snow
      Bit 6-7: Clear

    Args:
        qa_path: QA_PIXEL GeoTIFF 路径
        mask_cloud: 屏蔽云像元
        mask_shadow: 屏蔽云阴影
        mask_snow: 屏蔽雪
        mask_cirrus: 屏蔽卷云

    Returns:
        bool 数组: True=有效像元
    """
    with rasterio.open(qa_path) as src:
        qa = src.read(1)

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
#  无云合成
# ============================================================================

def median_composite(band_stacks: dict[str, list[np.ndarray]],
                     valid_mask: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """对多景影像做中位数合成。

    Args:
        band_stacks: {波段名: [数组1, 数组2, ...]}
        valid_mask: 可选，只在此区域内合成

    Returns:
        {波段名: 合成数组}
    """
    composite = {}
    for name, arrays in band_stacks.items():
        if not arrays:
            continue
        stack = np.stack(arrays, axis=0)
        if valid_mask is not None:
            # 对每个波段，在无效位置填充 NaN
            stack[:, ~valid_mask] = np.nan
        composite[name] = np.nanmedian(stack, axis=0)
    return composite


def best_available_composite(
    band_stacks: dict[str, list[np.ndarray]],
    quality_masks: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """最佳像元合成：优先使用质量最高的观测。

    策略: 按质量掩膜累计有效观测，取第一个有效像元。
    若多景均有效，取第一景（按输入顺序，期望已按质量排序）。

    Args:
        band_stacks: {波段名: [数组1, ...]}
        quality_masks: 每景的质量掩膜，True=有效

    Returns:
        {波段名: 合成数组}
    """
    if not quality_masks:
        raise ValueError("quality_masks required")

    ref_shape = quality_masks[0].shape
    n_scenes = len(quality_masks)

    composite = {}
    for name, arrays in band_stacks.items():
        if not arrays:
            continue
        result = np.full(ref_shape, np.nan, dtype=np.float32)
        # 从最佳到最差，取第一个有效值
        for i in range(n_scenes):
            if i >= len(arrays):
                break
            mask = quality_masks[i]
            if mask is None:
                continue
            result[mask] = np.where(
                np.isnan(result[mask]),
                arrays[i].astype(np.float32)[mask],
                result[mask],
            )
        composite[name] = result
    return composite


# ============================================================================
#  统一预处理入口
# ============================================================================

def preprocess_bands(
    band_paths: dict[str, str],
    data_source: str = "sentinel2",
    apply_scale: bool = True,
    cloud_mask: bool = True,
    scl_path: Optional[str] = None,
    qa_path: Optional[str] = None,
    target_shape: Optional[tuple[int, int]] = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """统一预处理入口：加载、缩放、掩膜。

    Args:
        band_paths: {波段名: GeoTIFF 路径}
        data_source: 数据源名称
        apply_scale: 是否应用比例因子
        cloud_mask: 是否应用云掩膜
        scl_path: Sentinel-2 SCL 文件路径（可选）
        qa_path: Landsat QA_PIXEL 文件路径（可选）
        target_shape: 目标数组尺寸 (H, W)

    Returns:
        (band_arrays, valid_mask)
        band_arrays: {波段名: np.ndarray} 预处理后的波段
        valid_mask: bool 数组，True=有效像元
    """
    # 1. 加载波段
    bands = {}
    first_meta = None
    for name, path in band_paths.items():
        with rasterio.open(path) as src:
            if first_meta is None:
                first_meta = src.meta
            bands[name] = src.read(1).astype(np.float32)

    if not bands:
        raise ValueError("No bands loaded")

    ref_shape = list(bands.values())[0].shape

    # 2. 传感器饱和检测
    valid = detect_saturation(bands, data_source)

    # 3. 云掩膜
    if cloud_mask:
        if data_source in ("sentinel2",) and scl_path and os.path.exists(scl_path):
            scl_valid = apply_scl_mask(scl_path)
            if scl_valid.shape == ref_shape:
                valid &= scl_valid
        elif data_source in ("landsat8", "landsat9") and qa_path and os.path.exists(qa_path):
            qa_valid = apply_landsat_qa(qa_path)
            if qa_valid.shape == ref_shape:
                valid &= qa_valid

    # 4. 比例因子
    if apply_scale:
        bands = apply_scale_factors(bands, data_source, valid)

    # 5. 在无效位置填 0
    for name in bands:
        bands[name][~valid] = 0.0

    return bands, valid


import os  # needed for path checks in preprocess_bands
