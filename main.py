"""
遥感解译助手 v3 — 核心引擎入口

向后兼容 v2 API，内部委托给 engine 包。
可直接运行: python main.py
或作为模块导入: from main import run_pipeline, run_pipeline_v3
"""

import os
import sys
import json
import warnings
import time

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.features import geometry_mask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import shape, box as sbox, Point, Polygon
from shapely.ops import transform as shp_transform
from shapely.prepared import prep
from pyproj import Transformer
import cv2
import pandas as pd

import config

# 从 engine 包导入核心功能
from engine.utils import (
    log, get_boundary, boundary_pixel_mask,
    setup_matplotlib_font, get_needed_bands, INDEX_BAND_REQUIREMENTS,
)
from engine.core import (
    calc_ndvi,
    run_change_detection_v2,
    run_change_detection_generic,
    generate_sample_points_v2,
    generate_report_v2,
    run_pipeline,
    run_pipeline_v3,
)
from engine.indices import (
    compute_index, compute_all_indices,
    INDEX_FUNCTIONS, INDEX_THRESHOLDS,
    INDEX_CLASS_NAMES, INDEX_COLORS,
)
from engine.data_sources import (
    get_data_source,
    Sentinel2DataSource, LandsatDataSource, Sentinel1DataSource,
)
from engine.ml_detector import (
    run_ml_change_detection,
    RandomForestChangeDetector,
)
from engine.timeseries import TimeSeriesAnalyzer
from engine.report_generator import PDFReportGenerator, generate_report_pdf

warnings.filterwarnings("ignore")

# 确保中文字体配置
setup_matplotlib_font()


# ============================================================================
#  向后兼容导出 (v2 API)
# ============================================================================

def find_best_scenes(boundary, year):
    """[v2 兼容] 搜索最佳 Sentinel-2 场景。

    内部使用 Sentinel2DataSource.search_scenes()。
    """
    source = get_data_source("sentinel2")
    return source.search_scenes(
        boundary, year, config.MONTH_START,
        config.MONTH_END, config.MAX_CLOUD_PCT)


def download_and_mosaic(boundary, year, out_dir):
    """[v2 兼容] 下载并拼接 Sentinel-2 影像。

    内部使用 Sentinel2DataSource.download_bands()。
    """
    source = get_data_source("sentinel2")
    scenes = source.search_scenes(
        boundary, year, config.MONTH_START,
        config.MONTH_END, config.MAX_CLOUD_PCT)
    if not scenes:
        log(f"  {year}: no suitable scenes found")
        return None, None

    log(f"  {year}: {len(scenes)} tile(s)")
    for s in scenes:
        item = s["item"]
        log(f"    {s['tile']}: cloud={s['cloud']:.0f}% "
            f"cov={s['coverage']*100:.1f}% "
            f"{item.datetime.date()}")

    bands = source.download_bands(scenes, out_dir, ['red', 'nir'])
    b4_path = bands.get('red')
    b8_path = bands.get('nir')
    return b4_path, b8_path


# ============================================================================
#  main() — 命令行入口
# ============================================================================

def main():
    """命令行入口：运行完整的 v3 变化检测流程。"""
    print("=" * 60)
    print("  遥感解译助手 v3 — 多源遥感变化检测")
    print(f"  区域: {config.REGION_NAME}")
    print(f"  时相: {config.YEAR1} vs {config.YEAR2}")
    print(f"  数据源: {getattr(config, 'DATA_SOURCE', 'sentinel2')}")
    print(f"  指数: {getattr(config, 'INDEX_TYPE', 'NDVI')}")
    print(f"  模式: {config.DETECTION_MODE}")
    print("=" * 60)

    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")
    out_dir = os.path.join(base, config.OUTPUT_DIR)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    boundary = get_boundary()
    print(f"  Boundary loaded")

    # 使用 v3 流水线
    data_source = getattr(config, 'DATA_SOURCE', 'sentinel2')
    index_type = getattr(config, 'INDEX_TYPE', 'NDVI')

    print(f"\n[1/3] Downloading imagery ({data_source})...")
    # v3 流水线内部处理下载
    results = run_pipeline_v3(
        data_source_name=data_source,
        data_dir=data_dir,
        out_dir=out_dir,
        boundary=boundary,
        index_type=index_type,
        detection_mode=config.DETECTION_MODE,
    )

    if results.get("error"):
        print(f"  Error: {results['error']}")
        return

    # 打印统计
    stats = results.get("stats", {})
    if config.DETECTION_MODE == "multi":
        print(f"\n  ==== 变化统计 ({index_type}) ====")
        for cls_val in [-3, -2, 1, 2, 3]:
            s = stats.get(cls_val, {})
            if s:
                print(f"  {s['name']:8s}: {s['area_ha']:>8.0f} ha  "
                      f"({s['percent']:>5.1f}%)")
        total = stats.get("changed_total", {})
        if total:
            print(f"  {'变化合计':8s}: {total['area_ha']:>8.0f} ha  "
                  f"({total['percent']:>5.1f}%)")
    else:
        print(f"  Changed: {stats.get('changed_area_ha', 0):.0f} ha  "
              f"({stats.get('changed_percent', 0):.1f}%)")

    print(f"\n  Sampling: {results.get('num_sample_points', 0)} points")
    timing = results.get("timing", {})
    print(f"  Total time: {timing.get('total', 0):.0f}s")

    print(f"\n{'=' * 60}")
    print(f"  Done! Output: {out_dir}")
    print(f"{'=' * 60}")


# ============================================================================
#  __all__ 导出
# ============================================================================

__all__ = [
    # v2 API (backward compat)
    "find_best_scenes",
    "download_and_mosaic",
    "calc_ndvi",
    "get_boundary",
    "boundary_pixel_mask",
    "run_change_detection_v2",
    "generate_sample_points_v2",
    "generate_report_v2",
    "run_pipeline",
    # v3 API
    "run_pipeline_v3",
    "run_change_detection_generic",
    "run_ml_change_detection",
    "TimeSeriesAnalyzer",
    "compute_index",
    "compute_all_indices",
    "get_data_source",
    "get_needed_bands",
]


if __name__ == "__main__":
    main()
