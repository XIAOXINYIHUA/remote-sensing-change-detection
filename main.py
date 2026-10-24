"""
遥感解译助手 v3 — 核心引擎入口

向后兼容 v2 API，内部委托给 engine 包。
可直接运行: python main.py
或作为模块导入: from main import run_pipeline, run_pipeline_v3
"""

import argparse
import os
import sys
import warnings

import config

# 从 engine 包导入核心功能
from engine.utils import (
    log, get_boundary, boundary_pixel_mask,
    setup_matplotlib_font, get_needed_bands,
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
from engine.indices import compute_index, compute_all_indices
from engine.data_sources import get_data_source
from engine.ml_detector import run_ml_change_detection
from engine.params import RunParams
from engine.timeseries import TimeSeriesAnalyzer

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

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="rs-change", description="多源遥感变化检测（未指定的参数取 config.py 默认值）")
    ap.add_argument("--boundary", help="AOI 边界 GeoJSON（WGS84）")
    ap.add_argument("--region", help="区域名称（用于输出文件名）")
    ap.add_argument("--year1", type=int, help="前时相年份")
    ap.add_argument("--year2", type=int, help="后时相年份")
    ap.add_argument("--months", help="月份范围，如 6-10")
    ap.add_argument("--source", dest="data_source",
                    choices=["sentinel2", "landsat8", "landsat9", "sentinel1"])
    ap.add_argument("--index", dest="index_type", help="NDVI/NDWI/MNDWI/NBR/EVI/SAVI/NDBI")
    ap.add_argument("--mode", dest="detection_mode", choices=["multi", "binary", "ml"])
    ap.add_argument("--cloud", dest="max_cloud_pct", type=float, help="场景云量上限 (%%)")
    ap.add_argument("--threshold", type=float, help="变化阈值")
    ap.add_argument("--severe", dest="severe_threshold", type=float, help="强烈变化阈值")
    ap.add_argument("--adaptive", dest="adaptive_threshold", action="store_true", default=None,
                    help="按稳定区噪声自动估计阈值")
    ap.add_argument("--min-area", dest="min_area_m2", type=float, help="最小斑块面积（平方米）")
    ap.add_argument("--formats", dest="output_formats", help="采样点格式，如 geojson,csv,shp")
    ap.add_argument("--data-dir", help="数据/缓存目录（默认 ./data）")
    ap.add_argument("--out", dest="out_dir", help="输出目录（默认 config.OUTPUT_DIR）")
    return ap.parse_args(argv)


def main(argv=None):
    """命令行入口：运行完整的 v3 变化检测流程。成功返回 0，失败返回非 0。"""
    # 中文 Windows 控制台默认 GBK：无法编码的字符替换输出，而不是让程序崩溃
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = _parse_args(argv)
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("boundary", "months", "data_dir", "out_dir")}
    overrides["boundary_file"] = args.boundary
    if args.months:
        ms, _, me = args.months.partition("-")
        overrides.update(month_start=int(ms), month_end=int(me or ms))
    try:
        params = RunParams.from_config(**overrides)
    except ValueError as e:
        print(f"参数错误: {e}", file=sys.stderr)
        return 2

    print("=" * 60)
    print("  遥感解译助手 v3 — 多源遥感变化检测")
    print(f"  区域: {params.region}")
    print(f"  时相: {params.year1} vs {params.year2}（{params.month_start}-{params.month_end} 月）")
    print(f"  数据源: {params.data_source}   指数: {params.index_type}   模式: {params.detection_mode}")
    print("=" * 60)

    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(base, config.OUTPUT_DIR)
    results = run_pipeline_v3(params=params, data_dir=args.data_dir, out_dir=out_dir)

    if results.get("error"):
        print(f"  Error: {results['error']}", file=sys.stderr)
        return 1

    stats = results.get("stats", {})
    if params.detection_mode != "binary":
        print(f"\n  ==== 变化统计 ({params.index_type}) ====")
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
    print(f"  Total time: {results.get('timing', {}).get('total', 0):.0f}s")
    print(f"\n{'=' * 60}")
    print(f"  Done! Output: {out_dir}")
    print(f"{'=' * 60}")
    return 0


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
    "RunParams",
]


if __name__ == "__main__":
    sys.exit(main())
