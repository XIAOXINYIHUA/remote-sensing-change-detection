"""
遥感解译助手 v3 — 时间序列分析模块

功能:
  - 多期指数年际变化趋势分析
  - Mann-Kendall 趋势检验 + Sen's slope
  - 年均变化率计算
  - 统计摘要生成
"""

import os
import time
import numpy as np


class TimeSeriesAnalyzer:
    """多期遥感指数时间序列分析器。

    逐像元计算多年变化趋势，使用 Mann-Kendall 检验
    和 Theil-Sen 斜率估计。

    Usage:
        analyzer = TimeSeriesAnalyzer(boundary, data_source="sentinel2")
        ts_data = analyzer.collect_time_series([2020,2021,2022], "NDVI")
        trend = analyzer.compute_trend(ts_data)
        summary = analyzer.summarize(ts_data, trend)
    """

    def __init__(self, boundary, data_source="sentinel2"):
        """初始化分析器。

        Args:
            boundary: Shapely 边界几何对象 (WGS84)。
            data_source: 数据源名称 ("sentinel2" | "landsat8" | ...)。
        """
        self.boundary = boundary
        self.data_source = data_source

    def collect_time_series(self, years, index_type="NDVI",
                            month_start=6, month_end=10,
                            data_dir="data/timeseries"):
        """收集多年指数数据。

        Args:
            years: 年份列表 (如 [2020, 2021, 2022, 2023, 2024])。
            index_type: 指数类型 ("NDVI"/"NDWI"/"NBR"/etc)。
            month_start/end: 影像搜索月份范围。
            data_dir: 数据缓存目录。

        Returns:
            dict: {
                "years": [2020, 2021, ...],
                "stacked": np.ndarray (T, H, W),
                "index_maps": {year: np.ndarray},
                "valid_mask": np.ndarray (H, W),
            }
        """
        from engine.data_sources import get_data_source
        from engine.indices import compute_index
        from engine.utils import get_needed_bands, log
        from engine.grid import GridSpec, reproject_to_grid, as_gridspec
        import rasterio

        source = get_data_source(self.data_source)
        os.makedirs(data_dir, exist_ok=True)

        index_maps = {}
        needed_bands = get_needed_bands(index_type)
        first_meta = None

        for year in years:
            log(f"  Processing {year}...")
            try:
                scenes = source.search_scenes(
                    self.boundary, year, month_start, month_end)

                if not scenes:
                    log(f"    No scenes for {year}, skipping")
                    continue

                bands = source.download_bands(
                    scenes, data_dir, needed_bands)

                band_arrays = {}
                for bname, bpath in bands.items():
                    if os.path.exists(bpath):
                        with rasterio.open(bpath) as src:
                            if first_meta is None:
                                first_meta = src.meta
                            band_arrays[bname] = (src.read(1).astype(np.float32), src.meta)

                if not band_arrays:
                    continue

                # 各波段配准到统一网格后再算指数
                if first_meta is None:
                    raise RuntimeError("No valid reference grid")
                ref_grid = GridSpec.from_meta(first_meta)
                aligned = {}
                for bname, (arr, bmeta) in band_arrays.items():
                    gs = as_gridspec(bmeta) if bmeta else ref_grid
                    if gs != ref_grid:
                        arr = reproject_to_grid((arr, bmeta), ref_grid)
                    aligned[bname] = arr

                idx = compute_index(index_type, aligned)
                index_maps[year] = idx
                log(f"    {year}: index computed, shape={idx.shape}")

            except Exception as e:
                log(f"    {year} failed: {e}")
                continue

        if not index_maps:
            raise RuntimeError(f"No valid data for any year in {years}")

        # 对齐到统一网格：基于 CRS+transform 重投影，而非 cv2.resize
        if first_meta is None:
            raise RuntimeError("No valid band files to determine reference grid")
        ref_grid = GridSpec.from_meta(first_meta)

        aligned = {}
        for year, idx in index_maps.items():
            if idx.shape != (ref_grid.height, ref_grid.width):
                log(f"    {year}: reprojecting {idx.shape}→"
                    f"({ref_grid.height},{ref_grid.width})")
                idx = reproject_to_grid((idx, first_meta), ref_grid)
            aligned[year] = idx

        # 有效区域：所有年份均非 NaN（NDVI=0 是合法值，不作为排除条件）
        valid = np.ones(ref_grid.shape, dtype=bool)
        for idx in aligned.values():
            valid &= ~np.isnan(idx)

        years_sorted = sorted(aligned.keys())
        stacked = np.stack([aligned[y] for y in years_sorted])

        return {
            "years": years_sorted,
            "stacked": stacked,
            "index_maps": aligned,
            "valid_mask": valid,
        }

    def compute_trend(self, ts_data):
        """逐像元计算变化趋势 (Mann-Kendall + Sen's slope)。

        Args:
            ts_data: collect_time_series() 的返回值。

        Returns:
            dict: {
                "slope": (H, W) 年均变化率,
                "p_value": (H, W) 显著性,
                "significant": (H, W) p<0.05,
                "trend_type": (H, W) -1退化/0不显著/1改善,
            }
        """
        from scipy import stats as sp_stats
        from engine.utils import log

        years = np.array(ts_data["years"])
        stacked = ts_data["stacked"]
        valid = ts_data["valid_mask"]
        T, H, W = stacked.shape

        slope = np.full((H, W), np.nan, dtype=np.float32)
        p_value = np.full((H, W), np.nan, dtype=np.float32)

        valid_idx = np.where(valid)
        n_valid = valid_idx[0].shape[0]

        log(f"  Computing pixel-wise trend for {n_valid:,} pixels...")
        t0 = time.time()

        batch_size = 10000
        n_batches = (n_valid + batch_size - 1) // batch_size

        for batch_i in range(0, n_valid, batch_size):
            batch_end = min(batch_i + batch_size, n_valid)
            rows = valid_idx[0][batch_i:batch_end]
            cols = valid_idx[1][batch_i:batch_end]

            for j in range(len(rows)):
                r, c = rows[j], cols[j]
                values = stacked[:, r, c]
                mask = ~np.isnan(values)
                if mask.sum() < 4:
                    continue
                try:
                    res = sp_stats.theilslopes(values[mask], years[mask])
                    slope[r, c] = res[0]
                    tau, p = sp_stats.kendalltau(years[mask], values[mask])
                    p_value[r, c] = p
                except (ValueError, RuntimeError):
                    continue

            if (batch_i // batch_size) % 10 == 0 and batch_i > 0:
                log(f"    {batch_end}/{n_valid} "
                    f"({time.time()-t0:.0f}s)")

        log(f"  Trend computation done: {time.time()-t0:.1f}s")

        # 分类
        significant = (p_value < 0.05) & ~np.isnan(slope)
        trend_type = np.zeros((H, W), dtype=np.int8)
        trend_type[significant & (slope > 0)] = 1
        trend_type[significant & (slope < 0)] = -1
        trend_type[~valid] = 0

        return {
            "slope": slope,
            "p_value": p_value,
            "significant": significant,
            "trend_type": trend_type,
        }

    def summarize(self, ts_data, trend_data):
        """生成时间序列统计摘要。

        Args:
            ts_data: collect_time_series() 返回值。
            trend_data: compute_trend() 返回值。

        Returns:
            dict: 摘要统计。
        """
        valid = ts_data["valid_mask"]
        trend = trend_data["trend_type"]
        slope = trend_data["slope"]

        n_valid = valid.sum()
        n_improve = (trend == 1).sum()
        n_degrade = (trend == -1).sum()
        n_stable = n_valid - n_improve - n_degrade

        mean_slope = float(np.nanmean(slope[valid])) if n_valid > 0 else 0.0

        yearly_means = {}
        years = ts_data["years"]
        stacked = ts_data["stacked"]
        for i, year in enumerate(years):
            yearly_means[int(year)] = float(
                np.nanmean(stacked[i][valid])) if n_valid > 0 else 0.0

        return {
            "years": [int(y) for y in years],
            "total_pixels": int(n_valid),
            "improve_pixels": int(n_improve),
            "degrade_pixels": int(n_degrade),
            "stable_pixels": int(n_stable),
            "improve_percent": round(
                n_improve / max(n_valid, 1) * 100, 2),
            "degrade_percent": round(
                n_degrade / max(n_valid, 1) * 100, 2),
            "mean_slope": round(mean_slope, 6),
            "yearly_mean_index": yearly_means,
        }
