"""
遥感解译助手 v3 — 时间序列分析模块

功能:
  - 多期指数年际变化趋势分析（逐期云掩膜 + AOI 裁剪）
  - Mann-Kendall 趋势检验 + Sen's slope（全向量化，按行分块控制内存）
  - 年均变化率计算
  - 统计摘要生成

Mann-Kendall p 值: n <= 10 用精确分布（Mahonian 数），n > 10 用带连续性
校正的正态近似；缺测年份按像元跳过（每个像元用自己的有效观测数 n）。
"""

import os
import time
import warnings

import numpy as np

# 每个像元至少需要的有效观测年数
MIN_OBSERVATIONS = 4
# 精确 p 值的最大样本量
MK_EXACT_MAX_N = 10


# ============================================================================
#  Mann-Kendall + Sen's slope（向量化）
# ============================================================================

def _mk_exact_tables(max_n=MK_EXACT_MAX_N):
    """预计算无结时 |S| -> 双侧 p 值的查找表。

    S = M - 2·inv，M = n(n-1)/2，inv 为逆序对数，其分布是 Mahonian 数。
    """
    tables = {}
    dist = np.array([1.0])
    for n in range(2, max_n + 1):
        dist = np.convolve(dist, np.ones(n))        # n 元排列的逆序数分布
        m = n * (n - 1) // 2
        s_vals = m - 2 * np.arange(m + 1)
        prob = dist / dist.sum()
        abs_s = np.abs(s_vals)
        uniq = np.unique(abs_s)                      # 升序
        tail = np.array([prob[abs_s >= u].sum() for u in uniq])
        tables[n] = (uniq, tail)
    return tables


_MK_TABLES = _mk_exact_tables()


def mk_pvalue(S, n):
    """Mann-Kendall 双侧 p 值（向量化）。

    Args:
        S: MK 统计量数组
        n: 每个元素对应的有效样本量数组

    Returns:
        p 值数组；n < 3 处为 NaN。
    """
    from scipy.special import erfc

    S = np.asarray(S, dtype=np.float64)
    n = np.asarray(n)
    p = np.full(S.shape, np.nan)
    abs_s = np.abs(S)

    for nv in np.unique(n):
        sel = n == nv
        if nv < 3:
            continue
        if nv <= MK_EXACT_MAX_N:
            uniq, tail = _MK_TABLES[int(nv)]
            # P(|S'| >= |S|)：取第一个 >= |S| 的可达值
            idx = np.searchsorted(uniq, abs_s[sel] - 1e-9)
            p[sel] = np.where(idx < len(uniq), tail[np.minimum(idx, len(uniq) - 1)], 0.0)
        else:
            var = nv * (nv - 1) * (2 * nv + 5) / 18.0
            z = (abs_s[sel] - 1.0).clip(min=0) / np.sqrt(var)
            p[sel] = erfc(z / np.sqrt(2.0))
    return p


def sen_slope_mk(stacked, years, valid=None, chunk_rows=256):
    """逐像元 Sen 斜率与 Mann-Kendall p 值。

    Args:
        stacked: (T, H, W) 指数时序，缺测为 NaN
        years: (T,) 年份
        valid: (H, W) 只计算这些像元（None=全部）
        chunk_rows: 每块行数（内存约 T(T-1)/2 × chunk_rows × W × 4 B）

    Returns:
        (slope, p_value, n_obs)：(H, W)，无法计算处 slope/p 为 NaN
    """
    stacked = np.asarray(stacked, dtype=np.float32)
    years = np.asarray(years, dtype=np.float32)
    T, H, W = stacked.shape
    i, k = np.triu_indices(T, 1)
    dt = (years[k] - years[i])[:, None, None]

    slope = np.full((H, W), np.nan, dtype=np.float32)
    p_value = np.full((H, W), np.nan, dtype=np.float32)
    n_obs = np.isfinite(stacked).sum(axis=0)
    if valid is None:
        valid = np.ones((H, W), dtype=bool)
    compute = valid & (n_obs >= MIN_OBSERVATIONS)

    for r0 in range(0, H, chunk_rows):
        rows = slice(r0, min(r0 + chunk_rows, H))
        m = compute[rows]
        if not m.any():
            continue
        blk = stacked[:, rows]
        d = blk[k] - blk[i]                              # (P, h, W)，缺测传播 NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s_blk = np.nanmedian(d / dt, axis=0)
        S = np.nansum(np.sign(d), axis=0)
        p_blk = mk_pvalue(S, n_obs[rows])
        slope[rows][m] = s_blk[m]
        p_value[rows][m] = p_blk[m]

    return slope, p_value, n_obs


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
                            data_dir="data/timeseries", max_cloud_pct=50,
                            progress_callback=None):
        """收集多年指数数据（每期云掩膜，AOI 外与缺测为 NaN）。

        各年份经单期快照缓存取数（与变化检测 / 地物分类共用，已下载过的年份
        不再下载）；所有年份在同一 AOI 分析网格上，无需再重投影。
        无影像或下载失败的年份记为缺测并跳过。

        Args:
            data_dir: 下载临时目录所在位置（数据最终存入快照缓存）。
            progress_callback: 进度回调 (msg, pct)。

        Returns:
            dict: {
                "years": [2020, 2021, ...],
                "stacked": np.ndarray (T, H, W)，缺测为 NaN,
                "index_maps": {year: np.ndarray},
                "valid_mask": (H, W) AOI 内且有效观测 >= MIN_OBSERVATIONS,
                "meta": 网格元数据,
                "preprocessing": {year: {"cloud_mask", "cloud_pct"}},
            }
        """
        import tempfile

        from engine.core import _acquire_periods
        from engine.data_sources import get_data_source
        from engine.indices import compute_index
        from engine.preprocessing import load_period
        from engine.utils import get_needed_bands, boundary_pixel_mask, log

        def cb(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)
            log(msg)

        source = get_data_source(self.data_source)
        os.makedirs(data_dir, exist_ok=True)
        grid = source.analysis_grid(self.boundary)
        meta = grid.to_meta()
        in_aoi = boundary_pixel_mask(self.boundary, meta, grid.height, grid.width)
        needed = get_needed_bands(index_type)

        index_maps, prep = {}, {}
        with tempfile.TemporaryDirectory(prefix="ts_", dir=data_dir,
                                         ignore_cleanup_errors=True) as work_dir:
            got = _acquire_periods(source, self.data_source, self.boundary, list(years),
                                   month_start, month_end, max_cloud_pct, needed,
                                   work_dir, cb, require_all=False, pct=(5, 80))
            # 未写入缓存的期文件还在 work_dir 中，须在清理前读取
            for year in sorted(got):
                try:
                    bands, valid, info = load_period(
                        got[year]["paths"], got[year]["quality"], self.data_source, grid)
                    idx = compute_index(index_type, bands).astype(np.float32)
                    idx[~(valid & in_aoi)] = np.nan
                    index_maps[year] = idx
                    prep[year] = info
                    log(f"    {year}: {np.isfinite(idx).mean() * 100:.1f}% valid pixels")
                except Exception as e:
                    log(f"    {year} failed: {e}")

        if not index_maps:
            raise RuntimeError(f"No valid data for any year in {years}")

        years_sorted = sorted(index_maps)
        stacked = np.stack([index_maps[y] for y in years_sorted])
        n_obs = np.isfinite(stacked).sum(axis=0)

        return {
            "years": years_sorted,
            "stacked": stacked,
            "index_maps": index_maps,
            "valid_mask": in_aoi & (n_obs >= MIN_OBSERVATIONS),
            "meta": meta,
            "preprocessing": prep,
        }

    def compute_trend(self, ts_data, alpha=0.05):
        """逐像元计算变化趋势 (Mann-Kendall + Sen's slope)。

        Args:
            ts_data: collect_time_series() 的返回值。
            alpha: 显著性水平。

        Returns:
            dict: {
                "slope": (H, W) 年均变化率,
                "p_value": (H, W) 显著性,
                "significant": (H, W) p<alpha,
                "trend_type": (H, W) -1退化/0不显著/1改善,
            }
        """
        from engine.utils import log

        valid = ts_data["valid_mask"]
        log(f"  Computing pixel-wise trend for {int(valid.sum()):,} pixels...")
        t0 = time.time()
        slope, p_value, _ = sen_slope_mk(ts_data["stacked"], ts_data["years"], valid)
        log(f"  Trend computation done: {time.time()-t0:.1f}s")

        significant = (p_value < alpha) & np.isfinite(slope)
        trend_type = np.zeros(slope.shape, dtype=np.int8)
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

        n_valid = int(valid.sum())
        n_improve = int((trend == 1).sum())
        n_degrade = int((trend == -1).sum())
        n_stable = n_valid - n_improve - n_degrade

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_slope = float(np.nanmean(slope[valid])) if n_valid > 0 else 0.0
            yearly_means = {
                int(year): (float(np.nanmean(ts_data["stacked"][i][valid]))
                            if n_valid > 0 else 0.0)
                for i, year in enumerate(ts_data["years"])
            }

        return {
            "years": [int(y) for y in ts_data["years"]],
            "total_pixels": n_valid,
            "improve_pixels": n_improve,
            "degrade_pixels": n_degrade,
            "stable_pixels": n_stable,
            "improve_percent": round(n_improve / max(n_valid, 1) * 100, 2),
            "degrade_percent": round(n_degrade / max(n_valid, 1) * 100, 2),
            "mean_slope": round(mean_slope, 6),
            "yearly_mean_index": yearly_means,
        }
