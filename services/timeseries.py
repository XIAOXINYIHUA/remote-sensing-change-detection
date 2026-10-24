"""
时间序列服务：收集多期影像（共用单期快照缓存）→ 逐像元 Mann-Kendall + Sen 斜率 → 统计摘要
"""

from engine.timeseries import MIN_OBSERVATIONS
from services import ServiceError, report

# 趋势分析至少需要的有效时相数：每个像元至少要有 MIN_OBSERVATIONS 期有效观测
MIN_PERIODS = MIN_OBSERVATIONS


def run(boundary, years, index_type, data_source, month_start, month_end, max_cloud_pct,
        data_dir, progress_callback=None):
    """全流程：收集影像 → 计算指数 → 趋势分析。

    无影像或检索失败的年份跳过；有效时相不足 MIN_PERIODS 期时报错。
    进度：收集影像 15–50%，趋势检验 50–80%，统计摘要 80–100%。

    Returns:
        dict(ts_data, trend, summary)
    Raises:
        ServiceError: 有效时相不足
    """
    from engine.timeseries import TimeSeriesAnalyzer

    if len(years) < MIN_PERIODS:
        raise ServiceError(f"共 {len(years)} 期，趋势分析至少需要 {MIN_PERIODS} 期")
    report(progress_callback, "初始化分析器...", 5)
    analyzer = TimeSeriesAnalyzer(boundary, data_source=data_source)

    report(progress_callback, f"📥 收集 {len(years)} 期 {index_type} 影像...", 15)
    ts_data = analyzer.collect_time_series(
        years, index_type=index_type,
        month_start=month_start, month_end=month_end,
        data_dir=data_dir, max_cloud_pct=max_cloud_pct,
        progress_callback=lambda msg, pct: report(progress_callback, msg, 15 + pct * 0.35))
    got = ts_data["years"]
    if len(got) < MIN_PERIODS:
        missing = [y for y in years if y not in got]
        raise ServiceError(
            f"只取得 {len(got)} 期有效影像（{', '.join(map(str, got)) or '无'}），"
            f"趋势分析至少需要 {MIN_PERIODS} 期；无可用影像的年份: {', '.join(map(str, missing))}")

    report(progress_callback, f"📊 {len(got)} 期影像就绪，逐像元趋势分析 (Mann-Kendall + Sen's slope)...", 50)
    trend = analyzer.compute_trend(ts_data)
    report(progress_callback, "📋 生成统计摘要...", 80)
    summary = analyzer.summarize(ts_data, trend)
    report(progress_callback, "✅ 时间序列分析完成！", 100)
    return {"ts_data": ts_data, "trend": trend, "summary": summary}
