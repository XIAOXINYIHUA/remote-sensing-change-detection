"""
前端组件 — 图表组件

可复用的 matplotlib 图表封装。
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def render_stats_chart(stats, index_type="NDVI"):
    """渲染变化检测统计图表（柱状图 + 饼图）。

    Args:
        stats: 统计字典。
        index_type: 指数类型。
    """
    from engine.indices import INDEX_CLASS_NAMES, INDEX_COLORS

    cls_names = INDEX_CLASS_NAMES.get(index_type, INDEX_CLASS_NAMES["NDVI"])
    cls_colors = INDEX_COLORS.get(index_type, INDEX_COLORS["NDVI"])

    cd = [(c, stats.get(c, {}).get("area_ha", 0))
          for c in [-3, -2, 2, 3]]
    cd = [(n, a) for n, a in cd if a > 0]

    if not cd:
        st.info("无显著变化数据")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = [cls_names.get(c, str(c)) for c, _ in cd]
    sizes = [a for _, a in cd]
    colors = [cls_colors.get(c, "#888") for c, _ in cd]

    ax1.bar(labels, sizes, color=colors, edgecolor="white")
    ax1.set_ylabel("面积 (ha)")
    ax1.set_title("各级变化面积", fontweight="bold")
    ax1.tick_params(axis="x", rotation=15)

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90)
    for t in autotexts:
        t.set_fontweight("bold")
    ax2.set_title("变化面积占比", fontweight="bold")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_timeseries_chart(summary, index_type="NDVI"):
    """渲染时间序列分析图表。

    Args:
        summary: TimeSeriesAnalyzer.summarize() 的返回值。
        index_type: 指数类型。
    """
    years = summary.get("years", [])
    means = [summary["yearly_mean_index"].get(y, 0) for y in years]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 年际折线图
    ax = axes[0]
    ax.plot(years, means, 'o-', color='#2D65A1', linewidth=2, markersize=8)
    ax.fill_between(years, means, alpha=0.2, color='#2D65A1')
    ax.set_xlabel("年份")
    ax.set_ylabel(f"{index_type} 均值")
    ax.set_title(f"{index_type} 年际变化趋势")
    ax.grid(True, alpha=0.3)

    # 右: 趋势分布饼图
    ax = axes[1]
    improve = summary.get("improve_percent", 0)
    degrade = summary.get("degrade_percent", 0)
    stable = 100 - improve - degrade

    sizes = [improve, degrade, max(stable, 0)]
    labels = ['改善', '退化', '稳定']
    colors = ['#16806B', '#B8403A', '#7B8780']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title("趋势分布")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
