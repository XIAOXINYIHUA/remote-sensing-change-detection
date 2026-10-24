"""
遥感分析平台 v3 — 前端组件：地物分类结果可视化

功能:
  - 3D 光谱散点图（plotly / matplotlib 回退）
  - 分类结果地图
  - 斑块标注交互
"""

import numpy as np


# ============================================================================
#  3D 光谱散点图
# ============================================================================

def render_3d_scatter(pca_3d, labels, n_clusters, width=800, height=600):
    """渲染 3D PCA 光谱散点图。

    输入:
        pca_3d: (H, W, 3) PCA 降维后的 3 个分量
        labels: (H, W) 聚类标签（0..n_clusters-1）
        n_clusters: 类别数

    返回:
        plotly Figure（plotly 不可用时返回 matplotlib Figure）
    """
    h, w = labels.shape

    # 随机采样（全图点太多绘制卡顿）
    n_sample = min(5000, h * w)
    rng = np.random.RandomState(42)
    flat_idx = rng.choice(h * w, n_sample, replace=False)
    rows = flat_idx // w
    cols = flat_idx % w

    x = pca_3d[rows, cols, 0]
    y = pca_3d[rows, cols, 1]
    z = pca_3d[rows, cols, 2]
    sample_labels = labels[rows, cols]

    # 截断离群值到 2%-98% 分位数，避免少数极值把坐标轴拉散
    def _clip(arr):
        lo, hi = np.percentile(arr, [2, 98])
        return np.clip(arr, lo, hi)

    x = _clip(x)
    y = _clip(y)
    z = _clip(z)

    return _plotly_scatter(x, y, z, sample_labels, n_clusters)


def _plotly_scatter(x, y, z, labels, n_clusters):
    """尝试用 plotly 绘制 3D 散点图，失败时回退到 matplotlib。"""
    try:
        import plotly.graph_objects as go
        from plotly.colors import qualitative

        colors = qualitative.Plotly * (n_clusters // len(qualitative.Plotly) + 1)
        colors = colors[:n_clusters]

        color_scale = []
        for index, color in enumerate(colors):
            low = index / max(n_clusters, 1)
            high = (index + 1) / max(n_clusters, 1)
            color_scale.extend([[low, color], [high, color]])

        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=2,
                color=labels,
                colorscale=color_scale,
                cmin=-0.5,
                cmax=max(n_clusters - 0.5, 0.5),
                opacity=0.68,
            ),
            text=[f"Cluster {l}" for l in labels],
            hoverinfo="text",
        ))

        fig.update_layout(
            title="PCA 光谱空间 3D 散点图",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
                bgcolor="#FFFFFF",
                xaxis=dict(gridcolor="#D9E1DA", zerolinecolor="#C5D0C7"),
                yaxis=dict(gridcolor="#D9E1DA", zerolinecolor="#C5D0C7"),
                zaxis=dict(gridcolor="#D9E1DA", zerolinecolor="#C5D0C7"),
            ),
            width=800,
            height=600,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#18221D"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    except ImportError:
        return _matplotlib_3d_scatter(x, y, z, labels, n_clusters)


def _matplotlib_3d_scatter(x, y, z, labels, n_clusters):
    """matplotlib 回退版 3D 散点图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, c=labels, cmap="tab10", s=1, alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 光谱空间 (matplotlib)")

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.xaxis.label.set_color("#18221D")
    ax.yaxis.label.set_color("#18221D")
    ax.zaxis.label.set_color("#18221D")
    ax.title.set_color("#18221D")
    ax.tick_params(colors="#526159")

    return fig


# ============================================================================
#  标注斑块可视化
# ============================================================================

def render_annotation_patch(bands, row, col, patch_size=32):
    """截取待标注斑块的 RGB 小图。"""
    half = patch_size // 2
    r_slice = slice(row - half, row + half + 1)
    c_slice = slice(col - half, col + half + 1)

    # 用红绿蓝合成近似真彩色
    rgb = np.zeros((patch_size + 1, patch_size + 1, 3), dtype=np.uint8)
    for i, bname in enumerate(["red", "green", "blue"]):
        if bname in bands:
            patch = bands[bname][r_slice, c_slice]
            # 拉伸到 0-255
            p_min, p_max = patch.min(), patch.max()
            if p_max > p_min:
                rgb[:, :, 2 - i] = np.clip(
                    (patch - p_min) / (p_max - p_min) * 255, 0, 255
                ).astype(np.uint8)
            else:
                rgb[:, :, 2 - i] = 128

    return rgb
