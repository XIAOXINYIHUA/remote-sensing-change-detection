"""
遥感地物分类器 v1 — 分类引擎

三种分类模式:
  1. 多指数阈值法 — NDVI + MNDWI + NDBI 联合阈值，零依赖即时出图
  2. GMM 聚类      — PCA 降维 16→3 + 高斯混合模型，自动发现光谱簇
  3. 主动学习迭代   — 不确定性采样 → 用户标注 → RF 训练 → 精度提升

纹理特征（提升城市/农田分类精度）:
  - 局部方差（近似 GLCM Contrast）
  - 边缘密度
  - 均值比

依赖: numpy, scipy, sklearn（已有）
"""

import numpy as np
from scipy.ndimage import uniform_filter, sobel


# ============================================================================
#  地物类别定义
# ============================================================================

CLASS_NAMES = {
    0: "未分类",
    1: "水体",
    2: "植被",
    3: "建筑/不透水面",
    4: "裸土",
    5: "阴影",
    6: "湿地",
    7: "农田",
}

CLASS_COLORS = {
    0: "#282828",
    1: "#4169E1",    # 水体 → 蓝色
    2: "#00b450",    # 植被 → 绿色
    3: "#DC143C",    # 建筑 → 红色
    4: "#CD853F",    # 裸土 → 棕色
    5: "#555555",    # 阴影 → 灰色
    6: "#20B2AA",    # 湿地 → 青绿
    7: "#FFD700",    # 农田 → 金色
}

# 默认多指数阈值
INDEX_THRESHOLDS = {
    "mndwi": 0.0,       # MNDWI > 0 → 水体
    "ndvi_high": 0.3,   # NDVI > 0.3 → 植被
    "ndvi_low": 0.1,    # NDVI < 0.1 → 低植被
    "ndbi": 0.0,        # NDBI > 0 → 建筑
    "ndvi_water": 0.1,  # NDVI < 0.1 + MNDWI > 0 → 水体（预留）
}


# ============================================================================
#  多指数阈值分类
# ============================================================================

def classify_multi_index(bands):
    """NDVI + MNDWI + NDBI 联合阈值分类。

    自动跳过缺失波段对应的指数（如无 SWIR 则不计算 MNDWI/NDBI），
    至少需要 red+nir 做 NDVI。

    输入:
        bands: {band_name: np.ndarray} 波段字典
    输出:
        result: (H, W) int8, 值对应 CLASS_NAMES
        rgb_vis: (H, W, 3) uint8 分类彩色图
    """
    from engine.indices import compute_index

    # 逐个计算指数，缺失波段则跳过
    idx_data = {}
    for idx_name, needed in [("NDVI", ["nir","red"]),
                              ("MNDWI", ["green","swir1"]),
                              ("NDBI", ["swir1","nir"])]:
        if all(b in bands for b in needed):
            try:
                idx_data[idx_name] = compute_index(idx_name, bands)
            except Exception:
                continue

    if "NDVI" not in idx_data:
        # 保底：如果没有 NDVI，用 NIR-Red 简单差值
        if "nir" in bands and "red" in bands:
            n = bands["nir"].astype(np.float32)
            r = bands["red"].astype(np.float32)
            denom = n + r; denom[denom == 0] = 0.001
            idx_data["NDVI"] = (n - r) / denom
        else:
            raise ValueError("至少需要 NIR 和 Red 波段（B4/B8）才能分类")

    ndvi = idx_data["NDVI"]
    mndwi = idx_data.get("MNDWI")
    ndbi = idx_data.get("NDBI")
    has_mndwi = mndwi is not None
    has_ndbi = ndbi is not None

    h, w = ndvi.shape
    result = np.zeros((h, w), dtype=np.int8)

    # 优先级 1: 水体 (MNDWI > 0)
    if has_mndwi:
        water = mndwi > INDEX_THRESHOLDS["mndwi"]
        water = water & (ndvi < INDEX_THRESHOLDS["ndvi_low"] + 0.15)
        result[water] = 1

    # 优先级 2: 植被 (NDVI > 0.3)
    dense_veg = ndvi > INDEX_THRESHOLDS["ndvi_high"]
    result[dense_veg] = 2

    # 优先级 3: 建筑 (NDBI > 0 且非植被/水体)
    if has_ndbi:
        built = (ndbi > INDEX_THRESHOLDS["ndbi"]) & (result == 0)
        built = built & (ndvi < INDEX_THRESHOLDS["ndvi_high"])
        result[built] = 3

    # 优先级 4: 裸土 (低 NDVI + 可选条件)
    if has_mndwi and has_ndbi:
        bare = (result == 0) & (ndvi > -0.1) & (ndvi < INDEX_THRESHOLDS["ndvi_low"])
        bare = bare & (mndwi < -0.1) & (ndbi < INDEX_THRESHOLDS["ndbi"])
        result[bare] = 4
    else:
        # 没有 SWIR 波段时，仅用 NDVI 低值做裸土
        bare = (result == 0) & (ndvi > -0.1) & (ndvi < INDEX_THRESHOLDS["ndvi_low"])
        result[bare] = 4

    # 优先级 5: 阴影
    if "red" in bands:
        red = bands["red"].astype(np.float32)
        shadow = (result == 0) & (ndvi < 0) & (red < 0.1)
        result[shadow] = 5

    return result, _label_to_rgb(result)


def _label_to_rgb(result):
    """将分类标签数组转换为 RGB 彩色图。"""
    h, w = result.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_val, hex_color in CLASS_COLORS.items():
        mask = result == cls_val
        if mask.any():
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            rgb[mask] = [r, g, b]
    return rgb


# ============================================================================
#  纹理特征提取
# ============================================================================

def extract_texture(band, window_size=3):
    """从单波段提取纹理特征（快速卷积实现，零额外依赖）。

    返回:
        dict: {"variance": ..., "edge_density": ...}
    """
    b = band.astype(np.float32)

    # 局部方差: E[X²] - E[X]²
    mean = uniform_filter(b, size=window_size)
    mean_sq = uniform_filter(b ** 2, size=window_size)
    variance = np.maximum(mean_sq - mean ** 2, 0)

    # 边缘密度: Sobel 梯度幅度局部平均
    gx = sobel(b, axis=1)
    gy = sobel(b, axis=0)
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_density = uniform_filter(edge_mag, size=window_size)

    return {
        "variance": variance,
        "edge_density": edge_density,
    }


# ============================================================================
#  特征堆叠：10 波段 + 6 指数 + 纹理
# ============================================================================

def build_feature_stack(bands, add_texture=True):
    """构建分类用完整特征张量。

    输入:
        bands: {band_name: np.ndarray} 波段字典
        add_texture: 是否添加纹理特征

    返回:
        np.ndarray: (H, W, C) 特征张量
    """
    from engine.indices import compute_all_indices

    h, w = list(bands.values())[0].shape[:2]
    features = []

    # 1. 原始波段
    for bname in ["blue", "green", "red", "nir", "swir1", "swir2"]:
        if bname in bands:
            features.append(bands[bname].astype(np.float32))

    # 2. 光谱指数
    try:
        indices = compute_all_indices(bands)
        for name in ["NDVI", "NDWI", "MNDWI", "NBR", "EVI", "SAVI", "NDBI"]:
            if name in indices:
                features.append(indices[name])
    except Exception:
        pass

    # 3. 纹理特征
    if add_texture:
        # 仅对 NIR 和 Red 做纹理
        for bname in ["nir", "red"]:
            if bname in bands:
                tex = extract_texture(bands[bname])
                features.append(tex["variance"])
                features.append(tex["edge_density"])

    if not features:
        raise ValueError("No features could be built from bands")

    return np.stack(features, axis=-1)


# ============================================================================
#  PCA + GMM 聚类
# ============================================================================

def pca_gmm_classify(feature_stack, n_clusters=6, sample_ratio=0.05,
                      random_state=42, pca_dims=3, covariance_type="diag"):
    """PCA 降维 + GMM 聚类。

    流程:
        feature_stack (H, W, C)
        → reshape → 标准化 → 采样 5% 像元 → PCA(3) → GMM → 映射全图

    注意: GMM 标签为 0..n_clusters-1。
    返回值 labels 中，0 是合法聚类标签（与"未分类"值 0 不同）。
    apply_cluster_mapping() 中用 -1 作为起始值防止冲突。

    参数:
        feature_stack: (H, W, C) 特征张量
        n_clusters: GMM 类别数
        sample_ratio: PCA 拟合采样比例（大影像节省内存）
        random_state: 随机种子
        pca_dims: PCA 降维目标维度 (2-6)
        covariance_type: GMM 协方差类型 (diag/full/tied/spherical)

    返回:
        dict:
            labels: (H, W) 聚类标签 (0..n_clusters-1)
            probs: (H, W, n_clusters) 软分类概率
            pca_3d: (H, W, pca_dims) PCA 降维结果（用于 3D 可视化）
            gmm: 训练好的 GMM 模型
            pca: 训练好的 PCA 模型
            explained_variance: PCA 解释方差比
            scaler: 训练好的 StandardScaler
    """
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    h, w, c = feature_stack.shape
    pixels = feature_stack.reshape(-1, c)  # (N, C)

    # 剔除无效像元 (全 0 行) 避免干扰标准化
    valid_rows = ~(np.max(np.abs(pixels), axis=1) == 0)
    pixels_valid = pixels[valid_rows] if valid_rows.sum() > 100 else pixels

    # 标准化特征 (PCA 的前提条件)
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels_valid)

    # 随机采样用于 PCA 拟合
    n_sample = max(1000, min(int(pixels_scaled.shape[0] * sample_ratio), 50000))
    n_sample = min(n_sample, pixels_scaled.shape[0])
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(pixels_scaled.shape[0], n_sample, replace=False)
    sample_data = pixels_scaled[sample_idx]

    # PCA 降维
    pca = PCA(n_components=pca_dims, random_state=random_state)
    pca.fit(sample_data)

    # 映射全图到低维空间
    # 先用 scaler 标准化全图，再 PCA transform
    pixels_all_scaled = scaler.transform(pixels)
    pca_all = pca.transform(pixels_all_scaled).reshape(h, w, pca_dims)

    # 用降维后的数据做 GMM：只在有效像元的采样上拟合（大影像全图拟合
    # 既慢又占内存，且无效像元会被当成一个光谱簇），再预测全图
    pca_flat = pca_all.reshape(-1, pca_dims)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=200,
        n_init=3,
    )
    gmm.fit(pca.transform(sample_data))

    labels = gmm.predict(pca_flat).reshape(h, w)
    probs = gmm.predict_proba(pca_flat).reshape(h, w, n_clusters)

    return {
        "labels": labels.astype(np.int32),
        "probs": probs,
        "pca_3d": pca_all,
        "gmm": gmm,
        "pca": pca,
        "scaler": scaler,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
    }


# ============================================================================
#  聚类后标注合并
# ============================================================================

def apply_cluster_mapping(labels, mapping):
    """将聚类标签映射为地物类别。

    使用 -1 作为初始填充值避免 GMM 标签 0 与「未分类」值 0 冲突。
    未映射的聚类自动设为 0（未分类）。

    参数:
        labels: (H, W) GMM 聚类标签 (0..n_clusters-1)
        mapping: {cluster_id: class_value}
                 如 {0: 2, 1: 1, 2: 3, 3: 4, 4: 2, 5: 3}

    返回:
        result: (H, W) int8 地物分类结果 (0=未分类)
    """
    result = np.full(labels.shape, -1, dtype=np.int8)
    for cluster_id, class_val in mapping.items():
        result[labels == cluster_id] = class_val
    result[result == -1] = 0  # 未映射的聚落设为未分类
    return result


# ============================================================================
#  不确定性采样（主动学习用）
# ============================================================================

def uncertainty_sampling(pca_3d, gmm, n_samples=200):
    """从 GMM 结果中挑出最不确定的像元（供人工标注）。

    策略: 最大概率最低 + 类边界像元

    返回:
        indices: (n, 2) array of (row, col) 最不确定像元的坐标
        uncertainties: (n,) 不确定度分数（越高越不确定）
    """
    h, w, n_dims = pca_3d.shape
    pca_flat = pca_3d.reshape(-1, n_dims)
    probs = gmm.predict_proba(pca_flat)

    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)  # (N,)

    # 不确定性分数 = 熵（归一化）
    max_entropy = np.log(probs.shape[1])
    uncertainty = entropy / max_entropy  # [0, 1]

    # 取不确定性最高的像元
    flat_idx = np.argsort(-uncertainty)[:n_samples]
    rows, cols = np.unravel_index(flat_idx, (h, w))

    return np.stack([rows, cols], axis=1), uncertainty[flat_idx]


def sample_patches(bands, pca_3d, gmm, patch_size=32, n_patches=50):
    """采样不确定性最高区域的小斑块（比单像元更适合人工标注）。

    返回:
        patches: [(row, col, patch_array), ...]
        uncertainties: 每个斑块的平均不确定性
    """
    h, w = list(bands.values())[0].shape[:2]
    # coords 是全图 (row, col)，已按不确定性降序排列；
    # 候选数按斑块面积放大，高不确定区域集中时仍能凑够不重叠斑块
    coords, uncertainties = uncertainty_sampling(
        pca_3d, gmm, n_samples=min(n_patches * patch_size * patch_size, h * w))

    # 在不确定性高的像元处采样不重叠斑块
    half = patch_size // 2
    patches, patch_unc = [], []
    used = np.zeros((h, w), dtype=bool)

    for (r, c), u in zip(coords, uncertainties):
        r, c = int(r), int(c)
        if r - half < 0 or r + half >= h or c - half < 0 or c + half >= w:
            continue
        if used[r - half : r + half + 1, c - half : c + half + 1].any():
            continue
        patches.append((r, c))
        patch_unc.append(float(u))
        used[r - half : r + half + 1, c - half : c + half + 1] = True
        if len(patches) >= n_patches:
            break

    return patches, patch_unc


# ============================================================================
#  伪标签 + RF 训练（主动学习第二轮）
# ============================================================================

def train_from_annotations(feature_stack, annotation_labels, annotation_mask,
                            method="random_forest", **kwargs):
    """用标注数据训练分类器，预测全图。

    参数:
        feature_stack: (H, W, C) 特征张量
        annotation_labels: (H, W) int8 标注标签（0=未标注）
        annotation_mask: (H, W) bool 已标注有效区域
        method: "random_forest"

    返回:
        result: (H, W) int8 全图分类结果
        confidence: (H, W) float32 预测置信度
    """
    from sklearn.ensemble import RandomForestClassifier

    h, w, c = feature_stack.shape
    pixels = feature_stack.reshape(-1, c)  # (N, C)
    labels = annotation_labels.flatten()  # (N,)
    mask = annotation_mask.flatten()  # (N,)

    train_idx = np.where(mask & (labels > 0))[0]
    if len(train_idx) < 10:
        raise ValueError(f"Too few annotated pixels: {len(train_idx)}")

    X_train = pixels[train_idx]
    y_train = labels[train_idx]

    model = RandomForestClassifier(
        n_estimators=kwargs.get("n_estimators", 100),
        max_depth=kwargs.get("max_depth", 20),
        min_samples_leaf=kwargs.get("min_samples_leaf", 5),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 预测全图
    result = model.predict(pixels).reshape(h, w).astype(np.int8)
    probs = model.predict_proba(pixels)
    confidence = probs.max(axis=1).reshape(h, w).astype(np.float32)

    return result, confidence, model


# ============================================================================
#  统计与报告
# ============================================================================

def classification_statistics(result, meta=None, area_per_pixel_ha=None):
    """生成分类统计。

    优先从 meta 的 GridSpec 计算实际像元面积。

    返回:
        list[dict]: [
            {"class": 1, "name": "水体", "pixels": 5000, "area_ha": 50.0, "percent": 12.3},
            ...
        ]
    """
    if area_per_pixel_ha is None and meta is not None:
        try:
            from engine.grid import GridSpec
            gs = GridSpec.from_meta(meta)
            area_per_pixel_ha = gs.area_per_pixel_ha
        except Exception:
            area_per_pixel_ha = 0.01
    area_per_pixel_ha = area_per_pixel_ha or 0.01

    total = (result > 0).sum()
    stats = []
    for cv in sorted(CLASS_NAMES.keys()):
        if cv == 0:
            continue
        n_px = (result == cv).sum()
        if n_px == 0:
            continue
        stats.append({
            "class": cv,
            "name": CLASS_NAMES[cv],
            "pixels": int(n_px),
            "area_ha": round(n_px * area_per_pixel_ha, 2),
            "percent": round(n_px * 100 / max(total, 1), 2),
        })
    return stats
