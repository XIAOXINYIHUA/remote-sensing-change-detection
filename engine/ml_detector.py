"""
遥感解译助手 v3 — 机器学习变化检测模块

方法:
  - 随机森林 (Random Forest): 基于多指数+纹理特征
  - 轻量 CNN: 基于 PyTorch 的 patch-based 变化检测 (可选)

使用方式:
  from engine.ml_detector import run_ml_change_detection
  result = run_ml_change_detection(bands_y1, bands_y2, valid_mask,
                                    method="random_forest")
"""

import os
import time
import pickle
import numpy as np


class RandomForestChangeDetector:
    """基于随机森林的变化检测器。

    特征包括:
      - 单期指数值 (NDVI_y1, NDVI_y2, NDWI_y1, NDWI_y2, ...)
      - 指数差值 (dNDVI, dNDWI, ...)
      - 原始波段差值
      - 纹理特征 (3x3 局部方差)

    输出: 与阈值法相同的 5 级分类 (-3/-2/1/2/3)
    """

    def __init__(self, model_path=None):
        """初始化检测器。

        Args:
            model_path: 预训练模型路径 (.pkl)，None 表示需训练。
        """
        self.model = None
        self.feature_names = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _extract_features(self, bands_y1, bands_y2, valid_mask):
        """从多时相波段中提取特征矩阵。

        Args:
            bands_y1: {band_name: np.ndarray} 前时相。
            bands_y2: {band_name: np.ndarray} 后时相。
            valid_mask: bool 数组，True=有效像元。

        Returns:
            np.ndarray: (N_valid, n_features) 特征矩阵。
        """
        from engine.indices import compute_index, INDEX_FUNCTIONS

        h, w = list(bands_y1.values())[0].shape[:2]
        valid_idx = np.where(valid_mask)
        features = []

        # 1. 各指数单期值 + 差值
        for name in ["NDVI", "NDWI", "NBR"]:
            try:
                i1 = compute_index(name, bands_y1)
                i2 = compute_index(name, bands_y2)
                features.append(i1[valid_idx].reshape(-1, 1))
                features.append(i2[valid_idx].reshape(-1, 1))
                features.append((i2 - i1)[valid_idx].reshape(-1, 1))
            except KeyError:
                continue

        # 2. 原始波段差值
        for band_name in ['red', 'nir', 'swir1', 'swir2']:
            if band_name in bands_y1 and band_name in bands_y2:
                diff = (bands_y2[band_name].astype(np.float32) -
                        bands_y1[band_name].astype(np.float32))
                features.append(diff[valid_idx].reshape(-1, 1))

        # 3. 纹理特征：NDVI 差值图的 3x3 局部方差
        if 'nir' in bands_y1 and 'red' in bands_y1:
            ndvi1 = compute_index("NDVI", bands_y1)
            ndvi2 = compute_index("NDVI", bands_y2)
            diff_ndvi = ndvi2 - ndvi1
            try:
                from scipy.ndimage import generic_filter
                local_var = generic_filter(diff_ndvi, np.var, size=3)
                features.append(local_var[valid_idx].reshape(-1, 1))
            except ImportError:
                pass

        X = np.hstack(features)
        self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        return X

    def train(self, bands_y1, bands_y2, labels, valid_mask=None,
              test_size=0.2, **kwargs):
        """训练随机森林模型。

        Args:
            bands_y1/y2: 波段字典。
            labels: 真实标签数组（与阈值法相同的 5 级编码）。
            valid_mask: 有效像素掩膜（None 时从 labels 推断）。
            test_size: 验证集比例。
            **kwargs: 传递给 RandomForestClassifier 的参数。

        Returns:
            self.model
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        if valid_mask is None:
            valid_mask = labels != 0

        X = self._extract_features(bands_y1, bands_y2, valid_mask)
        y = labels[valid_mask]

        valid = y != 0
        X, y = X[valid], y[valid]

        if len(X) == 0:
            raise ValueError("No valid training samples")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)

        self.model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 15),
            min_samples_leaf=kwargs.get('min_samples_leaf', 10),
            class_weight='balanced',
            random_state=42, n_jobs=-1)

        print(f"  Training RF on {X_train.shape[0]} samples, "
              f"{X_train.shape[1]} features...")
        t0 = time.time()
        self.model.fit(X_train, y_train)
        print(f"  Training done: {time.time()-t0:.1f}s")

        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(f"  Accuracy: {(y_pred == y_test).mean():.3f}")

        return self.model

    def predict(self, bands_y1, bands_y2, valid_mask):
        """预测整个区域的变化类别。

        Args:
            bands_y1/y2: 波段字典。
            valid_mask: 有效像素掩膜。

        Returns:
            np.ndarray: 分类结果 (shape 同输入)。
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        X = self._extract_features(bands_y1, bands_y2, valid_mask)
        y_pred = self.model.predict(X)

        h, w = list(bands_y1.values())[0].shape[:2]
        result = np.zeros((h, w), dtype=np.int8)
        valid_idx = np.where(valid_mask)
        result[valid_idx] = y_pred[:valid_idx[0].shape[0]]

        return result

    def save(self, path):
        """保存模型到磁盘。"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
            }, f)
        print(f"  Model saved: {path}")

    def load(self, path):
        """从磁盘加载模型。"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = data.get('feature_names')
        print(f"  Model loaded: {path}")


# ============================================================================
#  ML 检测统一入口
# ============================================================================

def run_ml_change_detection(bands_y1, bands_y2, valid_mask,
                            method="random_forest", model_path=None,
                            train_mode=False, labels=None,
                            **kwargs):
    """机器学习变化检测统一入口。

    Args:
        bands_y1/y2: 前/后时相波段字典。
        valid_mask: 有效像素 bool 数组。
        method: "random_forest" (默认)。
        model_path: 模型文件路径。
        train_mode: True=训练（需 labels），False=推理。
        labels: 训练标签数组（train_mode=True 时必需）。
        **kwargs: 传递给具体检测器的参数。

    Returns:
        tuple: (result_array, stats_dict)
    """
    if method == "random_forest":
        detector = RandomForestChangeDetector(model_path)

        if train_mode and labels is not None:
            detector.train(bands_y1, bands_y2, labels, valid_mask, **kwargs)
            if model_path:
                detector.save(model_path)

        result = detector.predict(bands_y1, bands_y2, valid_mask)

        # 生成与阈值法兼容的统计格式
        h, w = result.shape
        cls_names = {-3: "严重退化", -2: "中度退化", 1: "稳定",
                      2: "中度恢复", 3: "强烈恢复"}
        stats = {}
        total_valid = valid_mask.sum()
        for cv in [-3, -2, 1, 2, 3]:
            n_px = (result == cv).sum()
            area_ha = n_px * 0.01
            pct = n_px * 100 / max(total_valid, 1)
            stats[cv] = {
                "name": cls_names[cv],
                "pixels": int(n_px),
                "area_ha": round(area_ha, 2),
                "percent": round(pct, 2)
            }
        changed_px = sum((result == c).sum() for c in [-3, -2, 2, 3])
        stats["changed_total"] = {
            "pixels": int(changed_px),
            "area_ha": round(changed_px * 0.01, 2),
            "percent": round(changed_px * 100 / max(total_valid, 1), 2)
        }
        stats["valid_pixels"] = int(total_valid)
        stats["total_pixels"] = h * w
        stats["ml_method"] = method

        return result, stats

    else:
        raise ValueError(f"Unknown ML method: {method}. "
                         f"Available: random_forest")
