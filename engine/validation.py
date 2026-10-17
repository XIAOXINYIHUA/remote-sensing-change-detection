"""
空间验证 + 置信度校准模块

功能:
  - 空间分块 K-fold 交叉验证
  - ECE (Expected Calibration Error) 计算
  - Platt 校准 / 温度缩放
  - 低置信度拒判
  - 分类报告生成
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ============================================================================
#  空间分块
# ============================================================================

def spatial_blocks(
    shape: tuple[int, int],
    n_blocks: int = 5,
    block_aspect: float = 1.0,
) -> list[slice]:
    """将图像按空间分块（为空间 K-fold 准备）。

    Args:
        shape: (H, W) 图像尺寸
        n_blocks: 分块数量
        block_aspect: 块宽高比

    Returns:
        slices: 每个块的行索引 [slice, slice, ...]
    """
    h, w = shape
    # 在行方向分块（列方向也可分，为简单起见只分行）
    block_h = h // n_blocks
    if block_h < 1:
        n_blocks = max(1, h)
        block_h = 1

    slices = []
    for i in range(n_blocks):
        r_start = i * block_h
        r_end = (i + 1) * block_h if i < n_blocks - 1 else h
        slices.append(slice(r_start, r_end))

    return slices


# ============================================================================
#  K-fold 空间交叉验证
# ============================================================================

def spatial_kfold_split(
    labels: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成空间 K-fold 训练/验证索引。

    将像元按空间块分组，每折留一个块作为验证集。

    Args:
        labels: (H, W) 标签数组
        n_folds: 折数
        random_seed: 随机种子（用于在块内采样）

    Returns:
        folds: [(train_idx, val_idx), ...]
             每个索引是 (N,) 的 1D 索引数组
    """
    blocks = spatial_blocks(labels.shape, n_blocks=n_folds)
    total_pixels = labels.shape[0] * labels.shape[1]

    folds = []
    for val_block_idx in range(n_folds):
        val_mask_lookup = np.zeros(labels.shape, dtype=bool)
        val_mask_lookup[blocks[val_block_idx], :] = True

        val_idx = np.where(val_mask_lookup.ravel())[0]
        train_idx = np.where(~val_mask_lookup.ravel())[0]

        # 只保留有效（标注过的）像元
        labels_flat = labels.ravel()
        train_labeled = train_idx[labels_flat[train_idx] > 0]
        val_labeled = val_idx[labels_flat[val_idx] > 0]

        if len(train_labeled) > 0 and len(val_labeled) > 0:
            folds.append((train_labeled, val_labeled))

    return folds


# ============================================================================
#  ECE 计算
# ============================================================================

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """计算 Expected Calibration Error (ECE)。

    Args:
        probs: (N, n_classes) 预测概率
        labels: (N,) 真实标签
        n_bins: 置信度分箱数

    Returns:
        dict: {"ece": float, "bins": [...], "accuracy": float}
    """
    if labels.ndim > 1:
        labels = labels.ravel()
    if probs.ndim == 2:
        confidence = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidence = probs
        predictions = (probs >= 0.5).astype(np.int32)

    correct = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bins_info = []

    for i in range(n_bins):
        in_bin = (confidence >= bin_lowers[i]) & (confidence < bin_uppers[i])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidence[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
            bins_info.append({
                "bin": i, "range": f"{bin_lowers[i]:.2f}-{bin_uppers[i]:.2f}",
                "count": int(in_bin.sum()),
                "accuracy": float(avg_accuracy),
                "confidence": float(avg_confidence),
            })

    return {
        "ece": float(ece),
        "accuracy": float(correct.mean()),
        "bins": bins_info,
    }


# ============================================================================
#  置信度校准
# ============================================================================

def platt_scale(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Platt 缩放（逻辑回归校准），用于二分类。

    训练参数 a, b 使得:
        P(y=1|x) = 1 / (1 + exp(a * logit + b))

    Args:
        logits: (N,) 模型输出 logits
        labels: (N,) 真实标签 {0, 1}
        max_iter: 最大迭代

    Returns:
        (a, b) 校准参数
    """
    from scipy.optimize import minimize

    def _neg_log_likelihood(params):
        a, b = params
        p = 1.0 / (1.0 + np.exp(-(a * logits + b)))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.sum(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    result = minimize(
        _neg_log_likelihood,
        x0=[1.0, 0.0],
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    return float(result.x[0]), float(result.x[1])


def apply_platt_scale(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    """应用 Platt 缩放。"""
    return 1.0 / (1.0 + np.exp(-(a * logits + b)))


# ============================================================================
#  低置信度拒判
# ============================================================================

def reject_low_confidence(
    labels: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.6,
    reject_label: int = 0,
) -> np.ndarray:
    """对低置信度预测设为「拒绝」标签。

    Args:
        labels: (H, W) 分类结果
        confidence: (H, W) 置信度（最大概率）
        threshold: 置信度阈值，低于此值设为 reject_label
        reject_label: 拒绝标签值（默认 0 = 未分类）

    Returns:
        处理后的标签数组
    """
    result = labels.copy()
    result[confidence < threshold] = reject_label
    return result


# ============================================================================
#  分类报告
# ============================================================================

def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[dict[int, str]] = None,
) -> dict:
    """生成分类精度报告（逐类 + 总体）。

    Args:
        y_true: (N,) 真实标签
        y_pred: (N,) 预测标签
        class_names: {class_value: name}

    Returns:
        dict: {"classes": [...], "overall": {...}}
    """
    from sklearn.metrics import classification_report as sk_report

    # 过滤掉 0（未分类）
    valid = (y_true > 0) & (y_pred > 0)
    yt = y_true[valid]
    yp = y_pred[valid]

    if len(yt) == 0:
        return {"error": "No valid samples"}

    classes = sorted(set(np.unique(yt)) | set(np.unique(yp)))
    class_map = {v: class_names.get(v, f"Class {v}") for v in classes
                 if class_names and v > 0}

    report = sk_report(yt, yp, labels=[c for c in classes if c > 0],
                       output_dict=True, zero_division=0)

    # 格式化输出
    result = {"overall": {
        "accuracy": report.get("accuracy", 0),
        "macro_avg": report.get("macro avg", {}),
        "weighted_avg": report.get("weighted avg", {}),
        "n_samples": len(yt),
    }}

    classes_out = []
    for c in classes:
        if c > 0 and c in report:
            info = report[c]
            info["name"] = class_map.get(c, f"Class {c}")
            classes_out.append(info)
    result["classes"] = classes_out

    return result
