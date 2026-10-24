"""
遥感解译助手 v3 — 模块化引擎包

公共 API:
  - core: 变化检测核心算法
  - data_sources: 多源遥感数据接口
  - indices: 遥感指数计算
  - ml_detector: 机器学习变化检测
  - timeseries: 时间序列分析
  - report_generator: PDF 专业报告生成
  - utils: 共用工具函数
"""

from engine.utils import (
    log,
    get_boundary,
    boundary_pixel_mask,
    setup_matplotlib_font,
    get_needed_bands,
    INDEX_BAND_REQUIREMENTS,
)

from engine.core import (
    run_change_detection_v2,
    generate_sample_points,
    generate_sample_points_v2,
    generate_report,
    generate_report_v2,
    run_pipeline,
    run_pipeline_v3,
    prepare_data,
    prepare_period,
    find_cached_snapshot,
    find_cached_period,
)

from engine.params import RunParams

from engine.report_generator import PDFReportGenerator, generate_report_pdf

from engine.timeseries import TimeSeriesAnalyzer

from engine.cache import SnapshotCache, compute_aoi_hash, DatasetSnapshot

from engine.classifier import (
    classify_multi_index,
    pca_gmm_classify,
    build_feature_stack,
    extract_texture,
    uncertainty_sampling,
    sample_patches,
    train_from_annotations,
    apply_cluster_mapping,
    classification_statistics,
    CLASS_NAMES,
    CLASS_COLORS,
)

__all__ = [
    "log",
    "get_boundary",
    "boundary_pixel_mask",
    "setup_matplotlib_font",
    "get_needed_bands",
    "INDEX_BAND_REQUIREMENTS",
    "run_change_detection_v2",
    "generate_sample_points_v2",
    "generate_report_v2",
    "run_pipeline",
    "run_pipeline_v3",
    "prepare_data",
    "prepare_period",
    "find_cached_snapshot",
    "find_cached_period",
    "generate_sample_points",
    "generate_report",
    "RunParams",
    "PDFReportGenerator",
    "generate_report_pdf",
    "TimeSeriesAnalyzer",
    "SnapshotCache",
    "compute_aoi_hash",
    "DatasetSnapshot",
    "classify_multi_index",
    "pca_gmm_classify",
    "build_feature_stack",
    "extract_texture",
    "uncertainty_sampling",
    "sample_patches",
    "train_from_annotations",
    "apply_cluster_mapping",
    "classification_statistics",
    "CLASS_NAMES",
    "CLASS_COLORS",
]
