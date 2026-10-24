"""
RunParams — 一次分析运行的全部参数

流水线只读取显式传入的 RunParams，不读写全局 config：
Streamlit 的多个会话共享同一进程，改写模块级 config 会让并发运行互相覆盖参数。
config.py 只作为默认值来源（RunParams.from_config）。
"""

from __future__ import annotations

import dataclasses
from typing import Optional

VALID_SOURCES = ("sentinel2", "landsat8", "landsat9", "sentinel1")
VALID_MODES = ("multi", "binary", "ml")
VALID_FORMATS = ("geojson", "csv", "shp")


@dataclasses.dataclass
class RunParams:
    """变化检测运行参数。"""

    region: str
    year1: int
    year2: int
    boundary_file: Optional[str] = None
    month_start: int = 6
    month_end: int = 10
    data_source: str = "sentinel2"
    index_type: str = "NDVI"
    detection_mode: str = "multi"
    max_cloud_pct: float = 50

    # 阈值：None = INDEX_THRESHOLDS 注册表 > config 默认
    threshold: Optional[float] = None
    severe_threshold: Optional[float] = None
    adaptive_threshold: bool = False

    # 斑块面积：min_area_m2 非 None 时统一覆盖 min_area_by_class
    min_area_m2: Optional[float] = None
    min_area_by_class: dict = dataclasses.field(
        default_factory=lambda: {-3: 3000, -2: 5000, 2: 5000, 3: 3000})
    binary_min_area_m2: float = 5000
    morph_kernel_size: int = 4
    use_bilateral_filter: bool = False

    # 采样
    num_sample_points: Optional[int] = None
    min_point_distance_m: Optional[float] = None

    # 输出
    output_formats: tuple = ("geojson", "csv")
    generate_report: bool = True

    # 机器学习
    ml_model: str = "random_forest"
    ml_model_path: Optional[str] = None

    def __post_init__(self):
        self.year1, self.year2 = int(self.year1), int(self.year2)
        self.month_start, self.month_end = int(self.month_start), int(self.month_end)
        self.index_type = self.index_type.upper()
        if isinstance(self.output_formats, str):
            self.output_formats = tuple(
                f.strip() for f in self.output_formats.split(",") if f.strip())
        self.output_formats = tuple(self.output_formats)
        self.validate()

    def validate(self):
        """参数校验，错误时抛 ValueError（在下载前尽早失败）。"""
        errors = []
        if self.data_source not in VALID_SOURCES:
            errors.append(f"未知数据源 {self.data_source}，可选 {VALID_SOURCES}")
        if self.detection_mode not in VALID_MODES:
            errors.append(f"未知检测模式 {self.detection_mode}，可选 {VALID_MODES}")
        if self.year1 == self.year2:
            errors.append("前后时相年份不能相同")
        if not (1 <= self.month_start <= 12 and 1 <= self.month_end <= 12):
            errors.append("月份需在 1-12 之间")
        elif self.month_start > self.month_end:
            errors.append("起始月份不能晚于结束月份")
        if not 0 < self.max_cloud_pct <= 100:
            errors.append("云量阈值需在 (0, 100] 之间")
        if self.threshold is not None and self.severe_threshold is not None \
                and self.severe_threshold <= self.threshold:
            errors.append("强烈变化阈值必须大于变化阈值")
        bad_fmt = [f for f in self.output_formats if f not in VALID_FORMATS]
        if bad_fmt:
            errors.append(f"未知输出格式 {bad_fmt}，可选 {VALID_FORMATS}")
        if self.num_sample_points is not None and self.num_sample_points < 1:
            errors.append("采样数量上限需 >= 1（不限请用 None）")
        if errors:
            raise ValueError("；".join(errors))

    @property
    def change_class_names(self) -> dict:
        from engine.indices import INDEX_CLASS_NAMES
        return INDEX_CLASS_NAMES.get(self.index_type, INDEX_CLASS_NAMES["NDVI"])

    @classmethod
    def from_config(cls, **overrides) -> "RunParams":
        """以 config.py 为默认值构造（只读 config，不修改）。

        overrides 中值为 None 的键视为「未指定」，沿用 config。
        """
        import config
        base = dict(
            region=config.REGION_NAME,
            boundary_file=config.BOUNDARY_FILE,
            year1=config.YEAR1,
            year2=config.YEAR2,
            month_start=config.MONTH_START,
            month_end=config.MONTH_END,
            data_source=getattr(config, "DATA_SOURCE", "sentinel2"),
            index_type=getattr(config, "INDEX_TYPE", "NDVI"),
            detection_mode=config.DETECTION_MODE,
            max_cloud_pct=config.MAX_CLOUD_PCT,
            min_area_by_class=dict(config.MIN_AREA_BY_CLASS),
            binary_min_area_m2=config.MIN_CHANGE_AREA_M2,
            morph_kernel_size=config.MORPH_KERNEL_SIZE,
            use_bilateral_filter=config.USE_BILATERAL_FILTER,
            num_sample_points=config.NUM_SAMPLE_POINTS,
            min_point_distance_m=config.MIN_POINT_DISTANCE_M,
            output_formats=config.OUTPUT_FORMATS,
            generate_report=config.GENERATE_REPORT,
            ml_model=getattr(config, "ML_MODEL", "random_forest"),
            ml_model_path=getattr(config, "ML_MODEL_PATH", None),
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["min_area_by_class"] = {str(k): v for k, v in self.min_area_by_class.items()}
        d["output_formats"] = list(self.output_formats)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunParams":
        """to_dict() 的逆操作（JSON 往返后 min_area_by_class 的键是字符串）。"""
        d = dict(d)
        if "min_area_by_class" in d:
            d["min_area_by_class"] = {int(k): v for k, v in d["min_area_by_class"].items()}
        return cls(**d)
