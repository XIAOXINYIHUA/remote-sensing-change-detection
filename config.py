"""
遥感解译助手 v2 - 显著变化检测与采样点生成
修改这里来适配不同的区域和时间
"""

# ============ 区域设置 ============
REGION_NAME = "灵川县"
# 行政区边界文件（GeoJSON Polygon）
BOUNDARY_FILE = "lingchuan_boundary.geojson"

# ============ 时间设置 ============
# 前一时相年份
YEAR1 = 2025
# 后一时相年份
YEAR2 = 2026

# 月份范围（6-10月植被生长季，云量较少）
MONTH_START = 6
MONTH_END = 10

# ============ 数据源设置 ============
# 云量过滤（%）
MAX_CLOUD_PCT = 50
# 瓦片最小覆盖比例（低于此值忽略该瓦片）
MIN_TILE_COVERAGE = 0.01

# ============ 性能优化设置 ============
# 并行下载多瓦片（⚠ rasterio/GDAL 线程不安全，默认关闭。如需加速可手动开启）
PARALLEL_DOWNLOADS = False
# 最大并行下载线程数
MAX_PARALLEL_WORKERS = 4

# ============ 变化检测参数 ============

# --- 检测模式 ---
# "binary" : 原始的二值检测模式（变化/无变化）
# "multi"  : 多级分级检测（严重退化/中度退化/稳定/中度恢复/强烈恢复）
DETECTION_MODE = "multi"

# --- NDVI变化阈值 ---
# 二值模式: CHANGE_THRESHOLD 决定变化临界值
# 多级模式:
#   NDVI_diff < -0.25  → 严重退化 (class -3)
#   -0.25 ~ -0.15      → 中度退化 (class -2)
#   -0.15 ~ +0.15      → 稳定     (class 1)
#   +0.15 ~ +0.25      → 中度恢复 (class 2)
#   > +0.25            → 强烈恢复 (class 3)
CHANGE_THRESHOLD = 0.15
SEVERE_THRESHOLD = 0.25

# --- 双边预滤波 ---
USE_BILATERAL_FILTER = False
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 0.1
BILATERAL_SIGMA_SPACE = 5

# --- 形态学开运算核大小（像素）---
MORPH_KERNEL_SIZE = 4

# --- 连通域分析参数（多级模式）---
# 每个变化级别的独立最小斑块面积（平方米）
MIN_AREA_BY_CLASS = {
    -3: 3000,
    -2: 5000,
     2: 5000,
     3: 3000,
}

# 最小变化面积（平方米）- 二值模式用
MIN_CHANGE_AREA_M2 = 5000
AREA_PER_PIXEL_HECTARES = 0.01
MIN_CHANGE_AREA_PX = max(1, MIN_CHANGE_AREA_M2 // 100)

# ============ 采样点参数 ============
NUM_SAMPLE_POINTS = None
MIN_POINT_DISTANCE_M = None
SAMPLE_STRATEGY = "centroid"  # centroid, random, grid

# ============ 输出设置 ============
OUTPUT_DIR = "output"
SAMPLE_OUTPUT = "sample_points.geojson"
OUTPUT_FORMATS = "geojson,csv"
GENERATE_REPORT = True
