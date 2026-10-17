# 🛰️ 遥感分析平台 RSAP v3

一站式遥感影像分析平台：**数据获取 → 变化检测 → 地物分类 → 时间序列 → 报告生成**。
基于 Sentinel-2 / Landsat 多源影像，提供 Streamlit Web 工作台、CLI 与批量任务三种使用方式。

## 功能模块

### 变化检测（engine/core.py）
- **7 种光谱指数**：NDVI / NDWI / MNDWI / NBR / EVI / SAVI / NDBI
- **三种检测模式**：多级 5 级分类（严重退化→强烈恢复）/ 二值 / 随机森林（ML）
- **连通域逐斑块面积过滤**：按斑块实际面积过滤，比传统形态学开运算更精确（numpy 向量化）
- **采样点自动生成**：质心法提取变化斑块坐标，GeoJSON / CSV / Shapefile 导出
- **暗色主题报告图**：分级图 + 面积饼图 + 统计信息

### 地物分类（engine/classifier.py）
- 多指数阈值联合分类（NDVI + MNDWI + NDBI）
- GMM 聚类（PCA 降维）+ 纹理特征（局部方差 / 边缘密度）
- 聚类标注映射界面，分类统计与 GeoTIFF / GeoJSON / CSV 导出

### 时间序列（engine/timeseries.py）
- 多期影像收集，逐像元 **Mann-Kendall 趋势检验 + Theil-Sen 斜率估计**
- 内存有界的窗口式栅格计算

### 数据获取（engine/data_sources.py）
- 经 **Microsoft Planetary Computer STAC API** 自动搜索 / 下载 Sentinel-2 L2A
- 按"边界覆盖率 × 云量"评分逐瓦片优选时相，多瓦片自动镶嵌与行政区裁剪

### 工程能力
- **Streamlit 多页工作台**：变化检测 / 地物分类 / 时间序列 / 批量处理 / 工具箱 / 任务中心 / 成果库
- **批量任务**：`batch/` 配置文件驱动 + 定时调度
- **报告生成**：PDF 专业报告（封面 / 方法 / 结果 / 采样点表）
- **Docker 部署**、`setup.py` pip 安装、pytest 测试

## 方法原理

```
指数计算:  NDVI = (NIR - Red) / (NIR + Red)
差值分级:  ΔNDVI = NDVI_year2 - NDVI_year1

  ΔNDVI < -0.25  → 严重退化       -0.25 ~ -0.15 → 中度退化
  -0.15 ~ +0.15  → 稳定            +0.15 ~ +0.25 → 中度恢复
  > +0.25        → 强烈恢复

后处理:    连通域分析 → 按斑块实际面积过滤 → 质心采样点
```

## 快速开始

```bash
# 环境要求: Python 3.10+，Windows / macOS / Linux
pip install -r requirements.txt

# Web 工作台（推荐）
streamlit run app.py

# CLI 批处理（参数见 config.py）
python main.py

# 批量任务
python batch/batch_runner.py --config batch/batch_config.json

# Docker
docker compose -f docker/docker-compose.yml up
```

无需 API Key：影像数据来自公开的 [Planetary Computer](https://planetarycomputer.microsoft.com/) STAC 目录（Sentinel-2 L2A 由 ESA 提供）。

## 配置说明

编辑 `config.py` 修改参数：

```python
REGION_NAME = "灵川县"            # 区域与边界
BOUNDARY_FILE = "lingchuan_boundary.geojson"
YEAR1, YEAR2 = 2025, 2026         # 时相（生长季 6-10 月）
DETECTION_MODE = "multi"          # "multi" 多级 | "binary" 二值 | "ml" 随机森林
INDEX_TYPE = "NDVI"               # 7 种指数可选
CHANGE_THRESHOLD = 0.15           # 变化阈值
SEVERE_THRESHOLD = 0.25           # 强烈变化阈值
MIN_AREA_BY_CLASS = {-3: 3000, -2: 5000, 2: 5000, 3: 3000}  # 每级最小斑块 m²
OUTPUT_FORMATS = "geojson,csv"    # geojson / csv / shp
```

## 输出文件

运行完成后在 `output/<区域名>/` 目录生成：

| 文件 | 说明 |
|------|------|
| `<区域>_变化检测.tif` | 多级分类结果 GeoTIFF（LZW 压缩） |
| `sample_points.geojson` | 采样点 GeoJSON（含类别属性） |
| `sample_points.csv` | 采样点 CSV（UTF-8 BOM，Excel 直接打开） |
| `<区域>_变化检测报告.png` | 综合解译报告（分级图 + 饼图 + 统计） |

## 项目结构

```
engine/      核心算法（core 变化检测 / classifier 分类 / timeseries 时序 /
             data_sources 数据源 / indices 指数 / ml_detector / grid 内存有界计算 /
             preprocessing / validation / report_generator / cache / utils）
frontend/    Streamlit 组件与共享工具
pages/       多页工作台（01 变化检测 ~ 07 成果库）
batch/       批量任务与调度
docker/      Dockerfile + docker-compose
tests/       pytest 测试
main.py      CLI 入口
app.py       Web 入口
config.py    全局参数
```

## 版本历史

### v3.0 (2026-07) — 平台化重构
- 与地物分类器合并为统一分析平台，算法层模块化拆分至 `engine/`
- 新增：7 种光谱指数、随机森林检测模式、GMM 聚类 + 纹理特征分类
- 新增：Mann-Kendall + Theil-Sen 逐像元时序趋势模块
- 新增：Landsat 8/9 数据源、批量任务调度、PDF 报告、Docker 部署
- 新增：Streamlit 多页工作台（7 页）与任务中心 / 成果库

### v2.1 (2026-06-29) — 性能优化
- 修复 matplotlib 报告中文乱码（自动检测系统字体）
- 修复 Windows 控制台输出乱码（Unicode 容错）
- 连通域过滤向量化（numpy 索引替代 Python 循环）
- 采样点生成批量坐标变换 + prepared geometry 空间查询
- 地图自适应降采样渲染

### v2.0 — 算法升级
- 多级 5 级分级检测、连通域逐斑块面积过滤、可选双边预滤波
- GeoJSON + CSV + Shapefile 多格式导出、Streamlit Web 前端

### v1.0 — 基础版本
- 二值 NDVI 变化检测、形态学开运算去噪、单格式 GeoJSON 导出

## License

MIT
