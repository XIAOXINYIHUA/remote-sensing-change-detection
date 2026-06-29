# 🛰️ 遥感解译助手 v2

基于 Sentinel-2 卫星影像的**植被变化检测**与**采样点自动生成**工具。支持多级 NDVI 差值分级（严重退化 → 强烈恢复），连通域斑块过滤，交互式 Web 前端。

## 功能特性

- **多级分级检测**：5 级分类（严重退化 / 中度退化 / 稳定 / 中度恢复 / 强烈恢复）
- **连通域分析**：逐斑块面积过滤，比传统形态学开运算更精确
- **自动数据获取**：通过 Microsoft Planetary Computer STAC API 搜索并下载 Sentinel-2 L2A 影像
- **多瓦片拼接**：自动处理跨瓦片区域
- **采样点生成**：质心法提取变化斑块坐标，支持 GeoJSON / CSV / Shapefile 导出
- **综合报告**：暗色主题 PNG 报告（分级图 + 饼图 + 统计信息）
- **Web 前端**：Streamlit 交互界面，folium 地图可视化，参数面板配置

## 方法原理

```
NDVI = (NIR - Red) / (NIR + Red)
ΔNDVI = NDVI_year2 - NDVI_year1

分类规则:
  ΔNDVI < -0.25  → 严重退化
  -0.25 ~ -0.15  → 中度退化
  -0.15 ~ +0.15  → 稳定
  +0.15 ~ +0.25  → 中度恢复
  > +0.25        → 强烈恢复

后处理:
  连通域分析 → 按斑块实际面积过滤小斑块 → 边界裁剪
```

## 快速开始

### 环境要求

- Python 3.10+
- Windows / macOS / Linux

### 安装

```bash
cd E:\解译工具
pip install -r requirements.txt
```

### 启动 Web 界面（推荐）

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，在侧边栏选择地区 → 下载影像 → 运行分析 → 查看结果。

### 命令行运行

```bash
python main.py
```

根据 `config.py` 中的参数配置直接运行完整流程。

## 配置说明

编辑 `config.py` 修改参数：

```python
# 区域设置
REGION_NAME = "灵川县"
BOUNDARY_FILE = "lingchuan_boundary.geojson"

# 时相
YEAR1 = 2025
YEAR2 = 2026
MONTH_START = 6   # 植被生长季
MONTH_END = 10

# 检测模式
DETECTION_MODE = "multi"        # "multi" 多级 | "binary" 二值
CHANGE_THRESHOLD = 0.15         # NDVI 变化阈值
SEVERE_THRESHOLD = 0.25         # 强烈变化阈值

# 连通域过滤（每级独立最小面积 m²）
MIN_AREA_BY_CLASS = {
    -3: 3000,  # 严重退化
    -2: 5000,  # 中度退化
     2: 5000,  # 中度恢复
     3: 3000,  # 强烈恢复
}

# 采样点
NUM_SAMPLE_POINTS = None        # 上限（None=不限）
MIN_POINT_DISTANCE_M = None     # 最小间距（None=不限）

# 输出
OUTPUT_FORMATS = "geojson,csv"  # 逗号分隔: geojson,csv,shp
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
解译工具/
├── main.py                    # 核心引擎：STAC搜索/下载/检测/采样/报告
├── app.py                     # Streamlit Web 前端
├── config.py                  # 全局参数配置
├── requirements.txt           # Python 依赖
├── run_me.bat                 # Windows 批处理启动
├── README.md                  # 本文件
├── *.geojson                  # 行政区边界文件
├── data/                      # 影像缓存目录
│   └── <区域>/
│       ├── B4_2025.tif        # 红波段
│       ├── B8_2025.tif        # 近红外波段
│       ├── B4_2026.tif
│       └── B8_2026.tif
└── output/                    # 分析输出目录
    └── <区域>/
        ├── <区域>_变化检测.tif
        ├── sample_points.geojson
        ├── sample_points.csv
        └── <区域>_变化检测报告.png
```

## 依赖

```
numpy, rasterio, opencv-python, matplotlib
shapely, pyproj, pandas
pystac-client, planetary-computer
streamlit, folium, streamlit-folium
fiona (可选, Shapefile导出)
```

## 版本历史

### v2.1 (2026-06-29) — 性能优化

- 🐛 修复 matplotlib 报告 PNG 中文乱码（自动检测系统字体）
- 🐛 修复 Windows 控制台输出乱码（`chcp 65001` + Unicode 容错）
- ⚡ 连通域过滤向量化（numpy 索引替代 Python 循环，加速 10-50x）
- ⚡ 采样点生成批量坐标变换 + prepared geometry 空间查询
- ⚡ 地图自适应降采样渲染（大区域不卡顿）

### v2.0 — 算法升级

- 多级 5 级分级检测
- 连通域分析逐斑块面积过滤
- 可选双边预滤波降噪
- GeoJSON + CSV + Shapefile 多格式导出
- Streamlit 交互式 Web 前端

### v1.0 — 基础版本

- 二值 NDVI 变化检测
- 形态学开运算去噪
- 单格式 GeoJSON 导出

## 数据来源

Sentinel-2 L2A 影像通过 [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC 目录获取，数据由 ESA 提供。

## License

MIT
