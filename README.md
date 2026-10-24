# 🛰️ 遥感分析平台 RSAP v3

一站式遥感影像分析平台：**数据获取 → 变化检测 → 地物分类 → 时间序列 → 报告生成**。
基于 Sentinel-2 / Landsat 多源影像，提供 Streamlit Web 工作台、CLI 与批量任务三种使用方式。

## 功能模块

### 变化检测（engine/core.py）
- **7 种光谱指数**：NDVI / NDWI / MNDWI / NBR / EVI / SAVI / NDBI
- **三种检测模式**：多级 5 级分类（严重退化→强烈恢复）/ 二值 / 随机森林（ML，需自备训练好的模型）
- **阈值**：每种指数有独立默认阈值，可手动指定，或按稳定区噪声（稳健 σ）自适应估计
- **连通域逐斑块面积过滤**：按各级变化类别分别做连通域，按实际像元面积（10 m / 30 m）过滤
- **采样点自动生成**：每个斑块取一个内部代表点（凹形 / 环形斑块也保证落在斑块内），
  按投影坐标（米）做最小间距稀疏；GeoJSON / CSV / Shapefile (EPSG:4326) 导出
- **报告图**：类别名、配色、阈值随指数与实际运行参数变化

### 地物分类（engine/classifier.py）
- 多指数阈值联合分类（NDVI + MNDWI + NDBI）
- GMM 聚类（PCA 降维）+ 纹理特征（局部方差 / 边缘密度）
- 聚类标注映射界面，分类统计与 GeoTIFF / GeoJSON / CSV 导出

### 时间序列（engine/timeseries.py）
- 多期影像统一到同一 AOI 网格，每期做云掩膜，缺测记为 NaN
- 逐像元 **Mann-Kendall 趋势检验 + Theil-Sen 斜率估计**（向量化、按行分块；
  n ≤ 10 用精确分布求 p 值，每像元至少 4 期有效观测）

### 数据获取与预处理（engine/data_sources.py / preprocessing.py）
- 经 **Microsoft Planetary Computer STAC API** 检索 Sentinel-2 L2A、Landsat 8/9 C2 L2
  （Sentinel-1 GRD 为实验性，暂无可用指数）
- 按"边界覆盖率 × 云量"评分逐瓦片优选时相；只窗口读取 AOI 范围（COG），
  所有时相、波段对齐到同一确定性网格（AOI 质心所在 UTM 带，网格角点对齐数据源像元）
- Sentinel-2 处理基线 ≥ 04.00 的 +1000 DN 偏移逐景归一化
- 云 / 阴影掩膜：Sentinel-2 用 SCL，Landsat 用 QA_PIXEL，在对齐后的网格上执行
- 单期快照缓存：原始 DN + 质量波段按「AOI / 数据源 / 年份 / 月份 / 云量 / 波段 / 预处理版本」
  逐期缓存，变化检测、地物分类、时间序列共用（波段为超集即可复用，同一时相只下载一次）；
  云量阈值统一取项目设置

### 工程能力
- **Streamlit 多页工作台**：变化检测 / 地物分类 / 时间序列 / 批量处理 / 工具箱 / 任务中心 / 成果库
- **批量任务**：`batch/` 配置文件驱动 + 定时调度
- **报告生成**：PDF 专业报告（封面 / 方法 / 结果 / 采样点表）
- **运行参数显式化**：所有入口通过 `RunParams` 传参并在下载前校验，流水线不修改全局 `config`，
  多个 Web 会话 / 批量任务可安全并发
- **Docker 部署**、`pyproject.toml` 安装、离线 pytest 回归测试、GitHub Actions CI

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
pip install -e ".[web]"          # 或 pip install -r requirements.txt

# Web 工作台（推荐）
streamlit run app.py

# CLI：用仓库自带的约 5 km × 5 km 示例区域试跑（未指定的参数取 config.py 默认值）
python main.py --boundary examples/sample_aoi.geojson --region 示例区 --year1 2023 --year2 2024 --index NDVI
python main.py --help

# 批量任务
python batch/batch_runner.py --config batch/batch_config.json

# Docker
docker compose -f docker/docker-compose.yml up
# 国内网络构建镜像时可指定镜像源（不指定则用官方源）
docker compose -f docker/docker-compose.yml build \
  --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn \
  --build-arg PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
```

无需 API Key：影像数据来自公开的 [Planetary Computer](https://planetarycomputer.microsoft.com/) STAC 目录（Sentinel-2 L2A 由 ESA 提供）。

## 配置说明

`config.py` 只提供默认值；一次运行的实际参数由 Web 界面、CLI 参数或批量任务字段构造为
`engine.params.RunParams`（非法组合如两期年份相同、强烈阈值 ≤ 变化阈值会在下载前报错）。
修改默认值：

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
| `<区域>_<指数>_变化检测.tif` | 分类结果 Cloud Optimized GeoTIFF（int8，LZW） |
| `sample_points.geojson` | 采样点 GeoJSON（含类别、面积属性与运行元数据） |
| `sample_points.csv` | 采样点 CSV（UTF-8 BOM，Excel 直接打开） |
| `sample_points.shp` | 采样点 Shapefile（需在格式中加 `shp`） |
| `<区域>_<指数>_变化检测报告.png` | 综合解译报告（分级图 + 饼图 + 统计） |

## 项目结构

```
engine/      核心算法（core 变化检测 / classifier 分类 / timeseries 时序 /
             data_sources 数据源 / indices 指数 / ml_detector / grid 统一分析网格 /
             preprocessing / params 运行参数 / validation / report_generator / cache / utils）
services/    服务层：变化检测 / 地物分类 / 时间序列 / 批量任务的取数与分析流程
             （不依赖 streamlit，进度走回调）；jobs 本机任务执行器，job_tasks 后台任务入口
frontend/    Streamlit 组件（含区域 / 时相 / 数据源共用侧栏）与共享工具
pages/       多页工作台（01 变化检测 ~ 07 成果库），只负责界面与会话状态
batch/       批量任务与调度（定时调度需另装 schedule）
docker/      Dockerfile + docker-compose
examples/    示例 AOI
tests/       pytest 测试（全部离线）
main.py      CLI 入口
app.py       Web 入口
config.py    全局参数
```

## 版本历史

### v3.3 — 前端修正与统一取数
- 面积按椭球计算；结果地图改为逐类图片图层（全部像元、可缓存）
- 成果库列出未归档输出；首页区分可复用影像与旧版数据；移除无法运行的 SAR 选项
- 地物分类、时间序列与变化检测共用单期快照缓存
- 页面冒烟测试（`RSAP_STATE_DIR` 把界面状态文件隔离到临时目录）
- 取数与分析流程移到 `services/`；三个分析页共用区域 / 时相 / 数据源侧栏组件
- 本机任务执行器：收集影像、变化检测、分类影像下载、时间序列、批量任务在独立工作进程中执行，
  页面只提交任务并显示进度（可取消）；切换页面、刷新或关闭浏览器不影响任务，回到页面自动载入结果。
  任务中心列出全部后台任务（进度、取消、重试、日志）。任务记录在 `data/jobs/`，
  同时运行数见 `config.JOB_MAX_WORKERS`；应用关闭时未结束的任务下次启动记为「已中断」，可重试。
  环境变量：`RSAP_JOBS_DIR` 任务目录、`RSAP_JOB_WORKERS` 同时运行数、`RSAP_CACHE_DIR` 快照缓存目录、
  `RSAP_JOB_MODE=thread` 改在服务进程的线程中执行（测试用）
- 一个项目一个状态文件 `.project_state.json`（目录可用 `RSAP_STATE_DIR` 改，Docker 镜像放在 `data/`）：区域、项目设置、
  分析记录与各页需要保留的状态都在里面。分析区域以项目为准，任一页面切换区域，其他页面清掉旧区域的
  影像与结果；多个浏览器标签页共用同一文件。首次启动时自动并入旧版的 `.shared_state.json`、
  `.app_state.json`、`.classify_state.json`、`.ts_state.json`、`.batch_state.json`，
  旧文件不再使用，确认无误后可删除
- 清理：删除无引用的函数、旧版前端组件（`frontend/components/params.py`、`charts.py`、
  `frontend/pages/`）；按钮等组件的 `use_container_width` 改为 Streamlit 1.58 的 `width` 参数
  （依赖下限相应提到 `streamlit>=1.58`）；ruff 启用未使用导入 / 变量检查
- Docker：状态文件与缓存目录改到容器用户可写的位置；构建参数 `APT_MIRROR` / `PIP_MIRROR` 可指定镜像源
- 修正：地物分类面积统计按实际像元大小（Landsat 30 m）；主动学习斑块数设置生效；
  时间序列至少 4 期（与逐像元最少观测数一致）；PDF 报告填入区域面积

### v3.2 — 商业化加固
- 数据获取改为 AOI 窗口读取 + 确定性 UTM 分析网格，跨 UTM 带瓦片正确镶嵌
- 时间序列：多期统一网格 + 云掩膜，Mann-Kendall / Sen 斜率向量化
- 采样点：按类别分别做连通域、内部代表点、按米稀疏、真实像元面积；Shapefile 坐标系修正
- 报告：随指数与实际阈值变化，补齐 NDBI 类别名 / 配色 / 阈值
- `RunParams` 取代运行时改写全局 `config`；Web 端资料收集与分析共享快照缓存
- `pyproject.toml`、精简 Docker 镜像、示例 AOI、GitHub Actions CI

### v3.1 — P0 正确性修复
- 云掩膜在对齐网格上执行；Sentinel-2 基线偏移归一化；自适应阈值改用稳健 σ
- 批量运行器崩溃修复；主动学习斑块落在高不确定区域

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

MIT —— 见 [LICENSE](LICENSE)
