"""
侧栏共用组件：分析区域、时间范围、数据源与指数

变化检测 / 地物分类 / 时间序列三个分析页用同一套控件。区域的会话键带页面前缀
（cd / cl / ts）：{prefix}_boundary、{prefix}_region、{prefix}_slug；切换区域时同步到 ProjectState。
"""

import json
from pathlib import Path

import streamlit as st

from engine.utils import geodesic_area_km2
from frontend.project_state import ProjectState
from frontend.utils import (
    ANALYSIS_SOURCES,
    CAPABILITY_MATRIX,
    DATA_SOURCE_LABELS,
    INDEX_HELP,
    detect_region,
    fetch_boundary_osm,
    load_gj,
    slug,
)

# 边界文件默认放在项目根目录（OSM 搜索结果也写到这里）
BOUNDARY_DIR = Path(__file__).resolve().parents[2]

YEAR_MIN, YEAR_MAX = 2015, 2027


def _same_file(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except (TypeError, OSError):
        return False


def _boundary_label(path):
    try:
        return Path(path).resolve().relative_to(BOUNDARY_DIR).as_posix()
    except ValueError:
        return str(path)


# =====================================================================
#  分析区域
# =====================================================================

def region_picker(prefix, on_change=None, boundary_dir=None):
    """分析区域：边界文件选择、OSM 搜索，以及中心 / 面积说明。

    可选的边界文件为 boundary_dir（默认项目根目录）下的 *.geojson；当前边界在别处
    （如向导或 CLI 用过的 examples/ 下的文件）时也列出，不会被换成别的区域。
    区域变化时写入会话键与 ProjectState，调用 on_change()（页面在其中重置已下载 /
    已分析的状态并保存页面状态），然后 rerun。

    Returns:
        当前边界几何（WGS84）；没有可用边界时 None。
    """
    S = st.session_state
    root = Path(boundary_dir) if boundary_dir else BOUNDARY_DIR
    k_boundary, k_region, k_slug = f"{prefix}_boundary", f"{prefix}_region", f"{prefix}_slug"

    def select(path, region):
        S[k_boundary], S[k_region], S[k_slug] = str(path), region, slug(region)
        ProjectState.set_aoi(region, str(path), S[k_slug])
        if on_change:
            on_change()
        st.rerun()

    st.sidebar.subheader("分析区域")
    options = {f.name: str(f) for f in sorted(root.glob("*.geojson"))}
    cur = S.get(k_boundary)
    if cur and Path(cur).exists() and not any(_same_file(p, cur) for p in options.values()):
        options[_boundary_label(cur)] = cur
    if options:
        names = list(options)
        cur_name = next((n for n, p in options.items() if cur and _same_file(p, cur)), None)
        sel = st.sidebar.selectbox("边界文件", names,
                                   index=names.index(cur_name) if cur_name else 0)
        if sel != cur_name:
            select(options[sel], detect_region(options[sel]))

    query = st.sidebar.text_input("搜索地区", "", placeholder="城市/区县名").strip()
    if st.sidebar.button("获取边界", icon=":material/search:", width="stretch",
                         disabled=not query):
        with st.sidebar.status(f"正在搜索 {query}..."):
            gj, err = fetch_boundary_osm(query)
        if err:
            st.sidebar.error(err)
        else:
            fp = root / f"{slug(query)}.geojson"
            fp.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
            select(fp, query)

    cur = S.get(k_boundary)
    if not cur or not Path(cur).exists():
        st.sidebar.error("请选择边界文件")
        return None
    try:
        geom = load_gj(cur)[1]
    except Exception:
        st.sidebar.error("边界解析失败")
        return None
    st.sidebar.caption(f"中心: {geom.centroid.y:.3f}, {geom.centroid.x:.3f}  "
                       f"面积: {geodesic_area_km2(geom):,.1f} km²")
    return geom


# =====================================================================
#  时间范围
# =====================================================================

def _clamp_year(year):
    return min(max(int(year), YEAR_MIN), YEAR_MAX)


def year_pair(default1, default2):
    """变化检测的前 / 后时相年份。Returns (year1, year2)。"""
    y1 = st.sidebar.number_input("前时相年份", YEAR_MIN, YEAR_MAX, _clamp_year(default1), 1)
    y2 = st.sidebar.number_input("后时相年份", YEAR_MIN, YEAR_MAX, _clamp_year(default2), 1)
    if y1 == y2:
        st.sidebar.error("两期年份不能相同")
    return int(y1), int(y2)


def single_year(default):
    """单期年份。"""
    return int(st.sidebar.number_input("年份", YEAR_MIN, YEAR_MAX, _clamp_year(default), 1))


def year_series(default_start=2020, default_end=2026):
    """多期年份序列。Returns (years, start, end, step)；结束早于起始时 years 为空。"""
    start = st.sidebar.number_input("起始年份", YEAR_MIN, YEAR_MAX, _clamp_year(default_start), 1)
    end = st.sidebar.number_input("结束年份", YEAR_MIN, YEAR_MAX, _clamp_year(default_end), 1)
    step = st.sidebar.selectbox("步长", [1, 2, 3], index=0)
    years = list(range(int(start), int(end) + 1, step))
    if years:
        st.sidebar.caption(f"共 {len(years)} 期: {years[0]} 至 {years[-1]}")
    else:
        st.sidebar.error("结束年份必须晚于起始年份")
    return years, int(start), int(end), step


def month_range(default_start=6, default_end=10):
    """每年取影像的月份范围（含首尾月）。Returns (month_start, month_end)。"""
    ms = st.sidebar.slider("起始月份", 1, 12, int(default_start))
    me = st.sidebar.slider("结束月份", 1, 12, int(default_end))
    if me < ms:
        st.sidebar.error("结束月份不能早于起始月份")
    return int(ms), int(me)


# =====================================================================
#  数据源与指数
# =====================================================================

def data_source_picker(default=None):
    """数据源：只列出有可用指数的数据源（Sentinel-1 雷达指数尚未实现）。"""
    return st.sidebar.selectbox(
        "数据源",
        ANALYSIS_SOURCES,
        index=ANALYSIS_SOURCES.index(default) if default in ANALYSIS_SOURCES else 0,
        format_func=lambda x: DATA_SOURCE_LABELS.get(x, x),
        help="sentinel2: 10m 多光谱 | landsat: 30m",
    )


def index_picker(data_source, default=None, allowed=None, remember_as=None):
    """遥感指数：按数据源能力（CAPABILITY_MATRIX）过滤，allowed 可进一步限定。

    remember_as 给出会话键时记住上次的选择，切换数据源后仍尽量保留。
    """
    available = [i for i in CAPABILITY_MATRIX.get(data_source, [])
                 if allowed is None or i in allowed]
    cur = st.session_state.get(remember_as, default) if remember_as else default
    cur = cur if cur in available else available[0]
    idx = st.sidebar.selectbox(
        "遥感指数",
        available,
        index=available.index(cur),
        format_func=lambda x: f"{x} ({INDEX_HELP.get(x, '')})",
    )
    if remember_as:
        st.session_state[remember_as] = idx
    return idx
