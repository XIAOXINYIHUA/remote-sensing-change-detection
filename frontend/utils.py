"""
遥感分析平台 v3 — 前端共享工具函数

包含: 全局CSS, 地图工具, 边界工具, 字体配置, 常量
"""

import os
import re
import json
from pathlib import Path

import streamlit as st
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from shapely.geometry import shape
import requests


# ============================================================================
#  全局 CSS
# ============================================================================

def inject_global_css():
    """Inject the shared workbench visual system."""
    st.markdown("""
    <style>
        :root {
            --rs-ink: #18221D;
            --rs-muted: #627068;
            --rs-faint: #849087;
            --rs-canvas: #F3F6F3;
            --rs-surface: #FFFFFF;
            --rs-surface-soft: #EDF2EE;
            --rs-line: #D9E1DA;
            --rs-line-strong: #C5D0C7;
            --rs-accent: #0F766E;
            --rs-accent-strong: #0A5D57;
            --rs-blue: #2D65A1;
            --rs-amber: #A86413;
            --rs-red: #B8403A;
            --rs-sidebar: #17211C;
            --rs-sidebar-soft: #223029;
        }

        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"],
        button, input, textarea, select {
            font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
            letter-spacing: 0 !important;
        }

        .stApp,
        [data-testid="stAppViewContainer"] {
            background: var(--rs-canvas);
            color: var(--rs-ink);
        }

        [data-testid="stHeader"] {
            background: rgba(243, 246, 243, 0.94);
            border-bottom: 1px solid rgba(217, 225, 218, 0.75);
        }

        .main .block-container {
            max-width: 1480px;
            padding: 2.3rem 2.6rem 4rem;
        }

        h1, h2, h3, h4, h5, h6, p, label {
            color: var(--rs-ink);
            letter-spacing: 0 !important;
        }

        h1 { font-size: 2rem; line-height: 1.2; }
        h2 { font-size: 1.25rem; line-height: 1.35; }
        h3 { font-size: 1.04rem; line-height: 1.4; }
        p { line-height: 1.65; }

        [data-testid="stSidebar"] {
            background: var(--rs-sidebar);
            border-right: 1px solid #2C3932;
            min-width: 292px;
        }

        [data-testid="stSidebar"] * {
            letter-spacing: 0 !important;
        }

        [data-testid="stIconMaterial"] {
            font-family: "Material Symbols Rounded" !important;
            font-weight: normal !important;
            font-style: normal !important;
            line-height: 1 !important;
            white-space: nowrap !important;
            text-transform: none !important;
            font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24;
        }

        [data-testid="stSidebarContent"] {
            padding: 0.7rem 0.75rem 1.25rem;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #E9F0EB;
        }

        [data-testid="stSidebarNav"] {
            padding-top: 0.35rem;
        }

        [data-testid="stSidebarNav"] span,
        [data-testid="stSidebarNav"] p {
            font-size: 0.9rem;
        }

        [data-testid="stSidebarNav"] a {
            border-radius: 5px;
            min-height: 2.45rem;
            color: #C8D4CC;
        }

        [data-testid="stSidebarNav"] a:hover {
            background: #26352D;
            color: #FFFFFF;
        }

        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: #2C4338;
            color: #FFFFFF;
            box-shadow: inset 3px 0 0 #54B9AC;
        }

        [data-testid="stSidebarNavSeparator"] {
            border-color: #314038;
        }

        [data-testid="stSidebar"] hr {
            border-color: #334138;
            margin: 1rem 0;
        }

        .rs-brand {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            padding: 0.65rem 0.55rem 1rem;
            border-bottom: 1px solid #314038;
            margin-bottom: 0.45rem;
        }

        .rs-brand-mark {
            display: grid;
            place-items: center;
            width: 2.25rem;
            height: 2.25rem;
            border-radius: 5px;
            background: #D9EFEA;
            color: #0A5D57;
            font-size: 0.78rem;
            font-weight: 800;
        }

        .rs-brand b { color: #FFFFFF; font-size: 0.98rem; display: block; }
        .rs-brand span { color: #91A39A; font-size: 0.72rem; display: block; margin-top: 0.12rem; }

        .rs-sidebar-panel {
            padding: 0.65rem 0.35rem 0.4rem;
            margin: 0.15rem 0 0.55rem;
        }

        .rs-sidebar-panel b {
            color: #FFFFFF;
            display: block;
            font-size: 0.9rem;
        }

        .rs-sidebar-panel p {
            color: #92A39A !important;
            font-size: 0.74rem;
            margin: 0.2rem 0 0;
        }

        .rs-sidebar-context {
            padding: 0.85rem;
            margin: 0.35rem 0 0.2rem;
            border: 1px solid #34443B;
            border-radius: 6px;
            background: #1D2923;
        }

        .rs-sidebar-context small { color: #8EA096; display: block; }
        .rs-sidebar-context b { color: #F5F8F6; display: block; margin: 0.2rem 0 0.35rem; }
        .rs-sidebar-context span { color: #B7C5BD; font-size: 0.74rem; line-height: 1.45; display: block; }

        .rs-page-header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 0.8rem 1.5rem;
            align-items: start;
            padding: 0.15rem 0 1.35rem;
            border-bottom: 1px solid var(--rs-line);
            margin-bottom: 1.35rem;
        }

        .rs-page-heading { min-width: 0; }
        .rs-eyebrow,
        .rs-section-overline {
            color: var(--rs-accent);
            font-weight: 750;
            font-size: 0.72rem;
            text-transform: uppercase;
        }

        .rs-page-header h1 {
            margin: 0.24rem 0 0;
            font-size: 2rem;
            line-height: 1.16;
            color: var(--rs-ink);
            font-weight: 720;
        }

        .rs-page-description {
            color: var(--rs-muted);
            font-size: 0.91rem;
            margin: 0.45rem 0 0;
            max-width: 68rem;
        }

        .rs-page-meta {
            grid-column: 1 / -1;
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            margin-top: 0.25rem;
        }

        .rs-meta-chip,
        .rs-kind {
            display: inline-flex;
            align-items: center;
            gap: 0.32rem;
            min-height: 1.72rem;
            padding: 0.26rem 0.52rem;
            border-radius: 4px;
            background: var(--rs-surface-soft);
            border: 1px solid var(--rs-line);
            color: #526159;
            font-size: 0.73rem;
            white-space: normal;
        }

        .rs-meta-chip b { color: #233029; font-weight: 650; }

        .rs-status {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            color: #526159;
            background: #EEF1EF;
            border: 1px solid var(--rs-line);
            border-radius: 999px;
            padding: 0.3rem 0.62rem;
            font-size: 0.73rem;
            font-weight: 650;
            white-space: nowrap;
        }

        .rs-status i {
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 50%;
            background: #839087;
        }

        .rs-status--success { color: #0A6554; background: #E5F4EF; border-color: #C3E4D9; }
        .rs-status--success i { background: #15916F; }
        .rs-status--warning { color: #8B5510; background: #FBF0DD; border-color: #EFD5A9; }
        .rs-status--warning i { background: #C9821A; }
        .rs-status--danger { color: #943630; background: #FAEAE8; border-color: #EDC5C1; }
        .rs-status--danger i { background: #C44C45; }
        .rs-status--info { color: #285B8E; background: #E9F1F9; border-color: #C9DBEE; }
        .rs-status--info i { background: #3476B8; }

        .rs-section-header {
            margin: 1.35rem 0 0.72rem;
        }

        .rs-section-header h2 {
            margin: 0.18rem 0 0;
            color: var(--rs-ink);
            font-size: 1.18rem;
            font-weight: 700;
        }

        .rs-section-header p {
            color: var(--rs-muted);
            font-size: 0.84rem;
            margin: 0.25rem 0 0;
        }

        .rs-workflow {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            border: 1px solid var(--rs-line);
            border-radius: 6px;
            overflow: hidden;
            background: var(--rs-surface);
            margin: 0 0 1rem;
        }

        .rs-workflow-step {
            display: flex;
            gap: 0.55rem;
            align-items: center;
            min-height: 3.15rem;
            padding: 0.65rem 0.78rem;
            border-right: 1px solid var(--rs-line);
            color: #718078;
            font-size: 0.78rem;
        }

        .rs-workflow-step:last-child { border-right: 0; }
        .rs-workflow-step span { font-size: 0.68rem; font-weight: 800; color: #8A968F; }
        .rs-workflow-step.is-active { background: #E8F3F0; color: #0A5D57; }
        .rs-workflow-step.is-active span { color: #0F766E; }
        .rs-workflow-step.is-complete { color: #2D5C4F; background: #F0F7F4; }
        .rs-workflow-step.is-complete span { color: #16806B; }

        .rs-banner {
            display: flex;
            align-items: flex-start;
            gap: 0.72rem;
            padding: 0.85rem 0.95rem;
            border: 1px solid var(--rs-line);
            border-left-width: 3px;
            border-radius: 5px;
            background: var(--rs-surface);
            margin: 0.7rem 0;
        }

        .rs-banner-mark { width: 0.5rem; height: 0.5rem; margin-top: 0.35rem; border-radius: 50%; background: #75827A; flex: none; }
        .rs-banner b { color: var(--rs-ink); font-size: 0.86rem; }
        .rs-banner p { margin: 0.15rem 0 0; color: var(--rs-muted); font-size: 0.79rem; }
        .rs-banner--success { border-left-color: #16806B; background: #F1F8F5; }
        .rs-banner--success .rs-banner-mark { background: #16806B; }
        .rs-banner--warning { border-left-color: #BE781A; background: #FCF7ED; }
        .rs-banner--warning .rs-banner-mark { background: #BE781A; }
        .rs-banner--danger { border-left-color: #BD4942; background: #FCF2F1; }
        .rs-banner--danger .rs-banner-mark { background: #BD4942; }
        .rs-banner--info { border-left-color: #3476B8; background: #F1F6FB; }
        .rs-banner--info .rs-banner-mark { background: #3476B8; }

        .rs-empty-state {
            min-height: 9rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            padding: 1.5rem;
            border: 1px dashed var(--rs-line-strong);
            border-radius: 6px;
            background: rgba(255,255,255,0.48);
            text-align: left;
        }

        .rs-empty-state > span {
            display: grid;
            place-items: center;
            width: 2.5rem;
            height: 2.5rem;
            flex: none;
            border-radius: 5px;
            background: #E3EAE4;
            color: #65736B;
            font-size: 0.72rem;
            font-weight: 800;
        }

        .rs-empty-state b { color: var(--rs-ink); font-size: 0.93rem; }
        .rs-empty-state p { margin: 0.22rem 0 0; color: var(--rs-muted); font-size: 0.8rem; }

        .rs-task-heading { padding: 0.1rem 0; min-width: 0; }
        .rs-task-heading > div { display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .rs-task-heading h3 { margin: 0.5rem 0 0; font-size: 0.96rem; }
        .rs-task-heading p { margin: 0.2rem 0 0; color: var(--rs-muted); font-size: 0.78rem; overflow-wrap: anywhere; }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--rs-line) !important;
            border-radius: 6px !important;
            background: var(--rs-surface);
            box-shadow: none !important;
        }

        [data-testid="stMetric"] {
            min-height: 6.65rem;
            padding: 0.9rem 1rem;
            border: 1px solid var(--rs-line);
            border-radius: 6px;
            background: var(--rs-surface);
        }

        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4n+1) [data-testid="stMetric"] {
            border-top: 2px solid var(--rs-accent);
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4n+2) [data-testid="stMetric"] {
            border-top: 2px solid var(--rs-blue);
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4n+3) [data-testid="stMetric"] {
            border-top: 2px solid var(--rs-amber);
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4n+4) [data-testid="stMetric"] {
            border-top: 2px solid var(--rs-red);
        }

        [data-testid="stMetricLabel"] p {
            color: var(--rs-muted);
            font-size: 0.76rem;
            font-weight: 620;
        }

        [data-testid="stMetricValue"] {
            color: var(--rs-ink);
            font-size: 1.55rem;
            line-height: 1.3;
        }

        [data-testid="stMetricDelta"] { font-size: 0.72rem; }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stPageLink"] a {
            border-radius: 4px !important;
            min-height: 2.45rem;
            border: 1px solid var(--rs-line-strong);
            background: var(--rs-surface);
            color: #223029;
            box-shadow: none !important;
            font-weight: 650;
            white-space: normal;
            line-height: 1.25;
            transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stPageLink"] a:hover {
            color: var(--rs-accent-strong);
            border-color: #7FABA3;
            background: #F4FAF8;
        }

        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            background: var(--rs-accent);
            border-color: var(--rs-accent);
            color: #FFFFFF;
        }

        .stButton > button[kind="primary"]:hover,
        .stDownloadButton > button[kind="primary"]:hover {
            background: var(--rs-accent-strong);
            border-color: var(--rs-accent-strong);
            color: #FFFFFF;
        }

        .stButton > button:disabled {
            color: #8C9790;
            background: #E9EDE9;
            border-color: #D8DED9;
        }

        [data-baseweb="tab-list"] {
            gap: 0.15rem;
            border-bottom: 1px solid var(--rs-line);
            margin-bottom: 0.9rem;
        }

        [data-baseweb="tab"] {
            min-height: 2.7rem;
            padding: 0.55rem 0.82rem;
            border-radius: 4px 4px 0 0;
            color: var(--rs-muted);
            font-size: 0.82rem;
            font-weight: 620;
            white-space: normal;
        }

        [data-baseweb="tab"][aria-selected="true"] {
            color: var(--rs-accent-strong);
            background: #EAF3F0;
        }

        [data-baseweb="tab-highlight"] { background: var(--rs-accent); }

        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-testid="stNumberInputContainer"],
        [data-testid="stTextInputRootElement"] {
            border-color: var(--rs-line-strong) !important;
            border-radius: 4px !important;
            background: var(--rs-surface) !important;
            box-shadow: none !important;
        }

        [data-baseweb="popover"] { border-radius: 5px !important; }
        [data-baseweb="menu"] { background: #FFFFFF; }

        [data-testid="stFileUploaderDropzone"] {
            border-radius: 6px;
            border: 1px dashed #9EADA3;
            background: #F8FAF8;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--rs-line);
            border-radius: 5px;
            background: var(--rs-surface);
            box-shadow: none;
        }

        [data-testid="stAlert"] {
            border-radius: 5px;
            border-width: 1px;
            box-shadow: none;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border: 1px solid var(--rs-line);
            border-radius: 6px;
            overflow: hidden;
            background: #FFFFFF;
        }

        [data-testid="stProgressBar"] > div > div,
        .stProgress > div > div > div {
            background: var(--rs-accent) !important;
        }

        iframe,
        [data-testid="stImage"] img,
        [data-testid="stPlotlyChart"] {
            border-radius: 6px;
        }

        .leaflet-container { background: #DDE4DE !important; }

        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: var(--rs-sidebar-soft);
            border-color: #35443B;
        }

        [data-testid="stSidebar"] [data-testid="stMetricValue"],
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
            color: #EEF4F0;
        }

        @media (max-width: 900px) {
            .main .block-container { padding: 1.25rem 1rem 3rem; }
            .rs-page-header { grid-template-columns: 1fr; gap: 0.65rem; }
            .rs-page-header h1 { font-size: 1.62rem; overflow-wrap: anywhere; }
            .rs-page-header > .rs-status { justify-self: start; }
            .rs-workflow { grid-template-columns: 1fr 1fr; }
            .rs-workflow-step { border-bottom: 1px solid var(--rs-line); }
            [data-baseweb="tab-list"] { overflow-x: auto; }
            [data-testid="stMetric"] { min-height: 5.8rem; }
        }

        @media (max-width: 560px) {
            .main .block-container { padding-left: 0.72rem; padding-right: 0.72rem; }
            .rs-workflow { grid-template-columns: 1fr; }
            .rs-workflow-step { border-right: 0; }
            .rs-empty-state { align-items: flex-start; justify-content: flex-start; }
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
#  Matplotlib 中文字体
# ============================================================================

def setup_matplotlib_font():
    """配置 Matplotlib 中文字体，返回使用中的字体名。"""
    candidates = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
        'Noto Sans CJK SC', 'Noto Sans SC', 'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei', 'AR PL UMing CN', 'AR PL UKai CN',
        'Source Han Sans SC', 'PingFang SC', 'Heiti SC', 'STHeiti',
        'STSong', 'STKaiti', 'STFangsong',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            return font
    for f in fm.fontManager.ttflist:
        name_lower = f.name.lower()
        if any(kw in name_lower for kw in
               ['cjk', 'chinese', 'cn', 'sc', 'han', 'hei', 'ming', 'song', 'kai', 'fang']):
            plt.rcParams['font.sans-serif'] = [f.name, 'DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            return f.name
    return None


# ============================================================================
#  GeoJSON 边界工具
# ============================================================================

def detect_region(path):
    """从 GeoJSON 文件中自动检测地区名称。"""
    try:
        with open(path, encoding="utf-8") as f:
            gj = json.load(f)
        for feat in (gj.get("features") or [gj]):
            props = feat.get("properties") or {}
            for k in ["name", "NAME", "Name", "地区", "区域", "地名",
                      "city", "City", "district", "District", "county", "County"]:
                if k in props and props[k]:
                    return str(props[k]).strip()
            break
        stem = Path(path).stem
        clean = re.sub(r"(_boundary|_polygon|_geojson|_border|_area)$", "", stem, flags=re.I)
        return clean
    except Exception:
        return Path(path).stem


def slug(name):
    """将名称转为文件系统友好的短标识。"""
    slug_str = re.sub(r"[^\w一-鿿]+", "_", name)
    return slug_str.strip("_")[:30] or "default"


def load_gj(path):
    """加载 GeoJSON 文件，返回 (geojson_dict, shapely_geometry)。

    支持多要素 FeatureCollection（Union 所有要素保持数据完整性）。
    """
    from shapely.ops import unary_union
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    if gj["type"] == "FeatureCollection":
        geoms = []
        for feat in gj["features"]:
            try:
                g = shape(feat["geometry"])
                if g and not g.is_empty:
                    geoms.append(g)
            except Exception:
                continue
        if not geoms:
            raise ValueError("FeatureCollection 中无有效几何")
        geom = unary_union(geoms) if len(geoms) > 1 else geoms[0]
    elif gj["type"] in ("Polygon", "MultiPolygon"):
        geom = shape(gj)
        gj = {"type": "FeatureCollection",
              "features": [{"type": "Feature", "geometry": gj, "properties": {}}]}
    else:
        geom = shape(gj)
    return gj, geom


def fetch_boundary_osm(name):
    """通过 OSM Nominatim 搜索城市/区县边界。"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": name, "format": "json", "polygon_geojson": 1, "limit": 1}
    headers = {"User-Agent": "RemoteSensingTool/3.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None, f"未找到「{name}」的边界数据"
        gj = {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "geometry": data[0].get("geojson"),
            "properties": {"name": name},
        }]}
        return gj, None
    except Exception as e:
        return None, f"获取失败: {e}"


# ============================================================================
#  共享常量
# ============================================================================

DATA_SOURCE_LABELS = {
    "sentinel2": "Sentinel-2 L2A (10m)",
    "landsat8": "Landsat 8 (30m)",
    "landsat9": "Landsat 9 (30m)",
    "sentinel1": "Sentinel-1 SAR (仅SAR指数)",
}

INDEX_HELP = {
    "NDVI": "植被覆盖变化",
    "NDWI": "水体面积变化",
    "MNDWI": "建筑区水体提取",
    "NBR": "火烧迹地检测",
    "EVI": "高植被区（抗大气）",
    "SAVI": "低植被覆盖区",
    "NDBI": "建设用地与城市扩张",
}

# 能力矩阵：数据源 → 可用指数列表
# Sentinel-1 无红光/近红外波段，不支持光学指数
CAPABILITY_MATRIX = {
    "sentinel2": ["NDVI", "NDWI", "MNDWI", "NBR", "EVI", "SAVI", "NDBI"],
    "landsat8":  ["NDVI", "NDWI", "MNDWI", "NBR", "EVI", "SAVI", "NDBI"],
    "landsat9":  ["NDVI", "NDWI", "MNDWI", "NBR", "EVI", "SAVI", "NDBI"],
    "sentinel1": [],  # 需要 VV/VH 指数（RVI 等），暂未实现
}

# 界面可选的分析数据源：只列出至少有一个可用指数的数据源，
# 否则选中后无法收集影像也无法运行分析
ANALYSIS_SOURCES = [src for src in DATA_SOURCE_LABELS if CAPABILITY_MATRIX.get(src)]

MODE_LABELS = {
    "multi": "多级分级（推荐）",
    "binary": "二值检测",
    "ml": "机器学习",
}


# ============================================================================
#  状态持久化
# ============================================================================

def state_dir():
    """界面状态文件（项目状态 .project_state.json）所在目录。

    默认为项目根目录；设置环境变量 RSAP_STATE_DIR 可改到别处
    （测试用临时目录，避免改动用户的工作状态）。
    """
    root = Path(os.environ.get("RSAP_STATE_DIR") or Path(__file__).resolve().parent.parent)
    root.mkdir(parents=True, exist_ok=True)
    return root
