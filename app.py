"""
遥感解译助手 v2 — 两阶段工作流
阶段1: 选择地区 → 自动收集影像资料
阶段2: 资料就绪 → 用户一键运行分析

v2.1 优化:
  - matplotlib 中文字体自动配置（修复报告/图表乱码）
  - 地图矢量渲染自适应降采样（大幅提升大区域显示速度）
"""

import os, sys, json, io, re, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import requests

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
from shapely.geometry import shape
import rasterio

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
import config
import main as engine


# =====================================================================
#  中文字体配置（修复 Streamlit 图表中文乱码）
# =====================================================================

def _setup_matplotlib_font():
    """自动检测并配置 matplotlib 中文字体"""
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

_setup_matplotlib_font()


# =====================================================================
#  工具函数
# =====================================================================

def _detect_region(path):
    try:
        with open(path, encoding="utf-8") as f:
            gj = json.load(f)
        for feat in (gj.get("features") or [gj]):
            props = feat.get("properties") or {}
            for k in ["name","NAME","Name","地区","区域","地名",
                      "city","City","district","District","county","County"]:
                if k in props and props[k]:
                    return str(props[k]).strip()
            break
        stem = Path(path).stem
        clean = re.sub(r"(_boundary|_polygon|_geojson|_border|_area)$", "", stem, flags=re.I)
        return clean
    except Exception:
        return Path(path).stem


def _slug(name):
    slug = re.sub(r"[^\w一-鿿]+", "_", name)
    return slug.strip("_")[:30] or "default"


def _load_gj(path):
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    if gj["type"] == "FeatureCollection":
        geom = shape(gj["features"][0]["geometry"])
    elif gj["type"] in ("Polygon", "MultiPolygon"):
        geom = shape(gj)
        gj = {"type": "FeatureCollection",
              "features": [{"type": "Feature", "geometry": gj, "properties": {}}]}
    else:
        geom = shape(gj)
    return gj, geom


def _fetch_boundary_osm(name):
    """从 OpenStreetMap Nominatim 获取行政区边界 GeoJSON"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": name, "format": "json", "polygon_geojson": 1, "limit": 1}
    headers = {"User-Agent": "RemoteSensingTool/2.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None, f"未找到「{name}」的边界数据"
        gj = {"type":"FeatureCollection","features":[{
            "type":"Feature",
            "geometry":data[0].get("geojson"),
            "properties":{"name":name}
        }]}
        return gj, None
    except Exception as e:
        return None, f"获取失败: {e}"


# =====================================================================
#  步骤定义与进度系统
# =====================================================================

STEPS_COLLECT = [
    ("🔍", "搜索 Y1 影像"),
    ("🔍", "搜索 Y2 影像"),
    ("📥", "下载 Y1 影像"),
    ("📥", "下载 Y2 影像"),
    ("🧩", "拼接与裁剪"),
    ("✅", "资料就绪"),
]

STEPS_ANALYSIS = [
    ("📂", "加载波段"),
    ("🧮", "NDVI 计算"),
    ("📊", "NDVI 差值"),
    ("🏷️", "多级分类"),
    ("🔬", "连通域过滤"),
    ("📍", "生成采样点"),
    ("📋", "输出报告"),
    ("🏁", "完成"),
]


def _init_progress(n_labels, labels):
    st.session_state.progress = 0
    st.session_state.progress_msg = ""
    st.session_state.progress_start = time.time()
    st.session_state.progress_n = n_labels
    st.session_state.progress_labels = labels
    st.session_state.progress_active = 0


def _render_progress(active, msg, elapsed, detail=""):
    labels = st.session_state.get("progress_labels", [])
    n = len(labels)
    if n == 0:
        return
    parts = []
    for i, label in enumerate(labels):
        if i < active:
            parts.append(f"<span style='color:#0a0'>✅{label}</span>")
        elif i == active:
            parts.append(f"<span style='color:#08f;font-weight:bold'>⏳{label}</span>")
        else:
            parts.append(f"<span style='color:#888'>⏹️{label}</span>")
    step_bar = " &nbsp;→&nbsp; ".join(parts)
    ts = f"⏱️ {elapsed:.0f}s" if elapsed < 120 else f"⏱️ {elapsed/60:.1f}min"
    pct = st.session_state.get("progress", 0)
    prog_bar.progress(pct / 100.0)
    stat_ph.markdown(
        f"<div style='font-size:13px'>{step_bar}</div>"
        f"<div><b>{msg}</b> &nbsp; {ts}"
        f"{' &nbsp;—&nbsp; ' + detail if detail else ''}</div>",
        unsafe_allow_html=True,
    )


def _advance_step(idx, msg, pct, detail=""):
    st.session_state.progress = int(pct)
    st.session_state.progress_msg = msg
    st.session_state.progress_active = idx
    elapsed = time.time() - st.session_state.get("progress_start", time.time())
    _render_progress(idx, msg, elapsed, detail)


def _cb(msg, pct):
    st.session_state.progress = int(pct)
    st.session_state.progress_msg = msg
    elapsed = time.time() - st.session_state.get("progress_start", time.time())
    prog_bar.progress(int(pct) / 100.0)
    ts = f"⏱️ {elapsed:.0f}s" if elapsed < 120 else f"⏱️ {elapsed/60:.1f}min"
    stat_ph.info(f"**{msg}**  {ts}")


def _cb_auto(msg, pct):
    st.session_state.progress = int(pct)
    st.session_state.progress_msg = msg
    elapsed = time.time() - st.session_state.get("progress_start", time.time())
    prog_bar.progress(int(pct) / 100.0)
    labels = st.session_state.get("progress_labels", [])
    if pct < 35:
        active = 0
    elif pct < 50:
        active = 1
    elif pct < 60:
        active = 2
    elif pct < 70:
        active = 3
    elif pct < 80:
        active = 4
    elif pct < 85:
        active = 5
    elif pct < 95:
        active = 6
    else:
        active = min(len(labels) - 1, 7)
    st.session_state.progress_active = active
    _render_progress(active, msg, elapsed)


# =====================================================================
#  核心流程
# =====================================================================

def _collect_data(y1, y2, ms, me):
    st.session_state.collecting = True
    st.session_state.data_ok = False
    labels = [s[1] for s in STEPS_COLLECT]
    _init_progress(len(labels), labels)
    stat_ph.info("⏳ 初始化..."); prog_bar.progress(0)

    try:
        region = st.session_state.region or "未知"
        slug = st.session_state.slug or _slug(region)

        config.REGION_NAME = region
        config.YEAR1 = int(y1); config.YEAR2 = int(y2)
        config.MONTH_START = int(ms); config.MONTH_END = int(me)
        if st.session_state.boundary:
            config.BOUNDARY_FILE = st.session_state.boundary

        data_dir = str(BASE_DIR / "data" / slug)
        os.makedirs(data_dir, exist_ok=True)

        _advance_step(0, f"搜索 {y1} 年可用影像...", 5)
        b = engine.get_boundary()
        s1 = engine.find_best_scenes(b, y1)
        _advance_step(0, f"搜索 {y1} 完成: {len(s1)} 个瓦片", 12, f"{len(s1)} tiles")

        _advance_step(1, f"搜索 {y2} 年可用影像...", 15)
        s2 = engine.find_best_scenes(b, y2)
        _advance_step(1, f"搜索 {y2} 完成: {len(s2)} 个瓦片", 25, f"{len(s2)} tiles")

        st.session_state.collected = {
            "context": {"region": region, "year1": y1, "year2": y2,
                         "tiles_found": len(s1)+len(s2)}}

        _advance_step(2, f"下载 {y1} 年影像 ({len(s1)} 瓦片)...", 30)
        b4_1, b8_1 = engine.download_and_mosaic(b, y1, data_dir)
        if b4_1 is None: raise RuntimeError(f"{y1} 下载失败")
        m1 = sum(Path(f).stat().st_size for f in [b4_1,b8_1] if f and Path(f).exists()) / 1024/1024
        _advance_step(2, f"{y1} 年下载完成", 50, f"{m1:.0f} MB")

        _advance_step(3, f"下载 {y2} 年影像 ({len(s2)} 瓦片)...", 55)
        b4_2, b8_2 = engine.download_and_mosaic(b, y2, data_dir)
        if b4_2 is None: raise RuntimeError(f"{y2} 下载失败")
        m2 = sum(Path(f).stat().st_size for f in [b4_2,b8_2] if f and Path(f).exists()) / 1024/1024
        _advance_step(3, f"{y2} 年下载完成", 80, f"{m2:.0f} MB")

        _advance_step(4, "影像拼接与边界裁剪...", 85)
        total_mb = m1 + m2
        st.session_state.collected["context"]["size_mb"] = total_mb
        st.session_state.data_ok = True
        _advance_step(4, f"拼接完成，共 {total_mb:.0f} MB", 95, f"{total_mb:.0f} MB")
        _advance_step(5, f"✅ {region} 资料收集完成", 100, f"共 {total_mb:.0f} MB")

    except Exception as e:
        st.error(f"❌ 资料收集失败: {e}")
        import traceback; st.code(traceback.format_exc())
    finally:
        st.session_state.collecting = False


def _run_analysis(y1, y2, ms, me, dmode, ct, ma, bf,
                  strat, np_pts, md_pts, greport):
    st.session_state.running = True
    labels = [s[1] for s in STEPS_ANALYSIS]
    _init_progress(len(labels), labels)
    stat_ph.info("⏳ 初始化分析..."); prog_bar.progress(0)

    try:
        region = st.session_state.region or "未知"
        slug = st.session_state.slug or _slug(region)

        config.REGION_NAME = region
        config.YEAR1 = int(y1); config.YEAR2 = int(y2)
        config.MONTH_START = int(ms); config.MONTH_END = int(me)
        config.DETECTION_MODE = dmode
        config.CHANGE_THRESHOLD = ct; config.SEVERE_THRESHOLD = 0.25
        config.MORPH_KERNEL_SIZE = 4
        config.MIN_CHANGE_AREA_M2 = int(ma)
        config.MIN_CHANGE_AREA_PX = max(1, int(ma)//100)
        config.USE_BILATERAL_FILTER = bf
        config.SAMPLE_STRATEGY = strat
        config.NUM_SAMPLE_POINTS = int(np_pts) if int(np_pts)>0 else None
        config.MIN_POINT_DISTANCE_M = int(md_pts) if int(md_pts)>0 else None
        config.OUTPUT_FORMATS = "geojson,csv"
        config.GENERATE_REPORT = greport
        if st.session_state.boundary:
            config.BOUNDARY_FILE = st.session_state.boundary

        _advance_step(0, "加载波段数据...", 5)
        region_data_dir = str(BASE_DIR / "data" / slug)
        region_out_dir = str(BASE_DIR / "output" / slug)
        os.makedirs(region_out_dir, exist_ok=True)

        res = engine.run_pipeline(
            data_dir=region_data_dir,
            out_dir=region_out_dir,
            progress_callback=_cb_auto,
        )
        st.session_state.results = res
        st.session_state.stats = res.get("stats")

        _advance_step(6, "输出报告...", 90)
        gtif = res.get("files",{}).get("geotiff")
        if gtif and os.path.exists(gtif):
            with rasterio.open(gtif) as src:
                st.session_state.arr = src.read(1)
                st.session_state.meta = src.meta

        gjp = res.get("files",{}).get("geojson")
        if gjp and os.path.exists(gjp):
            with open(gjp, encoding="utf-8") as f:
                st.session_state.cands = json.load(f).get("features", [])

        timing = res.get("timing",{}).get("total",0)
        st.session_state.analysis_done = True
        _advance_step(7, f"✅ {region} 分析完成", 100, f"耗时 {timing:.0f}s")
        _save_state()
        st.balloons()

    except Exception as e:
        st.error(f"❌ 分析失败: {e}")
        import traceback; st.code(traceback.format_exc())
    finally:
        st.session_state.running = False


# ---- 展示函数 ----

def _show_map():
    if st.session_state.arr is None or not st.session_state.boundary: return
    try:
        gj, geom = _load_gj(st.session_state.boundary)
    except Exception:
        st.error("边界解析失败"); return
    m = folium.Map(location=[geom.centroid.y, geom.centroid.x], zoom_start=11)
    Fullscreen().add_to(m)
    folium.GeoJson(gj, name="边界",
                   style_function=lambda x:
                   {"fillColor":"#3388ff","color":"#3388ff",
                    "weight":2,"fillOpacity":0.08}).add_to(m)
    if config.DETECTION_MODE == "multi":
        _add_multi(m, st.session_state.arr, st.session_state.meta)
    else:
        _add_binary(m, st.session_state.arr, st.session_state.meta)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend()))
    st_folium(m, width="100%", height=600)
    fs = st.session_state.results.get("files",{}) if st.session_state.results else {}
    if fs:
        cols = st.columns(len(fs))
        for i,(fmt,p) in enumerate(fs.items()):
            if os.path.exists(p):
                with open(p,"rb") as fh:
                    cols[i].download_button(f"⬇ {Path(p).name}", fh.read(), Path(p).name)


def _add_multi(m, result, meta):
    """多级结果叠加到 folium 地图（自适应降采样优化）"""
    from rasterio.features import shapes as rshapes
    from rasterio import Affine
    import cv2

    cfg = {-3:("严重退化","#b40000"),-2:("中度退化","#ff6400"),
            2:("中度恢复","#00b450"),3:("强烈恢复","#005000")}

    h, w = result.shape
    # 自适应降采样：大图用更大缩放因子保证渲染性能
    if w * h > 4_000_000:       # > 2000x2000
        scale = 8
    elif w * h > 1_000_000:     # > 1000x1000
        scale = 4
    elif w > 800 or h > 800:
        scale = 2
    else:
        scale = 1

    if scale > 1:
        small = cv2.resize(result.astype(np.int8), (w // scale, h // scale),
                           interpolation=cv2.INTER_NEAREST)
        tf = meta["transform"]
        small_meta = meta.copy()
        small_meta.update(
            transform=Affine(tf.a * scale, tf.b, tf.c, tf.d, tf.e * scale, tf.f),
            width=w // scale, height=h // scale)
        result_small, meta_small = small, small_meta
    else:
        result_small, meta_small = result, meta

    mask = (result_small == -3) | (result_small == -2) | (result_small == 2) | (result_small == 3)
    if mask.sum() == 0:
        return

    raw = list(rshapes(result_small.astype(np.int8), mask=mask,
                       transform=meta_small["transform"], connectivity=8))

    # 限制矢量要素数量防止浏览器卡顿
    max_features = 30000
    if len(raw) > max_features:
        import random
        random.seed(42)
        raw = random.sample(raw, max_features)

    for cv_val, (cn, co) in cfg.items():
        feats = [{"type": "Feature", "geometry": g, "properties": {"t": cn}}
                 for g, v in raw if v == cv_val]
        if not feats:
            continue
        fg = folium.FeatureGroup(name=cn, show=True)
        folium.GeoJson(
            {"type": "FeatureCollection", "features": feats}, name=cn,
            style_function=lambda x, c=co:
                {"fillColor": c, "color": c, "weight": 0, "fillOpacity": 0.6},
            tooltip=folium.GeoJsonTooltip(fields=["t"], aliases=["类型:"])
        ).add_to(fg)
        fg.add_to(m)


def _add_binary(m, result, meta):
    """二值结果叠加到 folium 地图"""
    from rasterio.features import shapes as rshapes
    raw = list(rshapes(result.astype(np.uint8), mask=result == 1,
                       transform=meta["transform"], connectivity=8))
    feats = [{"type": "Feature", "geometry": g, "properties": {"t": "变化"}}
             for g, v in raw if v == 1]
    if not feats:
        return
    fg = folium.FeatureGroup(name="显著变化", show=True)
    folium.GeoJson(
        {"type": "FeatureCollection", "features": feats}, name="显著变化",
        style_function=lambda x:
            {"fillColor": "#ff3c3c", "color": "#ff3c3c",
             "weight": 0, "fillOpacity": 0.5}
    ).add_to(fg)
    fg.add_to(m)


def _legend():
    if config.DETECTION_MODE == "multi":
        return """<div style="position:fixed;bottom:20px;right:20px;z-index:9999;
            background:white;padding:10px;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,.3);
            font-size:13px;"><b>分级</b><br>
            <span style="display:inline-block;width:12px;height:12px;background:#b40000;border-radius:2px;"></span> 严重退化<br>
            <span style="display:inline-block;width:12px;height:12px;background:#ff6400;border-radius:2px;"></span> 中度退化<br>
            <span style="display:inline-block;width:12px;height:12px;background:#00b450;border-radius:2px;"></span> 中度恢复<br>
            <span style="display:inline-block;width:12px;height:12px;background:#005000;border-radius:2px;"></span> 强烈恢复</div>"""
    return """<div style="position:fixed;bottom:20px;right:20px;z-index:9999;
        background:white;padding:10px;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,.3);
        font-size:13px;"><b>图例</b><br>
        <span style="display:inline-block;width:12px;height:12px;background:#ff3c3c;border-radius:2px;"></span> 显著变化</div>"""


def _show_stats():
    s = st.session_state.stats
    if config.DETECTION_MODE != "multi":
        st.metric("变化面积", f"{s.get('changed_area_ha',0):,.0f} ha"); return
    cls_n = {-3: "严重退化", -2: "中度退化", 1: "稳定", 2: "中度恢复", 3: "强烈恢复"}
    cls_c = {-3: "#b40000", -2: "#ff6400", 2: "#00b450", 3: "#005000"}
    rows = []
    for cv in [-3, -2, 1, 2, 3]:
        if s.get(cv):
            rows.append({"类别": s[cv]["name"], "面积(ha)": f"{s[cv]['area_ha']:,.0f}",
                          "占比": f"{s[cv]['percent']:.1f}%"})
    if "changed_total" in s:
        t = s["changed_total"]
        rows.append({"类别": "📊合计", "面积(ha)": f"{t['area_ha']:,.0f}",
                      "占比": f"{t['percent']:.1f}%"})
    st.table(pd.DataFrame(rows))
    cd = [(c, s.get(c, {}).get("area_ha", 0)) for c in [-3, -2, 2, 3]]
    cd = [(n, a) for n, a in cd if a > 0]
    if cd:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        labs = [cls_n[c] for c, _ in cd]; vals = [a for _, a in cd]
        cols = [cls_c[c] for c, _ in cd]
        a1.bar(labs, vals, color=cols, edgecolor="white")
        a1.set_ylabel("面积(ha)"); a1.set_title("各级变化面积", fontweight="bold")
        a1.tick_params(axis="x", rotation=15)
        a2.pie(vals, labels=labs, autopct="%1.1f%%", colors=cols, startangle=90)
        a2.set_title("变化占比", fontweight="bold")
        fig.tight_layout(); st.pyplot(fig); plt.close()
        t = s["changed_total"]
        st.info(f"**结论**: {config.REGION_NAME} {config.YEAR1}-{config.YEAR2} "
                f"显著变化 **{t['area_ha']:,.0f} ha** ({t['percent']:.1f}%)")


def _show_points():
    if not st.session_state.cands:
        st.warning("无采样点"); return
    rows = [{"序号": f["properties"].get("id", ""),
             "经度": round(f["geometry"]["coordinates"][0], 6),
             "纬度": round(f["geometry"]["coordinates"][1], 6),
             "面积(ha)": f["properties"].get("area_ha", ""),
             "类别": f["properties"].get("class_name", "")} for f in st.session_state.cands]
    df = pd.DataFrame(rows)
    st.subheader(f"📍 {len(rows)} 个采样点")
    st.dataframe(df, use_container_width=True, height=400)
    buf = io.BytesIO(); df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button("⬇ CSV", buf.getvalue(), "sample_points.csv", "text/csv")
    if "类别" in df.columns:
        st.bar_chart(df["类别"].value_counts())


# =====================================================================
#  页面 & 状态
# =====================================================================

st.set_page_config(page_title="遥感解译助手 v2", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

S = st.session_state
_INIT = {
    "boundary": None, "region": None, "slug": None,
    "data_ok": False, "collecting": False, "running": False,
    "collected": None,
    "results": None, "arr": None, "meta": None,
    "stats": None, "cands": None,
    "progress": 0, "progress_msg": "",
}
for k, v in _INIT.items():
    if k not in S:
        S[k] = v

# ---- 状态持久化 ----
STATE_FILE = BASE_DIR / ".app_state.json"

def _save_state():
    """保存可持久化的状态到文件"""
    state = {
        "boundary": S.get("boundary"),
        "region": S.get("region"),
        "slug": S.get("slug"),
        "data_ok": S.get("data_ok", False),
        "collected": S.get("collected"),
    }
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception:
        pass

def _load_state():
    """从文件恢复持久化状态"""
    if not STATE_FILE.exists():
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        for k in ["boundary", "region", "slug", "data_ok", "collected"]:
            if k in state and state[k] is not None:
                S[k] = state[k]
    except Exception:
        pass

_load_state()

# =====================================================================
#  Sidebar
# =====================================================================

st.sidebar.markdown("## 🛰️ 遥感解译")
st.sidebar.markdown("---")

st.sidebar.subheader("📍 第一步：选定地区")
exist = {f.name: str(f) for f in BASE_DIR.glob("*.geojson")}
cur_name = Path(S.boundary).name if S.boundary else None
if exist:
    sel_name = st.sidebar.selectbox(
        "已有边界文件", list(exist.keys()),
        index=list(exist.keys()).index(cur_name) if cur_name in exist else 0)
    chosen = exist[sel_name]
    if chosen != S.boundary:
        S.boundary = chosen; r = _detect_region(chosen)
        S.region = r; S.slug = _slug(r)
        S.data_ok = False; S.collected = None
        _save_state()
        st.rerun()

search_name = st.sidebar.text_input(
    "搜索地区", "",
    placeholder="输入城市/区县名，如 桂林市")
if st.sidebar.button("🔍 获取边界", use_container_width=True,
                      disabled=not search_name.strip()):
    with st.sidebar.status(f"正在搜索 {search_name}..."):
        gj, err = _fetch_boundary_osm(search_name.strip())
        if err:
            st.sidebar.error(err)
        else:
            fp = BASE_DIR / f"{_slug(search_name)}.geojson"
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(gj, f, ensure_ascii=False)
            S.boundary = str(fp)
            S.region = search_name.strip()
            S.slug = _slug(search_name)
            S.data_ok = False; S.collected = None
            _save_state()
            st.rerun()

if S.boundary and Path(S.boundary).exists():
    try:
        _, geom = _load_gj(S.boundary)
        st.sidebar.caption(
            f"中心: {geom.centroid.y:.3f}, {geom.centroid.x:.3f}  "
            f"面积: {geom.area:.2f}°²")
    except Exception:
        st.sidebar.error("边界解析失败")
else:
    st.sidebar.error("请选择边界文件")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 时相")
y1 = st.sidebar.number_input("前时相年份", 2015, 2026, config.YEAR1, 1)
y2 = st.sidebar.number_input("后时相年份", 2015, 2026, config.YEAR2, 1)
ms = st.sidebar.slider("起始月份", 1, 12, config.MONTH_START)
me = st.sidebar.slider("结束月份", 1, 12, config.MONTH_END)
st.sidebar.markdown("---")

with st.sidebar.expander("🔬 检测参数"):
    dmode = st.selectbox("模式", ["multi","binary"],
                          format_func=lambda x: {"multi":"多级分级（推荐）","binary":"二值"}[x])
    ct = st.slider("NDVI 阈值", 0.05, 0.30, config.CHANGE_THRESHOLD, 0.01)
    ma = st.number_input("最小斑块 m²", 100, 50000, config.MIN_CHANGE_AREA_M2, 100)
    bf = st.checkbox("双边预滤波", config.USE_BILATERAL_FILTER)

with st.sidebar.expander("📊 采样"):
    strat = st.selectbox("策略", ["centroid","random","grid"],
                          format_func=lambda x: {"centroid":"质心","random":"随机","grid":"网格"}[x])
    np_pts = st.number_input("上限(0=不限)", 0, 100000, 0)
    md_pts = st.number_input("间距 m", 0, 10000, 0)

with st.sidebar.expander("📥 输出"):
    greport = st.checkbox("报告 PNG", True)

# =====================================================================
#  主面板
# =====================================================================

region_display = S.region or "未选择"
st.title(f"🛰️ {region_display} · 植被变化遥感解译")
st.markdown(f"**{y1}** → **{y2}**  |  {ms}月-{me}月  |  NDVI 阈值 ±{ct}")

stat_ph = st.empty()
prog_bar = st.progress(0)

tab_dl, tab_run, tab_map, tab_stat, tab_pt = st.tabs(
    ["📥 资料收集", "🚀 运行分析", "🗺️ 结果地图", "📈 统计数据", "📍 采样点"])

# =====================================================================
#  Tab 1: 资料收集
# =====================================================================

with tab_dl:
    st.subheader("阶段一：自动收集遥感影像资料")
    can_collect = bool(S.boundary) and Path(S.boundary).exists() and not S.collecting
    if st.button(
        f"📥 开始收集 — {S.region or '当前地区'} {y1}/{y2} 影像",
        type="primary", use_container_width=True, disabled=not can_collect):
        _collect_data(y1, y2, ms, me)

    if S.data_ok:
        info = S.collected or {}
        ctx = info.get("context", {})
        st.success(f"✅ **资料收集完成！** 可以进入「运行分析」标签页了")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("地区", ctx.get("region","-"))
        with c2: st.metric("瓦片数", ctx.get("tiles_found",0))
        with c3: st.metric("数据量", f"{ctx.get('size_mb',0):.0f} MB")
        if S.boundary:
            try:
                gj, geom = _load_gj(S.boundary)
                m = folium.Map(location=[geom.centroid.y, geom.centroid.x], zoom_start=11)
                folium.GeoJson(gj, style_function=lambda x:
                               {"fillColor":"#3388ff","color":"#3388ff",
                                "weight":2,"fillOpacity":0.1}).add_to(m)
                st_folium(m, width="100%", height=350)
            except Exception:
                pass
    elif S.collecting:
        st.info("⏳ 正在收集资料，请稍候...")
    else:
        cache_slug = S.slug or _slug(S.region or "")
        cache_dir = BASE_DIR / "data" / cache_slug
        cached = all((cache_dir / f).exists() for f in
                     [f"B4_{y1}.tif", f"B8_{y1}.tif", f"B4_{y2}.tif", f"B8_{y2}.tif"])
        if cached:
            mb = sum(f.stat().st_size for f in cache_dir.iterdir()
                     if f.suffix == ".tif") / 1024 / 1024
            st.success(f"✅ data/{cache_slug}/ 已有缓存影像 ({mb:.0f} MB), 无需下载")
        else:
            st.info("📡 data 目录无缓存，点击上方按钮下载 Sentinel-2 影像")

        if S.boundary and Path(S.boundary).exists():
            try:
                gj, geom = _load_gj(S.boundary)
                m = folium.Map(location=[geom.centroid.y, geom.centroid.x], zoom_start=11)
                folium.GeoJson(gj, style_function=lambda x:
                               {"fillColor":"#3388ff","color":"#3388ff",
                                "weight":2,"fillOpacity":0.1}).add_to(m)
                st.caption("📍 待分析区域")
                st_folium(m, width="100%", height=350)
            except Exception:
                pass

# =====================================================================
#  Tab 2: 运行分析
# =====================================================================

with tab_run:
    st.subheader("阶段二：运行变化检测分析")
    ready = S.data_ok
    if not ready:
        slug = S.slug or _slug(S.region or "default")
        data_dir = BASE_DIR / "data" / slug
        cached = all((data_dir/f).exists() for f in
                     [f"B4_{y1}.tif",f"B8_{y1}.tif",f"B4_{y2}.tif",f"B8_{y2}.tif"])
        if cached:
            mb = sum((data_dir/f).stat().st_size for f in os.listdir(data_dir)
                     if f.endswith(".tif")) / 1024/1024
            st.success(f"检测到已有缓存 ({mb:.0f} MB)，可直接运行")
            ready = True
        else:
            st.warning("⚠️ 请先在「资料收集」标签页下载影像")
    else:
        st.success("✅ 资料已就绪，可以开始分析")

    st.markdown("---")
    can_run = ready and not S.running
    if st.button(
        f"🚀 运行 — {S.region or ''} {y1}→{y2} 变化检测",
        type="primary", use_container_width=True, disabled=not can_run):
        _run_analysis(y1, y2, ms, me, dmode, ct, ma, bf,
                      strat, np_pts, md_pts, greport)

    if S.results:
        st.markdown("---")
        st.subheader("✅ 分析完成")
        res = S.results
        a,b,c,d = st.columns(4)
        with a: st.metric("采样点数", res.get("num_sample_points",0))
        with b:
            s = res.get("stats",{})
            ah = s.get("changed_total",{}).get("area_ha") or s.get("changed_area_ha",0)
            st.metric("变化面积", f"{ah:,.0f} ha")
        with c: st.metric("总耗时", f"{res.get('timing',{}).get('total',0):.0f}s")
        with d: st.metric("输出文件", len(res.get("files",{})))
        with st.expander("📁 输出文件"):
            for fmt, path in res.get("files",{}).items():
                if os.path.exists(path):
                    sz = os.path.getsize(path)/1024
                    st.text(f"  {fmt.upper():8s}  {Path(path).name}  ({sz:,.0f} KB)")

# =====================================================================
#  Tab 3-5: 结果
# =====================================================================

with tab_map:
    if S.arr is not None:
        _show_map()
    elif S.results:
        st.warning("地图数据不可用，GeoTIFF 文件可能不存在")
        with st.expander("🔍 调试信息"):
            gtif = S.results.get("files",{}).get("geotiff","?")
            st.code(f"arr={type(S.arr)}\n边界={S.boundary}\ngeotiff路径={gtif}\n存在={os.path.exists(gtif) if gtif and gtif!='?' else '?'}")
    else:
        st.info("运行分析后查看结果地图")
with tab_stat:
    if S.stats:
        _show_stats()
    elif S.results:
        st.warning("统计数据不可用")
    else:
        st.info("运行分析后查看统计")
with tab_pt:
    if S.cands:
        _show_points()
    elif S.results:
        st.warning("采样点数据不可用")
        with st.expander("🔍 调试"):
            gjp = S.results.get("files",{}).get("geojson","?")
            st.code(f"cands={type(S.cands)}\ngeojson路径={gjp}\n存在={os.path.exists(gjp) if gjp and gjp!='?' else '?'}")
    else:
        st.info("运行分析后查看采样点")

if __name__ == "__main__":
    prog_bar.empty(); stat_ph.empty()
