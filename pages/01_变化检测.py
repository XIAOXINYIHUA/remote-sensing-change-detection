"""
遥感分析平台 v3 — 变化检测页面

两阶段工作流:
  阶段1: 选择地区 → 自动收集遥感影像（多源/多指数）
  阶段2: 一键运行变化检测分析
"""

import os, sys, json, io, re, time, warnings
warnings.filterwarnings("ignore")

# GDAL 临时文件指向项目盘（避免写 C 盘）
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["GDAL_TMPDIR"] = os.path.join(_base, "tmp")
os.environ["TMP"] = os.environ["GDAL_TMPDIR"]
os.environ["TEMP"] = os.environ["GDAL_TMPDIR"]
os.makedirs(os.environ["GDAL_TMPDIR"], exist_ok=True)

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
from shapely.geometry import shape
import rasterio
import requests

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from engine.utils import log, get_needed_bands, INDEX_BAND_REQUIREMENTS
from engine.data_sources import get_data_source
from engine.core import run_pipeline_v3

from frontend.utils import (
    inject_global_css,
    setup_matplotlib_font,
    detect_region, slug as _slug,
    load_gj, fetch_boundary_osm,
    DATA_SOURCE_LABELS, INDEX_HELP, MODE_LABELS,
    CAPABILITY_MATRIX,
    save_app_state, load_app_state,
)
from frontend.project_state import ProjectState
from frontend.components.ui import (
    empty_state,
    page_header,
    section_header,
    sidebar_header,
    status_banner,
    workflow_bar,
)

setup_matplotlib_font()


# =====================================================================
#  进度系统
# =====================================================================

STEPS_ANALYSIS = [
    ("📂", "加载波段"),
    ("🧮", "指数计算"),
    ("📊", "差值分析"),
    ("🏷️", "多级分类"),
    ("🔬", "连通域过滤"),
    ("📍", "生成采样点"),
    ("📋", "输出报告"),
    ("🏁", "完成"),
]


def _init_progress(n, labels):
    st.session_state.cd_progress = 0
    st.session_state.cd_progress_msg = ""
    st.session_state.cd_progress_start = time.time()
    st.session_state.cd_progress_n = n
    st.session_state.cd_progress_labels = labels
    st.session_state.cd_progress_active = 0


def _render_progress(active, msg, elapsed, detail=""):
    labels = st.session_state.get("cd_progress_labels", [])
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
    ts = f"⏱️ {elapsed:.0f}s" if elapsed < 120 else f"⏱️ {elapsed / 60:.1f}min"
    stat_ph.markdown(
        f"<div style='font-size:13px'>{step_bar}</div>"
        f"<div><b>{msg}</b> &nbsp; {ts}"
        f"{' &nbsp;—&nbsp; ' + detail if detail else ''}</div>",
        unsafe_allow_html=True,
    )


def _advance_step(idx, msg, pct, detail=""):
    st.session_state.cd_progress = int(pct)
    st.session_state.cd_progress_msg = msg
    st.session_state.cd_progress_active = idx
    elapsed = time.time() - st.session_state.get("cd_progress_start", time.time())
    _render_progress(idx, msg, elapsed, detail)


def _cb_auto(msg, pct):
    st.session_state.cd_progress = int(pct)
    st.session_state.cd_progress_msg = msg
    elapsed = time.time() - st.session_state.get("cd_progress_start", time.time())
    prog_bar.progress(int(pct) / 100.0)
    labels = st.session_state.get("cd_progress_labels", [])
    if pct < 10:
        active = 0
    elif pct < 20:
        active = 1
    elif pct < 35:
        active = 2
    elif pct < 55:
        active = 3
    elif pct < 70:
        active = 4
    elif pct < 80:
        active = 5
    elif pct < 95:
        active = 6
    else:
        active = min(len(labels) - 1, 7)
    st.session_state.cd_progress_active = active
    _render_progress(active, msg, elapsed)


# =====================================================================
#  v3 数据收集
# =====================================================================

def _collect_data(y1, y2, ms, me, data_src, idx_type):
    """使用 v3 数据源工厂收集影像。"""
    st.session_state.cd_collecting = True
    st.session_state.cd_data_ok = False

    stat_ph.info("⏳ 初始化...")
    prog_bar.progress(0)

    try:
        region = st.session_state.cd_region or "未知"
        slug_name = st.session_state.cd_slug or _slug(region)

        config.REGION_NAME = region
        config.YEAR1 = int(y1)
        config.YEAR2 = int(y2)
        config.MONTH_START = int(ms)
        config.MONTH_END = int(me)
        config.DATA_SOURCE = data_src
        config.INDEX_TYPE = idx_type
        if st.session_state.cd_boundary:
            config.BOUNDARY_FILE = st.session_state.cd_boundary

        # 同步到 ProjectState
        ps = ProjectState.get()
        ps.region = region
        ps.data_source = data_src
        ps.index_type = idx_type
        ps.year1 = int(y1)
        ps.year2 = int(y2)
        ProjectState.commit()

        data_dir = str(BASE_DIR / "data" / slug_name)
        os.makedirs(data_dir, exist_ok=True)
        boundary = load_gj(st.session_state.cd_boundary)[1]

        # 确定需要的波段
        needed = get_needed_bands(idx_type)
        source = get_data_source(data_src)

        label = DATA_SOURCE_LABELS.get(data_src, data_src)
        info_msg = st.empty()

        # === Y1 ===
        info_msg.info(f"🔍 搜索 {y1} 年 {label} 影像 ({idx_type} 需要 {needed})...")
        s1 = source.search_scenes(boundary, y1, ms, me, config.MAX_CLOUD_PCT)
        if not s1:
            raise RuntimeError(f"{y1} 年未找到可用 {data_src} 影像")

        info_msg.info(f"📥 下载 {y1} 年影像 ({len(s1)} 瓦片, {len(needed)} 波段)...")
        b1 = source.download_bands(s1, data_dir, needed)
        m1 = sum(os.path.getsize(p) for p in b1.values() if os.path.exists(p)) / 1024 / 1024
        info_msg.success(f"✅ {y1} 年完成 ({m1:.0f} MB)")

        # === Y2 ===
        info_msg.info(f"🔍 搜索 {y2} 年 {label} 影像 ...")
        s2 = source.search_scenes(boundary, y2, ms, me, config.MAX_CLOUD_PCT)
        if not s2:
            raise RuntimeError(f"{y2} 年未找到可用 {data_src} 影像")

        info_msg.info(f"📥 下载 {y2} 年影像 ({len(s2)} 瓦片, {len(needed)} 波段)...")
        b2 = source.download_bands(s2, data_dir, needed)
        m2 = sum(os.path.getsize(p) for p in b2.values() if os.path.exists(p)) / 1024 / 1024
        info_msg.success(f"✅ {y2} 年完成 ({m2:.0f} MB)")

        total_mb = m1 + m2

        # —— 登记到 SnapshotCache，避免后续分析重复下载 ——
        try:
            from engine.cache import SnapshotCache, compute_aoi_hash
            import hashlib
            sc = SnapshotCache(cache_root=str(BASE_DIR / "data" / "cache"))
            aoi_hash = compute_aoi_hash(boundary)
            raw_key = f"{aoi_hash}|{data_src}|{idx_type}|{y1}|{y2}"
            new_key = hashlib.md5(raw_key.encode()).hexdigest()[:16]
            if not sc.has(new_key):
                snap_dict = {
                    "region": region, "aoi_hash": aoi_hash,
                    "data_source": data_src, "index_type": idx_type,
                    "year1": int(y1), "year2": int(y2),
                    "month_start": int(ms), "month_end": int(me),
                    "max_cloud_pct": config.MAX_CLOUD_PCT,
                    "bands": list(needed), "tile_ids": [],
                    "preprocessing_version": "v3.1",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                sc.put_snapshot(new_key, snap_dict)
                import shutil
                cache_dir = sc.get_band_dir(new_key)
                for fpath in list(b1.values()) + list(b2.values()):
                    if os.path.exists(fpath):
                        shutil.copy2(fpath, cache_dir)
                sc.mark_complete(new_key)
        except Exception as e:
            log(f"  Cache registration skipped: {e}")

        st.session_state.cd_collected = {
            "context": {
                "region": region, "year1": y1, "year2": y2,
                "data_source": data_src, "index_type": idx_type,
                "tiles_found": len(s1) + len(s2),
                "bands": needed, "size_mb": total_mb,
            }
        }
        st.session_state.cd_data_ok = True
        info_msg.success(f"✅ {region} 资料收集完成 ({total_mb:.0f} MB)")

    except Exception as e:
        st.error(f"❌ 资料收集失败: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        st.session_state.cd_collecting = False


def _check_cache(slug_name, y1, y2, idx_type):
    """检查缓存是否满足指数需求。"""
    data_dir = BASE_DIR / "data" / slug_name
    if not data_dir.exists():
        return False, []
    needed = get_needed_bands(idx_type)
    from engine.data_sources import Sentinel2DataSource
    s2 = Sentinel2DataSource()
    existing = set(os.listdir(str(data_dir)))
    missing = []
    for band in needed:
        fname = s2.FILE_NAME_MAP.get(band, band.upper())
        f1 = f"{fname}_{y1}.tif"
        f2 = f"{fname}_{y2}.tif"
        if f1 not in existing:
            missing.append(f1)
        if f2 not in existing:
            missing.append(f2)
    return len(missing) == 0, missing


# =====================================================================
#  v3 分析运行
# =====================================================================

def _run_analysis(y1, y2, ms, me, dmode, ct, sthr, ma, bf,
                  strat, np_pts, md_pts, greport,
                  idx_type, data_src):
    """v3 分析流水线。"""
    st.session_state.cd_running = True
    labels = [s[1] for s in STEPS_ANALYSIS]
    _init_progress(len(labels), labels)
    stat_ph.info("⏳ 初始化分析...")
    prog_bar.progress(0)

    try:
        region = st.session_state.cd_region or "未知"
        slug_name = st.session_state.cd_slug or _slug(region)

        config.REGION_NAME = region
        config.YEAR1 = int(y1)
        config.YEAR2 = int(y2)
        config.MONTH_START = int(ms)
        config.MONTH_END = int(me)
        config.DETECTION_MODE = dmode
        config.CHANGE_THRESHOLD = ct
        config.SEVERE_THRESHOLD = sthr
        config.MORPH_KERNEL_SIZE = 4
        config.MIN_CHANGE_AREA_M2 = int(ma)
        config.MIN_CHANGE_AREA_PX = max(1, int(ma) // 100)
        config.USE_BILATERAL_FILTER = bf
        config.SAMPLE_STRATEGY = strat
        config.NUM_SAMPLE_POINTS = int(np_pts) if int(np_pts) > 0 else None
        config.MIN_POINT_DISTANCE_M = int(md_pts) if int(md_pts) > 0 else None
        config.OUTPUT_FORMATS = "geojson,csv"
        config.GENERATE_REPORT = greport
        config.INDEX_TYPE = idx_type
        config.DATA_SOURCE = data_src
        if st.session_state.cd_boundary:
            config.BOUNDARY_FILE = st.session_state.cd_boundary

        region_data_dir = str(BASE_DIR / "data" / slug_name)
        region_out_dir = str(BASE_DIR / "output" / slug_name)
        os.makedirs(region_out_dir, exist_ok=True)

        res = run_pipeline_v3(
            data_source_name=data_src,
            data_dir=region_data_dir,
            out_dir=region_out_dir,
            index_type=idx_type,
            detection_mode=dmode,
            threshold=ct,
            severe_threshold=sthr,
            min_area_m2=ma,
            progress_callback=_cb_auto,
        )

        # —— P0-7 修复：检查错误返回 ——
        if "error" in res:
            st.error(f"❌ 分析失败: {res['error']}")
            st.session_state.cd_running = False
            return

        st.session_state.cd_results = res
        st.session_state.cd_stats = res.get("stats")

        # 加载 GeoTIFF 结果用于地图
        gtif = res.get("files", {}).get("geotiff")
        if gtif and os.path.exists(gtif):
            with rasterio.open(gtif) as src:
                st.session_state.cd_arr = src.read(1)
                st.session_state.cd_meta = src.meta

        # 加载采样点
        gjp = res.get("files", {}).get("geojson")
        if gjp and os.path.exists(gjp):
            with open(gjp, encoding="utf-8") as f:
                st.session_state.cd_cands = json.load(f).get("features", [])

        timing = res.get("timing", {}).get("total", 0)
        st.session_state.cd_analysis_done = True

        # —— 记录任务到 ProjectState ——
        try:
            from frontend.project_state import ProjectState
            s = res.get("stats", {})
            ah = (s.get("changed_total", {}).get("area_ha")
                  or s.get("changed_area_ha", 0))
            task_record = {
                "id": f"cd_{slug_name}_{y1}_{y2}_{idx_type}",
                "mode": "change_detection",
                "region": region,
                "data_source": data_src,
                "index_type": idx_type,
                "year1": int(y1), "year2": int(y2),
                "status": "success",
                "started_at": st.session_state.get("cd_progress_start", ""),
                "duration_s": round(timing, 1),
                "stats_summary": f"变化 {ah:,.0f} ha" if ah else "完成",
                "output_files": res.get("files", {}),
            }
            ProjectState.add_task(task_record)
        except Exception:
            pass

        # —— 自动生成 PDF 报告（根据 checkbox 设置）——
        if st.session_state.get("cd_gen_pdf", True):
            try:
                from engine.report_generator import generate_report_pdf
                meta_pdf = {
                    "region": region,
                    "year1": config.YEAR1,
                    "year2": config.YEAR2,
                    "index_type": idx_type,
                    "data_source": DATA_SOURCE_LABELS.get(data_src, data_src),
                    "threshold": ct,
                    "detection_mode": dmode,
                }
                report_image = res.get("files", {}).get("report", "")
                sample_pts = None
                if st.session_state.cd_cands:
                    sample_pts = [
                        {"lon": f["geometry"]["coordinates"][0],
                         "lat": f["geometry"]["coordinates"][1],
                         "class_name": f["properties"].get("class_name", "")}
                        for f in st.session_state.cd_cands[:50]
                    ]
                pdf_path = generate_report_pdf(
                    stats=res.get("stats", {}),
                    report_image=report_image,
                    sample_points=sample_pts,
                    boundary_info={"area_km2": ""},
                    metadata=meta_pdf,
                    output_dir=region_out_dir,
                )
                if pdf_path and os.path.exists(pdf_path):
                    res["files"]["pdf"] = pdf_path
            except Exception as e:
                print(f"  PDF auto-generation skipped: {e}")

        idx_label = idx_type
        st.success(
            f"✅ {region} · {idx_label} · {data_src} · "
            f"{y1}→{y2}  分析完成！耗时 {timing:.0f}s"
        )
        _save_state()

    except Exception as e:
        st.error(f"❌ 分析失败: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        st.session_state.cd_running = False


# =====================================================================
#  结果展示
# =====================================================================

def _show_map():
    if st.session_state.cd_arr is None or not st.session_state.cd_boundary:
        return
    try:
        gj, geom = load_gj(st.session_state.cd_boundary)
    except Exception:
        st.error("边界解析失败")
        return

    from frontend.components.map_view import (
        add_boundary_layer, legend_html,
    )

    idx = getattr(config, "INDEX_TYPE", "NDVI")
    meta = st.session_state.cd_meta

    # 透明度控制
    map_opacity = st.session_state.get("cd_map_opacity", 0.6)
    map_opacity = st.slider(
        "结果图层透明度", 0.0, 1.0, map_opacity, 0.05,
        key="cd_opacity_slider",
        on_change=lambda: st.session_state.__setitem__(
            "cd_map_opacity", st.session_state.cd_opacity_slider))

    m = folium.Map(
        location=[geom.centroid.y, geom.centroid.x],
        zoom_start=11, control_scale=True,
        tiles="CartoDB positron",
    )
    minx, miny, maxx, maxy = geom.bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(20, 20))
    Fullscreen().add_to(m)
    add_boundary_layer(m, gj)

    # 使用透明度参数
    from frontend.components.map_view import add_change_result_layer as _add_layer
    _add_layer(m, st.session_state.cd_arr, meta,
               index_type=idx, opacity=map_opacity)

    # 底图切换
    folium.TileLayer("OpenStreetMap", name="街道图", show=False).add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="卫星影像", attr="Esri", show=False
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html(idx)))
    output = st_folium(m, width=None, height=600, key="change_result_map")

    # ---- 点击交互：显示像元信息 ----
    clicked = output.get("last_clicked")
    if clicked:
        lat, lng = clicked["lat"], clicked["lng"]
        arr = st.session_state.cd_arr
        # 将经纬度转回像素坐标
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
        x, y = t.transform(lng, lat)
        from rasterio.transform import rowcol
        if meta and meta.get("transform"):
            row, col = rowcol(meta["transform"], x, y)
            row, col = int(row), int(col)
            if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                cls_val = int(arr[row, col])
                from engine.indices import INDEX_CLASS_NAMES
                names = INDEX_CLASS_NAMES.get(idx, INDEX_CLASS_NAMES["NDVI"])
                cls_name = names.get(cls_val, f"未知({cls_val})")
                status_banner(
                    "info", f"{idx} · {cls_name}",
                    f"位置 {lat:.5f}, {lng:.5f} · 像素 ({row}, {col})",
                )

    # 下载按钮
    fs = st.session_state.cd_results.get("files", {}) if st.session_state.cd_results else {}
    if fs:
        cols = st.columns(len(fs))
        for i, (fmt, p) in enumerate(fs.items()):
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    cols[i].download_button(
                        Path(p).name, fh.read(), Path(p).name,
                        icon=":material/download:")


def _show_stats():
    s = st.session_state.cd_stats
    idx = getattr(config, "INDEX_TYPE", "NDVI")
    dmode = config.DETECTION_MODE

    if dmode != "multi":
        st.metric("变化面积", f"{s.get('changed_area_ha', 0):,.0f} ha")
        return

    from engine.indices import INDEX_CLASS_NAMES, INDEX_COLORS
    cls_n = INDEX_CLASS_NAMES.get(idx, INDEX_CLASS_NAMES["NDVI"])
    cls_c = INDEX_COLORS.get(idx, INDEX_COLORS["NDVI"])

    rows = []
    for cv in [-3, -2, 1, 2, 3]:
        if s.get(cv):
            rows.append({
                "类别": s[cv]["name"],
                "面积(ha)": f"{s[cv]['area_ha']:,.0f}",
                "占比": f"{s[cv]['percent']:.1f}%",
            })
    if "changed_total" in s:
        t = s["changed_total"]
        rows.append({"类别": "合计", "面积(ha)": f"{t['area_ha']:,.0f}",
                      "占比": f"{t['percent']:.1f}%"})
    st.table(pd.DataFrame(rows))

    cd = [(c, s.get(c, {}).get("area_ha", 0)) for c in [-3, -2, 2, 3]]
    cd = [(n, a) for n, a in cd if a > 0]
    if cd:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        labs = [cls_n[c] for c, _ in cd]
        vals = [a for _, a in cd]
        colors = [cls_c[c] for c, _ in cd]
        a1.bar(labs, vals, color=colors, edgecolor="white")
        a1.set_ylabel("面积 (ha)")
        a1.set_title(f"{idx} 各级变化面积", fontweight="bold")
        a1.tick_params(axis="x", rotation=15)
        a2.pie(vals, labels=labs, autopct="%1.1f%%", colors=colors, startangle=90)
        a2.set_title("变化占比", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        t = s["changed_total"]
        status_banner(
            "info", "变化检测摘要",
            f"{config.REGION_NAME} · {config.YEAR1}-{config.YEAR2} · "
            f"{idx} 显著变化 {t['area_ha']:,.0f} ha ({t['percent']:.1f}%)",
        )

        # —— 多期对比视图 ——
        with st.expander("变化分级对比"):
            arr = st.session_state.get("cd_arr")
            if arr is not None:
                h, w = arr.shape
                scale = max(1, min(h, w) // 400)
                small = arr[::scale, ::scale]

                from engine.core import _get_change_mask_multi
                chg = _get_change_mask_multi(small)
                cls_small = small.copy()
                cls_small[cls_small == 0] = 5  # no-data

                colors_rgb = { -3: (180,0,0), -2: (255,100,0), 1: (40,40,40),
                               2: (0,180,80), 3: (0,80,0), 5: (10,10,10) }
                rgb = np.zeros((*cls_small.shape, 3), dtype=np.uint8)
                for cv, c in colors_rgb.items():
                    rgb[cls_small == cv] = c
                change_overlay = np.full((*cls_small.shape, 4), 0, dtype=np.uint8)
                change_overlay[chg] = [255, 60, 60, 180]

                fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
                axes2[0].imshow(rgb)
                axes2[0].set_title(f"{idx} 分级结果", fontsize=13, color="#18221D")
                axes2[0].axis("off")

                axes2[1].imshow(rgb)
                axes2[1].imshow(change_overlay, interpolation="nearest")
                axes2[1].set_title("变化区域高亮", fontsize=13, color="#18221D")
                axes2[1].axis("off")

                axes2[2].axis("off")
                from matplotlib.patches import Patch as MPatch
                patches_val = [
                    MPatch(color="#b40000", label=f"严重退化 {s.get(-3,{}).get('area_ha',0):.0f}ha"),
                    MPatch(color="#ff6400", label=f"中度退化 {s.get(-2,{}).get('area_ha',0):.0f}ha"),
                    MPatch(color="#282828", label=f"稳定"),
                    MPatch(color="#00b450", label=f"中度恢复 {s.get(2,{}).get('area_ha',0):.0f}ha"),
                    MPatch(color="#005000", label=f"强烈恢复 {s.get(3,{}).get('area_ha',0):.0f}ha"),
                    MPatch(color="#ff3c3c", label=f"变化合计 {s.get('changed_total',{}).get('area_ha',0):.0f}ha"),
                ]
                axes2[2].legend(handles=patches_val, loc="center", fontsize=11,
                                facecolor="#FFFFFF", labelcolor="#18221D")
                axes2[2].set_title("图例", fontsize=13, color="#18221D")

                fig2.patch.set_facecolor("#FFFFFF")
                for ax in axes2:
                    ax.set_facecolor("#FFFFFF")
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close()


def _show_points():
    if not st.session_state.cd_cands:
        st.warning("无采样点")
        return
    rows = [{
        "序号": f["properties"].get("id", ""),
        "经度": round(f["geometry"]["coordinates"][0], 6),
        "纬度": round(f["geometry"]["coordinates"][1], 6),
        "面积(ha)": f["properties"].get("area_ha", ""),
        "类别": f["properties"].get("class_name", ""),
    } for f in st.session_state.cd_cands]
    df = pd.DataFrame(rows)
    section_header(f"采样点 · {len(rows)}", "候选斑块中心点与类别属性。")
    st.dataframe(df, use_container_width=True, height=400)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "下载采样点 CSV", buf.getvalue(), "sample_points.csv", "text/csv",
        icon=":material/download:",
    )
    if "类别" in df.columns:
        st.bar_chart(df["类别"].value_counts())


# =====================================================================
#  Session State 初始化
# =====================================================================

S = st.session_state

# 从 ProjectState 初始化
ps = ProjectState.get()
if ps.is_ready():
    S.setdefault("cd_boundary", ps.boundary)
    S.setdefault("cd_region", ps.region)
    S.setdefault("cd_slug", ps.slug)

_INIT = {
    "cd_boundary": None,
    "cd_region": None,
    "cd_slug": None,
    "cd_data_ok": False,
    "cd_collecting": False,
    "cd_running": False,
    "cd_collected": None,
    "cd_results": None,
    "cd_arr": None,
    "cd_meta": None,
    "cd_stats": None,
    "cd_cands": None,
    "cd_progress": 0,
    "cd_progress_msg": "",
    "cd_analysis_done": False,
    "cd_map_opacity": 0.6,
}
for k, v in _INIT.items():
    if k not in S:
        S[k] = v

# ---- 状态持久化 ----
STATE_FILE = BASE_DIR / ".app_state.json"

def _save_state():
    state = {
        "cd_boundary": S.get("cd_boundary"),
        "cd_region": S.get("cd_region"),
        "cd_slug": S.get("cd_slug"),
        "cd_data_ok": S.get("cd_data_ok", False),
        "cd_collected": S.get("cd_collected"),
    }
    save_app_state(STATE_FILE, state)

def _load_state():
    loaded = load_app_state(STATE_FILE, ["cd_boundary", "cd_region", "cd_slug", "cd_data_ok", "cd_collected"])
    for k, v in loaded.items():
        S[k] = v

_load_state()

# =====================================================================
#  Sidebar 参数面板
# =====================================================================

sidebar_header("变化检测参数", "分析范围、时相与检测阈值")

# --- 地区选择 ---
st.sidebar.subheader("分析区域")
exist = {f.name: str(f) for f in BASE_DIR.glob("*.geojson")}
cur_name = Path(S.cd_boundary).name if S.cd_boundary else None
if exist:
    sel_name = st.sidebar.selectbox(
        "已有边界文件",
        list(exist.keys()),
        index=list(exist.keys()).index(cur_name) if cur_name in exist else 0,
    )
    chosen = exist[sel_name]
    if chosen != S.cd_boundary:
        S.cd_boundary = chosen
        r = detect_region(chosen)
        S.cd_region = r
        S.cd_slug = _slug(r)
        S.cd_data_ok = False
        S.cd_collected = None
        ProjectState.set_aoi(r, chosen)
        _save_state()
        st.rerun()

search_name = st.sidebar.text_input("搜索地区", "", placeholder="城市/区县名")
if st.sidebar.button("获取边界", icon=":material/search:", use_container_width=True,
                      disabled=not search_name.strip()):
    with st.sidebar.status(f"正在搜索 {search_name}..."):
        gj, err = fetch_boundary_osm(search_name.strip())
        if err:
            st.sidebar.error(err)
        else:
            fp = BASE_DIR / f"{_slug(search_name)}.geojson"
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(gj, f, ensure_ascii=False)
            S.cd_boundary = str(fp)
            S.cd_region = search_name.strip()
            S.cd_slug = _slug(search_name)
            S.cd_data_ok = False
            S.cd_collected = None
            ProjectState.set_aoi(S.cd_region, S.cd_boundary, S.cd_slug)
            _save_state()
            st.rerun()

if S.cd_boundary and Path(S.cd_boundary).exists():
    try:
        _, geom = load_gj(S.cd_boundary)
        # 将面积从度²转为 km²（粗略估算：1° ≈ 111 km）
        area_deg2 = geom.area
        area_km2 = area_deg2 * (111.32 ** 2)
        st.sidebar.caption(
            f"中心: {geom.centroid.y:.3f}, {geom.centroid.x:.3f}  "
            f"面积: {area_km2:.1f} km²"
        )
    except Exception:
        st.sidebar.error("边界解析失败")
else:
    st.sidebar.error("请选择边界文件")

st.sidebar.markdown("---")

# --- 时相 ---
st.sidebar.subheader("时间范围")
y1 = st.sidebar.number_input("前时相年份", 2015, 2027, config.YEAR1, 1)
y2 = st.sidebar.number_input("后时相年份", 2015, 2027, config.YEAR2, 1)
ms = st.sidebar.slider("起始月份", 1, 12, config.MONTH_START)
me = st.sidebar.slider("结束月份", 1, 12, config.MONTH_END)
st.sidebar.markdown("---")

# --- 数据源与指数 ---
st.sidebar.subheader("数据源与指数")

# 能力矩阵：根据数据源限制可用指数
all_sources = ["sentinel2", "landsat8", "landsat9", "sentinel1"]
data_src = st.sidebar.selectbox(
    "数据源",
    all_sources,
    format_func=lambda x: DATA_SOURCE_LABELS.get(x, x),
    help="sentinel2: 10m 多光谱 | landsat: 30m | sentinel1: SAR雷达（不支持光学指数）"
)

# 根据能力矩阵过滤可用指数
available_indices = CAPABILITY_MATRIX.get(data_src, ["NDVI", "NDWI", "MNDWI", "NBR", "EVI", "SAVI"])
if not available_indices:
    st.sidebar.warning("⚠️ 该数据源暂不支持光谱指数。请切换至 Sentinel-2 或 Landsat。")
    available_indices = ["—"]

# 初始化/保留 idx_type
cur_idx = st.session_state.get("cd_cur_idx", "NDVI")
cur_idx = cur_idx if cur_idx in available_indices else available_indices[0]

idx_type = st.sidebar.selectbox(
    "遥感指数",
    available_indices,
    index=available_indices.index(cur_idx) if cur_idx in available_indices else 0,
    format_func=lambda x: f"{x} ({INDEX_HELP.get(x, '')})" if x != "—" else "（无可用指数）",
)
st.session_state.cd_cur_idx = idx_type
st.sidebar.markdown("---")

# --- 检测参数 ---
with st.sidebar.expander("检测参数", expanded=True):
    # 检查 ML 模型是否可用
    ml_model_path = getattr(config, 'ML_MODEL_PATH', 'models/change_detector.pkl')
    ml_available = ml_model_path and os.path.exists(os.path.join(BASE_DIR, ml_model_path))
    all_modes = ["multi", "binary"]
    if ml_available:
        all_modes.append("ml")
    else:
        all_modes.append("ml_disabled")

    dmode_idx = 0
    cur_dmode = st.session_state.get("cd_cur_dmode", "multi")
    if cur_dmode in all_modes:
        dmode_idx = all_modes.index(cur_dmode)
    elif cur_dmode == "ml" and not ml_available:
        dmode_idx = 0  # fallback to multi

    def _fmt_mode(m):
        if m == "ml_disabled":
            return "机器学习 ⚠️（无模型）"
        return MODE_LABELS.get(m, m)

    dmode = st.selectbox(
        "模式",
        all_modes,
        index=dmode_idx,
        format_func=_fmt_mode,
    )
    if dmode == "ml_disabled":
        st.caption("⚠️ 未检测到 ML 模型文件。请先训练模型或切换为多级/二值模式。")
        dmode = "multi"
    st.session_state.cd_cur_dmode = dmode

    ct = st.slider(
        f"{idx_type} 阈值", 0.05, 0.50, config.CHANGE_THRESHOLD, 0.01,
        help=f"{idx_type} 差值超过此值视为变化",
    )
    sthr = st.slider("强烈阈值", 0.10, 0.50, config.SEVERE_THRESHOLD, 0.01)
    ma = st.number_input("最小斑块 (m²)", 100, 50000, config.MIN_CHANGE_AREA_M2, 100)
    bf = st.checkbox("双边预滤波", config.USE_BILATERAL_FILTER)

with st.sidebar.expander("采样设置"):
    strat = st.selectbox(
        "策略", ["centroid", "random", "grid"],
        format_func=lambda x: {"centroid": "质心", "random": "随机", "grid": "网格"}[x],
    )
    np_pts = st.number_input("上限 (0=不限)", 0, 100000, 0)
    md_pts = st.number_input("间距 (m)", 0, 10000, 0)

with st.sidebar.expander("输出设置"):
    greport = st.checkbox("报告 PNG", True)
    gen_pdf = st.checkbox("自动 PDF 报告", True, key="cd_gen_pdf",
                          help="分析完成后自动生成专业 PDF 报告（需 reportlab）")

# =====================================================================
#  引导式分析向导（无历史分析时显示）
# =====================================================================

wizard_key = "_cd_wizard_active"
if not S.cd_data_ok and not S.cd_results and not S.get(wizard_key, False):
    from frontend.components.wizard import render_wizard
    wizard_done = render_wizard()
    if wizard_done:
        S[wizard_key] = True
        st.rerun()
    # wizard 未完成时隐藏下方内容
    st.stop()

# =====================================================================
#  主面板
# =====================================================================

region_display = S.cd_region or "未选择"
src_label = DATA_SOURCE_LABELS.get(data_src, data_src)
page_header(
    "变化检测",
    "分析工作台",
    f"{region_display} 的两时相遥感指数差异分析。",
    meta={
        "区域": region_display,
        "时相": f"{y1} - {y2}",
        "月份": f"{ms} - {me} 月",
        "数据": src_label,
        "指数": idx_type,
        "阈值": f"±{ct}",
    },
    status=("分析完成" if S.cd_results else "数据已就绪" if S.cd_data_ok else "等待数据"),
    status_tone=("success" if S.cd_results else "info" if S.cd_data_ok else "warning"),
)

workflow_bar(
    ["准备影像", "运行检测", "空间结果", "统计与采样"],
    active=2 if S.cd_results else 1 if S.cd_data_ok else 0,
    completed=[0, 1] if S.cd_results else [0] if S.cd_data_ok else [],
)

stat_ph = st.empty()
prog_bar = st.progress(0)

tab_dl, tab_run, tab_map, tab_stat, tab_pt = st.tabs(
    ["影像数据", "运行分析", "结果地图", "统计摘要", "采样点"]
)

# =====================================================================
#  Tab 1 — 资料收集
# =====================================================================

with tab_dl:
    section_header("准备遥感影像", "检查本地缓存，按需检索并下载所需波段。", "阶段 01")

    can_collect = bool(S.cd_boundary) and Path(S.cd_boundary).exists() and not S.cd_collecting
    band_str = ", ".join(get_needed_bands(idx_type))
    btn_label = f"收集 {y1} / {y2} 影像 · {idx_type}"
    if st.button(btn_label, type="primary", icon=":material/download:", use_container_width=True,
                  disabled=not can_collect):
        _collect_data(y1, y2, ms, me, data_src, idx_type)

    if S.cd_data_ok:
        info = S.cd_collected or {}
        ctx = info.get("context", {})
        status_banner("success", "影像数据已就绪", "可进入“运行分析”执行变化检测。")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("地区", ctx.get("region", "-"))
        with c2:
            st.metric("数据量", f"{ctx.get('size_mb', 0):.0f} MB")
        with c3:
            c3.metric("数据源", src_label)
        with c4:
            c4.metric("指数", idx_type)
        if S.cd_boundary:
            try:
                gj, geom = load_gj(S.cd_boundary)
                m = folium.Map(
                    location=[geom.centroid.y, geom.centroid.x],
                    zoom_start=11, control_scale=True,
                )
                folium.GeoJson(gj, style_function=lambda x:
                               {"fillColor": "#3388ff", "color": "#3388ff",
                                "weight": 2, "fillOpacity": 0.1}).add_to(m)
                st_folium(m, width="100%", height=350)
            except Exception:
                pass
    elif S.cd_collecting:
        status_banner("info", "正在收集影像", "检索、下载与波段裁剪正在进行。")
    else:
        # 检查缓存
        cache_slug = S.cd_slug or _slug(S.cd_region or "")
        cached_ok, missing = _check_cache(cache_slug, y1, y2, idx_type)
        if cached_ok:
            data_dir = BASE_DIR / "data" / cache_slug
            mb = sum(f.stat().st_size for f in data_dir.iterdir()
                     if f.suffix == ".tif") / 1024 / 1024
            status_banner("success", "检测到完整本地缓存", f"data/{cache_slug}/ · {mb:.0f} MB")
        elif missing:
            status_banner(
                "warning",
                f"缺少 {len(missing)} 个波段文件",
                f"{src_label} · {idx_type} · 需要 {band_str}",
            )
        else:
            status_banner("info", "尚未收集影像", f"{src_label} · {idx_type}")

        if S.cd_boundary and Path(S.cd_boundary).exists():
            try:
                gj, geom = load_gj(S.cd_boundary)
                m = folium.Map(
                    location=[geom.centroid.y, geom.centroid.x],
                    zoom_start=11, control_scale=True,
                )
                folium.GeoJson(gj, style_function=lambda x:
                               {"fillColor": "#3388ff", "color": "#3388ff",
                                "weight": 2, "fillOpacity": 0.1}).add_to(m)
                st.caption("待分析区域")
                st_folium(m, width="100%", height=350)
            except Exception:
                pass

# =====================================================================
#  Tab 2 — 运行分析
# =====================================================================

with tab_run:
    section_header("执行变化检测", "使用当前侧栏参数计算差值、分级与斑块。", "阶段 02")

    # 检查数据是否就绪
    ready = S.cd_data_ok
    if not ready:
        slug_name = S.cd_slug or _slug(S.cd_region or "default")
        cached_ok, _ = _check_cache(slug_name, y1, y2, idx_type)
        if cached_ok:
            data_dir = BASE_DIR / "data" / slug_name
            mb = sum((data_dir / f).stat().st_size for f in os.listdir(str(data_dir))
                     if f.endswith(".tif")) / 1024 / 1024
            status_banner("success", "检测到可用缓存", f"{mb:.0f} MB · 可直接运行")
            ready = True
        else:
            status_banner("warning", "影像数据未就绪", "请先在“影像数据”中完成收集。")

    can_run = ready and not S.cd_running
    btn_run = st.button(
        f"运行 {idx_type} 变化检测",
        type="primary", icon=":material/play_arrow:",
        use_container_width=True, disabled=not can_run,
    )
    if btn_run:
        _run_analysis(y1, y2, ms, me, dmode, ct, sthr, ma, bf,
                      strat, np_pts, md_pts, greport,
                      idx_type, data_src)

    if S.cd_results:
        section_header("分析结果", "本次执行的核心指标与输出文件。", "完成")
        res = S.cd_results
        a, b, c, d = st.columns(4)
        with a:
            st.metric("采样点", res.get("num_sample_points", 0))
        with b:
            s = res.get("stats", {})
            ah = (s.get("changed_total", {}).get("area_ha")
                  or s.get("changed_area_ha", 0))
            st.metric("变化面积", f"{ah:,.0f} ha")
        with c:
            st.metric("耗时", f"{res.get('timing', {}).get('total', 0):.0f}s")
        with d:
            st.metric("输出文件", len(res.get("files", {})))

        # ---- 实际执行参数（验证 UI 值是否生效） ----
        st.markdown("**实际执行参数**")
        s = res.get("stats", {})
        params_display = {
            "指数": res.get("index_type", idx_type),
            "数据源": res.get("data_source", data_src),
            "变化阈值": s.get("_threshold", ct),
            "强烈阈值": s.get("_severe_threshold", "—"),
            "检测模式": res.get("detection_mode", dmode),
            "像元面积": f"{s.get('_area_per_px_ha', 0.01)*10000:.0f} m²",
            "最小斑块面积": f"{ma:,} m²",
        }
        st.caption(" | ".join(f"{k}: {v}" for k, v in params_display.items()))

        with st.expander("输出文件"):
            for fmt, path in res.get("files", {}).items():
                if os.path.exists(path):
                    sz = os.path.getsize(path) / 1024
                    st.text(f"  {fmt.upper():8s}  {Path(path).name}  ({sz:,.0f} KB)")

            # PDF 报告自动生成后提供下载
            pdf_path = res.get("files", {}).get("pdf")
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as fh:
                    st.download_button(
                        "下载 PDF 报告", fh.read(),
                        Path(pdf_path).name, "application/pdf",
                        icon=":material/download:", use_container_width=True)

# =====================================================================
#  Tab 3-5 — 结果展示
# =====================================================================

with tab_map:
    if S.cd_arr is not None:
        _show_map()
    elif S.cd_results:
        status_banner("warning", "地图数据不可用")
    else:
        empty_state("暂无结果地图", "完成变化检测后显示空间分布。", "MAP")

with tab_stat:
    if S.cd_stats:
        _show_stats()
    elif S.cd_results:
        status_banner("warning", "统计数据不可用")
    else:
        empty_state("暂无统计摘要", "完成变化检测后显示分类面积与占比。", "SUM")

with tab_pt:
    if S.cd_cands:
        _show_points()
    elif S.cd_results:
        status_banner("warning", "采样点数据不可用")
    else:
        empty_state("暂无采样点", "完成变化检测后显示候选采样位置。", "PTS")
