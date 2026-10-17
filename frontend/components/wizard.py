"""Guided setup flow shared by the analysis workbenches."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from frontend.components.ui import page_header, section_header, status_banner, workflow_bar
from frontend.project_state import ProjectState
from frontend.utils import (
    CAPABILITY_MATRIX,
    DATA_SOURCE_LABELS,
    INDEX_HELP,
    detect_region,
    fetch_boundary_osm,
    load_gj,
    slug as _slug,
)


PROBLEM_TYPES = {
    "植被退化与恢复": {
        "recommended": {"data_source": "sentinel2", "index_type": "NDVI"},
        "description": "森林、草地与农作物覆盖的退化或恢复。",
        "available_indices": ["NDVI", "EVI", "SAVI", "NDWI"],
        "available_sources": ["sentinel2", "landsat8", "landsat9"],
    },
    "水体变化": {
        "recommended": {"data_source": "sentinel2", "index_type": "MNDWI"},
        "description": "湖泊、水库与河流范围的时相变化。",
        "available_indices": ["MNDWI", "NDWI", "NBR"],
        "available_sources": ["sentinel2", "landsat8", "landsat9"],
    },
    "火烧迹地": {
        "recommended": {"data_source": "sentinel2", "index_type": "NBR"},
        "description": "火灾影响范围与燃烧严重程度。",
        "available_indices": ["NBR", "NDVI"],
        "available_sources": ["sentinel2", "landsat8", "landsat9"],
    },
    "建设用地扩张": {
        "recommended": {"data_source": "sentinel2", "index_type": "NDBI"},
        "description": "城市扩张与新增建设用地。",
        "available_indices": ["NDBI", "MNDWI", "NDVI"],
        "available_sources": ["sentinel2", "landsat8", "landsat9"],
    },
    "地物分类": {
        "recommended": {"data_source": "sentinel2", "index_type": "ALL"},
        "description": "水体、植被、建筑与裸土等多类别分类。",
        "available_indices": ["ALL"],
        "available_sources": ["sentinel2", "landsat8", "landsat9"],
    },
    "长期趋势分析": {
        "recommended": {"data_source": "sentinel2", "index_type": "NDVI"},
        "description": "五年及以上时间序列的趋势与显著性。",
        "available_indices": ["NDVI", "NDWI", "NBR", "EVI"],
        "available_sources": ["sentinel2", "landsat8", "landsat9"],
    },
    "SAR 洪水与湿度": {
        "recommended": {"data_source": "sentinel1", "index_type": ""},
        "description": "基于 Sentinel-1 雷达数据的洪水与湿度变化。",
        "available_indices": [],
        "available_sources": ["sentinel1"],
    },
}


def render_wizard() -> bool:
    """Render the three-step setup flow and return True when complete."""
    ps = ProjectState.get()
    wizard_step = int(st.session_state.get("wizard_step", 0))

    page_header(
        "新建分析",
        "配置向导",
        ["选择分析区域", "选择分析目标", "确认执行参数"][min(wizard_step, 2)],
        status=f"第 {min(wizard_step + 1, 3)} / 3 步",
        status_tone="info",
    )
    workflow_bar(
        ["分析区域", "分析目标", "参数确认"],
        active=min(wizard_step, 2),
        completed=range(min(wizard_step, 3)),
    )

    if wizard_step == 0:
        return _wizard_step_1_region(ps)
    if wizard_step == 1:
        return _wizard_step_2_target(ps)
    if wizard_step == 2:
        return _wizard_step_3_confirm(ps)
    return True


def _wizard_step_1_region(ps) -> bool:
    base_dir = Path(__file__).parent.parent.parent
    existing = {path.name: str(path) for path in base_dir.glob("*.geojson")}

    section_header("选择分析区域", "选择本地边界、上传文件或检索行政区。", "步骤 01")
    local_col, search_col = st.columns([1, 1], gap="large")

    with local_col:
        st.markdown("**本地边界**")
        if existing:
            names = list(existing)
            current_name = Path(ps.boundary).name if ps.boundary else None
            default_index = names.index(current_name) if current_name in names else 0
            selected = st.selectbox(
                "边界文件", names, index=default_index,
                label_visibility="collapsed",
            )
            if st.button(
                "使用所选边界", type="primary", icon=":material/check:",
                use_container_width=True,
            ):
                chosen = existing[selected]
                ProjectState.set_aoi(detect_region(chosen), chosen)
                st.session_state.wizard_step = 1
                st.rerun()
        else:
            status_banner("warning", "未发现本地 GeoJSON")

        st.markdown("**上传边界**")
        uploaded = st.file_uploader(
            "上传 GeoJSON、KML 或 GPKG",
            type=["geojson", "json", "kml", "gpkg"],
            label_visibility="collapsed",
        )
        if uploaded:
            ext = Path(uploaded.name).suffix.lower()
            stem = _slug(Path(uploaded.name).stem)
            output_path = base_dir / f"uploaded_{stem}.geojson"
            try:
                if ext in (".geojson", ".json"):
                    output_path.write_bytes(uploaded.getvalue())
                elif ext in (".kml", ".gpkg"):
                    import fiona

                    temp_path = base_dir / f"_tmp_{stem}{ext}"
                    temp_path.write_bytes(uploaded.getvalue())
                    features = []
                    with fiona.open(str(temp_path)) as source:
                        for feature in source:
                            features.append({
                                "type": "Feature",
                                "geometry": feature.geometry,
                                "properties": feature.properties,
                            })
                    output_path.write_text(
                        json.dumps(
                            {"type": "FeatureCollection", "features": features},
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    temp_path.unlink(missing_ok=True)
                else:
                    raise ValueError(f"不支持的文件格式: {ext}")

                region = detect_region(str(output_path))
                ProjectState.set_aoi(region, str(output_path))
                st.session_state.wizard_step = 1
                st.rerun()
            except ImportError:
                st.error("KML/GPKG 转换需要 fiona 依赖。")
            except Exception as exc:
                st.error(f"边界文件处理失败: {exc}")

    with search_col:
        st.markdown("**检索行政区**")
        search = st.text_input(
            "城市或区县名称",
            placeholder="例如：新余市、灵川县",
            label_visibility="collapsed",
        )
        if st.button(
            "检索并获取边界", icon=":material/search:",
            use_container_width=True, disabled=not search.strip(),
        ):
            with st.spinner(f"正在检索 {search.strip()}..."):
                geojson, error = fetch_boundary_osm(search.strip())
            if error:
                st.error(error)
            else:
                path = base_dir / f"{_slug(search)}.geojson"
                path.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
                ProjectState.set_aoi(search.strip(), str(path))
                st.session_state.wizard_step = 1
                st.rerun()

        if ps.is_ready():
            status_banner("success", "已选择分析区域", ps.region)

    if ps.is_ready():
        try:
            import folium
            from streamlit_folium import st_folium

            geojson, geometry = load_gj(ps.boundary)
            region_map = folium.Map(
                location=[geometry.centroid.y, geometry.centroid.x],
                zoom_start=9,
                control_scale=True,
                tiles="CartoDB positron",
            )
            folium.GeoJson(
                geojson,
                style_function=lambda _feature: {
                    "fillColor": "#0F766E", "color": "#0A5D57",
                    "weight": 2, "fillOpacity": 0.12,
                },
            ).add_to(region_map)
            minx, miny, maxx, maxy = geometry.bounds
            region_map.fit_bounds([[miny, minx], [maxy, maxx]], padding=(20, 20))
            st_folium(region_map, width=None, height=300, key="wizard_region_map", returned_objects=[])
        except Exception:
            pass

        _, next_col = st.columns([2, 1])
        with next_col:
            if st.button(
                "下一步", type="primary", icon=":material/arrow_forward:",
                use_container_width=True,
            ):
                st.session_state.wizard_step = 1
                st.rerun()
    return False


def _wizard_step_2_target(ps) -> bool:
    section_header("选择分析目标", "目标将限定可用数据源与遥感指数。", "步骤 02")
    option_col, config_col = st.columns([1, 1.35], gap="large")

    options = list(PROBLEM_TYPES)
    current_index = min(int(st.session_state.get("wizard_prob_idx", 0)), len(options) - 1)
    with option_col:
        selected_problem = st.radio(
            "分析目标",
            options,
            index=current_index,
            key="wizard_problem_selector",
        )
        st.session_state.wizard_prob_idx = options.index(selected_problem)

    problem = PROBLEM_TYPES[selected_problem]
    recommended = problem["recommended"]
    with config_col:
        status_banner(
            "info",
            selected_problem,
            problem["description"],
        )
        st.caption("推荐配置")
        rec_cols = st.columns(2)
        rec_cols[0].metric(
            "数据源",
            DATA_SOURCE_LABELS.get(recommended["data_source"], recommended["data_source"]).split(" (")[0],
        )
        rec_cols[1].metric("指数", recommended["index_type"] or "自动")

        source_options = problem["available_sources"]
        source_index = source_options.index(ps.data_source) if ps.data_source in source_options else 0
        chosen_source = st.selectbox(
            "数据源",
            source_options,
            index=source_index,
            format_func=lambda value: DATA_SOURCE_LABELS.get(value, value),
            key="wizard_ds_selector",
        )

        allowed_for_source = CAPABILITY_MATRIX.get(
            chosen_source, problem["available_indices"]
        )
        index_options = [
            item for item in problem["available_indices"]
            if item == "ALL" or item in allowed_for_source
        ]
        if chosen_source == "sentinel1" and not index_options:
            index_options = ["--"]
        if not index_options:
            index_options = allowed_for_source or ["--"]

        if index_options[0] != "--":
            current_index_type = (
                ps.index_type if ps.index_type in index_options
                else recommended["index_type"] if recommended["index_type"] in index_options
                else index_options[0]
            )
            chosen_index = st.selectbox(
                "遥感指数",
                index_options,
                index=index_options.index(current_index_type),
                format_func=lambda value: (
                    f"{value} · {INDEX_HELP.get(value, '综合特征')}"
                    if value != "ALL" else "全部波段与指数"
                ),
                key="wizard_idx_selector",
            )
        else:
            chosen_index = "--"
            status_banner("warning", "当前数据源不提供光谱指数")

        ps.data_source = chosen_source
        if chosen_index != "--":
            ps.index_type = chosen_index
        ProjectState.commit()

    problem_to_page = {
        "植被退化与恢复": "pages/01_变化检测.py",
        "水体变化": "pages/01_变化检测.py",
        "火烧迹地": "pages/01_变化检测.py",
        "建设用地扩张": "pages/01_变化检测.py",
        "地物分类": "pages/02_地物分类.py",
        "长期趋势分析": "pages/03_时间序列.py",
        "SAR 洪水与湿度": "pages/01_变化检测.py",
    }

    back_col, _, next_col = st.columns([1, 2, 1])
    with back_col:
        if st.button(
            "上一步", icon=":material/arrow_back:", use_container_width=True,
        ):
            st.session_state.wizard_step = 0
            st.rerun()
    with next_col:
        if st.button(
            "下一步", type="primary", icon=":material/arrow_forward:",
            use_container_width=True,
        ):
            st.session_state.wizard_target_page = problem_to_page[selected_problem]
            st.session_state.wizard_step = 2
            st.rerun()
    return False


def _wizard_step_3_confirm(ps) -> bool:
    section_header("确认执行参数", "检查区域、数据源与时间范围。", "步骤 03")
    summary_col, time_col = st.columns([1, 1], gap="large")

    with summary_col:
        st.markdown("**分析配置**")
        area_text = "--"
        center_text = "--"
        try:
            _, geometry = load_gj(ps.boundary)
            area_text = f"{geometry.area * (111.32 ** 2):,.1f} km²"
            center_text = f"{geometry.centroid.y:.3f}, {geometry.centroid.x:.3f}"
        except Exception:
            pass
        st.dataframe(
            {
                "项目": ["地区", "边界", "中心点", "面积", "数据源", "指数"],
                "配置": [
                    ps.region,
                    Path(ps.boundary).name if ps.boundary else "--",
                    center_text,
                    area_text,
                    DATA_SOURCE_LABELS.get(ps.data_source, ps.data_source),
                    ps.index_type,
                ],
            },
            use_container_width=True,
            hide_index=True,
        )

    with time_col:
        st.markdown("**时间范围**")
        year_cols = st.columns(2)
        with year_cols[0]:
            year1 = st.number_input("前时相年份", 2015, 2027, int(ps.year1))
        with year_cols[1]:
            year2 = st.number_input("后时相年份", 2015, 2027, int(ps.year2))
        month_start, month_end = st.select_slider(
            "月份范围",
            options=list(range(1, 13)),
            value=(int(ps.month_start), int(ps.month_end)),
            format_func=lambda value: f"{value} 月",
        )
        ps.year1 = int(year1)
        ps.year2 = int(year2)
        ps.month_start = int(month_start)
        ps.month_end = int(month_end)
        ProjectState.commit()

    valid_period = year2 > year1 and month_end >= month_start
    if valid_period:
        status_banner(
            "success",
            "配置完整",
            f"{year1} 至 {year2} 年 · {month_start} 至 {month_end} 月 · {ps.index_type}",
        )
    else:
        status_banner("danger", "时间范围无效", "后时相年份必须晚于前时相年份。")

    back_col, _, start_col = st.columns([1, 2, 1])
    with back_col:
        if st.button(
            "上一步", icon=":material/arrow_back:", use_container_width=True,
        ):
            st.session_state.wizard_step = 1
            st.rerun()
    with start_col:
        if st.button(
            "进入工作台",
            type="primary",
            icon=":material/arrow_forward:",
            use_container_width=True,
            disabled=not valid_period,
        ):
            import config as cfg

            cfg.REGION_NAME = ps.region
            cfg.YEAR1 = ps.year1
            cfg.YEAR2 = ps.year2
            cfg.MONTH_START = ps.month_start
            cfg.MONTH_END = ps.month_end
            cfg.DATA_SOURCE = ps.data_source
            cfg.INDEX_TYPE = ps.index_type
            cfg.BOUNDARY_FILE = ps.boundary

            st.session_state.wizard_complete = True
            st.session_state._cd_wizard_active = True
            target = st.session_state.get("wizard_target_page", "")
            if target:
                st.switch_page(target)
            else:
                return True
    return False
