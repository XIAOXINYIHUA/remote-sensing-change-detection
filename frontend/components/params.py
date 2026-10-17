"""
前端组件 — 参数面板

可复用的 Streamlit 参数输入组件。
"""

import streamlit as st


def render_region_params(boundary_files, current_boundary):
    """地区参数面板。

    Returns:
        str: 选中的边界文件路径。
    """
    names = list(boundary_files.keys())
    if not names:
        st.info("未发现边界文件")
        return None
    current_name = next(
        (name for name, path in boundary_files.items() if path == current_boundary),
        names[0] if names else None,
    )
    sel_name = st.selectbox(
        "已有边界文件", names,
        index=names.index(current_name) if current_name in names else 0)
    return boundary_files.get(sel_name)


def render_time_params():
    """时相参数面板。

    Returns:
        tuple: (year1, year2, month_start, month_end)
    """
    import config
    y1 = st.number_input("前时相年份", 2015, 2027, config.YEAR1, 1)
    y2 = st.number_input("后时相年份", 2015, 2027, config.YEAR2, 1)
    ms, me = st.select_slider(
        "月份范围", options=list(range(1, 13)),
        value=(config.MONTH_START, config.MONTH_END),
        format_func=lambda value: f"{value} 月",
    )
    return y1, y2, ms, me


def render_detection_params():
    """检测参数面板：指数选择、检测模式、阈值。

    Returns:
        dict: {
            "index_type", "detection_mode", "threshold",
            "severe_threshold", "ml_model", "data_source"
        }
    """
    import config

    index_type = st.selectbox(
        "遥感指数",
        ["NDVI", "NDWI", "MNDWI", "NBR", "EVI", "SAVI", "NDBI"],
        index=0,
        help="NDVI/EVI/SAVI: 植被 | NDWI/MNDWI: 水体 | NBR: 火烧 | NDBI: 建设用地"
    )

    data_source = st.selectbox(
        "数据源",
        ["sentinel2", "landsat8", "landsat9", "sentinel1"],
        index=0,
        help="sentinel2: Sentinel-2 | landsat8/9: Landsat | sentinel1: SAR雷达"
    )

    mode_options = {
        "multi": "多级分级（推荐）",
        "binary": "二值检测",
        "ml": "机器学习"
    }
    mode = st.selectbox(
        "检测模式",
        list(mode_options.keys()),
        format_func=lambda x: mode_options[x])

    threshold = st.slider(
        "变化阈值", 0.05, 0.50,
        config.CHANGE_THRESHOLD, 0.01,
        help="指数差值超过此值视为变化")

    severe_threshold = st.slider(
        "强烈阈值", 0.10, 0.50,
        config.SEVERE_THRESHOLD, 0.01,
        help="指数差值超过此值视为剧烈变化")

    ml_model = None
    if mode == "ml":
        ml_model = st.selectbox(
            "ML 模型",
            ["random_forest", "xgboost", "cnn"],
            help="random_forest: 随机森林 | cnn: 卷积网络(需PyTorch)"
        )

    return {
        "index_type": index_type,
        "data_source": data_source,
        "detection_mode": mode,
        "threshold": threshold,
        "severe_threshold": severe_threshold,
        "ml_model": ml_model,
    }


def render_sample_params():
    """采样参数面板。

    Returns:
        dict: 采样参数。
    """
    strat = st.selectbox(
        "策略", ["centroid", "random", "grid"],
        format_func=lambda x: {
            "centroid": "质心", "random": "随机", "grid": "网格"
        }[x])
    n_pts = st.number_input("数量上限（0 为不限）", 0, 100000, 0)
    min_dist = st.number_input("最小间距（米，0 为不限）", 0, 10000, 0)
    return {
        "strategy": strat,
        "num_points": n_pts if n_pts > 0 else None,
        "min_distance": min_dist if min_dist > 0 else None,
    }


def render_output_params():
    """输出参数面板。

    Returns:
        dict: 输出参数。
    """
    formats = st.multiselect(
        "导出格式",
        ["geojson", "csv", "shp"],
        default=["geojson", "csv"])
    gen_report = st.checkbox("生成 PNG 报告", True)
    return {
        "formats": ",".join(formats),
        "generate_report": gen_report,
    }
