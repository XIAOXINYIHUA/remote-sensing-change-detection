"""
前端组件 — 增强版交互地图

folium 地图封装，支持:
  - 浅色底图 + 多个底图选项
  - 全屏 / 绘制ROI / 测量 / 鹰眼
  - 变化检测结果叠加（自适应降采样）
  - 图例面板
"""

import folium
from folium.plugins import Fullscreen, Draw, MeasureControl, MiniMap


def create_interactive_map(center_lat, center_lon, zoom_start=11):
    """创建增强版交互地图。

    Args:
        center_lat: 中心纬度。
        center_lon: 中心经度。
        zoom_start: 初始缩放级别。

    Returns:
        folium.Map: 配置好的地图对象。
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        control_scale=True,
        tiles="CartoDB positron",
    )

    # 插件
    Fullscreen().add_to(m)
    Draw(export=True).add_to(m)
    MeasureControl(position="bottomleft").add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)

    # 额外底图
    folium.TileLayer(
        "OpenStreetMap", name="街道图", show=False
    ).add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="卫星影像", attr="Esri", show=False
    ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


def add_boundary_layer(m, geojson_data, name="边界",
                       fill_color="#0F766E", fill_opacity=0.10):
    """向地图添加边界 GeoJSON 图层。

    Args:
        m: folium.Map 对象。
        geojson_data: GeoJSON dict。
        name: 图层名称。
        fill_color: 填充色。
        fill_opacity: 填充透明度。
    """
    folium.GeoJson(
        geojson_data, name=name,
        style_function=lambda x, fc=fill_color, fo=fill_opacity: {
            "fillColor": fc, "color": fc,
            "weight": 2, "fillOpacity": fo
        }
    ).add_to(m)


def add_change_result_layer(m, result, meta, index_type="NDVI", opacity=0.6):
    """向地图添加变化检测结果图层（自适应降采样）。

    Args:
        m: folium.Map 对象。
        result: 变化检测结果数组。
        meta: rasterio 元数据。
        index_type: 指数类型（用于颜色映射）。
        opacity: 图层透明度 (0-1)。
    """
    from rasterio.features import shapes as rshapes
    from rasterio import Affine
    import cv2
    import numpy as np
    import random

    from engine.indices import INDEX_COLORS, INDEX_CLASS_NAMES

    colors = INDEX_COLORS.get(index_type, INDEX_COLORS["NDVI"])
    cls_names = INDEX_CLASS_NAMES.get(index_type, INDEX_CLASS_NAMES["NDVI"])

    h, w = result.shape

    # 自适应降采样
    if w * h > 4_000_000:
        scale = 8
    elif w * h > 1_000_000:
        scale = 4
    elif w > 800 or h > 800:
        scale = 2
    else:
        scale = 1

    if scale > 1:
        small = cv2.resize(
            result.astype(np.int8), (w // scale, h // scale),
            interpolation=cv2.INTER_NEAREST)
        tf = meta["transform"]
        small_meta = meta.copy()
        small_meta.update(
            transform=Affine(tf.a * scale, tf.b, tf.c,
                             tf.d, tf.e * scale, tf.f),
            width=w // scale, height=h // scale)
        result_small, meta_small = small, small_meta
    else:
        result_small, meta_small = result, meta

    # 仅渲染变化类别
    change_classes = [-3, -2, 2, 3]
    for cv_val in change_classes:
        mask = (result_small == cv_val)
        if mask.sum() == 0:
            continue

        raw = list(rshapes(
            result_small.astype(np.int8), mask=mask,
            transform=meta_small["transform"], connectivity=8))

        max_features = 30000
        if len(raw) > max_features:
            random.seed(42)
            raw = random.sample(raw, max_features)

        cn = cls_names.get(cv_val, str(cv_val))
        co = colors.get(cv_val, "#888888")
        feats = [{"type": "Feature", "geometry": g,
                  "properties": {"t": cn}}
                 for g, v in raw if v == cv_val]
        if not feats:
            continue

        fg = folium.FeatureGroup(name=cn, show=True)
        folium.GeoJson(
            {"type": "FeatureCollection", "features": feats},
            name=cn,
            style_function=lambda x, c=co, o=opacity: {
                "fillColor": c, "color": c,
                "weight": 0, "fillOpacity": o
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["t"], aliases=["类型:"])
        ).add_to(fg)
        fg.add_to(m)


def legend_html(index_type="NDVI"):
    """生成图例 HTML。

    Args:
        index_type: 指数类型。

    Returns:
        str: HTML 字符串。
    """
    from engine.indices import INDEX_COLORS, INDEX_CLASS_NAMES
    colors = INDEX_COLORS.get(index_type, INDEX_COLORS["NDVI"])
    cls_names = INDEX_CLASS_NAMES.get(index_type, INDEX_CLASS_NAMES["NDVI"])

    items = []
    for cv in [-3, -2, 2, 3]:
        co = colors.get(cv, "#888")
        cn = cls_names.get(cv, str(cv))
        items.append(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{co};border-radius:2px;"></span> {cn}'
        )

    return f"""<div style="position:fixed;bottom:20px;right:20px;z-index:9999;
        background:#fff;padding:10px 12px;border-radius:5px;
        border:1px solid #d9e1da;box-shadow:0 4px 16px rgba(24,34,29,.12);
        color:#18221d;font-family:Segoe UI,Microsoft YaHei,sans-serif;font-size:13px;">
        <b style="display:block;margin-bottom:4px">{index_type} 分级</b>
        {'<br>'.join(items)}
        </div>"""
