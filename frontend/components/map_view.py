"""
前端组件 — 增强版交互地图

folium 地图封装，支持:
  - 浅色底图 + 多个底图选项
  - 全屏 / 绘制ROI / 测量 / 鹰眼
  - 变化检测结果叠加（逐类图片图层，全部像元、可缓存）
  - 图例面板
"""

import folium


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


CHANGE_CLASSES = (-3, -2, 2, 3)
MAX_OVERLAY_SIDE = 2048  # 叠加图最长边（像素），超过则按块取「任一」缩小


def change_overlays(result, meta, index_type="NDVI", max_side=MAX_OVERLAY_SIDE):
    """把分级结果渲染为每个变化类别一张 Web 墨卡托 PNG 叠加图。

    全部像元参与渲染、不抽样：
      - 默认按 2 倍超采样 + 最近邻重投影，形状与原栅格一致，
        且每个像元都至少落到一个输出像元上；
      - 图太大需要缩小时改用 max 重采样，块内任一像元属于该类即着色，
        小斑块不会消失（代价是边缘最多外扩一个输出像元）。
    Leaflet 的 ImageOverlay 在墨卡托平面内线性拉伸图片，因此先重投影到
    EPSG:3857，叠加位置才与底图严格一致。

    Args:
        result: 分级结果数组（投影坐标系，如 UTM）。
        meta: rasterio 元数据（含 crs / transform）。
        index_type: 指数类型（决定类别名与配色）。
        max_side: 叠加图最长边像素数上限。

    Returns:
        list[dict]: 每个出现的类别一项，
            value / name / color / image (PNG data URL) /
            bounds [[south, west], [north, east]] / pixels（原分辨率像元数）。
    """
    import base64

    import cv2
    import numpy as np
    from pyproj import Transformer
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    from engine.indices import INDEX_CLASS_NAMES, INDEX_COLORS

    colors = INDEX_COLORS.get(index_type, INDEX_COLORS["NDVI"])
    cls_names = INDEX_CLASS_NAMES.get(index_type, INDEX_CLASS_NAMES["NDVI"])

    result = np.asarray(result)
    h, w = result.shape
    src_crs, src_tf = meta["crs"], meta["transform"]
    bounds = array_bounds(h, w, src_tf)
    native_tf, dw, dh = calculate_default_transform(src_crs, "EPSG:3857", w, h, *bounds)
    scale, resampling = 0.5, Resampling.nearest
    if max(dw, dh) * 2 > max_side:
        scale, resampling = max(dw, dh) / (max_side - 1), Resampling.max
    dst_tf, dw, dh = calculate_default_transform(
        src_crs, "EPSG:3857", w, h, *bounds,
        resolution=(native_tf.a * scale, -native_tf.e * scale))

    west, south, east, north = array_bounds(dh, dw, dst_tf)
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon0, lat0 = to_wgs84.transform(west, south)
    lon1, lat1 = to_wgs84.transform(east, north)

    layers = []
    for value in CHANGE_CLASSES:
        mask = result == value
        pixels = int(mask.sum())
        if not pixels:
            continue
        warped = np.zeros((dh, dw), np.uint8)
        reproject(mask.astype(np.uint8), warped,
                  src_transform=src_tf, src_crs=src_crs,
                  dst_transform=dst_tf, dst_crs="EPSG:3857",
                  resampling=resampling)
        color = colors.get(value, "#888888")
        r, g, b = (int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        bgra = np.zeros((dh, dw, 4), np.uint8)
        bgra[warped > 0] = (b, g, r, 255)
        ok, png = cv2.imencode(".png", bgra)
        if not ok:
            continue
        layers.append({
            "value": value,
            "name": cls_names.get(value, str(value)),
            "color": color,
            "image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(),
            "bounds": [[lat0, lon0], [lat1, lon1]],
            "pixels": pixels,
        })
    return layers


def cached_change_overlays(store, key, result, meta, index_type="NDVI"):
    """按 key（如结果文件路径 + 修改时间 + 指数）缓存叠加图。

    拖动透明度、点击地图等重跑只复用已渲染的图片，不再重算。
    store 为 st.session_state 或任意 dict。
    """
    entry = store.get("_change_overlay_cache")
    if entry and entry[0] == key:
        return entry[1]
    layers = change_overlays(result, meta, index_type)
    store["_change_overlay_cache"] = (key, layers)
    return layers


def add_change_result_layer(m, result=None, meta=None, index_type="NDVI",
                            opacity=0.6, overlays=None):
    """向地图添加变化检测结果图层（每类一个可开关的图片图层）。

    Args:
        m: folium.Map 对象。
        result / meta: 分级结果数组与 rasterio 元数据（未传 overlays 时现算）。
        index_type: 指数类型（用于颜色映射）。
        opacity: 图层透明度 (0-1)。
        overlays: change_overlays() / cached_change_overlays() 的结果。

    Returns:
        list[dict]: 实际使用的叠加图（含各类像元数）。
    """
    if overlays is None:
        overlays = change_overlays(result, meta, index_type)
    for layer in overlays:
        folium.raster_layers.ImageOverlay(
            image=layer["image"],
            bounds=layer["bounds"],
            opacity=opacity,
            name=layer["name"],
            interactive=False,
            cross_origin=False,
        ).add_to(m)
    return overlays


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
