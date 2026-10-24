"""
变化检测服务：资料收集 → 运行流水线 → PDF 报告 / 任务记录 → 读取结果

参数一律通过 engine.params.RunParams 传入；边界为 WGS84 几何。
"""

import json
import os
from datetime import datetime

from engine.core import prepare_data, run_pipeline_v3
from engine.utils import geodesic_area_km2, get_needed_bands, log
from services import ServiceError


def collect(params, boundary, data_dir, progress_callback=None):
    """检索 + 下载两期影像并写入快照缓存；之后以相同参数运行分析直接命中缓存。

    Returns:
        资料摘要 dict(region, year1, year2, data_source, index_type,
        tiles_found, bands, size_mb, cache_key)
    Raises:
        ServiceError: 任一期无影像、下载或写入缓存失败
    """
    info = prepare_data(params, boundary, data_dir, progress_callback=progress_callback)
    if "error" in info:
        raise ServiceError(info["error"])
    return {
        "region": params.region, "year1": params.year1, "year2": params.year2,
        "data_source": params.data_source, "index_type": params.index_type,
        "tiles_found": info.get("tiles", 0),
        "bands": get_needed_bands(params.index_type),
        "size_mb": info.get("size_mb", 0),
        "cache_key": info.get("key"),
    }


def run(params, boundary, data_dir, out_dir, *, slug, source_label=None, pdf=True,
        progress_callback=None):
    """运行变化检测流水线，可选生成 PDF 报告。

    Returns:
        run_pipeline_v3 的结果，另加:
          unmasked_years  缺少质量波段、未做云掩膜的年份
          task            任务记录（写入任务历史由调用方负责）
        生成了 PDF 时 files["pdf"] 为其路径。
    Raises:
        ServiceError: 流水线返回错误（无影像、参数组合无效等）
    """
    started_at = datetime.now().isoformat(timespec="seconds")
    os.makedirs(out_dir, exist_ok=True)
    res = run_pipeline_v3(params=params, boundary=boundary, data_dir=data_dir,
                          out_dir=out_dir, progress_callback=progress_callback)
    if "error" in res:
        raise ServiceError(res["error"])

    years = {"y1": params.year1, "y2": params.year2}
    res["unmasked_years"] = [years[k] for k, v in res.get("preprocessing", {}).items()
                             if not v.get("cloud_mask")]
    if pdf:
        pdf_path = _pdf_report(res, params, boundary, out_dir, source_label)
        if pdf_path:
            res["files"]["pdf"] = pdf_path
    res["task"] = task_record(res, params, slug, started_at)
    return res


def changed_area_ha(stats):
    """变化总面积（ha）；兼容二值模式的统计字段。"""
    return (stats.get("changed_total", {}).get("area_ha")
            or stats.get("changed_area_ha", 0))


def task_record(res, params, slug, started_at=""):
    """任务历史中的一条记录。"""
    ah = changed_area_ha(res.get("stats", {}))
    return {
        "id": f"cd_{slug}_{params.year1}_{params.year2}_{params.index_type}",
        "mode": "change_detection",
        "region": params.region,
        "data_source": params.data_source,
        "index_type": params.index_type,
        "year1": params.year1, "year2": params.year2,
        "status": "success",
        "started_at": started_at,
        "duration_s": round(res.get("timing", {}).get("total", 0), 1),
        "stats_summary": f"变化 {ah:,.0f} ha" if ah else "完成",
        "output_files": dict(res.get("files", {})),
    }


def load_candidates(geojson_path):
    """读取采样点 GeoJSON 的要素列表；文件不存在时返回空列表。"""
    if not geojson_path or not os.path.exists(geojson_path):
        return []
    with open(geojson_path, encoding="utf-8") as f:
        return json.load(f).get("features", [])


def load_outputs(res):
    """读取结果 GeoTIFF（地图用）与采样点。

    Returns:
        (arr, meta, candidates)；没有 GeoTIFF 时 arr / meta 为 None。
    """
    import rasterio

    files = res.get("files", {})
    arr = meta = None
    gtif = files.get("geotiff")
    if gtif and os.path.exists(gtif):
        with rasterio.open(gtif) as src:
            arr, meta = src.read(1), src.meta
    return arr, meta, load_candidates(files.get("geojson"))


def _pdf_report(res, params, boundary, out_dir, source_label=None):
    """生成 PDF 报告（需 reportlab）；失败只记日志，不影响分析结果。"""
    try:
        from engine.report_generator import generate_report_pdf

        points = [
            {"lon": f["geometry"]["coordinates"][0],
             "lat": f["geometry"]["coordinates"][1],
             "class_name": f["properties"].get("class_name", "")}
            for f in load_candidates(res["files"].get("geojson"))[:50]
        ]
        pdf_path = generate_report_pdf(
            stats=res.get("stats", {}),
            report_image=res["files"].get("report", ""),
            sample_points=points or None,
            boundary_info={"area_km2": f"{geodesic_area_km2(boundary):,.1f}"},
            metadata={
                "region": params.region,
                "year1": params.year1,
                "year2": params.year2,
                "index_type": params.index_type,
                "data_source": source_label or params.data_source,
                "threshold": params.threshold,
                "detection_mode": params.detection_mode,
            },
            output_dir=out_dir,
        )
        return pdf_path if pdf_path and os.path.exists(pdf_path) else None
    except Exception as e:
        log(f"  PDF auto-generation skipped: {e}")
        return None
