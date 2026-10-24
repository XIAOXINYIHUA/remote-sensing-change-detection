"""
后台任务入口（在工作进程中执行）：把任务参数（JSON）还原为服务调用

每个函数签名 fn(params: dict, progress_callback) -> 结果；结果以 pickle 保存在任务目录，
页面取回后写入会话状态。边界一律从 GeoJSON 文件读取（engine.utils.get_boundary，
与页面的 frontend.utils.load_gj 合并要素的规则相同，影像缓存键一致）。
"""

from engine.params import RunParams
from engine.utils import get_boundary
from services import BASE_DIR, batch, change_detection, classification, timeseries


def cd_collect(params, progress_callback):
    """params: run_params（RunParams.to_dict()）、boundary_file、data_dir"""
    return change_detection.collect(
        RunParams.from_dict(params["run_params"]), get_boundary(params["boundary_file"]),
        params["data_dir"], progress_callback=progress_callback)


def cd_run(params, progress_callback):
    """params: run_params、boundary_file、data_dir、out_dir、slug，可选 source_label、pdf"""
    return change_detection.run(
        RunParams.from_dict(params["run_params"]), get_boundary(params["boundary_file"]),
        params["data_dir"], params["out_dir"], slug=params["slug"],
        source_label=params.get("source_label"), pdf=params.get("pdf", True),
        progress_callback=progress_callback)


def cl_download(params, progress_callback):
    """params: data_source、boundary_file、year、month_start、month_end、max_cloud_pct、data_dir"""
    return classification.download(
        params["data_source"], get_boundary(params["boundary_file"]), params["year"],
        params["month_start"], params["month_end"], params["max_cloud_pct"], params["data_dir"],
        progress_callback=progress_callback)


def ts_run(params, progress_callback):
    """params: boundary_file、years、index_type、data_source、month_start、month_end、
    max_cloud_pct、data_dir

    页面只用年份、趋势图层与摘要：逐年指数图（时相数 × 行 × 列）与 p 值不随结果保存。
    """
    res = timeseries.run(
        get_boundary(params["boundary_file"]), params["years"], params["index_type"],
        params["data_source"], params["month_start"], params["month_end"],
        params["max_cloud_pct"], params["data_dir"], progress_callback=progress_callback)
    ts = res["ts_data"]
    return {
        "ts_data": {key: ts[key] for key in ("years", "meta", "valid_mask", "preprocessing")},
        "trend": {key: value for key, value in res["trend"].items() if key != "p_value"},
        "summary": res["summary"],
    }


def batch_task(params, progress_callback):
    """params: task（batch_config.json 中的一项），可选 base_dir"""
    return batch.run_task(params["task"], base_dir=params.get("base_dir") or BASE_DIR,
                          progress_callback=progress_callback)
