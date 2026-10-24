"""
批量任务服务：运行批量配置（batch_config.json）中的单个变化检测任务

命令行的断点续跑与失败重试见 batch/batch_runner.py。
"""

import os
import time

from services import BASE_DIR, ServiceError


def task_id(task):
    """任务标识：配置中的 id，缺省为「区域_年份1_年份2」。"""
    return task.get("id") or f"{task['region']}_{task['year1']}_{task['year2']}"


def run_task(task, base_dir=BASE_DIR, progress_callback=None):
    """运行一个任务，输出到 output/<任务标识>/。

    Returns:
        run_pipeline_v3 的结果，另加 task_id、elapsed（秒）
    Raises:
        ValueError: 任务参数无效
        ServiceError: 流水线返回错误
    """
    from batch.batch_runner import task_params
    from engine.core import run_pipeline_v3

    tid = task_id(task)
    boundary_file = task["boundary"]
    if not os.path.isabs(boundary_file):
        boundary_file = os.path.join(base_dir, boundary_file)
    params = task_params(task, boundary_file)

    t0 = time.time()
    res = run_pipeline_v3(
        params=params,
        data_dir=os.path.join(base_dir, "data", tid),
        out_dir=os.path.join(base_dir, "output", tid),
        progress_callback=progress_callback,
    )
    if res.get("error"):
        raise ServiceError(res["error"])
    res["task_id"] = tid
    res["elapsed"] = round(time.time() - t0, 1)
    return res
