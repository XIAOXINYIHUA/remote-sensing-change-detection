"""tests/test_jobs.py 用的后台任务入口（在工作进程中执行，须为可导入的模块级函数）。"""

import os
import time

from services import ServiceError


def count(params, progress_callback):
    n = params["n"]
    for i in range(n):
        progress_callback(f"第 {i + 1} 步", (i + 1) * 100 / n)
        time.sleep(params.get("sleep", 0))
    return {"n": n, "squares": [i * i for i in range(n)]}


def fail(params, progress_callback):
    if params.get("expected"):
        raise ServiceError("没有可用影像")
    raise KeyError("boom")


def wait(params, progress_callback):
    """运行到被取消或超时；cooperative=False 时从不调用进度回调（只能被终止）。"""
    deadline = time.time() + params.get("timeout", 60)
    while time.time() < deadline:
        if params.get("cooperative", True):
            progress_callback("等待中", 50)
        time.sleep(0.05)
    return "timeout"


def crash(params, progress_callback):
    os._exit(3)


def env(params, progress_callback):
    return {key: os.environ.get(key) for key in params["keys"]}
