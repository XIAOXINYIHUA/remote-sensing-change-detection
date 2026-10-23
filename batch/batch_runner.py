"""
遥感解译助手 v3 — 批量运行器 (增强版)

支持:
  - 每个任务持有独立配置（不再污染全局 config）
  - 失败重试和断点续跑
  - 任务结果清单
  - 资源控制

Usage:
  python batch/batch_runner.py                          # 运行所有任务
  python batch/batch_runner.py --task task_id [--retry] # 运行/重试指定任务
  python batch/batch_runner.py --config my.json         # 使用自定义配置
"""

import os
import sys
import json
import time
import argparse
import traceback
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config
from engine.utils import log


def _task_overrides(task, boundary_file):
    """把任务字段映射为 config 属性覆盖值（未指定的沿用当前 config）。"""
    return {
        "REGION_NAME": task.get("region", config.REGION_NAME),
        "BOUNDARY_FILE": boundary_file,
        "YEAR1": task.get("year1", config.YEAR1),
        "YEAR2": task.get("year2", config.YEAR2),
        "MONTH_START": task.get("month_start", config.MONTH_START),
        "MONTH_END": task.get("month_end", config.MONTH_END),
        "DETECTION_MODE": task.get("detection_mode", config.DETECTION_MODE),
        "INDEX_TYPE": task.get("index_type", config.INDEX_TYPE),
        "DATA_SOURCE": task.get("data_source", config.DATA_SOURCE),
        "CHANGE_THRESHOLD": task.get("threshold", config.CHANGE_THRESHOLD),
        "SEVERE_THRESHOLD": task.get("severe_threshold", config.SEVERE_THRESHOLD),
        "NUM_SAMPLE_POINTS": task.get("sample_points", config.NUM_SAMPLE_POINTS),
        "OUTPUT_FORMATS": task.get("formats", config.OUTPUT_FORMATS),
        "GENERATE_REPORT": task.get("report", config.GENERATE_REPORT),
    }


@contextmanager
def _config_overrides(overrides):
    """临时覆盖全局 config 属性，退出时（含异常）还原，任务之间互不污染。

    模块对象不能 copy.copy（会抛 TypeError），所以只保存/还原被覆盖的属性。
    """
    saved = {k: getattr(config, k) for k in overrides if hasattr(config, k)}
    try:
        for k, v in overrides.items():
            setattr(config, k, v)
        yield
    finally:
        for k in overrides:
            if k in saved:
                setattr(config, k, saved[k])
            else:
                delattr(config, k)


class BatchRunner:
    """批量任务运行器 — 每个任务独立配置。"""

    def __init__(self, batch_file="batch/batch_config.json"):
        self.batch_file = os.path.join(BASE_DIR, batch_file)
        self.configs = self._load()

    def _load(self):
        if not os.path.exists(self.batch_file):
            default = {
                "tasks": [{
                    "id": "lingchuan_2024_2025",
                    "region": "灵川县",
                    "boundary": "lingchuan_boundary.geojson",
                    "year1": 2024,
                    "year2": 2025,
                    "month_start": 6, "month_end": 10,
                    "data_source": "sentinel2",
                    "detection_mode": "multi",
                    "index_type": "NDVI",
                    "threshold": 0.15,
                    "formats": "geojson,csv",
                    "report": True,
                    "retry_on_fail": True,
                    "max_retries": 2,
                }],
                "output_base": "output",
                "parallel": False,
                "max_workers": 2,
            }
            os.makedirs(os.path.dirname(self.batch_file) or ".", exist_ok=True)
            with open(self.batch_file, "w", encoding="utf-8") as f:
                json.dump(default, f, ensure_ascii=False, indent=2)
            log(f"  Created default batch config: {self.batch_file}")
            return default
        with open(self.batch_file, encoding="utf-8") as f:
            return json.load(f)

    def _save_result(self, task_id, result):
        """将单个任务结果写入独立日志文件。"""
        result_dir = os.path.join(
            BASE_DIR, self.configs.get("output_base", "output"),
            task_id)
        os.makedirs(result_dir, exist_ok=True)
        path = os.path.join(result_dir, ".batch_result.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2,
                          default=str)
        except Exception:
            pass

    def _load_result(self, task_id):
        """检查任务是否已完成（断点续跑支持）。"""
        result_dir = os.path.join(
            BASE_DIR, self.configs.get("output_base", "output"),
            task_id)
        path = os.path.join(result_dir, ".batch_result.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    r = json.load(f)
                if r.get("status") == "success":
                    return r
            except Exception:
                pass
        return None

    def _run_single_task(self, task, task_idx, total):
        """运行单个任务（含失败重试）。"""
        task_id = task.get("id", f"task_{task_idx}")
        boundary_file = task.get("boundary", "")
        if not os.path.isabs(boundary_file):
            boundary_file = os.path.join(BASE_DIR, boundary_file)

        # 检查断点
        existing = self._load_result(task_id)
        if existing:
            log(f"  [{task_idx}/{total}] {task_id} — 已完成，跳过")
            return existing

        result = {
            "task_id": task_id,
            "status": "failed",
            "elapsed": 0,
            "error": "",
        }

        max_retries = task.get("max_retries", 2) if task.get("retry_on_fail", True) else 0

        for attempt in range(1 + max_retries):
            t0 = time.time()
            try:
                log(f"\n[{task_idx}/{total}] {task_id} (attempt {attempt + 1})")

                overrides = _task_overrides(task, boundary_file)
                out_dir = os.path.join(
                    BASE_DIR, self.configs.get("output_base", "output"), task_id)
                data_dir = os.path.join(
                    BASE_DIR, "data", overrides["REGION_NAME"] or task_id)

                from engine.core import run_pipeline_v3
                # run_pipeline_v3 仍读取全局 config：任务期间临时覆盖，结束后还原
                with _config_overrides(overrides):
                    r = run_pipeline_v3(
                        data_source_name=overrides["DATA_SOURCE"],
                        data_dir=data_dir,
                        out_dir=out_dir,
                        index_type=overrides["INDEX_TYPE"],
                        detection_mode=overrides["DETECTION_MODE"],
                        threshold=task.get("threshold"),
                        severe_threshold=task.get("severe_threshold"),
                    )

                elapsed = time.time() - t0
                r["task_id"] = task_id
                r["elapsed"] = round(elapsed, 1)
                r["status"] = "success"

                self._save_result(task_id, r)
                log(f"  ✅ {task_id} — {elapsed:.0f}s")
                return r

            except Exception as e:
                elapsed = time.time() - t0
                log(f"  ❌ {task_id} attempt {attempt + 1} failed: {e}")
                traceback.print_exc()

                if attempt < max_retries:
                    wait = 2 ** attempt
                    log(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    result["error"] = str(e)
                    result["elapsed"] = round(elapsed, 1)

        self._save_result(task_id, result)
        return result

    def run_all(self):
        tasks = self.configs.get("tasks", [])
        if not tasks:
            log("  No tasks defined")
            return []

        total = len(tasks)
        log(f"\n{'='*60}")
        log(f"  Batch Runner — {total} task(s)")
        log(f"{'='*60}")

        results = []
        for i, task in enumerate(tasks):
            r = self._run_single_task(task, i + 1, total)
            results.append(r)

        self._summary(results)
        return results

    def run_single(self, task_id):
        tasks = self.configs.get("tasks", [])
        for i, task in enumerate(tasks):
            if task.get("id") == task_id:
                return self._run_single_task(task, i + 1, len(tasks))
        log(f"  Task '{task_id}' not found")
        return None

    def _summary(self, results):
        log(f"\n{'='*60}")
        log(f"  Batch Summary")
        log(f"{'='*60}")
        success = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]
        total_changed = 0

        for r in success:
            s = r.get("stats", {})
            if "changed_total" in s:
                area = s["changed_total"]["area_ha"]
            else:
                area = s.get("changed_area_ha", 0)
            total_changed += area
            log(f"  ✅ {r['task_id']:30s}  "
                f"变化 {area:>10,.0f} ha  {r.get('elapsed', 0):>6.0f}s")

        for r in failed:
            log(f"  ❌ {r['task_id']:30s}  {r.get('error', 'unknown')}")

        log(f"  {'─'*56}")
        log(f"  {'Total':30s}  {total_changed:>10,.0f} ha")
        log(f"  {'Success':30s}  {len(success)}/{len(results)}")
        log(f"{'='*60}")


def batch_main():
    parser = argparse.ArgumentParser(description="遥感解译批量处理")
    parser.add_argument("--config", default="batch/batch_config.json",
                        help="批量配置文件路径")
    parser.add_argument("--task", help="只运行指定任务 ID")
    parser.add_argument("--retry", action="store_true",
                        help="重试失败任务")
    args = parser.parse_args()

    runner = BatchRunner(args.config)
    if args.task:
        runner.run_single(args.task)
    else:
        results = runner.run_all()
        if args.retry:
            failed = [r for r in results if r.get("status") == "failed"]
            if failed:
                log(f"\n重试 {len(failed)} 个失败任务...")
                for r in failed:
                    runner.run_single(r["task_id"])


if __name__ == "__main__":
    batch_main()
