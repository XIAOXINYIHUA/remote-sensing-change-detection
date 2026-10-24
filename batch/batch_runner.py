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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from engine.utils import log


def task_params(task, boundary_file=None):
    """把任务字段映射为 RunParams（未指定的沿用 config 默认值；参数错误抛 ValueError）。

    每个任务持有独立的 RunParams，不修改全局 config，任务之间互不污染。
    """
    from engine.params import RunParams
    return RunParams.from_config(
        region=task.get("region"),
        boundary_file=boundary_file or task.get("boundary"),
        year1=task.get("year1"), year2=task.get("year2"),
        month_start=task.get("month_start"), month_end=task.get("month_end"),
        data_source=task.get("data_source"),
        index_type=task.get("index_type"),
        detection_mode=task.get("detection_mode"),
        max_cloud_pct=task.get("max_cloud_pct"),
        threshold=task.get("threshold"),
        severe_threshold=task.get("severe_threshold"),
        adaptive_threshold=task.get("adaptive_threshold"),
        min_area_m2=task.get("min_area_m2"),
        num_sample_points=task.get("sample_points"),
        min_point_distance_m=task.get("min_point_distance_m"),
        output_formats=task.get("formats"),
        generate_report=task.get("report"),
    )


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

                params = task_params(task, boundary_file)
                out_dir = os.path.join(
                    BASE_DIR, self.configs.get("output_base", "output"), task_id)
                data_dir = os.path.join(BASE_DIR, "data", params.region or task_id)

                from engine.core import run_pipeline_v3
                r = run_pipeline_v3(params=params, data_dir=data_dir, out_dir=out_dir)
                if r.get("error"):
                    raise RuntimeError(r["error"])

                elapsed = time.time() - t0
                r["task_id"] = task_id
                r["elapsed"] = round(elapsed, 1)
                r["status"] = "success"

                self._save_result(task_id, r)
                log(f"  ✅ {task_id} — {elapsed:.0f}s")
                return r

            except ValueError as e:
                # 参数错误重试也不会成功
                log(f"  ❌ {task_id} 参数无效: {e}")
                result["error"] = str(e)
                break
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
        log("  Batch Summary")
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
