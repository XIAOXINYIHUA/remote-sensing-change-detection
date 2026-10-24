"""
本机任务执行器测试（全部离线）

  - 真实工作进程：结果回传、失败与崩溃、协作取消与强制终止、并发上限与排队、环境变量继承
  - 执行器重启：未结束任务记为 interrupted，可重试；记录删除与保留上限
  - 任务入口（线程方式，可用假数据源）：变化检测收集 / 分析、地物分类下载、时间序列、批量任务

用法:
    python -m pytest tests/test_jobs.py -v
"""

import json
import os
import time

import pytest

from services import jobs
from services.jobs import JobRunner

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_AOI = os.path.join(BASE, "examples", "sample_aoi.geojson")


@pytest.fixture
def runner(tmp_path):
    r = JobRunner(tmp_path / "jobs", max_workers=2, mode="process")
    yield r
    r.shutdown()


def _wait(runner, job_id, until=lambda job: not job.active, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = runner.get(job_id)
        if until(job):
            return job
        time.sleep(0.1)
    raise AssertionError(f"等待任务超时: {runner.get(job_id)}")


def _running(job):
    return job.state == "running"


# =====================================================================
#  工作进程
# =====================================================================

def test_job_runs_in_worker_process_and_returns_result(runner):
    jid = runner.submit("job_targets:count", {"n": 5, "sleep": 0.05}, title="计数",
                        meta={"slot": "x"})
    job = runner.get(jid)
    assert job.active and job.title == "计数" and job.meta == {"slot": "x"}

    job = _wait(runner, jid)
    assert job.state == "succeeded" and job.progress == 100 and job.finished_at
    assert runner.result(jid) == {"n": 5, "squares": [0, 1, 4, 9, 16]}
    status = json.loads((runner.dir / jid / "status.json").read_text(encoding="utf-8"))
    assert status["pid"] != os.getpid(), "任务在独立进程中执行"


def test_failures_are_recorded(runner):
    expected = runner.submit("job_targets:fail", {"expected": True})
    unexpected = runner.submit("job_targets:fail", {})
    crashed = runner.submit("job_targets:crash", {})

    job = _wait(runner, expected)
    assert job.state == "failed" and job.error == "没有可用影像"
    assert runner.result(expected) is None
    job = _wait(runner, unexpected)
    assert job.state == "failed" and job.error.startswith("KeyError")
    assert "Traceback" in runner.log_tail(unexpected), "意外异常的堆栈写入任务日志"
    job = _wait(runner, crashed)
    assert job.state == "failed" and "退出码 3" in job.error


def test_cancel_cooperative_and_stuck_jobs(runner, monkeypatch):
    monkeypatch.setattr(jobs, "CANCEL_GRACE_S", 0.5)
    coop = runner.submit("job_targets:wait", {})
    stuck = runner.submit("job_targets:wait", {"cooperative": False})
    for jid in (coop, stuck):
        _wait(runner, jid, until=_running)

    assert runner.cancel(coop) and runner.cancel(stuck)
    assert runner.get(coop).cancel_requested
    assert _wait(runner, coop).state == "cancelled"
    assert _wait(runner, stuck).state == "cancelled", "不调用进度回调的任务在宽限期后被终止"
    assert not runner.cancel(coop), "已结束的任务不能再取消"


def test_queue_respects_max_workers(tmp_path):
    r = JobRunner(tmp_path / "jobs", max_workers=1, mode="process")
    try:
        first = r.submit("job_targets:wait", {})
        second = r.submit("job_targets:count", {"n": 1})
        third = r.submit("job_targets:count", {"n": 1})
        _wait(r, first, until=_running)
        time.sleep(0.5)
        assert r.get(second).state == "queued"
        assert (r.queue_position(second), r.queue_position(third)) == (1, 2)

        assert r.cancel(third) and r.get(third).state == "cancelled"
        r.cancel(first)
        assert _wait(r, second).state == "succeeded"
        assert r.get(third).started_at == "", "排队时取消的任务不会启动"
    finally:
        r.shutdown()


def test_worker_inherits_isolated_environment(runner):
    jid = runner.submit("job_targets:env", {"keys": ["RSAP_CACHE_DIR", "RSAP_JOBS_DIR"]})
    assert _wait(runner, jid).state == "succeeded"
    assert runner.result(jid) == {key: os.environ[key]
                                  for key in ("RSAP_CACHE_DIR", "RSAP_JOBS_DIR")}


# =====================================================================
#  记录管理
# =====================================================================

def test_restart_marks_unfinished_jobs_interrupted(tmp_path):
    first = JobRunner(tmp_path / "jobs", max_workers=1, mode="thread")
    try:
        running = first.submit("job_targets:wait", {}, title="长任务", meta={"slot": "s"})
        queued = first.submit("job_targets:count", {"n": 1})
        _wait(first, running, until=_running)

        restarted = JobRunner(tmp_path / "jobs", mode="thread")
        for jid in (running, queued):
            job = restarted.get(jid)
            assert job.state == "interrupted" and "重试" in job.error

        new = restarted.retry(running)
        job = restarted.get(new)
        assert new != running and (job.title, job.meta, job.params) == ("长任务", {"slot": "s"}, {})
        restarted.cancel(new)
        assert _wait(restarted, new).state == "cancelled"
        restarted.shutdown()
    finally:
        first.shutdown()


def test_list_delete_and_prune(tmp_path):
    r = JobRunner(tmp_path / "jobs", mode="thread")
    try:
        done = [r.submit("job_targets:count", {"n": 1}) for _ in range(3)]
        for jid in done:
            _wait(r, jid)
        active = r.submit("job_targets:wait", {})
        _wait(r, active, until=_running)
        assert [job.id for job in r.list()] == [active, *done[::-1]]

        assert not r.delete(active), "运行中的任务不能删除"
        assert r.delete(done[0]) and r.get(done[0]) is None
        r.prune(keep=1)
        assert [job.id for job in r.list()] == [active, done[2]]
    finally:
        r.shutdown()


def test_submit_validates_input(runner):
    with pytest.raises(ValueError, match="未知任务类型"):
        runner.submit("nope", {})
    with pytest.raises(TypeError):
        runner.submit("job_targets:count", {"n": object()})
    assert list(runner.dir.iterdir()) == [], "无效提交不留下任务目录"
    assert runner.get("../etc") is None and runner.result("../etc") is None


def test_get_runner_is_shared_per_directory(tmp_path):
    try:
        a = jobs.get_runner(tmp_path / "a", max_workers=1, mode="thread")
        assert jobs.get_runner(str(tmp_path / "a"), max_workers=3, mode="thread") is a
        assert a.max_workers == 3
        assert jobs.get_runner(tmp_path / "b", mode="thread") is not a
    finally:
        jobs.shutdown_all()


# =====================================================================
#  任务入口（线程方式，假数据源对其可见）
# =====================================================================

def _aoi_file(world):
    from shapely.geometry import mapping
    path = world.tmp / "aoi.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"name": "测试区"},
                      "geometry": mapping(world.aoi)}],
    }), encoding="utf-8")
    return str(path)


def test_job_boundary_matches_page_boundary(tmp_path):
    """任务从文件读边界（engine），页面用 load_gj（frontend）：两者的缓存键必须一致。"""
    from shapely.geometry import box, mapping

    from engine.cache import compute_aoi_hash
    from engine.utils import get_boundary
    from frontend.utils import load_gj

    multi = tmp_path / "multi.geojson"
    multi.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(box(110.0, 25.0, 110.1, 25.1))},
        {"type": "Feature", "properties": {}, "geometry": mapping(box(110.2, 25.0, 110.3, 25.1))},
    ]}), encoding="utf-8")
    for path in (SAMPLE_AOI, str(multi)):
        assert compute_aoi_hash(get_boundary(path)) == compute_aoi_hash(load_gj(path)[1])


def test_job_tasks_with_fake_source(world, tmp_path):
    from engine.params import RunParams

    r = JobRunner(tmp_path / "jobs", max_workers=2, mode="thread")
    try:
        aoi = _aoi_file(world)
        run_params = RunParams(region="测试区", year1=2021, year2=2022,
                               output_formats=("geojson",), generate_report=False).to_dict()
        base = {"run_params": run_params, "boundary_file": aoi, "data_dir": world.data}

        collect = r.submit("cd.collect", base)
        assert _wait(r, collect).state == "succeeded"
        assert "+" in r.result(collect)["cache_key"]
        run = r.submit("cd.run", {**base, "out_dir": str(tmp_path / "out"), "slug": "t",
                                  "pdf": False})
        assert _wait(r, run).state == "succeeded"
        res = r.result(run)
        assert res["task"]["id"] == "cd_t_2021_2022_NDVI" and len(world.calls.download) == 2

        ts = r.submit("ts.run", {"boundary_file": aoi, "years": [2019, 2020, 2021, 2022],
                                 "index_type": "NDVI", "data_source": "sentinel2",
                                 "month_start": 6, "month_end": 10, "max_cloud_pct": 50,
                                 "data_dir": world.data})
        assert _wait(r, ts).state == "succeeded"
        res = r.result(ts)
        assert res["ts_data"]["years"] == [2019, 2020, 2021, 2022]
        assert "stacked" not in res["ts_data"] and "p_value" not in res["trend"]
        assert res["trend"]["trend_type"].shape == res["ts_data"]["valid_mask"].shape

        cl = r.submit("cl.download", {"data_source": "sentinel2", "boundary_file": aoi,
                                      "year": 2021, "month_start": 6, "month_end": 10,
                                      "max_cloud_pct": 50, "data_dir": world.data})
        assert _wait(r, cl).state == "succeeded" and os.path.isdir(r.result(cl)["dir"])

        task = {"id": "b1", "region": "测试区", "boundary": aoi, "year1": 2021, "year2": 2022,
                "formats": "geojson", "report": False}
        bt = r.submit("batch.task", {"task": task, "base_dir": str(tmp_path)})
        assert _wait(r, bt).state == "succeeded" and r.result(bt)["task_id"] == "b1"

        missing = r.submit("cd.collect", {**base, "run_params": {**run_params, "year2": 2024}})
        job = _wait(r, missing)
        assert job.state == "failed" and "2024" in job.error
    finally:
        r.shutdown()
