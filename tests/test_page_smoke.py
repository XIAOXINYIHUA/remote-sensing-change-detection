"""
页面冒煙测试（全部离线）：从 app.py 进入，逐页渲染，不能有异常，也不能改动项目根目录的状态文件。

界面状态文件通过 RSAP_STATE_DIR 指到临时目录；快照缓存指到临时目录。
另有几条页面 → 后台任务 → 服务层的流程测试：用假数据源（conftest.world）点按钮提交任务，
反复重跑页面直到任务结束、结果载入会话。任务在线程中执行（RSAP_JOB_MODE=thread），
假数据源与 monkeypatch 对它可见；服务函数的数据与输出目录改指到临时目录，不写项目的 data/、output/。
这是后续重构（抽服务层、统一取数）的安全网。

用法:
    python -m pytest tests/test_page_smoke.py -v
"""

import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

SAMPLE_AOI = os.path.join(BASE, "examples", "sample_aoi.geojson")
ROOT_STATE_FILES = (".project_state.json", ".shared_state.json", ".app_state.json",
                    ".classify_state.json", ".ts_state.json", ".batch_state.json")

PAGES = {
    "pages/00_首页.py": "项目总览",
    "pages/06_任务中心.py": "任务中心",
    "pages/07_成果库.py": "成果库",
    "pages/01_变化检测.py": "新建分析",        # 无数据时先显示配置向导
    "pages/02_地物分类.py": "地物分类",
    "pages/03_时间序列.py": "时间序列",
    "pages/04_批量处理.py": "批量任务",
    "pages/05_工具箱.py": "系统与数据",
}


def _root_state():
    return {
        name: os.path.getmtime(os.path.join(BASE, name))
        if os.path.exists(os.path.join(BASE, name)) else None
        for name in ROOT_STATE_FILES
    }


@pytest.fixture
def app(tmp_path, monkeypatch):
    """返回 make(project_ready, **session) -> 已运行到首页的 AppTest。"""
    from streamlit.testing.v1 import AppTest
    import engine.core as core
    import frontend.project_state as project_state

    state = tmp_path / "state"
    monkeypatch.setenv("RSAP_STATE_DIR", str(state))
    monkeypatch.setenv("RSAP_JOB_MODE", "thread")
    # app.py / 页面会改写临时目录环境变量，测试结束后恢复
    for key in ("TMP", "TEMP", "GDAL_TMPDIR"):
        monkeypatch.setenv(key, os.environ.get(key, str(tmp_path)))
    monkeypatch.setattr(project_state, "_PROJECT_STATE_FILE", "")
    monkeypatch.setattr(core, "_cache_root", lambda: str(tmp_path / "cache"))
    before = _root_state()

    def make(project_ready=False, **session):
        at = AppTest.from_file(os.path.join(BASE, "app.py"), default_timeout=120)
        if project_ready:
            at.session_state["_shared_project_state"] = project_state._ProjectState(
                region="示例区", boundary=SAMPLE_AOI, slug="sample")
        for key, value in session.items():
            at.session_state[key] = value
        return at.run()

    yield make
    assert _root_state() == before, "测试不应改动项目根目录的状态文件"


def _texts(at):
    return " ".join(str(m.value) for m in at.markdown)


@pytest.mark.parametrize("project_ready", [False, True], ids=["fresh", "project"])
@pytest.mark.parametrize("page", list(PAGES))
def test_page_renders(app, page, project_ready):
    at = app(project_ready)
    assert not at.exception, [e.value for e in at.exception]
    at = at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert PAGES[page] in _texts(at)


def _seed_period(boundary_path, year, bands, source="sentinel2", months=(6, 10), cloud=50):
    """在（已隔离的）快照缓存中放一个完整的单期条目，文件内容不重要。"""
    from pathlib import Path

    import engine.core as core
    from engine.cache import PERIOD_KIND, SnapshotCache, compute_aoi_hash, period_key
    from engine.data_sources import get_data_source
    from engine.preprocessing import PREPROCESSING_VERSION
    from frontend.utils import load_gj

    aoi_hash = compute_aoi_hash(load_gj(boundary_path)[1])
    sc = SnapshotCache(core._cache_root())
    key = period_key(aoi_hash, source, year, *months, cloud, bands, PREPROCESSING_VERSION)
    sc.put_snapshot(key, {
        "kind": PERIOD_KIND, "aoi_hash": aoi_hash, "data_source": source, "year": year,
        "month_start": months[0], "month_end": months[1], "max_cloud_pct": cloud,
        "bands": sorted(bands), "tile_ids": ["T1"], "preprocessing_version": PREPROCESSING_VERSION,
    })
    fmap = get_data_source(source).FILE_NAME_MAP
    for band in bands:
        Path(sc.get_band_dir(key), f"{fmap[band]}_{year}.tif").write_bytes(b"\0" * 1024)
    sc.mark_complete(key)


def test_cached_periods_are_shared_between_pages(app):
    """地物分类下载过的时相（6 波段），地物分类页与变化检测工作台都识别为可用缓存。"""
    from frontend.project_state import _ProjectState

    defaults = _ProjectState()
    at = app(project_ready=True, _cd_wizard_active=True)
    at = at.switch_page("pages/02_地物分类.py").run()
    assert "检测到匹配的本地缓存" not in _texts(at)

    # 页面可能按本机边界文件切换了区域，以页面实际使用的边界为准
    boundary = at.session_state["cl_boundary"]
    for year in (defaults.year1, defaults.year2):
        _seed_period(boundary, year, ["blue", "green", "red", "nir", "swir1", "swir2"])

    at = at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "检测到匹配的本地缓存" in _texts(at)

    at = at.switch_page("pages/01_变化检测.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["cd_boundary"] == boundary
    assert "检测到匹配的本地缓存" in _texts(at)
    run = next(b for b in at.button if b.label.startswith("运行"))
    assert not run.disabled, "两期都已缓存，可以直接运行分析"


def test_change_detection_workbench_renders(app):
    """跳过向导后渲染工作台（影像 / 运行 / 地图 / 统计 / 采样点各标签页）。"""
    at = app(project_ready=True, _cd_wizard_active=True)
    at = at.switch_page("pages/01_变化检测.py").run()
    assert not at.exception, [e.value for e in at.exception]
    texts = _texts(at)
    assert "准备遥感影像" in texts and "执行变化检测" in texts
    run = next(b for b in at.button if "变化检测" in b.label and b.label.startswith("运行"))
    assert run.disabled, "没有影像时不能运行分析"


# =====================================================================
#  页面 → 服务层
# =====================================================================

def _world_project(world, tmp_path, **fields):
    """把 world 的 AOI 写成边界文件，返回以它为当前区域的 ProjectState。"""
    import json

    from shapely.geometry import mapping

    from frontend.project_state import _ProjectState

    path = tmp_path / "world_aoi.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"name": "测试区"},
                      "geometry": mapping(world.aoi)}],
    }, ensure_ascii=False), encoding="utf-8")
    return _ProjectState(region="测试区", boundary=str(path), slug="pytest_world", **fields)


def _button(at, prefix, contains=""):
    return next(b for b in at.button if b.label.startswith(prefix) and contains in b.label)


def _finish(at, slot, timeout=60):
    """slot 中的后台任务结束前反复重跑页面（页面在 settle() 中取回结果）。"""
    import time

    deadline = time.time() + timeout
    while at.session_state[slot]:
        assert time.time() < deadline, f"等待 {slot} 超时"
        time.sleep(0.1)
        at = at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _wait_job(runner, job_id, timeout=60):
    import time

    deadline = time.time() + timeout
    while runner.get(job_id).active:
        assert time.time() < deadline, "等待任务超时"
        time.sleep(0.1)
    return runner.get(job_id)


def test_change_detection_page_collect_and_run(app, world, tmp_path, monkeypatch):
    from services import change_detection as cd_svc

    real_collect, real_run = cd_svc.collect, cd_svc.run
    monkeypatch.setattr(cd_svc, "collect", lambda params, boundary, data_dir, **kw:
                        real_collect(params, boundary, world.data, **kw))
    monkeypatch.setattr(cd_svc, "run", lambda params, boundary, data_dir, out_dir, **kw:
                        real_run(params, boundary, world.data, str(tmp_path / "out"),
                                 **{**kw, "pdf": False}))

    project = _world_project(world, tmp_path, year1=2021, year2=2022)
    at = app(_shared_project_state=project, _cd_wizard_active=True)
    at = at.switch_page("pages/01_变化检测.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["cd_boundary"] == project.boundary, "项目边界在根目录之外也保留"

    at = _button(at, "收集").click().run()
    assert not at.exception, [e.value for e in at.exception]
    from services import jobs
    assert [job.kind for job in jobs.get_runner(mode="thread").list()] == ["cd.collect"], \
        "收集影像提交为后台任务"
    at = _finish(at, "cd_collect_job")
    assert not at.error, [e.value for e in at.error]
    assert "影像收集完成" in _texts(at)
    assert at.session_state["cd_data_ok"]
    assert at.session_state["cd_collected"]["context"]["year2"] == 2022

    at = _button(at, "运行", "变化检测").click().run()
    assert not at.exception, [e.value for e in at.exception]
    at = _finish(at, "cd_run_job")
    assert not at.error, [e.value for e in at.error]
    assert "变化检测完成" in _texts(at)
    S = at.session_state
    assert S["cd_results"]["task"]["id"] == "cd_pytest_world_2021_2022_NDVI"
    assert S["cd_arr"] is not None and S["cd_cands"]
    assert [t["id"] for t in S["_shared_project_state"].task_history] == \
        ["cd_pytest_world_2021_2022_NDVI"]
    assert len(world.calls.download) == 2, "运行分析命中收集阶段的缓存"

    # 一个项目一个状态文件：区域、分析记录与本页状态都在里面，新会话（刷新浏览器）照样恢复
    import json
    saved = json.loads((tmp_path / "state" / ".project_state.json").read_text(encoding="utf-8"))
    assert saved["boundary"] == project.boundary and len(saved["task_history"]) == 1
    assert saved["pages"]["cd"]["aoi"] == project.boundary and saved["pages"]["cd"]["cd_data_ok"]
    assert sorted(os.listdir(tmp_path / "state")) == [".project_state.json"]

    fresh = app(_cd_wizard_active=True).switch_page("pages/01_变化检测.py").run()
    assert not fresh.exception, [e.value for e in fresh.exception]
    assert fresh.session_state["cd_boundary"] == project.boundary
    assert fresh.session_state["cd_data_ok"]
    assert fresh.session_state["cd_collected"]["context"]["year2"] == 2022


def test_region_change_on_one_page_resets_other_pages(app, world, tmp_path, monkeypatch):
    """区域以项目为准：在地物分类页换了区域，变化检测页清掉旧区域的影像状态。"""
    import shutil

    import frontend.components.sidebar as sidebar

    boundaries = tmp_path / "boundaries"
    boundaries.mkdir()
    other = boundaries / "other.geojson"
    shutil.copy(SAMPLE_AOI, other)
    monkeypatch.setattr(sidebar, "BOUNDARY_DIR", boundaries)

    project = _world_project(world, tmp_path)
    at = app(_shared_project_state=project, _cd_wizard_active=True, cd_data_ok=True,
             cd_collected={"context": {"region": "测试区", "size_mb": 1.0}})
    at = at.switch_page("pages/01_变化检测.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["cd_data_ok"]

    at = at.switch_page("pages/02_地物分类.py").run()
    assert at.session_state["cl_boundary"] == project.boundary
    at = at.sidebar.selectbox[0].select("other.geojson").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["_shared_project_state"].boundary == str(other)

    at = at.switch_page("pages/01_变化检测.py").run()
    assert not at.exception, [e.value for e in at.exception]
    S = at.session_state
    assert S["cd_boundary"] == str(other)
    assert not S["cd_data_ok"] and S["cd_collected"] is None


def test_classification_page_load_classify_and_cluster(app, world, tmp_path, monkeypatch):
    import numpy as np

    from services import classification as cl_svc

    real_download = cl_svc.download
    monkeypatch.setattr(cl_svc, "download", lambda *a, progress_callback=None: real_download(
        *a[:6], world.data, progress_callback=progress_callback))
    # world 影像各波段为常数，GMM 需要一点噪声
    real_gmm = cl_svc.gmm
    rng = np.random.default_rng(0)
    monkeypatch.setattr(cl_svc, "gmm", lambda bands, meta, **kw: real_gmm(
        {k: v + rng.normal(0, 0.01, v.shape).astype(np.float32) for k, v in bands.items()},
        meta, **kw))

    at = app(_shared_project_state=_world_project(world, tmp_path))
    at = at.switch_page("pages/02_地物分类.py").run()
    at = at.sidebar.number_input[0].set_value(2021).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "检测到匹配的本地缓存" not in _texts(at)

    at = _button(at, "下载 2021").click().run()
    at = _finish(at, "cl_download_job")
    assert not at.error, [e.value for e in at.error]
    assert at.session_state["cl_download_ok"] and "影像下载完成" in _texts(at)
    assert "检测到匹配的本地缓存" in _texts(at)

    at = _button(at, "加载波段数据").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["cl_valid"].all()
    at = _button(at, "运行快速分类").click().run()
    assert not at.exception and not at.error, [e.value for e in at.exception + at.error]
    assert at.session_state["cl_class_mode"] == "quick" and at.session_state["cl_stats"]

    at = at.sidebar.slider[2].set_value(3).run()          # 聚类数
    at = _button(at, "运行 GMM 聚类").click().run()
    assert not at.exception and not at.error, [e.value for e in at.exception + at.error]
    S = at.session_state
    assert S["cl_class_mode"] == "gmm" and S["cl_n_clusters"] == 3
    assert S["cl_al_patches"], "主动学习按新的聚类结果采样"

    at = _button(at, "标注并继续").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["cl_al_current_idx"] == 1

    at = _button(at, "运行连通域过滤").click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_timeseries_page_run(app, world, tmp_path, monkeypatch):
    from services import timeseries as ts_svc

    real_run = ts_svc.run
    monkeypatch.setattr(ts_svc, "run", lambda *a, progress_callback=None: real_run(
        *a[:7], world.data, progress_callback=progress_callback))

    at = app(_shared_project_state=_world_project(world, tmp_path))
    at = at.switch_page("pages/03_时间序列.py").run()
    at = at.sidebar.number_input[0].set_value(2019).run()
    at = at.sidebar.number_input[1].set_value(2022).run()
    assert not at.exception, [e.value for e in at.exception]

    at = _button(at, "运行").click().run()
    assert not at.exception, [e.value for e in at.exception]
    at = _finish(at, "ts_job")
    assert not at.error, [e.value for e in at.error]
    assert at.session_state["ts_years"] == [2019, 2020, 2021, 2022]
    assert "趋势摘要" in _texts(at)


def test_page_shows_progress_and_cancels_job(app, world, tmp_path):
    """页面跟踪的任务在运行：显示进度与「取消任务」；取消后记下结果提示。"""
    from services import jobs

    runner = jobs.get_runner(mode="thread")
    job_id = runner.submit("job_targets:wait", {}, title="长任务")
    at = app(_shared_project_state=_world_project(world, tmp_path), ts_job=job_id)
    at = at.switch_page("pages/03_时间序列.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert _button(at, "运行").disabled, "任务进行中不能再提交"
    assert "长任务" in " ".join(str(c.value) for c in at.caption)

    at = _button(at, "取消任务").click().run()
    at = _finish(at, "ts_job")
    assert runner.get(job_id).state == "cancelled"
    assert "趋势分析已取消" in _texts(at)


def test_batch_page_runs_tasks_in_background(app, world, tmp_path, monkeypatch):
    import time

    from services import batch

    real_run_task = batch.run_task
    monkeypatch.setattr(batch, "run_task", lambda task, base_dir=None, progress_callback=None:
                        real_run_task(task, base_dir=str(tmp_path),
                                      progress_callback=progress_callback))
    boundary = _world_project(world, tmp_path).boundary
    good = {"id": "b_ok", "region": "测试区", "boundary": boundary, "year1": 2021, "year2": 2022,
            "formats": "geojson", "report": False}
    bad = {**good, "id": "b_bad", "year2": 2024}         # 2024 年没有影像

    at = app(bt_tasks=[good, bad], bt_defaults_loaded=True)
    at = at.switch_page("pages/04_批量处理.py").run()
    at = _button(at, "执行全部任务").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.session_state["bt_jobs"]) == 2

    deadline = time.time() + 60
    while len(at.session_state["bt_result_cache"]) < 2:
        assert time.time() < deadline, "等待批量任务超时"
        time.sleep(0.1)
        at = at.run()
        assert not at.exception, [e.value for e in at.exception]
    results = {r["task_id"]: r for r in at.session_state["bt_result_cache"].values()}
    assert not results["b_ok"].get("error") and results["b_ok"]["stats"]
    assert results["b_bad"]["error"]
    assert (tmp_path / "output" / "b_ok").is_dir(), "输出写到 base_dir"


def test_task_center_retry_hands_job_back_to_page(app, world, tmp_path):
    """任务中心重试失败的任务：原页面没有跟踪中的任务、区域没变时交回原页面。"""
    from engine.params import RunParams
    from services import jobs

    project = _world_project(world, tmp_path)
    runner = jobs.get_runner(mode="thread")
    meta = {"slot": "cd_collect_job", "page": "pages/01_变化检测.py", "region": "测试区",
            "boundary": project.boundary}
    params = {"run_params": RunParams(region="测试区", year1=2021, year2=2024).to_dict(),
              "boundary_file": project.boundary, "data_dir": world.data}
    first = runner.submit("cd.collect", params, title="测试区 · 2021/2024 影像", meta=meta)
    assert _wait_job(runner, first).state == "failed"

    at = app(_shared_project_state=project, _cd_wizard_active=True)
    at = at.switch_page("pages/06_任务中心.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "测试区 · 2021/2024 影像" in _texts(at)

    at = _button(at, "重试").click().run()
    assert not at.exception, [e.value for e in at.exception]
    new = at.session_state["cd_collect_job"]
    assert new and new != first and runner.get(new).meta == meta
    assert len(runner.list()) == 2

    at = at.switch_page("pages/01_变化检测.py").run()
    at = _finish(at, "cd_collect_job")
    assert "影像收集失败" in _texts(at)
