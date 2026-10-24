"""
项目状态测试（全部离线）：一个项目一个状态文件 .project_state.json

  - 首次启动把旧版 .shared_state.json 与各页状态文件并入新文件，旧文件不改动；之后不再迁移
  - 页面状态按区域保存：区域换了不载入旧区域的页面状态，换回来恢复
  - 多个会话（浏览器标签页）共用状态文件：别的会话写过后重新载入，区域变化各自察觉

用法:
    python -m pytest tests/test_project_state.py -v
"""

import json
import os

import pytest

import frontend.project_state as project_state
from frontend.project_state import ProjectState, same_path


@pytest.fixture(autouse=True)
def _restore_state_file(monkeypatch):
    # ProjectState.init 改写模块级的状态文件路径，测试后恢复
    monkeypatch.setattr(project_state, "_PROJECT_STATE_FILE", project_state._PROJECT_STATE_FILE)


def _dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


# =====================================================================
#  迁移旧版状态文件
# =====================================================================

def test_legacy_files_are_merged_once(tmp_path):
    lc, xy = str(tmp_path / "lc.geojson"), str(tmp_path / "xy.geojson")
    _dump(tmp_path / ".shared_state.json", {
        "region": "灵川", "boundary": lc, "slug": "lc", "year1": 2020,
        "max_cloud_pct": 30, "task_history": [{"id": "t1"}]})
    _dump(tmp_path / ".app_state.json", {
        "cd_boundary": lc, "cd_region": "灵川", "cd_slug": "lc", "cd_data_ok": True,
        "cd_collected": {"context": {"year2": 2022}}, "cd_run_job": None})
    # 更早版本的地物分类状态：键没有页面前缀
    _dump(tmp_path / ".classify_state.json", {
        "boundary": xy, "region": "新余", "slug": "新余", "download_ok": True, "data_dir": "d"})
    _dump(tmp_path / ".batch_state.json", {"bt_jobs": ["j1"]})
    legacy = {p.name: p.read_bytes() for p in tmp_path.glob(".*.json")}

    ProjectState.init(str(tmp_path))
    target = tmp_path / ".project_state.json"
    data = _load(target)
    assert data["version"] == 2
    assert (data["region"], data["boundary"], data["year1"], data["max_cloud_pct"]) == \
        ("灵川", lc, 2020, 30)
    assert data["task_history"] == [{"id": "t1"}]
    assert data["pages"] == {
        "cd": {"aoi": lc, "cd_data_ok": True, "cd_collected": {"context": {"year2": 2022}},
               "cd_run_job": None},
        "cl": {"aoi": xy, "cl_download_ok": True, "cl_data_dir": "d"},
        "bt": {"aoi": "", "bt_jobs": ["j1"]},
    }
    assert {p.name: p.read_bytes() for p in tmp_path.glob(".*.json")
            if p.name != target.name} == legacy, "旧文件保留不动"

    _dump(tmp_path / ".app_state.json", {"cd_data_ok": False})
    ProjectState.init(str(tmp_path))
    assert _load(target) == data, "已有新文件时不再迁移"


def test_migration_takes_region_from_pages_when_shared_has_none(tmp_path):
    xy = str(tmp_path / "xy.geojson")
    _dump(tmp_path / ".ts_state.json", {"ts_boundary": xy, "ts_region": "新余", "ts_slug": "xy"})
    ProjectState.init(str(tmp_path))
    data = _load(tmp_path / ".project_state.json")
    assert (data["region"], data["boundary"], data["slug"]) == ("新余", xy, "xy")
    assert data["pages"] == {"ts": {"aoi": xy}}


def test_no_legacy_files_writes_nothing(tmp_path):
    ProjectState.init(str(tmp_path))
    assert not (tmp_path / ".project_state.json").exists()


def test_same_path():
    here = os.path.abspath("x.geojson")
    assert same_path("x.geojson", here)
    assert same_path("", None) and not same_path("", here)
    assert not same_path(here, os.path.abspath("y.geojson"))


# =====================================================================
#  页面状态与多会话
# =====================================================================

def _session_script(state_dir):
    import streamlit as st

    from frontend.project_state import ProjectState

    ProjectState.init(state_dir)
    S = st.session_state
    action = S.pop("action", None)
    if action == "save":
        ProjectState.save_page("cd", {"cd_data_ok": True, "cd_run_job": None})
    elif action:
        ProjectState.set_aoi(*action)
    S.changed = ProjectState.aoi_changed("cd")
    S.loaded = ProjectState.load_page("cd", ["cd_data_ok", "cd_run_job"])
    S.region = ProjectState.get().region


def test_page_state_follows_region_across_sessions(tmp_path):
    from streamlit.testing.v1 import AppTest

    a, b = str(tmp_path / "a.geojson"), str(tmp_path / "b.geojson")
    _dump(tmp_path / ".project_state.json", {"region": "甲", "boundary": a, "slug": "a"})

    def session():
        return AppTest.from_function(_session_script, args=(str(tmp_path),), default_timeout=30)

    def run(at, action=None):
        if action:
            at.session_state["action"] = action
        at = at.run()
        assert not at.exception, [e.value for e in at.exception]
        S = at.session_state
        return S["region"], S["changed"], S["loaded"]

    one, two = session(), session()
    assert run(one) == ("甲", False, {})
    assert run(one, "save") == ("甲", False, {"cd_data_ok": True})
    saved = _load(tmp_path / ".project_state.json")["pages"]["cd"]
    assert saved == {"aoi": a, "cd_data_ok": True, "cd_run_job": None}
    assert run(two) == ("甲", False, {"cd_data_ok": True}), "另一个会话读到同一个文件"

    assert run(one, ("乙", b)) == ("乙", True, {}), "区域换了：本页数据作废，旧区域状态不载入"
    assert run(two) == ("乙", True, {}), "另一个会话重新载入文件并察觉区域变化"
    assert run(two) == ("乙", False, {})

    assert run(one, ("甲", a)) == ("甲", True, {"cd_data_ok": True}), "换回原区域恢复其页面状态"
