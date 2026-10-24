"""
侧栏共用组件测试（全部离线，AppTest 直接运行组件）

  - 区域：首次进入选第一个边界并同步到 ProjectState；切换时调用 on_change；
    当前边界在边界目录之外（或写成相对路径）时保持不变
  - 时间范围 / 数据源 / 指数：越界默认值收窄、无效组合提示、指数按能力过滤并记住选择

用法:
    python -m pytest tests/test_sidebar.py -v
"""

import json
import os

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_AOI = os.path.join(BASE, "examples", "sample_aoi.geojson")


def _region_script(boundary_dir):
    import streamlit as st

    from frontend.components.sidebar import region_picker

    st.session_state.setdefault("changes", 0)

    def on_change():
        st.session_state.changes += 1

    geom = region_picker("t", on_change=on_change, boundary_dir=boundary_dir)
    st.write(f"geom={geom is not None}")


def _period_script():
    import streamlit as st

    from frontend.components.sidebar import (
        data_source_picker, index_picker, month_range, year_pair,
    )

    y1, y2 = year_pair(2030, 2020)
    ms, me = month_range(9, 3)
    src = data_source_picker("landsat8")
    idx = index_picker(src, default="NDBI", allowed=["NDVI", "NDWI"], remember_as="t_idx")
    st.write(f"result={y1}|{y2}|{ms}|{me}|{src}|{idx}")


@pytest.fixture
def run(monkeypatch):
    """run(script, *args, **session) -> 已运行一次的 AppTest；ProjectState 不写文件。"""
    from streamlit.testing.v1 import AppTest

    import frontend.project_state as project_state

    monkeypatch.setattr(project_state, "_PROJECT_STATE_FILE", "")

    def make(script, *args, **session):
        at = AppTest.from_function(script, args=args, default_timeout=30)
        for key, value in session.items():
            at.session_state[key] = value
        return at.run()

    return make


def _write_aoi(path, name=None):
    with open(SAMPLE_AOI, encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj.get("features", []):
        feat["properties"] = {"name": name} if name else {}
    path.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _texts(at):
    return " ".join(str(m.value) for m in at.markdown)


def test_region_picker_selects_and_syncs_project(run, tmp_path):
    a = _write_aoi(tmp_path / "a_boundary.geojson", "甲区")
    b = _write_aoi(tmp_path / "b.geojson")
    at = run(_region_script, str(tmp_path))
    assert not at.exception, [e.value for e in at.exception]
    S = at.session_state
    assert (S["t_boundary"], S["t_region"], S["t_slug"], S["changes"]) == (a, "甲区", "甲区", 1)
    project = S["_shared_project_state"]
    assert (project.boundary, project.region) == (a, "甲区")
    assert "面积" in " ".join(c.value for c in at.sidebar.caption)
    assert "geom=True" in _texts(at)

    at = at.sidebar.selectbox[0].select("b.geojson").run()
    assert (S["t_boundary"], S["t_region"], S["changes"]) == (b, "b", 2)
    assert S["_shared_project_state"].boundary == b


def test_region_picker_keeps_boundary_outside_dir(run, tmp_path):
    _write_aoi(tmp_path / "a.geojson")
    at = run(_region_script, str(tmp_path), t_boundary=SAMPLE_AOI)
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["t_boundary"] == SAMPLE_AOI and at.session_state["changes"] == 0
    box = at.sidebar.selectbox[0]
    assert box.value == "examples/sample_aoi.geojson" and "a.geojson" in box.options

    # 同一文件写成相对路径也不算切换区域
    at = run(_region_script, os.path.dirname(SAMPLE_AOI), t_boundary=os.path.relpath(SAMPLE_AOI))
    assert at.session_state["changes"] == 0 and at.sidebar.selectbox[0].value == "sample_aoi.geojson"


def test_region_picker_without_boundaries(run, tmp_path):
    at = run(_region_script, str(tmp_path))
    assert not at.exception, [e.value for e in at.exception]
    assert "请选择边界文件" in [e.value for e in at.sidebar.error]
    assert "geom=False" in _texts(at) and at.session_state["changes"] == 0


def test_period_and_source_widgets(run):
    at = run(_period_script)
    assert not at.exception, [e.value for e in at.exception]
    # 2030 超出可选范围 → 收窄到 2027；月份 9 → 3 提示；NDBI 不在允许列表 → 第一个可用指数
    assert "result=2027|2020|9|3|landsat8|NDVI" in _texts(at)
    assert "结束月份不能早于起始月份" in [e.value for e in at.sidebar.error]

    at = at.sidebar.number_input[1].set_value(2027).run()
    assert "两期年份不能相同" in [e.value for e in at.sidebar.error]

    at = at.sidebar.selectbox[1].select("NDWI").run()
    assert at.session_state["t_idx"] == "NDWI"
    at = at.sidebar.selectbox[0].select("sentinel2").run()
    assert "|sentinel2|NDWI" in _texts(at), "切换数据源后保留上次选择的指数"
