"""Shared UI primitives for the Streamlit workbench."""

from __future__ import annotations

from html import escape
from typing import Iterable, Mapping, Sequence

import streamlit as st


def _text(value) -> str:
    return escape(str(value if value is not None else ""))


def page_header(
    title: str,
    eyebrow: str,
    description: str = "",
    meta: Mapping[str, object] | Sequence[tuple[str, object]] | None = None,
    status: str = "",
    status_tone: str = "neutral",
) -> None:
    """Render the compact, shared page heading used across the app."""
    items = meta.items() if isinstance(meta, Mapping) else (meta or [])
    meta_html = "".join(
        f'<span class="rs-meta-chip"><b>{_text(label)}</b>{_text(value)}</span>'
        for label, value in items
        if value not in (None, "")
    )
    status_html = ""
    if status:
        status_html = (
            f'<span class="rs-status rs-status--{_text(status_tone)}">'
            f'<i></i>{_text(status)}</span>'
        )
    description_html = (
        f'<p class="rs-page-description">{_text(description)}</p>'
        if description else ""
    )
    meta_block = (
        f'<div class="rs-page-meta">{meta_html}</div>' if meta_html else ""
    )
    st.markdown(
        f"""
        <header class="rs-page-header">
          <div class="rs-page-heading">
            <div class="rs-eyebrow">{_text(eyebrow)}</div>
            <h1>{_text(title)}</h1>
            {description_html}
          </div>
          {status_html}
          {meta_block}
        </header>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str = "", overline: str = "") -> None:
    """Render a section heading without introducing a decorative card."""
    overline_html = (
        f'<span class="rs-section-overline">{_text(overline)}</span>'
        if overline else ""
    )
    caption_html = f'<p>{_text(caption)}</p>' if caption else ""
    st.markdown(
        f"""
        <div class="rs-section-header">
          {overline_html}
          <h2>{_text(title)}</h2>
          {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_bar(steps: Sequence[str], active: int, completed: Iterable[int] = ()) -> None:
    """Render a stable workflow rail for multi-stage analysis pages."""
    completed_set = set(completed)
    step_html = []
    for index, label in enumerate(steps):
        state = "is-complete" if index in completed_set else ""
        if index == active:
            state = "is-active"
        marker = "OK" if index in completed_set else f"{index + 1:02d}"
        step_html.append(
            f'<div class="rs-workflow-step {state}">'
            f'<span>{marker}</span><b>{_text(label)}</b></div>'
        )
    st.markdown(
        f'<div class="rs-workflow">{"".join(step_html)}</div>',
        unsafe_allow_html=True,
    )


def status_banner(tone: str, title: str, detail: str = "") -> None:
    """Render a compact operational status message."""
    detail_html = f'<p>{_text(detail)}</p>' if detail else ""
    st.markdown(
        f"""
        <div class="rs-banner rs-banner--{_text(tone)}">
          <span class="rs-banner-mark"></span>
          <div><b>{_text(title)}</b>{detail_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, detail: str = "", code: str = "--") -> None:
    """Render an empty state; actions stay as native Streamlit buttons below it."""
    detail_html = f'<p>{_text(detail)}</p>' if detail else ""
    st.markdown(
        f"""
        <div class="rs-empty-state">
          <span>{_text(code)}</span>
          <div><b>{_text(title)}</b>{detail_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_header(title: str, caption: str = "") -> None:
    """Render a restrained heading for page-specific sidebar controls."""
    caption_html = f'<p>{_text(caption)}</p>' if caption else ""
    st.sidebar.markdown(
        f'<div class="rs-sidebar-panel"><b>{_text(title)}</b>{caption_html}</div>',
        unsafe_allow_html=True,
    )


def task_row_heading(
    title: str,
    kind: str,
    status: str,
    status_tone: str,
    detail: str = "",
) -> None:
    """Render the textual part of task/result rows consistently."""
    detail_html = f'<p>{_text(detail)}</p>' if detail else ""
    st.markdown(
        f"""
        <div class="rs-task-heading">
          <div><span class="rs-kind">{_text(kind)}</span>
          <span class="rs-status rs-status--{_text(status_tone)}"><i></i>{_text(status)}</span></div>
          <h3>{_text(title)}</h3>
          {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
