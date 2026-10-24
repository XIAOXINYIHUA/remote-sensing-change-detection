"""本地影像与成果文件盘点（不依赖 Streamlit，首页 / 系统页 / 成果库共用）。

首页曾把 data/ 下所有 .tif 都算作「本地影像」，其中包括旧版下载（没有质量波段）
和预处理版本已过期的缓存，分析时并不会复用，于是出现「首页有 546 MB 影像、
工作台却提示尚未收集」的矛盾。这里把两者分开统计。
"""

from __future__ import annotations

from pathlib import Path

from engine.cache import PERIOD_KIND, SnapshotCache
from engine.preprocessing import PREPROCESSING_VERSION

SHAPEFILE_SIDECARS = (".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx")


def _mb(paths) -> float:
    return sum(path.stat().st_size for path in paths) / 1024 / 1024


def imagery_summary(data_root) -> dict:
    """统计 data/ 下的影像：可被分析复用的数据 vs 旧版下载 / 过期缓存。

    可复用：
      - 单期快照缓存：data/cache/<key>/ 已完整收集（有 .collected 标记）、
        预处理版本与当前一致，变化检测 / 地物分类 / 时间序列以相同参数取数时直接命中；
      - 旧版地物分类下载：含 source.json 的目录（下载成功后才写入），
        地物分类页仍可直接加载。
    旧版两期快照（无 kind 字段）已不再被查找，计入不可复用。

    Returns:
        dict(snapshots, snapshot_files, snapshot_mb,
             classify_sets, classify_files, classify_mb,
             reusable_sets, reusable_mb, stale_files, stale_mb)
    """
    data_root = Path(data_root)
    cache_root = data_root / "cache"
    snapshot_dirs = {
        (cache_root / entry["cache_key"]).resolve()
        for entry in SnapshotCache(str(cache_root)).list_entries()
        if entry["collected"] and entry.get("kind") == PERIOD_KIND
        and entry.get("preprocessing_version") == PREPROCESSING_VERSION
    }
    exists = data_root.exists()
    classify_dirs = {
        marker.parent.resolve() for marker in data_root.rglob("source.json")
        if cache_root not in marker.parents
    } if exists else set()
    tifs = list(data_root.rglob("*.tif")) if exists else []
    snapshot = [path for path in tifs if path.parent.resolve() in snapshot_dirs]
    classify = [path for path in tifs if path.parent.resolve() in classify_dirs]
    stale = [path for path in tifs
             if path.parent.resolve() not in snapshot_dirs | classify_dirs]
    return {
        "snapshots": len(snapshot_dirs),
        "snapshot_files": len(snapshot),
        "snapshot_mb": _mb(snapshot),
        "classify_sets": len(classify_dirs),
        "classify_files": len(classify),
        "classify_mb": _mb(classify),
        "reusable_sets": len(snapshot_dirs) + len(classify_dirs),
        "reusable_mb": _mb(snapshot) + _mb(classify),
        "stale_files": len(stale),
        "stale_mb": _mb(stale),
    }


def _archived_keys(archived_paths) -> set:
    """已归档文件的 (目录, 文件名) 集合；.shp 连带其同名附属文件。"""
    keys = set()
    for raw in archived_paths:
        if not raw:
            continue
        path = Path(raw).resolve()
        keys.add((path.parent, path.name.lower()))
        if path.suffix.lower() == ".shp":
            keys.update((path.parent, (path.stem + ext).lower()) for ext in SHAPEFILE_SIDECARS)
    return keys


def unarchived_outputs(output_root, archived_paths) -> list[dict]:
    """output/ 中没有被任何成功任务记录引用的成果文件，按区域子目录分组。

    这些文件来自 CLI、批量任务、旧版本或已删除的任务记录。成果库只读任务记录时
    看不到它们，会出现「output 里有 8 个文件、成果库却显示 0」的情况。

    Returns:
        list[dict(region, files, size_mb, modified)]，最近修改的分组在前，
        组内文件按修改时间倒序。
    """
    output_root = Path(output_root)
    if not output_root.exists():
        return []
    archived = _archived_keys(archived_paths)
    groups: dict[str, list[Path]] = {}
    for path in output_root.rglob("*"):
        if (not path.is_file() or path.name.startswith(".")
                or path.suffix.lower() == ".tmp"
                or (path.parent.resolve(), path.name.lower()) in archived):
            continue
        rel = path.relative_to(output_root)
        region = rel.parts[0] if len(rel.parts) > 1 else "output"
        groups.setdefault(region, []).append(path)

    result = []
    for region, files in groups.items():
        files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        result.append({
            "region": region,
            "files": files,
            "size_mb": _mb(files),
            "modified": files[0].stat().st_mtime,
        })
    result.sort(key=lambda group: group["modified"], reverse=True)
    return result
