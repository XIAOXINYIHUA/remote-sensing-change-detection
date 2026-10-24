"""
数据快照缓存系统

缓存单位是「单期快照」：一个 AOI 在一个年份 / 月份范围 / 云量阈值下，
按确定性分析网格下载的原始 DN 波段与质量波段。缓存键内容:
  - AOI 几何哈希
  - 数据源（landsat8 / landsat9 分开）
  - 年份与月份范围
  - 云量阈值
  - 波段列表（查找时允许超集）
  - 预处理版本（版本变化时旧缓存失效）
"""

from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import typing as T

if T.TYPE_CHECKING:
    pass


@dataclasses.dataclass(frozen=True)
class DatasetSnapshot:
    """不可变数据快照 — 记录一次数据收集的全部元数据。"""

    # 识别字段
    region: str
    aoi_hash: str
    data_source: str
    index_type: str

    # 时相
    year1: int
    year2: int
    month_start: int
    month_end: int
    max_cloud_pct: float

    # 技术元数据
    bands: tuple[str, ...]
    tile_ids: tuple[str, ...]
    preprocessing_version: str = "v3.1"

    # 网格
    crs: str = ""
    transform: str = ""
    width: int = 0
    height: int = 0
    resolution: float = 0.0

    # 时间戳
    created_at: str = ""

    @property
    def cache_key(self) -> str:
        """确定性缓存键。"""
        raw = "|".join([
            self.aoi_hash,
            self.data_source,
            self.index_type,
            str(self.year1), str(self.year2),
            str(self.month_start), str(self.month_end),
            str(self.max_cloud_pct),
            ",".join(self.bands),
            str(self.crs),
            str(self.resolution),
            self.preprocessing_version,
        ])
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "aoi_hash": self.aoi_hash,
            "data_source": self.data_source,
            "index_type": self.index_type,
            "year1": self.year1,
            "year2": self.year2,
            "month_start": self.month_start,
            "month_end": self.month_end,
            "max_cloud_pct": self.max_cloud_pct,
            "bands": list(self.bands),
            "tile_ids": list(self.tile_ids),
            "preprocessing_version": self.preprocessing_version,
            "crs": self.crs,
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "created_at": self.created_at,
            "cache_key": self.cache_key,
        }


PERIOD_KIND = "period"


class SnapshotCache:
    """单期影像快照的持久化缓存。

    一个条目 = 一个时相（AOI / 数据源 / 年份 / 月份 / 云量 / 波段 / 预处理版本），
    变化检测（两期）、地物分类（一期）与时间序列（多期）共用：
    同一时相下载过一次，各模块都能复用，波段为所需波段的超集即可。

    缓存的结构:
        data/cache/<cache_key>/
            snapshot.json          # 元数据（kind="period"）
            .collected             # 标记文件：收集完成
            <前缀>_<年份>.tif       # 原始 DN 波段，如 B4_2024.tif
            SCL_<年份>.tif          # 质量波段（S2 SCL / Landsat QA_PIXEL）
    """

    def __init__(self, cache_root: str = "data/cache"):
        self.root = cache_root

    def _entry_path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def has(self, key: str) -> bool:
        """检查快照是否完整（存在 .collected 标记）。"""
        ep = self._entry_path(key)
        return os.path.exists(os.path.join(ep, ".collected"))

    def get_snapshot(self, key: str) -> dict | None:
        ep = self._entry_path(key)
        sp = os.path.join(ep, "snapshot.json")
        if not os.path.exists(sp):
            return None
        try:
            with open(sp, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def find_period(self, aoi_hash: str, data_source: str, year: int,
                    month_start: int, month_end: int, max_cloud_pct: float,
                    bands: T.Sequence[str], preprocessing_version: str) -> str | None:
        """查找已完整收集、波段包含 bands 的单期快照，返回 cache_key 或 None。

        多个条目都满足时取波段最少的（再按 key 排序），结果确定。
        预处理版本不同的旧缓存自动失效。
        """
        matches = []
        for snap in self.list_entries():
            if (snap["collected"]
                    and snap.get("kind") == PERIOD_KIND
                    and snap.get("aoi_hash") == aoi_hash
                    and snap.get("data_source") == data_source
                    and snap.get("year") == int(year)
                    and snap.get("month_start") == int(month_start)
                    and snap.get("month_end") == int(month_end)
                    and snap.get("max_cloud_pct") == max_cloud_pct
                    and set(bands) <= set(snap.get("bands", []))
                    and snap.get("preprocessing_version") == preprocessing_version):
                matches.append((len(snap.get("bands", [])), snap["cache_key"]))
        return min(matches)[1] if matches else None

    def put_snapshot(self, key: str, snapshot: dict) -> None:
        """保存快照元数据。"""
        ep = self._entry_path(key)
        os.makedirs(ep, exist_ok=True)
        sp = os.path.join(ep, "snapshot.json")
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    def mark_complete(self, key: str) -> None:
        """标记收集完成。"""
        ep = self._entry_path(key)
        os.makedirs(ep, exist_ok=True)
        Path(os.path.join(ep, ".collected")).write_text("1")

    def get_band_dir(self, key: str) -> str:
        """返回快照的波段文件目录。"""
        return self._entry_path(key)

    def list_entries(self) -> list[dict]:
        if not os.path.exists(self.root):
            return []
        entries = []
        for entry_name in os.listdir(self.root):
            ep = os.path.join(self.root, entry_name)
            sp = os.path.join(ep, "snapshot.json")
            collected = os.path.exists(os.path.join(ep, ".collected"))
            if not os.path.exists(sp):
                continue
            try:
                with open(sp, "r", encoding="utf-8") as f:
                    snap = json.load(f)
            except Exception:
                continue
            snap["cache_key"] = entry_name
            snap["collected"] = collected
            entries.append(snap)
        return entries

    def clear(self) -> None:
        import shutil
        if os.path.exists(self.root):
            shutil.rmtree(self.root)


from pathlib import Path


def period_key(aoi_hash: str, data_source: str, year: int,
               month_start: int, month_end: int, max_cloud_pct: float,
               bands: T.Sequence[str], preprocessing_version: str) -> str:
    """由单期快照的全部匹配字段生成缓存目录名，参数不同的快照互不覆盖。"""
    raw = "|".join([
        PERIOD_KIND, aoi_hash, data_source, str(int(year)),
        str(int(month_start)), str(int(month_end)),
        str(max_cloud_pct), ",".join(sorted(bands)),
        preprocessing_version,
    ])
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def compute_aoi_hash(boundary_wgs84) -> str:
    """计算 AOI 几何的确定性哈希。

    将边界简化为 ~100m 精度，取 WGS84 坐标的 MD5 哈希。
    支持 MultiPolygon（遍历所有子部件）。
    """
    try:
        simplified = boundary_wgs84.simplify(0.001, preserve_topology=True)
        # MultiPolygon: 遍历 .geoms, 连接所有外环
        if hasattr(simplified, "geoms"):
            all_coords = []
            for part in simplified.geoms:
                if part.exterior:
                    all_coords.extend(list(part.exterior.coords))
        elif simplified.exterior:
            all_coords = list(simplified.exterior.coords)
        else:
            all_coords = []
        coord_str = "|".join(f"{x:.6f},{y:.6f}" for x, y in all_coords)
        return hashlib.md5(coord_str.encode()).hexdigest()[:16]
    except Exception:
        return "unknown_aoi"
