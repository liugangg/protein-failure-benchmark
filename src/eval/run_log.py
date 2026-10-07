"""运行日志 —— SPEC v2 §1.4 与 §6「可复现性要求」。

v2 原文要求 (两处):
  §1.4 "**所有训练运行必须记录该次使用了哪些数据源**, 写入 reports/ 下的运行日志。"
  §6   "每次训练运行记录: 数据源清单、切分 hash、超参、环境版本, 写入 reports/runs/。"

为什么必须机器写而不是手记: DS1 是 CC BY-SA 4.0, 带传染性条款, "用它训练的模型权重
算不算演绎作品"法律上无定论。将来要拿出"这份权重没用过 TargetTrack"的证据时,
靠记忆或事后补写是没用的 —— 必须是训练当时就落盘、且能核对的记录。
§1.4 还要求阶段四额外保留一份不含 TargetTrack 的对照模型。

任何产生模型或评测数字的脚本, 结束时调 write_run_log()。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone

RUNS = pathlib.Path("reports/runs")
RECORDS = pathlib.Path("data/processed/records.parquet")
SPLITS = pathlib.Path("data/processed/splits")


def _sha256(p: pathlib.Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _pkg_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("polars", "numpy", "sklearn", "pyarrow", "torch", "transformers"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = "not installed"
    return out


def _git_rev() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def data_sources_used(sources: list[str]) -> dict:
    """数据源清单, 并显式标出是否含 TargetTrack (许可上最敏感的那个)。"""
    return {
        "sources": sorted(sources),
        "contains_targettrack": "ds1_targettrack" in sources,
        "license_note": (
            "含 TargetTrack (CC BY-SA 4.0, 带 share-alike)"
            if "ds1_targettrack" in sources
            else "不含 TargetTrack —— 可作为商业化的干净起点 (v2 §1.4)"
        ),
    }


def write_run_log(
    run_name: str,
    *,
    sources: list[str],
    splits_used: list[str],
    config: dict,
    seed: int | None = None,
    results: dict | None = None,
    notes: str | None = None,
) -> pathlib.Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M%S%z")

    split_hashes = {}
    mf = SPLITS / "MANIFEST.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        for k, v in m.get("splits", {}).items():
            split_hashes[k] = v.get("sha256")
        split_hashes["_records"] = m.get("records", {}).get("sha256")

    log = {
        "run_name": run_name,
        "timestamp": ts,
        "seed": seed if seed is not None else config.get("seed"),
        "data": data_sources_used(sources),
        "splits_used": splits_used,
        "split_hashes": split_hashes,
        "records_parquet_sha256": _sha256(RECORDS),
        "config": config,
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _pkg_versions(),
            "git_rev": _git_rev(),
        },
        "results": results or {},
        "notes": notes,
    }
    out = RUNS / f"{ts}_{run_name}.json"
    out.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"[run_log] {out}")
    return out
