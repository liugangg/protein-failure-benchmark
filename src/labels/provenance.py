"""产物指纹与上游校验 —— 替换掉散落的单点补丁。

为什么需要 (三次真实事故, 全部**不报错**地给出错误数字):
  1. `seq_uid` 曾用无序 .unique() 编号 -> 重跑 pool.py 就换号, 而 cluster30 的 tsv 还是旧号,
     join 上去序列与簇静默错位 (clone 阴性簇数 5,463 -> 9,260 才暴露)。
  2. SoluProt 的 clean 子集按 seq_uid 存, records.parquet 重建后对不上,
     交集从 2,377 静默缩到 289。
  3. mmseqs 崩掉时 split_groups.py 读到上一轮残留的 .m8, 建出一套看着正常的错误分组。

共同点: **下游读到了与当前上游不匹配的中间产物**。所以机制是:
  产物落盘时, 把它**所有上游输入的指纹**记进 `<产物>.prov.json`;
  下游读取前调 require(), 重算上游当前指纹并比对, 不一致就硬失败。

指纹用 sha256 + 大小 + (parquet 的) 行数。行数是廉价的二次确认: 文件被原地改写但大小
碰巧相同的情况 sha256 会抓到, 而行数让报错信息更可读。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Iterable

_CHUNK = 1 << 20


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def _rows(p: pathlib.Path) -> int | None:
    if p.suffix == ".parquet":
        try:
            import polars as pl
            return int(pl.scan_parquet(p).select(pl.len()).collect().item())
        except Exception:
            return None
    if p.suffix in (".tsv", ".m8", ".fasta", ".fa", ".csv"):
        try:
            with p.open("rb") as fh:
                return sum(1 for _ in fh)
        except Exception:
            return None
    return None


def fingerprint(path: str | pathlib.Path) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"上游产物不存在: {p}")
    return {"path": str(p), "size": p.stat().st_size,
            "sha256": _sha256(p), "rows": _rows(p)}


def _prov_path(artifact: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(artifact)
    return p.with_suffix(p.suffix + ".prov.json")


def stamp(artifact: str | pathlib.Path, inputs: Iterable[str | pathlib.Path],
          produced_by: str, extra: dict | None = None) -> pathlib.Path:
    """产物落盘后调用: 记录它的上游指纹。"""
    art = pathlib.Path(artifact)
    rec = {
        "artifact": fingerprint(art),
        "produced_by": produced_by,
        "inputs": [fingerprint(i) for i in inputs],
        "extra": extra or {},
    }
    out = _prov_path(art)
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return out


def require(artifacts: Iterable[str | pathlib.Path], *, strict: bool = True) -> None:
    """读取这些产物之前调用: 校验它们记录的上游指纹与当前上游一致。

    strict=True (默认): 缺 .prov.json 也算失败 —— 没有指纹就无法保证不是陈旧产物。
    对还没接入指纹的历史产物, 显式传 strict=False 并在调用处写清为什么可以放过。
    """
    problems: list[str] = []
    for a in artifacts:
        art = pathlib.Path(a)
        if not art.exists():
            problems.append(f"{art}: 不存在")
            continue
        pp = _prov_path(art)
        if not pp.exists():
            if strict:
                problems.append(f"{art}: 缺 {pp.name}, 无法确认上游 —— 重跑生成它的脚本")
            continue
        try:
            rec = json.loads(pp.read_text())
        except Exception as e:
            problems.append(f"{pp}: 读不出来 ({e})")
            continue
        # 产物自身有没有被改过
        cur = fingerprint(art)
        old = rec.get("artifact", {})
        if old.get("sha256") and cur["sha256"] != old["sha256"]:
            problems.append(
                f"{art}: 产物本身与记录不符 (rows {old.get('rows')} -> {cur['rows']}) "
                f"—— 它被改写过但指纹没更新")
        # 上游有没有变
        for inp in rec.get("inputs", []):
            ip = pathlib.Path(inp["path"])
            if not ip.exists():
                problems.append(f"{art}: 上游 {ip} 已不存在")
                continue
            now = fingerprint(ip)
            if now["sha256"] != inp["sha256"]:
                problems.append(
                    f"{art}: 上游 {ip} 已变 (rows {inp.get('rows')} -> {now['rows']}) "
                    f"—— {art.name} 是用旧版上游生成的, 先重跑它")
    if problems:
        msg = "上游指纹校验失败, 拒绝继续 (防止陈旧产物静默污染结果):\n" + \
              "\n".join(f"  - {m}" for m in problems)
        raise SystemExit(msg)
