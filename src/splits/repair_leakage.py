"""把泄漏检查找到的越界序列对并进 split_group, 直到检查通过。

为什么需要这一步 (不是设计缺陷, 是 mmseqs 的性质):
  split_groups.py 的 all-vs-all 和 check_leakage.py 的 train-vs-test 是**两次独立的**
  mmseqs 检索。mmseqs 的预筛 (prefilter) 是启发式的 k-mer 匹配, 不保证召回全部同源对,
  两次运行的召回集合不完全一致。所以建组那次漏掉的对, 检查那次可能找出来。
  实测第一轮修补前剩 16 / 1 / 8 对 (最高 fident 0.56)。

做法: 把检查找到的对当成新的边并进并查集, 重写 split_groups.parquet,
      重跑 make_splits, 再检查。反复直到三套切分都 PASS。
      每轮都会让 split_group 变少 (合并), 收敛是单调的。
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import polars as pl

_sys_path_added = True
import sys as _s, pathlib as _pl
_s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from labels import provenance

ROOT = pathlib.Path(__file__).resolve().parents[2]
GROUPS = pathlib.Path("data/interim/split_groups.parquet")
REPORTS = pathlib.Path("reports")
PY = ROOT / ".venv/bin/python"


class DSU:
    def __init__(self) -> None:
        self.p: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def merge_pairs() -> int:
    """把各切分的越界对并进 split_group。返回并进去的边数。"""
    g = pl.read_parquet(GROUPS)
    dsu = DSU()
    # 先按现有分组建并查集
    for uid, grp in g.select("seq_uid", "split_group").iter_rows():
        dsu.union(int(grp), int(uid))

    n_edges = 0
    for name in ("sequence", "lab", "time", "bind_target"):
        f = REPORTS / f"leakage_pairs_{name}.parquet"
        if not f.exists():
            continue
        pairs = pl.read_parquet(f)
        for a, b in pairs.select("seq_uid_a", "seq_uid_b").iter_rows():
            dsu.union(int(a), int(b))
            n_edges += 1
    if not n_edges:
        return 0

    uids = g["seq_uid"].to_list()
    newg = pl.DataFrame(
        {"seq_uid": uids, "split_group": [dsu.find(int(u)) for u in uids]},
        schema={"seq_uid": pl.Int64, "split_group": pl.Int64},
    )
    out = g.drop("split_group").join(newg, on="seq_uid", how="left")
    before = g["split_group"].n_unique()
    after = out["split_group"].n_unique()
    out.write_parquet(GROUPS)
    # 本脚本**故意**原地改写 split_groups.parquet (合并泄漏检查找到的边),
    # 所以必须重新盖章, 否则下游 provenance.require() 会把这次合法修改当成陈旧产物拦下。
    # 上游沿用原来的 (fasta + 簇文件), 并在 extra 里记下修补轮次与并入边数。
    prev = {}
    pp = GROUPS.with_suffix(GROUPS.suffix + ".prov.json")
    if pp.exists():
        try:
            prev = json.loads(pp.read_text())
        except Exception:
            prev = {}
    ups = [i["path"] for i in prev.get("inputs", [])] or [
        "data/interim/pooled_unique_seqs.fasta",
        "data/interim/cluster30/c30_cluster.tsv"]
    extra = dict(prev.get("extra") or {})
    extra["repair_rounds"] = extra.get("repair_rounds", 0) + 1
    extra["edges_merged_total"] = extra.get("edges_merged_total", 0) + n_edges
    provenance.stamp(GROUPS, ups, produced_by="src/splits/repair_leakage.py", extra=extra)
    print(f"[repair] 并入 {n_edges} 条边; split_group {before:,} -> {after:,} (已重新盖章)")
    return n_edges


def run(*cmd: str) -> int:
    print(f"[repair] $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def pending_edges() -> int:
    """当前待并入的边数 (由上一次 check_leakage 落盘的 pairs 文件决定)。"""
    n = 0
    for name in ("sequence", "lab", "time", "bind_target"):
        f = REPORTS / f"leakage_pairs_{name}.parquet"
        if f.exists():
            n += pl.read_parquet(f).height
    return n


def main() -> None:
    max_rounds = 8
    # 先跑一次检查, 把待修补的对落盘。
    # 踩过的坑: 原来第一轮直接 merge_pairs(), 但那时 pairs 文件还没生成 (上一次检查通过
    # 会删掉它们), 于是 n=0, 紧接着的 "n==0 就退出" 守卫把循环打断 —— 检查明明报了
    # 10/1/8 对却一轮都没修。所以顺序必须是: 检查 -> 有对就并 -> 重切 -> 再检查。
    rc = run(str(PY), "src/splits/check_leakage.py", "--threads", "64")
    if rc == 0:
        print("[repair] 初次检查即通过, 无需修补")
        return

    for rnd in range(1, max_rounds + 1):
        print(f"\n===== 修补第 {rnd} 轮 =====")
        pend = pending_edges()
        if pend == 0:
            sys.exit("检查未通过但没有可并的对 —— 人工看 reports/leakage_check.json")
        n = merge_pairs()
        if run(str(PY), "src/splits/make_splits.py") != 0:
            sys.exit("make_splits 失败")
        rc = run(str(PY), "src/splits/check_leakage.py", "--threads", "64")
        if rc == 0:
            print(f"\n[repair] 第 {rnd} 轮后三套切分全部通过泄漏检查")
            return
    sys.exit(f"{max_rounds} 轮仍未收敛, 停下来人工看")


if __name__ == "__main__":
    main()
