"""构造"切分组" (split group): 保证切分两侧之间不存在 >30% 相似度序列对的最小单位。

为什么需要它 (2026-09-30 泄漏检查打出来的教训):
  MMseqs2 默认的级联聚类是贪心 set cover, 只保证"成员与自己簇的代表满足 30% 判据",
  **不保证不同簇的成员之间低于 30%**。实测按 set-cover 簇整体划分后, 序列切分的
  训练/测试之间仍有 10,841 对 >30% 的序列对, 最高一对 fident = 1.00 (短序列完整
  包含在长序列里, 贪心分簇时被分到了两个簇)。所以"按簇切分"只是必要条件。

  mmseqs 的 --cluster-mode 1 (连通分量) 能给传递闭包, 但它要求 --single-step-clustering,
  在 35.5 万条序列上是一次全量 all-vs-all, 代价高。这里换个便宜且等效的做法:
  **只在 92,779 条簇代表序列上做 all-vs-all 检索, 把有边相连的簇并成一个 split group。**
  簇内已经保证同源, 簇间的边由这一步补全 -> 组间无 >30% 的对, 组就是安全的切分单位。

输出: data/interim/split_groups.parquet (seq_uid -> cluster_rep -> split_group)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import polars as pl
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from labels import provenance

ROOT = pathlib.Path(__file__).resolve().parents[2]
MMSEQS = ROOT / ".tools/mmseqs/bin/mmseqs"
INTERIM = pathlib.Path("data/interim")
REPS = INTERIM / "cluster30" / "c30_rep_seq.fasta"
ALL_SEQS = INTERIM / "pooled_unique_seqs.fasta"
CLU = INTERIM / "cluster30" / "c30_cluster.tsv"
OUT = INTERIM / "split_groups.parquet"


def load_clusters() -> pl.DataFrame:
    c = pl.read_csv(CLU, separator="\t", has_header=False,
                    new_columns=["cluster_rep", "seq_uid"],
                    schema_overrides={"cluster_rep": pl.Int64, "seq_uid": pl.Int64})
    return (
        c.with_columns((pl.col("cluster_rep") == pl.col("seq_uid")).cast(pl.Int8).alias("_s"))
        .sort(["seq_uid", "_s"]).unique(subset=["seq_uid"], keep="first").drop("_s")
    )


class DSU:
    def __init__(self) -> None:
        self.p: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:          # 路径压缩
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--all-sequences", action="store_true", default=True,
                    help="在全部唯一序列上做 all-vs-all (默认, 唯一严谨的做法)")
    ap.add_argument("--reps-only", dest="all_sequences", action="store_false",
                    help="只搜簇代表 —— 快但**会漏**, 只用于调试")
    ap.add_argument("--sensitivity", type=float, default=7.5,
                    help="低相似度检索必须拉高灵敏度, 否则漏掉真同源对会假装'无泄漏'")
    args = ap.parse_args()

    # 必须在**全部**唯一序列上做 all-vs-all, 不能只搜簇代表。
    # 踩过的坑 (2026-09-30): 先只搜了 92,779 条簇代表, 建出来的组仍然漏 ——
    # 泄漏检查还能找到 2,302 对跨侧 >30% 的序列对 (最高 fident=1.00)。
    # 原因: A 簇的**非代表**成员可能和 B 簇的非代表成员高度相似, 而两个代表之间不相似。
    # 传递闭包必须建在所有成员上。
    # 上游校验: fasta 与簇文件必须都由当前的 pooled_records 生成
    provenance.require([ALL_SEQS])
    provenance.require([CLU])   # cluster30.sh 末尾会盖章
    query = ALL_SEQS if args.all_sequences else REPS
    if not query.exists():
        raise SystemExit(f"{query} 不存在, 先跑 src/labels/pool.py 和 cluster30.sh")

    tmp = ROOT / ".tools/tmp/splitgroups"
    tmp.mkdir(parents=True, exist_ok=True)
    hits = tmp / "allseq_vs_allseq.m8"
    cmd = [str(MMSEQS), "easy-search", str(query), str(query), str(hits), str(tmp / "stmp"),
           "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "1",
           "--alignment-mode", "3", "-s", str(args.sensitivity), "-e", "0.001",
           "--threads", str(args.threads), "-v", "1",
           "--format-output", "query,target,fident"]
    n_q = sum(1 for line in query.open() if line.startswith(">"))
    print(f"[split_groups] all-vs-all 检索中 ({n_q:,} 条序列)…")
    # 先删掉上一次的结果文件。否则这次 mmseqs 挂掉时会**静默读到上一轮的旧结果**,
    # 建出一套看起来正常、实际错误的切分组 —— 这种错不会报任何异常。
    hits.unlink(missing_ok=True)
    subprocess.run(cmd, check=True)
    if not hits.exists():
        raise SystemExit("mmseqs 没有产出结果文件, 不要继续")

    h = pl.read_csv(hits, separator="\t", has_header=False,
                    new_columns=["q", "t", "fident"],
                    schema_overrides={"q": pl.Int64, "t": pl.Int64})
    h = h.filter((pl.col("fident") > 0.30) & (pl.col("q") != pl.col("t")))
    print(f"[split_groups] 簇间 >30% 的边 {h.height:,} 条")

    clus = load_clusters()
    dsu = DSU()
    for rep in clus["cluster_rep"].unique():
        dsu.find(int(rep))
    # 先把簇内成员并起来 (簇本身就是同源的)
    for rep, m in clus.select("cluster_rep", "seq_uid").iter_rows():
        dsu.union(int(rep), int(m))
    # 再把检索到的跨序列相似边并起来
    for q, tt in h.select("q", "t").iter_rows():
        dsu.union(int(q), int(tt))

    uids = clus["seq_uid"].to_list()
    gmap = pl.DataFrame(
        {"seq_uid": uids, "split_group": [dsu.find(int(u)) for u in uids]},
        schema={"seq_uid": pl.Int64, "split_group": pl.Int64},
    )
    out = clus.join(gmap, on="seq_uid", how="left")
    out.write_parquet(OUT)
    provenance.stamp(OUT, [ALL_SEQS, CLU], produced_by="src/splits/split_groups.py")

    sizes = out.group_by("split_group").agg(pl.col("cluster_rep").n_unique().alias("k"))
    print(f"[split_groups] 序列 {out.height:,} -> 簇 {out['cluster_rep'].n_unique():,} "
          f"-> 切分组 {out['split_group'].n_unique():,}")
    stats = {
        "set_cover_clusters": int(clus["cluster_rep"].n_unique()),
        "split_groups": int(out["split_group"].n_unique()),
        "inter_cluster_edges_over_30pct": h.height,
        "largest_group_clusters": int(sizes["k"].max()),
        "singleton_groups": int((sizes["k"] == 1).sum()),
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    pathlib.Path("data/interim/split_groups.stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
