"""泄漏检查 (SPEC §3.1 / GATE 2): 训练集与测试集之间不许有 >30% 相似度的序列对。

为什么簇完整性不够、必须真的搜一遍:
  MMseqs2 的级联聚类是贪心 set cover, 它保证"每个成员与自己簇的代表满足判据",
  **不保证**不同簇的成员之间一定低于 30%。所以"按簇切分"只是必要条件。
  这里用 mmseqs search 把训练侧当库、测试侧当 query 真搜一遍, 数越界的对。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[2]
MMSEQS = ROOT / ".tools/mmseqs/bin/mmseqs"
SPLITS = pathlib.Path("data/processed/splits")
RECORDS = pathlib.Path("data/processed/records.parquet")
OUT = pathlib.Path("reports")


def write_fasta(df: pl.DataFrame, path: pathlib.Path) -> int:
    """写出该侧**全部**唯一序列。

    踩过的坑: 原来每个簇只取一条 (unique on cluster_rep), 等于只抽查了簇代表,
    漏掉"A 簇非代表 vs B 簇非代表"这类跨侧高相似对 —— 检查本身给出假通过。
    """
    u = df.select("seq_uid", "cluster_key_seq").unique(subset=["seq_uid"])
    with path.open("w") as fh:
        for uid, seq in u.iter_rows():
            fh.write(f">{uid}\n{seq}\n")
    return u.height


def check(split_name: str, tmp: pathlib.Path, threads: int) -> dict:
    rec = pl.read_parquet(RECORDS)
    sp = pl.read_parquet(SPLITS / f"{split_name}_split.parquet").select(
        "record_id", "split_group", "split")
    d = rec.join(sp, on="record_id", how="inner")

    tr = d.filter(pl.col("split") == "train")
    te = d.filter(pl.col("split") == "test")
    if te.height == 0 or tr.height == 0:
        return {"split": split_name, "error": "一侧为空"}

    # 组级断言: 同一个 split_group 不许同时出现在两侧
    overlap = set(tr["split_group"].unique()) & set(te["split_group"].unique())
    tmp.mkdir(parents=True, exist_ok=True)
    fa_tr, fa_te = tmp / f"{split_name}_train.fa", tmp / f"{split_name}_test.fa"
    n_tr, n_te = write_fasta(tr, fa_tr), write_fasta(te, fa_te)

    res = tmp / f"{split_name}_hits.m8"
    # -s 7.5: 低相似度检索要拉高灵敏度, 否则会漏掉真实的 30% 同源对而误判"无泄漏"。
    # --min-seq-id 0.3 + --alignment-mode 3: 与聚类同一把尺子。
    cmd = [str(MMSEQS), "easy-search", str(fa_te), str(fa_tr), str(res),
           str(tmp / f"{split_name}_stmp"),
           "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "1",
           "--alignment-mode", "3", "-s", "7.5", "-e", "0.001",
           "--threads", str(threads), "-v", "1",
           "--format-output", "query,target,fident,alnlen,evalue"]
    subprocess.run(cmd, check=True)

    n_hits = 0
    pairs: list[tuple[str, str, float]] = []
    pair_file = OUT / f"leakage_pairs_{split_name}.parquet"
    pair_file.unlink(missing_ok=True)
    if res.exists() and res.stat().st_size:
        h = pl.read_csv(res, separator="\t", has_header=False,
                        new_columns=["query", "target", "fident", "alnlen", "evalue"])
        h = h.filter(pl.col("fident") > 0.30)
        n_hits = h.height
        pairs = [(str(a), str(b), float(c))
                 for a, b, c in h.sort("fident", descending=True)
                 .head(10).select("query", "target", "fident").iter_rows()]
        if n_hits:
            # 全部越界对都要落盘, 供 repair_leakage.py 把它们并进 split_group。
            # 只留前 10 个例子的话没法修补, 只能干看着。
            h.select(
                pl.col("query").cast(pl.Int64).alias("seq_uid_a"),
                pl.col("target").cast(pl.Int64).alias("seq_uid_b"),
                pl.col("fident"),
            ).write_parquet(pair_file)

    return {
        "split": split_name,
        "train_sequences": n_tr,
        "test_sequences": n_te,
        "train_clusters": int(tr["cluster_rep"].n_unique()),
        "test_clusters": int(te["cluster_rep"].n_unique()),
        "split_group_overlap": len(overlap),
        "cross_pairs_over_30pct": n_hits,
        "worst_examples": pairs,
        "pairs_file": str(pair_file) if n_hits else None,
        "verdict": "PASS" if (not overlap and n_hits == 0) else "FAIL",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    tmp = ROOT / ".tools/tmp/leakcheck"
    names = ["sequence", "lab", "time"]
    if (SPLITS / "bind_target_split.parquet").exists():
        names.append("bind_target")
    out = [check(n, tmp, args.threads) for n in names]
    OUT.mkdir(exist_ok=True)
    (OUT / "leakage_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if any(o.get("verdict") == "FAIL" for o in out):
        print("\n!! 有切分未通过泄漏检查, 不要拿它跑任何模型")
        sys.exit(1)


if __name__ == "__main__":
    main()
