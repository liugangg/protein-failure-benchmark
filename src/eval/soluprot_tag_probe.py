"""探测 SoluProt 是否也吃 His 标签这条通道 (刘刚刚 2026-09-30 指定, 第 2 项)。

做法: 取同一批序列, 一份保留标签、一份剥掉标签, 都过一遍 SoluProt,
      比较**同一条序列**在两版下的预测位移。
为什么这是全文杀伤力最大的结果: 如果一个已发表、被广泛使用的可溶性预测器
在剥掉亲和标签后预测显著改变, 说明它的预测有相当部分建立在构建体残留上,
而构建体残留是实验室习惯 —— 那么"序列可见的实验室身份通道"就不是我们这套数据的毛病,
而是这个研究领域的系统性问题。

配对设计: 同一条序列的两个版本直接相减, 不需要跨样本比较, 统计效力最高。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import polars as pl
from scipy import stats as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.baseline_gbdt import strip_tags  # noqa: E402
from eval.run_log import write_run_log  # noqa: E402

ENV = ROOT / ".tools/soluprot-env"
RUN = ROOT / ".tools/soluprot-run"
RECORDS = pathlib.Path("data/processed/records.parquet")
OUT = pathlib.Path("reports/soluprot_tag_probe.json")


def run_sp(pairs: list[tuple[int, str]], tag: str, threads: int) -> pl.DataFrame:
    tmp = ROOT / ".tools/tmp/sp_probe" / tag
    tmp.mkdir(parents=True, exist_ok=True)
    fa = tmp / "in.fa"
    with fa.open("w") as fh:
        for uid, s in pairs:
            fh.write(f">{uid}\n{s}\n")
    out_csv = tmp / "out.csv"
    out_csv.unlink(missing_ok=True)
    r = subprocess.run(
        [str(ENV / "bin/python"), "soluprot.py",
         "--i_fa", str(fa.resolve()), "--o_csv", str(out_csv.resolve()),
         "--tmp_dir", str((tmp / "work").resolve()), "--no_tmhmm",
         "--usearch", str((ENV / "bin/usearch").resolve()), "--no_proc", str(threads)],
        cwd=RUN, capture_output=True, text=True)
    if not out_csv.exists():
        print(r.stderr[-1500:])
        raise SystemExit(f"SoluProt 在 {tag} 上没有产出")
    n_miss = r.stderr.count("can not be calculated")
    d = pl.read_csv(out_csv)
    return d.select(pl.col("fa_id").cast(pl.Int64).alias("seq_uid"),
                    pl.col("soluble").cast(pl.Float64).alias(f"sp_{tag}")), n_miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="抽多少条带标签的序列")
    ap.add_argument("--threads", type=int, default=48)
    args = ap.parse_args()

    rec = pl.read_parquet(RECORDS).filter(pl.col("source") == "ds1_targettrack")
    # 只取**确实带标签**的序列 —— 不带标签的剥了等于没剥, 会把位移稀释成 0
    tagged = (rec.filter(pl.col("sequence").str.contains("H{6,}")
                         & (pl.col("seq_len") >= 20))
                 .select("seq_uid", "sequence", "center")
                 .unique(subset=["seq_uid"]))
    print(f"带 His 标签的唯一序列 {tagged.height:,} 条")
    if tagged.height > args.n:
        tagged = tagged.sample(args.n, seed=20260930)
    seqs = tagged["sequence"].to_list()
    uids = tagged["seq_uid"].to_list()
    stripped, n_changed = strip_tags(seqs)
    print(f"抽样 {len(seqs):,} 条; 剥标签实际改动 {n_changed:,} 条")
    dl = np.array([len(a) - len(b) for a, b in zip(seqs, stripped)])
    print(f"剥掉的残基数: 中位 {np.median(dl):.0f}  均值 {dl.mean():.1f}  最大 {dl.max()}")

    keep, m1 = run_sp(list(zip(uids, seqs)), "keep", args.threads)
    strip, m2 = run_sp(list(zip(uids, stripped)), "strip", args.threads)
    print(f"特征缺失警告: 保留版 {m1:,}  剥离版 {m2:,}")

    j = keep.join(strip, on="seq_uid", how="inner")
    a = j["sp_keep"].to_numpy(); b = j["sp_strip"].to_numpy()
    diff = b - a          # 剥后 - 剥前; >0 表示剥掉标签后被判为更可溶
    w = st.wilcoxon(a, b)
    payload = {
        "n_paired": int(j.height),
        "n_stripped": int(n_changed),
        "residues_removed_median": float(np.median(dl)),
        "soluble_score_keep": {"mean": float(a.mean()), "median": float(np.median(a))},
        "soluble_score_strip": {"mean": float(b.mean()), "median": float(np.median(b))},
        "paired_shift": {
            "mean": float(diff.mean()), "median": float(np.median(diff)),
            "std": float(diff.std()),
            "abs_mean": float(np.abs(diff).mean()),
            "frac_increased": float((diff > 0).mean()),
            "frac_changed_gt_0.05": float((np.abs(diff) > 0.05).mean()),
            "frac_changed_gt_0.10": float((np.abs(diff) > 0.10).mean()),
            "wilcoxon_p": float(w.pvalue),
        },
        "feature_missing_warnings": {"keep": m1, "strip": m2},
        "interpretation_note": (
            "SoluProt 的 96 个特征里含氨基酸组成与与 PDB 序列的一致度, 剥掉 His 标签会同时"
            "改变组成 (H 频率) 与长度。所以位移量化的是'标签残留对已发表预测器的影响', "
            "不区分具体是哪个特征通道。"
        ),
    }
    print()
    print(f"=== 配对位移 (n={j.height:,}) ===")
    print(f"  保留标签 可溶分 均值 {a.mean():.4f} 中位 {np.median(a):.4f}")
    print(f"  剥掉标签 可溶分 均值 {b.mean():.4f} 中位 {np.median(b):.4f}")
    print(f"  配对差 (剥后-剥前): 均值 {diff.mean():+.4f} 中位 {np.median(diff):+.4f} "
          f"|差|均值 {np.abs(diff).mean():.4f}")
    print(f"  变大的比例 {(diff>0).mean()*100:.1f}%  |  |差|>0.05 的 {(np.abs(diff)>0.05).mean()*100:.1f}%"
          f"  |差|>0.10 的 {(np.abs(diff)>0.10).mean()*100:.1f}%")
    print(f"  配对 Wilcoxon p = {w.pvalue:.3e}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    write_run_log("soluprot_tag_probe", sources=["ds1_targettrack"], splits_used=[],
                  config={"n": args.n}, seed=20260930,
                  results=payload["paired_shift"],
                  notes="SoluProt 保留 vs 剥离 His 标签的配对预测位移")


if __name__ == "__main__":
    main()
