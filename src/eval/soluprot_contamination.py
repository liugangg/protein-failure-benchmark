"""检查 SoluProt 训练集与我们冻结测试集的污染程度。

刘刚刚 2026-09-30 的硬要求:
  "取 SoluProt 的训练集, 与我们冻结的三套测试集做 30% 相似度比对, 报告重叠数量。
   若有重叠, 额外在去重叠子集上跑一遍对比。**这一步不做完, SoluProt 的数字不许写进报告。**"

为什么必须做: SoluProt 官方 about 页原文 —— "The training set is based on the
TargetTrack database, which was carefully filtered to keep only targets expressed
in Escherichia coli."  它和我们的 DS1 同源, 重叠不是可能性问题而是程度问题。
如果不扣掉重叠就比, SoluProt 是在它自己见过的数据上被评估, 对比无效。

判据与我们自己的去冗余完全一致 (--min-seq-id 0.3 -c 0.8 --cov-mode 1
--alignment-mode 3 -s 7.5), 否则两套口径不可比。
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import polars as pl
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from labels import provenance

ROOT = pathlib.Path(__file__).resolve().parents[2]
MMSEQS = ROOT / ".tools/mmseqs/bin/mmseqs"
SP_DIR = pathlib.Path("data/interim/extracted/soluprot/soluprot_data")
RECORDS = pathlib.Path("data/processed/records.parquet")
SPLITS = pathlib.Path("data/processed/splits")
OUT = pathlib.Path("reports/soluprot_contamination.json")
SPLIT_NAMES = ["sequence", "lab", "time", "bind_target"]


def read_fasta(p: pathlib.Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name, buf = None, []
    for ln in p.read_text().split("\n"):
        if ln.startswith(">"):
            if name and buf:
                seqs[name] = "".join(buf).upper()
            name, buf = ln[1:].strip(), []
        else:
            buf.append(ln.strip())
    if name and buf:
        seqs[name] = "".join(buf).upper()
    return seqs


def main() -> None:
    tr_fa = SP_DIR / "training_set.fasta"
    if not tr_fa.exists():
        raise SystemExit(f"{tr_fa} 不存在; 先从 loschmidt.chemi.muni.cz/soluprot/?page=download 取 soluprot_data.zip")
    tr = read_fasta(tr_fa)
    te_sp = read_fasta(SP_DIR / "test_set.fasta")
    print(f"[soluprot] 训练集 {len(tr):,} 条, 其测试集 {len(te_sp):,} 条")

    tmp = ROOT / ".tools/tmp/soluprot_contam"
    tmp.mkdir(parents=True, exist_ok=True)
    sp_fa = tmp / "soluprot_train.fa"
    with sp_fa.open("w") as fh:
        for i, s in enumerate(tr.values()):
            fh.write(f">sp{i}\n{s}\n")

    provenance.require([RECORDS])
    rec = pl.read_parquet(RECORDS)
    tr_set, te_sp_set = set(tr.values()), set(te_sp.values())
    results: dict[str, dict] = {}

    for name in SPLIT_NAMES:
        p = SPLITS / f"{name}_split.parquet"
        if not p.exists():
            continue
        sp = pl.read_parquet(p).select("record_id", "split")
        d = rec.join(sp, on="record_id", how="inner").filter(pl.col("split") == "test")
        uniq = d.select("seq_uid", "sequence").unique(subset=["seq_uid"])
        our_fa = tmp / f"{name}_test.fa"
        with our_fa.open("w") as fh:
            for uid, s in uniq.iter_rows():
                fh.write(f">{uid}\n{s}\n")

        hits = tmp / f"{name}_hits.m8"
        hits.unlink(missing_ok=True)
        subprocess.run(
            [str(MMSEQS), "easy-search", str(our_fa), str(sp_fa), str(hits),
             str(tmp / f"{name}_stmp"),
             "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "1",
             "--alignment-mode", "3", "-s", "7.5", "-e", "0.001",
             "--threads", "48", "-v", "1",
             "--format-output", "query,target,fident"],
            check=True,
        )
        contaminated: set[int] = set()
        if hits.exists() and hits.stat().st_size:
            h = pl.read_csv(hits, separator="\t", has_header=False,
                            new_columns=["q", "t", "fident"],
                            schema_overrides={"q": pl.Int64})
            h = h.filter(pl.col("fident") > 0.30)
            contaminated = set(h["q"].unique().to_list())

        ours = set(uniq["sequence"].to_list())
        results[name] = {
            "test_unique_sequences": int(uniq.height),
            "exact_match_in_soluprot_train": len(ours & tr_set),
            "exact_match_in_soluprot_test": len(ours & te_sp_set),
            "similar_30pct_to_soluprot_train": len(contaminated),
            "contamination_rate_30pct": round(len(contaminated) / max(uniq.height, 1), 4),
            "clean_subset_size": int(uniq.height) - len(contaminated),
            "contaminated_seq_uids_file": None,
        }
        # 干净子集的 seq_uid 落盘, 供"去重叠后重跑对比"用
        clean = uniq.filter(~pl.col("seq_uid").is_in(list(contaminated))) if contaminated else uniq
        cp = pathlib.Path(f"data/processed/splits/{name}_soluprot_clean_seq_uids.parquet")
        clean.select("seq_uid").write_parquet(cp)
        provenance.stamp(cp, [RECORDS, SPLITS / f"{name}_split.parquet"],
                         produced_by="src/eval/soluprot_contamination.py")
        results[name]["clean_subset_file"] = str(cp)
        r = results[name]
        print(f"  {name:12s} test 唯一序列 {r['test_unique_sequences']:>7,} | "
              f"完全相同 {r['exact_match_in_soluprot_train']:>5,} | "
              f"30% 相似 {r['similar_30pct_to_soluprot_train']:>7,} "
              f"({r['contamination_rate_30pct']*100:5.1f}%) | "
              f"干净子集 {r['clean_subset_size']:>7,}")

    payload = {
        "soluprot_train_n": len(tr),
        "soluprot_test_n": len(te_sp),
        "soluprot_train_source_quote": (
            "SoluProt 官方 about 页: \"The training set is based on the TargetTrack "
            "database, which was carefully filtered to keep only targets expressed in "
            "Escherichia coli.\" —— 与我们的 DS1 同源。"
        ),
        "criteria": "--min-seq-id 0.3 -c 0.8 --cov-mode 1 --alignment-mode 3 -s 7.5",
        "splits": results,
        "verdict_note": (
            "污染率 >5% 的切分, SoluProt 的对比数字必须同时给全集和干净子集两版; "
            "只给全集的数字无效 (SoluProt 在自己见过的数据上被评估)。"
        ),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
