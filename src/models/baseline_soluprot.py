"""SoluProt 基线 (SPEC v2 §3.2) —— 在我们冻结的测试集上跑, 全集 + 去污染子集两版。

刘刚刚 2026-09-30 的硬要求已全部前置完成:
  检查一 (定义对齐): 过万的是 `express` (17,699 簇) 不是 `soluble` (3,447 簇)。
    SoluProt 预测的是"可溶表达"这一**单个**二元事件, 对应我们的派生复合标签
    `label_soluble_expression` (= express 成功 且 soluble 成功), 19,007 个阴性簇。
    拿它比我们的 express 头或 soluble 头都是在比两件不同的事, 见 labels/schema.py 的说明。
    本脚本**默认比复合标签**, 同时给出 express 与 soluble 两个单头的数字作对照
    (soluble 单头标 underpowered: 它的阴性 99.9% 来自 v2 §2.4 推断)。
  检查二 (训练集污染): SoluProt 官方 about 页写明训练集来自 TargetTrack, 与我们 DS1 同源。
    实测 30% 相似度污染率: 序列切分 12.4%, 实验室切分 10.3%, 时间 3.8%, bind 0%。
    因此**每个数字都给全集与干净子集两版**, 只给全集的无效。
    见 reports/soluprot_contamination.json。

安装与已知偏差 (必须随数字一起报):
  官方 standalone v1.0.1.0 + 官方 conda 环境 (py3.7 / sklearn 0.20.1 / biopython 1.74)。
  USEARCH 用 bioconda 的 12.0_beta 预编译包 (GPL-3.0, 零编译)。
  **打了一处兼容补丁**: v12 取消了 `-search_global`, 换成注册名 `-usearch_global`,
  除此之外未改 SoluProt 任何逻辑。用 TMHMM 时需要注册申请, 故走官方支持的 `--no_tmhmm`
  (官方文档说约 -0.5% 准确率, 有专门的 notmhmm 模型)。
  在官方自带 data/test.fa 上复现其参考输出: 21 条序列平均绝对差 0.0374, 最大 0.1613。
  差异同时来自 --no_tmhmm 与 usearch 版本, **无法分离**。因此本基线只用
  **排序类指标 (precision@k / PR-AUC)**, 不用绝对阈值下的 accuracy —— 排序对小幅分数
  漂移不敏感, 而 accuracy 会。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval.metrics import evaluate  # noqa: E402
from eval.run_log import write_run_log  # noqa: E402
from labels import provenance  # noqa: E402

ENV = ROOT / ".tools/soluprot-env"
RUN = ROOT / ".tools/soluprot-run"
RECORDS = pathlib.Path("data/processed/records.parquet")
SPLITS = pathlib.Path("data/processed/splits")
CONTAM = pathlib.Path("reports/soluprot_contamination.json")
OUT = pathlib.Path("reports/baseline_soluprot.json")

# SoluProt 输出的是"可溶概率", 我们的阳性是失败 -> 打分取负
TASKS = [
    ("soluble_expression", "label_soluble_expression", "对齐口径 (SoluProt 的定义)"),
    ("express", "label_express", "对照: 我们的 express 单头 (定义更窄)"),
    ("soluble", "label_soluble", "对照: 我们的 soluble 单头 (underpowered, 阴性 99.9% 来自推断)"),
]


def run_soluprot(seqs: pl.DataFrame, tag: str, threads: int) -> pl.DataFrame:
    tmp = ROOT / ".tools/tmp/soluprot" / tag
    tmp.mkdir(parents=True, exist_ok=True)
    fa = tmp / "in.fa"
    with fa.open("w") as fh:
        for uid, s in seqs.select("seq_uid", "sequence").iter_rows():
            fh.write(f">{uid}\n{s}\n")
    out_csv = tmp / "out.csv"
    out_csv.unlink(missing_ok=True)
    cmd = [str(ENV / "bin/python"), "soluprot.py",
           "--i_fa", str(fa.resolve()), "--o_csv", str(out_csv.resolve()),
           "--tmp_dir", str((tmp / "work").resolve()),
           "--no_tmhmm", "--usearch", str((ENV / "bin/usearch").resolve()),
           "--no_proc", str(threads)]
    print(f"[soluprot] 跑 {seqs.height:,} 条序列 ({tag}) …")
    r = subprocess.run(cmd, cwd=RUN, capture_output=True, text=True)
    if not out_csv.exists():
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit("SoluProt 没有产出结果")
    nwarn = r.stderr.count("can not be calculated")
    if nwarn:
        print(f"[soluprot] 有 {nwarn} 个特征值缺失 (usearch 无命中), SoluProt 用训练集均值填补")
    d = pl.read_csv(out_csv)
    return d.select(pl.col("fa_id").cast(pl.Int64).alias("seq_uid"),
                    pl.col("soluble").cast(pl.Float64).alias("soluprot_soluble"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["lab", "sequence"])
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    contam = json.loads(CONTAM.read_text()) if CONTAM.exists() else {"splits": {}}
    provenance.require([RECORDS])
    rec = pl.read_parquet(RECORDS)
    results: list[dict] = []

    for name in args.splits:
        p = SPLITS / f"{name}_split.parquet"
        if not p.exists():
            print(f"[skip] {name}")
            continue
        sp = pl.read_parquet(p).select("record_id", "split")
        d = rec.join(sp, on="record_id", how="inner").filter(pl.col("split") == "test")
        uniq = d.select("seq_uid", "sequence").unique(subset=["seq_uid"])
        # SoluProt 硬限制: 最短 20 aa (官方 web 与 standalone 都是)。
        # 我们的测试集里有更短的序列 (GATE 1 报告 §5 已标出长度异常), 必须显式剔除并记数,
        # 否则 SoluProt 直接拒跑整批。这部分样本**不参与 SoluProt 的对比**,
        # 所以对比表里的 n 会小于该切分测试侧的总数 —— 这是口径差异, 不是数据丢失。
        n_all = uniq.height
        uniq = uniq.filter(pl.col("sequence").str.len_chars() >= 20)
        n_short = n_all - uniq.height
        if n_short:
            print(f"[soluprot] {name}: 剔除 <20 aa 的序列 {n_short} 条 "
                  f"(SoluProt 硬限制), 余 {uniq.height:,} 条")
        pred = run_soluprot(uniq, name, args.threads)

        clean_f = SPLITS / f"{name}_soluprot_clean_seq_uids.parquet"
        # 陈旧产物防护改用通用机制 (src/labels/provenance.py):
        # clean 子集按 seq_uid 存, records.parquet 重建后 seq_uid 会变, 两边对不上且**不报错**
        # (实测交集从 2,377 静默缩到 289)。require() 会校验 clean 文件记录的上游指纹。
        if clean_f.exists():
            provenance.require([clean_f])
        clean_ids = (set(pl.read_parquet(clean_f)["seq_uid"].to_list())
                     if clean_f.exists() else None)
        cr = contam["splits"].get(name, {}).get("contamination_rate_30pct")

        for task, col, note in TASKS:
            base = d.join(pred, on="seq_uid", how="inner").filter(pl.col(col) != -1)
            # 去冗余: 每个 split_group 一条, 与其它基线同口径同种子
            base = (base.with_columns((pl.col("record_id") + pl.lit(":20260930")).hash().alias("_h"))
                        .sort(["split_group", "_h"])
                        .unique(subset=["split_group"], keep="first"))
            for variant in ("full", "clean"):
                sub = base
                if variant == "clean":
                    if clean_ids is None:
                        continue
                    sub = sub.filter(pl.col("seq_uid").is_in(list(clean_ids)))
                if sub.height < 200:
                    results.append({"split": name, "task": task, "variant": variant,
                                    "skipped": f"样本不足 ({sub.height})"})
                    continue
                y = (sub[col] == 0).to_numpy().astype(np.int8)     # 阳性 = 失败
                if y.sum() < 20:
                    results.append({"split": name, "task": task, "variant": variant,
                                    "skipped": f"阴性不足 ({int(y.sum())})"})
                    continue
                score = -sub["soluprot_soluble"].to_numpy()        # 可溶概率低 -> 更可能失败
                m = evaluate(y, score, ks=(20, 100))
                results.append({
                    "split": name, "task": task, "task_note": note, "variant": variant,
                    "contamination_rate_30pct": cr,
                    "excluded_shorter_than_20aa": n_short, **m,
                })
                print(f"  {name:9s} {task:19s} {variant:5s} n={m['n']:6d} "
                      f"fail={m['n_fail']:5d} base={m['base_rate']:.3f} "
                      f"P@20={m['precision@20']:.3f} P@100={m['precision@100']:.3f} "
                      f"lift@100={m['lift@100']:.2f} PR-AUC={m['pr_auc']:.3f}")

    payload = {
        "install": {
            "soluprot_version": "1.0.1.0 (官方 standalone)",
            "usearch": "bioconda 12.0_beta (GPL-3.0 预编译)",
            "compat_patch": "-search_global -> -usearch_global (v12 取消了前者), 仅此一处",
            "tmhmm": "未用 (走官方 --no_tmhmm, 因 TMHMM 需注册申请)",
            "official_test_reproduction": {"n": 21, "mean_abs_diff": 0.0374, "max_abs_diff": 0.1613},
        },
        "metric_choice_note": (
            "只用排序类指标 (precision@k / PR-AUC), 不用绝对阈值下的 accuracy: "
            "我们的分数与官方参考值有 0.037 量级的漂移 (来源 --no_tmhmm + usearch 版本, 无法分离), "
            "排序对此不敏感, accuracy 会。"
        ),
        "contamination": {k: v.get("contamination_rate_30pct")
                          for k, v in contam["splits"].items()},
        "results": results,
    }
    OUT.parent.mkdir(exist_ok=True)
    # 合并而不是覆盖: 本脚本常按切分分批跑 (sequence 那批 16k 条要跑十几分钟),
    # 原来每次 write 覆盖整个文件, 第二批会把第一批的结果冲掉 —— 实测踩到,
    # GATE 2 报告里少了整个实验室切分的行。
    if OUT.exists():
        try:
            old_payload = json.loads(OUT.read_text())
            kept = [r for r in old_payload.get("results", [])
                    if r.get("split") not in set(args.splits)]
            if kept:
                print(f"[soluprot] 保留此前已跑的 {len(kept)} 条结果 "
                      f"(切分: {sorted({r.get('split') for r in kept})})")
            payload["results"] = kept + payload["results"]
        except Exception as e:
            print(f"[soluprot] 旧结果读取失败, 按覆盖处理: {e}")
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    write_run_log("baseline_soluprot", sources=sorted(rec["source"].unique().to_list()),
                  splits_used=args.splits, config=payload["install"], seed=None,
                  results={f"{r.get('split')}/{r.get('task')}/{r.get('variant')}":
                           (r.get("skipped") or {"lift@100": r.get("lift@100"),
                                                 "pr_auc": r.get("pr_auc")})
                           for r in results},
                  notes="SoluProt 基线; 全集与去污染子集两版")


if __name__ == "__main__":
    main()
