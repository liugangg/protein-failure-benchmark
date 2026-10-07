"""中心内 CV vs 跨中心留出 —— 同模型、同任务、**同测试折**, 只换训练来源。

刘刚刚 2026-09-30 指定: 这一项决定论文的结论是
  (A) "去掉身份通道后近乎不可学"  还是
  (B) "有真实信号但被身份通道淹没, 需要中心分层协议"
摘要最后一句取决于此。

设计 (关键是测试折完全相同, 否则差异里混了测试集难度):
  对每个中心 C, 把 C 的 split_group 分成 K 折。对第 i 折:
    within : 训练 = C 的其余 K-1 折           -> 身份通道**不可用** (训练测试同中心)
    cross  : 训练 = 所有中心 ≠ C 的全部数据    -> 身份通道**可用** (模型可认出"这不是我见过的中心")
    测试 = C 的第 i 折 (两者完全相同)
  折的划分以 split_group 为单位, 所以中心内也没有同源泄漏。

读法:
  within ≈ 0.5 lift 而 cross 也 ≈ 0.5  -> (A) 这批数据在去身份通道后近乎不可学
  within 明显 > 1 而 cross ≈ 1         -> (B) 有真实信号, 只是跨中心时被身份通道淹没
  within < cross                        -> 反常, 要查 (可能是中心内样本量不足)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import polars as pl
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval.metrics import evaluate  # noqa: E402
from eval.run_log import write_run_log  # noqa: E402
from eval.caliber import caliber_dir, caliber_meta  # noqa: E402
from labels import provenance  # noqa: E402
from models.baseline_gbdt import featurize  # noqa: E402

RECORDS = pathlib.Path("data/processed/records.parquet")
CONFIG = pathlib.Path("configs/baseline_gbdt.yaml")
OUT_NAME = "within_vs_cross_center.json"
TASKS = [("express", "label_express"),
         ("soluble_expression", "label_soluble_expression"),
         ("purify", "label_purify")]
K = 5
MIN_TEST = 150          # 每折测试侧至少这么多样本
MIN_TEST_POS = 20       # 且至少这么多阴性


def fold_of(group: int, seed: int, k: int) -> int:
    h = hashlib.sha256(f"{seed}:cv:{group}".encode()).digest()
    return int.from_bytes(h[:8], "big") % k


def dedup(d: pl.DataFrame, seed: int) -> pl.DataFrame:
    return (d.with_columns((pl.col("record_id") + pl.lit(f":{seed}")).hash().alias("_h"))
             .sort(["split_group", "_h"]).unique(subset=["split_group"], keep="first")
             .drop("_h"))


def fit_eval(tr: pl.DataFrame, te: pl.DataFrame, col: str, cfg: dict) -> dict | None:
    ytr = (tr[col] == 0).to_numpy().astype(np.int8)
    yte = (te[col] == 0).to_numpy().astype(np.int8)
    if ytr.sum() < 30 or (len(ytr) - ytr.sum()) < 30:
        return None
    if yte.sum() < MIN_TEST_POS or len(yte) < MIN_TEST:
        return None
    m = cfg["model"]
    clf = HistGradientBoostingClassifier(
        max_iter=m["max_iter"], learning_rate=m["learning_rate"],
        max_leaf_nodes=m["max_leaf_nodes"], l2_regularization=m["l2_regularization"],
        early_stopping=m["early_stopping"], validation_fraction=m["validation_fraction"],
        random_state=cfg["seed"])
    clf.fit(featurize(tr["sequence"].to_list(), cfg), ytr)
    sc = clf.predict_proba(featurize(te["sequence"].to_list(), cfg))[:, 1]
    return evaluate(yte, sc, ks=(20, 100))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-center-n", type=int, default=3000)
    ap.add_argument("--keep-tags", dest="strip_tags", action="store_false", default=True,
                    help="消融: 保留构建体残留")
    ap.add_argument("--keep-length", action="store_true",
                    help="消融: 保留序列长度特征")
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    # 口径由命令行控制, 不再硬写 —— 主数字必须用新默认 (SPEC §3.1),
    # 旧口径只作消融。两者都跑才能回答"within>cross 是否只靠标签与长度"。
    cfg["strip_tags"] = bool(args.strip_tags)
    if args.keep_length:
        cfg["features"]["seq_len"] = True
        cfg["features"]["log_seq_len"] = True
    provenance.require([RECORDS])
    rec = pl.read_parquet(RECORDS).filter(pl.col("source") == "ds1_targettrack")
    seed = cfg["seed"]

    centers = (rec.group_by("center").agg(pl.len().alias("n"))
                  .filter(pl.col("n") >= args.min_center_n)
                  .sort("n", descending=True)["center"].to_list())
    print(f"参评中心 {len(centers)} 个 (记录数 >= {args.min_center_n:,})")

    results = []
    for task, col in TASKS:
        obs = rec.filter(pl.col(col) != -1)
        print(f"\n===== 任务 {task} =====")
        print(f"{'center':<10}{'折':>3}{'test_n':>8}{'阴性':>7}"
              f"{'within lift':>13}{'cross lift':>12}{'within PRAUC/base':>19}{'cross PRAUC/base':>18}")
        for ctr in centers:
            inside = dedup(obs.filter(pl.col("center") == ctr), seed)
            outside = dedup(obs.filter(pl.col("center") != ctr), seed)
            if inside.height < K * MIN_TEST:
                continue
            inside = inside.with_columns(
                pl.col("split_group").map_elements(
                    lambda g: fold_of(int(g), seed, K), return_dtype=pl.Int64).alias("_fold"))
            per_fold = []
            for i in range(K):
                te = inside.filter(pl.col("_fold") == i)
                tr_in = inside.filter(pl.col("_fold") != i)
                w = fit_eval(tr_in, te, col, cfg)
                c = fit_eval(outside, te, col, cfg)
                if w is None or c is None:
                    continue
                per_fold.append({"fold": i, "within": w, "cross": c})
                print(f"{ctr:<10}{i:>3}{w['n']:>8}{w['n_positive']:>7}"
                      f"{w['lift@100']:>13.2f}{c['lift@100']:>12.2f}"
                      f"{w['pr_auc_over_base']:>19.2f}{c['pr_auc_over_base']:>18.2f}")
            if not per_fold:
                continue
            agg = {"task": task, "center": ctr, "n_folds": len(per_fold),
                   "test_n_mean": float(np.mean([f["within"]["n"] for f in per_fold])),
                   "base_rate_mean": float(np.mean([f["within"]["base_rate"] for f in per_fold]))}
            for side in ("within", "cross"):
                for met in ("lift@100", "pr_auc", "pr_auc_over_base", "precision@100"):
                    vals = [f[side][met] for f in per_fold if not np.isnan(f[side][met])]
                    agg[f"{side}_{met}_mean"] = round(float(np.mean(vals)), 4) if vals else None
                    agg[f"{side}_{met}_std"] = round(float(np.std(vals)), 4) if vals else None
            results.append(agg)

    # 汇总
    print("\n" + "=" * 92)
    print("汇总 (每个中心 K 折平均; within = 同中心训练, cross = 其它中心训练, 测试折相同)")
    print(f"{'任务':<20}{'center':<10}{'折':>3}{'基础失败率':>11}"
          f"{'within lift':>12}{'cross lift':>11}{'within/cross':>13}")
    for r in results:
        w, c = r["within_lift@100_mean"], r["cross_lift@100_mean"]
        ratio = (w / c) if (w and c) else float("nan")
        print(f"{r['task']:<20}{r['center']:<10}{r['n_folds']:>3}{r['base_rate_mean']:>11.3f}"
              f"{w:>12.2f}{c:>11.2f}{ratio:>13.2f}")

    cmeta = caliber_meta(bool(cfg["strip_tags"]), bool(cfg["features"]["seq_len"]))
    OUT = caliber_dir(bool(cfg["strip_tags"]), bool(cfg["features"]["seq_len"])) / OUT_NAME
    OUT.write_text(json.dumps({"caliber": cmeta, "K": K, "results": results},
                              indent=2, ensure_ascii=False))
    provenance.stamp(OUT, [RECORDS], produced_by="src/eval/within_vs_cross_center.py",
                     extra=cmeta)
    print(f"\nwrote {OUT}  (口径: {cmeta['caliber']})")
    write_run_log("within_vs_cross_center", sources=["ds1_targettrack"], splits_used=[],
                  config={"K": K, "tasks": [t for t, _ in TASKS],
                          "min_center_n": args.min_center_n}, seed=seed,
                  results={f"{r['task']}/{r['center']}":
                           {"within_lift100": r["within_lift@100_mean"],
                            "cross_lift100": r["cross_lift@100_mean"]} for r in results},
                  notes="中心内 CV vs 跨中心留出, 同测试折")


if __name__ == "__main__":
    main()
