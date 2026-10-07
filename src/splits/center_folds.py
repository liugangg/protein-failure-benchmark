"""按中心分组的折 (GroupKFold by center) —— 本文的主评估设定。

── 两个设计决定, 都是被第一版的失败逼出来的 ──

**1. 排除是"按折"而不是"全局"。**
   第一版把任何跨折的 split_group 全局排除, 结果排掉 76% 的记录 (729,627 条),
   而且某折只剩 231 个 express 阴性簇。
   正确做法: 对第 j 折,
     test_j  = center ∈ fold_j 的记录
     train_j = center ∉ fold_j **且** 其 split_group 不含任何 center ∈ fold_j 的记录
   这样每折只损失"触碰该折留出中心"的同源组, 这些组在测试别的折时照样可用。
   同时两个性质都保住:
     - 无同源泄漏: split_group 互不相交, 同源性封闭在组内
     - 留出中心纯净: 训练侧完全不含留出中心的任何记录

**2. 中心到折的分配用"贪心均衡"而不是哈希。**
   哈希分配让折 4 独占 JCSG+NESG+NYSGXRC 三个最大中心 (147k 条 vs 其余 9-22k)。
   改为按中心大小降序、每次放进当前最小的折 —— 确定性, 且折间量级可比。

产物: data/processed/splits/center_folds.parquet 存 **center -> fold 映射** (不存逐记录的
排除标记, 因为排除是按折算的)。L0/L1/L2/L3 全部读同一份, 不允许任何模型重切。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import polars as pl
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from labels import provenance  # noqa: E402

RECORDS = pathlib.Path("data/processed/records.parquet")
OUT = pathlib.Path("data/processed/splits/center_folds.parquet")
CONFIG = pathlib.Path("configs/stage3_train.yaml")


def greedy_balance(sizes: list[tuple[str, int]], k: int) -> dict[str, int]:
    """按大小降序, 每个中心放进当前总量最小的折。确定性。"""
    load = [0] * k
    out: dict[str, int] = {}
    for name, n in sorted(sizes, key=lambda t: (-t[1], t[0])):
        j = min(range(k), key=lambda i: (load[i], i))
        out[name] = j
        load[j] += n
    return out


def fold_masks(d: pl.DataFrame, fold_of: dict[str, int], j: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """返回 (train_j, test_j)。见模块 docstring 的定义。"""
    held = [c for c, f in fold_of.items() if f == j]
    is_held = pl.col("center").is_in(held)
    test = d.filter(is_held)
    # 触碰留出中心的 split_group, 整组从训练侧剔除 (防同源泄漏)
    touched = d.filter(is_held)["split_group"].unique()
    train = d.filter(~is_held & ~pl.col("split_group").is_in(touched))
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-center-n", type=int, default=1000)
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    seed, k = cfg["seed"], cfg["eval"]["n_center_folds"]

    provenance.require([RECORDS])
    rec = pl.read_parquet(RECORDS).filter(pl.col("source") == "ds1_targettrack")

    cc = (rec.group_by("center").agg(pl.len().alias("n")).sort("n", descending=True))
    big = [(c, n) for c, n in cc.iter_rows() if n >= args.min_center_n]
    small = [c for c, n in cc.iter_rows() if n < args.min_center_n]
    print(f"参与折划分的中心 {len(big)} 个; 记录数 <{args.min_center_n:,} 的 {len(small)} 个 "
          f"中心只进训练侧、不做测试")

    fold_of = greedy_balance(big, k)
    mapping = pl.DataFrame(
        {"center": list(fold_of), "center_fold": [fold_of[c] for c in fold_of]},
        schema={"center": pl.Utf8, "center_fold": pl.Int8})
    # 小中心标 -1: 永远在训练侧
    mapping = pl.concat([mapping, pl.DataFrame(
        {"center": small, "center_fold": [-1] * len(small)},
        schema={"center": pl.Utf8, "center_fold": pl.Int8})], how="vertical")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_parquet(OUT)
    provenance.stamp(OUT, [RECORDS], produced_by="src/splits/center_folds.py",
                     extra={"k": k, "seed": seed, "min_center_n": args.min_center_n,
                            "assignment": "greedy_balance_by_size",
                            "exclusion": "per_fold (不是全局)"})

    d = rec.join(mapping, on="center", how="left")
    print(f"\n{'折':>3}{'中心数':>7}{'test记录':>10}{'train记录':>11}{'train损失%':>11}  中心")
    summary = []
    total = d.height
    for j in range(k):
        tr, te = fold_masks(d, fold_of, j)
        # 若不做同源排除, 训练侧本应有多少
        naive = d.filter(~pl.col("center").is_in([c for c, f in fold_of.items() if f == j])).height
        lost = (naive - tr.height) / naive * 100 if naive else 0.0
        cs = sorted([c for c, f in fold_of.items() if f == j])
        row = {"fold": j, "centers": cs, "n_test": te.height, "n_train": tr.height,
               "train_loss_pct_from_homology_exclusion": round(lost, 1)}
        for col in ("label_express", "label_soluble_expression"):
            row[f"test_neg_groups_{col.replace('label_','')}"] = int(
                te.filter(pl.col(col) == 0)["split_group"].n_unique())
            row[f"train_neg_groups_{col.replace('label_','')}"] = int(
                tr.filter(pl.col(col) == 0)["split_group"].n_unique())
        # 可评测性判定 (SPEC §3.1 的 min_test_n / min_test_positives):
        # 折 0 = JCSG 单独一折, 而 JCSG 在全库 378,364 条记录里**不记录任何失败**
        # (见 reports/gate1_data_inventory.md §2b), 所以它的 test 侧阴性为 0, 无法评测。
        # 必须显式标出来, 不能让一个 0 阴性的折悄悄混进平均值。
        row["evaluable_express"] = bool(
            te.height >= cfg["eval"]["min_test_n"]
            and row["test_neg_groups_express"] >= cfg["eval"]["min_test_positives"])
        row["evaluable_soluble_expression"] = bool(
            te.height >= cfg["eval"]["min_test_n"]
            and row["test_neg_groups_soluble_expression"] >= cfg["eval"]["min_test_positives"])
        if not row["evaluable_express"]:
            row["not_evaluable_reason"] = (
                f"test 侧 express 阴性簇仅 {row['test_neg_groups_express']} 个 "
                f"(门槛 {cfg['eval']['min_test_positives']}); "
                f"留出中心 {', '.join(cs)} 不记录失败"
            )
        summary.append(row)
        print(f"{j:>3}{len(cs):>7}{te.height:>10,}{tr.height:>11,}{lost:>10.1f}%  "
              f"{', '.join(cs)}")

    print(f"\n各折的阴性簇数 (test / train):")
    print(f"{'折':>3}{'express test':>14}{'express train':>15}"
          f"{'sol_expr test':>15}{'sol_expr train':>16}")
    for r in summary:
        print(f"{r['fold']:>3}{r['test_neg_groups_express']:>14,}"
              f"{r['train_neg_groups_express']:>15,}"
              f"{r['test_neg_groups_soluble_expression']:>15,}"
              f"{r['train_neg_groups_soluble_expression']:>16,}")

    pathlib.Path("reports/center_folds.json").write_text(json.dumps(
        {"k": k, "seed": seed, "assignment": "greedy_balance_by_size",
         "exclusion_policy": "per_fold",
         "centers_train_only": small, "folds": summary},
        indent=2, ensure_ascii=False))
    ev = [r["fold"] for r in summary if r["evaluable_express"]]
    nv = [r["fold"] for r in summary if not r["evaluable_express"]]
    print(f"\n可评测的折 (express): {ev}")
    if nv:
        print(f"不可评测: {nv}")
        for r in summary:
            if not r["evaluable_express"]:
                print(f"   折 {r['fold']}: {r['not_evaluable_reason']}")
        print("   -> 训练循环必须跳过这些折, 且报告里写明是哪些中心、为什么。")
    print(f"\nwrote {OUT} 和 reports/center_folds.json")


if __name__ == "__main__":
    main()
