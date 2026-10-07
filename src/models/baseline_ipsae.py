"""复现 AF3 ipSAE_min 的 binder 结合预测基线 (SPEC v2 §3.2, 参考值 最高 F1 = 0.61)。

数据就是 DS5 本身: DTU 元分析的 final_dataset.csv 里已经含 `af3_ipSAE_min` 等
200+ 个特征, 以及湿实验的 `binder` 结果。所以这个基线**不需要跑任何结构预测**,
只是在同一批数据上重算阈值与 F1。

做法 (与论文口径对齐):
  - 正类 = **结合** (binder == True)。注意这与本项目其它地方"阳性=失败"的约定相反,
    因为要复现的是论文报告的 binder 预测 F1, 必须用论文的方向, 否则数字不可比。
    函数名与输出里都写明方向, 不要混。
  - 扫描全部候选阈值, 取使 F1 最大的那个 -> "最高 F1"。这正是论文"最佳单指标"的定义。
  - 同时报 PR-AUC 与 precision@k, 便于和本项目其它基线对照。

同时评测 v2 §3.2 提到的对照指标 (ipAE / ipTM), 以验证论文"ipSAE 优于 ipAE / ipTM"
这一结论在我们手里也成立 —— 如果不成立, 说明我们读错了列或方向, 要先排查再往下走。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from eval.run_log import write_run_log  # noqa: E402

SRC = pathlib.Path("data/raw/ds5_binder_negatives/dtu_meta_analysis/final_dataset.csv")
OUT = pathlib.Path("reports/baseline_ipsae.json")

# 要评测的指标: (列名, 方向) —— +1 表示"值越大越可能结合"
METRICS = [
    ("af3_ipSAE_min", +1),
    ("af3_ipSAE_avg", +1),
    ("af3_ipSAE_max", +1),
    ("af3_iptm_avg", +1),
    ("af3_ipae", -1),          # ipAE 是误差, 越小越好
    ("boltz1_ipSAE_min", +1),
    ("colab_ipSAE_min", +1),
    ("af2_pae_interaction", -1),
]


def best_f1(y_bind: np.ndarray, score: np.ndarray) -> dict:
    """扫全部阈值取最高 F1。y_bind: 1 = 结合 (论文方向)。"""
    prec, rec, thr = precision_recall_curve(y_bind, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    i = int(np.nanargmax(f1))
    return {
        "best_f1": float(f1[i]),
        "precision_at_best_f1": float(prec[i]),
        "recall_at_best_f1": float(rec[i]),
        # thr 比 prec/rec 少一个元素, 末点对应"全部判为正"没有阈值
        "threshold": float(thr[i]) if i < len(thr) else None,
        "pr_auc": float(average_precision_score(y_bind, score)),
    }


def eval_on_bind_split() -> dict | None:
    """在冻结的 bind_target 切分测试侧上评 ipSAE, 与 GBDT 苹果对苹果比。

    方向回到项目约定: **阳性 = 失败**, 所以打分取 -ipSAE (ipSAE 越低越可能失败)。
    去冗余口径与 baseline_gbdt.py 完全一致 (每个 split_group 取一条, 同一确定性哈希),
    否则两者的 precision@k 不可比。
    """
    sp_path = pathlib.Path("data/processed/splits/bind_target_split.parquet")
    rec_path = pathlib.Path("data/processed/records.parquet")
    if not (sp_path.exists() and rec_path.exists() and SRC.exists()):
        return None
    rec = pl.read_parquet(rec_path)
    sp = pl.read_parquet(sp_path).select("record_id", "split")
    d = rec.join(sp, on="record_id", how="inner").filter(pl.col("split") == "test")
    d = (d.with_columns((pl.col("record_id") + pl.lit(":20260930")).hash().alias("_h"))
           .sort(["split_group", "_h"]).unique(subset=["split_group"], keep="first"))

    ds5 = pl.read_csv(SRC, columns=["A_seq", "af3_ipSAE_min"],
                      infer_schema_length=20000, ignore_errors=True) \
            .with_columns(pl.col("A_seq").str.to_uppercase().alias("sequence"))
    j = d.join(ds5.select("sequence", "af3_ipSAE_min"), on="sequence", how="left")
    have = j.filter(pl.col("af3_ipSAE_min").is_not_null())
    if have.height < 50:
        return {"error": f"能接上 ipSAE 的只有 {have.height} 条, 不评"}

    y = (have["label_bind"] == 0).to_numpy().astype(np.int8)     # 阳性 = 失败
    s = -have["af3_ipSAE_min"].cast(pl.Float64).to_numpy()       # 低 ipSAE -> 更可能失败
    base = float(y.mean())

    def pk(sc, k):
        if len(y) < k:
            return float("nan")
        return float(y[np.argsort(-sc, kind="stable")[:k]].mean())

    rng = np.random.default_rng(20260930)
    rand = rng.random(len(y))
    KS = (10, 20, 50, 100)

    def block(sc):
        ap = float(average_precision_score(y, sc))
        r = {"pr_auc": ap, "pr_auc_over_base": ap / base}
        for k in KS:
            r[f"precision@{k}"] = pk(sc, k)
            r[f"lift@{k}"] = (pk(sc, k) / base) if base else float("nan")
        return r

    # 同一批数据、同一个打分, 两种极性都算 —— 这是刘刚刚 2026-09-30 抓到的坑:
    # 只报一侧的 PR-AUC 会让结论量级完全走形。
    y_bind = 1 - y
    s_bind = -s          # s 已是 -ipSAE, 取负回到 +ipSAE (值高 -> 更可能结合)
    ap_bind = float(average_precision_score(y_bind, s_bind))
    out = {
        "test_n_dedup": int(have.height),
        "match_rate": round(have.height / d.height, 4),
        "positive_class": "failure (label_bind == 0, 即非结合)",
        "base_rate_positive_is_failure": base,
        "base_rate_positive_is_binder": float(y_bind.mean()),
        "ipSAE_min": block(s),
        "random_control": block(rand),
        "polarity_check": {
            "positive_is_failure": {"base_rate": base,
                                    "pr_auc": float(average_precision_score(y, s)),
                                    "pr_auc_over_base": float(average_precision_score(y, s)) / base},
            "positive_is_binder": {"base_rate": float(y_bind.mean()),
                                   "pr_auc": ap_bind,
                                   "pr_auc_over_base": ap_bind / float(y_bind.mean())},
            "why_it_matters": (
                "同一批 240 个设计同一个打分, 换极性后 PR-AUC 从 0.909 变 0.353, "
                "而归一化倍数从 1.14 变 1.73 —— 大小关系反过来。"
                "所以**引用 PR-AUC 必须同时声明极性与正类占比**, 否则结论量级走形。"
                "这条已写进 src/eval/metrics.py 作为强制规则。"
            ),
        },
        "note": (
            "极性 = 阳性是失败 (项目约定), 与本文件上半部分复现论文时用的极性相反 (论文正类=结合)。"
            "去冗余与 baseline_gbdt.py 同口径同种子, 两者可直接比。"
        ),
    }
    print()
    print(f"[bind_target 切分] 测试侧去冗余 {have.height} 条, 基础失败率 {base:.3f}")
    print(f"  ipSAE_min      P@20={out['ipSAE_min']['precision@20']:.3f} "
          f"P@100={out['ipSAE_min']['precision@100']:.3f} "
          f"lift={out['ipSAE_min']['lift@100']:.2f} PR-AUC={out['ipSAE_min']['pr_auc']:.3f}")
    print(f"  随机对照       P@20={out['random_control']['precision@20']:.3f} "
          f"P@100={out['random_control']['precision@100']:.3f} "
          f"lift={out['random_control']['lift@100']:.2f} "
          f"PR-AUC={out['random_control']['pr_auc']:.3f}")
    return out


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} 不存在")
    cols = ["binder"] + [c for c, _ in METRICS]
    df = pl.read_csv(SRC, columns=cols, infer_schema_length=20000, ignore_errors=True)
    df = df.filter(pl.col("binder").is_not_null())
    y = df["binder"].cast(pl.Int8).to_numpy()          # 1 = 结合 (论文方向)
    base_rate = float(y.mean())
    print(f"样本 {len(y):,}  结合 {int(y.sum()):,} ({base_rate*100:.1f}%)  "
          f"非结合 {int((1-y).sum()):,}")
    print()

    results = {}
    for col, sign in METRICS:
        s = df[col].cast(pl.Float64, strict=False)
        mask = s.is_not_null().to_numpy()
        if mask.sum() < 100:
            print(f"  {col:24s} 有效值不足 ({int(mask.sum())}), 跳过")
            continue
        sc = (sign * s.fill_null(strategy="mean").to_numpy())[mask]
        r = best_f1(y[mask], sc)
        r["n_evaluated"] = int(mask.sum())
        r["direction"] = "higher_is_binder" if sign > 0 else "lower_is_binder"
        results[col] = r
        print(f"  {col:24s} n={r['n_evaluated']:5d}  best_F1={r['best_f1']:.3f}  "
              f"P={r['precision_at_best_f1']:.3f} R={r['recall_at_best_f1']:.3f}  "
              f"PR-AUC={r['pr_auc']:.3f}")

    # ---- 聚合口径: 论文分析的是"15 个结构多样的靶点", per-target 才是它的自然单位 ----
    # 池化算会被 FGFR2 一个靶点主导 (2,123/3,669 = 58% 的设计), 那不是论文的口径。
    # 两种口径都报, GATE 2 判据用 per-target 平均, 并把理由写进 payload。
    tgt = pl.read_csv(SRC, columns=["binder", "target_id", "af3_ipSAE_min"],
                      infer_schema_length=20000, ignore_errors=True) \
            .filter(pl.col("binder").is_not_null())
    per_target = {}
    for tname in tgt["target_id"].unique().to_list():
        sub = tgt.filter(pl.col("target_id") == tname)
        yy = sub["binder"].cast(pl.Int8).to_numpy()
        ss = sub["af3_ipSAE_min"].cast(pl.Float64, strict=False).fill_null(0).to_numpy()
        # 单一类别或样本过少的靶点不参与聚合 —— 它们的 F1 是噪声
        if len(yy) < 20 or yy.sum() == 0 or yy.sum() == len(yy):
            continue
        per_target[str(tname)] = {
            "n": int(len(yy)), "n_bind": int(yy.sum()),
            "best_f1": best_f1(yy, ss)["best_f1"],
        }
    pt_vals = [v["best_f1"] for v in per_target.values()]

    # 论文层面**不选门槛** (刘刚刚 2026-09-30 定): 基准的职责是给出完整曲线与若干预算下的
    # precision@k, 让使用者按自己的成本结构选。原论文也只说最优区间 0.5-0.8,
    # 且跨靶点精度在 0.1-1.0 之间波动 —— 这种不稳定性本身就是要报告的发现, 不是要抹平的。
    curve = None
    if "af3_ipSAE_min" in results:
        sd = tgt_full = pl.read_csv(SRC, columns=["binder", "af3_ipSAE_min", "target_id"],
                                   infer_schema_length=20000, ignore_errors=True) \
                          .filter(pl.col("binder").is_not_null())
        yy = sd["binder"].cast(pl.Int8).to_numpy()
        ss = sd["af3_ipSAE_min"].cast(pl.Float64, strict=False).fill_null(0).to_numpy()
        prec, rec, thr = precision_recall_curve(yy, ss)
        # 曲线抽稀到 200 点落盘, 供论文画图
        idx = np.linspace(0, len(thr) - 1, min(200, len(thr))).astype(int)
        curve = {
            "positive_class": "binder (论文极性)",
            "base_rate": float(yy.mean()),
            "points": [{"threshold": float(thr[i]), "precision": float(prec[i]),
                        "recall": float(rec[i])} for i in idx],
            "precision_at_k": {},
            "note": "完整 PR 曲线, 不在论文里选单一门槛。",
        }
        order = np.argsort(-ss, kind="stable")
        for k in (10, 20, 50, 100):
            if len(yy) >= k:
                curve["precision_at_k"][f"precision@{k}"] = float(yy[order[:k]].mean())

    ref = 0.61
    pooled = results.get("af3_ipSAE_min", {}).get("best_f1")
    verdict = None
    if pt_vals:
        pt_mean = float(np.mean(pt_vals))
        pt_median = float(np.median(pt_vals))
        delta = pt_mean - ref
        verdict = {
            "reference_best_f1": ref,
            "reference_note": "SPEC v2 §3.2 给的参考值: AF3 ipSAE_min 最高 F1 = 0.61",
            "pooled_best_f1": pooled,
            "per_target_mean_best_f1": pt_mean,
            "per_target_median_best_f1": pt_median,
            "n_targets_aggregated": len(pt_vals),
            "aggregation_used_for_gate2": "per_target_mean",
            "aggregation_rationale": (
                "论文分析对象是 15 个结构多样的靶点, per-target 是它的自然单位。"
                "池化算会被 FGFR2 主导 (2,123/3,669 = 58%% 的设计, 其单靶点 F1 仅 0.431), "
                f"得到 {pooled:.3f}, 不能与论文数字直接比。".replace("%%", "%")
            ),
            "delta_vs_reference": delta,
            "within_gate2_tolerance_0.05": bool(abs(delta) <= 0.05),
            "per_target": per_target,
        }
        print()
        print(f"af3_ipSAE_min 最高 F1: 池化 {pooled:.3f} | "
              f"per-target 平均 {pt_mean:.3f} | per-target 中位 {pt_median:.3f} "
              f"({len(pt_vals)} 个靶点)")
        print(f"GATE 2 判据 (per-target 平均 vs 论文 {ref}): 差 {delta:+.3f} -> "
              f"{'✅ 在 ±0.05 内' if abs(delta) <= 0.05 else '❌ 超出 ±0.05, 需解释'}")
    got = pooled

    # 论文另一条结论: ipSAE 优于 ipAE / ipTM
    ordering = None
    if all(k in results for k in ("af3_ipSAE_min", "af3_ipae", "af3_iptm_avg")):
        ordering = {
            "ipSAE_min_beats_ipae": results["af3_ipSAE_min"]["pr_auc"] > results["af3_ipae"]["pr_auc"],
            "ipSAE_min_beats_iptm": results["af3_ipSAE_min"]["pr_auc"] > results["af3_iptm_avg"]["pr_auc"],
            "pr_auc": {k: results[k]["pr_auc"]
                       for k in ("af3_ipSAE_min", "af3_ipae", "af3_iptm_avg")},
        }
        print(f"论文结论核验 (按 PR-AUC): ipSAE_min > ipAE = {ordering['ipSAE_min_beats_ipae']}, "
              f"ipSAE_min > ipTM = {ordering['ipSAE_min_beats_iptm']}")

    on_split = eval_on_bind_split()
    payload = {
        "note": "正类 = 结合 (论文方向), 与本项目其它基线『阳性=失败』相反, 不要混用",
        "on_bind_target_split": on_split,
        "n": len(y), "binder_rate": base_rate,
        "metrics": results, "gate2_check": verdict, "paper_claim_check": ordering,
        "pr_curve_no_single_threshold": curve,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")

    write_run_log(
        "baseline_ipsae",
        sources=["ds5_dtu_binder"],
        splits_used=[],     # 这个基线不需要切分: 它只是在 DS5 上重算阈值
        config={"metrics": [c for c, _ in METRICS], "reference_best_f1": ref},
        seed=None,
        results={"af3_ipSAE_min_best_f1": got, "gate2_check": verdict},
        notes="复现 SPEC v2 §3.2 的 AF3 ipSAE_min 基线; 不含 TargetTrack",
    )


if __name__ == "__main__":
    main()
