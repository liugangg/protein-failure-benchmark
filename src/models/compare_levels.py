"""在**同一套 center_folds、同一测试折**上比较 L0 / L1 / 随机对照, 并做**取反验证**。

取反验证 (刘刚刚 2026-09-30 指定, 零成本):
  L0 cross lift 中位 0.61、L1 0.75, 系统性低于随机的 1.03, 且折 2 的 L0 lift 恰为 0.00
  (前 100 个最"可能失败"的预测里一个真失败都没有)。这指向的机制**不是"无信号"**,
  而是"排序系统性反转"。把 cross 的预测取反重算 lift@k: 若明显 >1, 机制确认。
  两者对论文的表述完全不同:
    无信号     -> "数据不可学、堆模型无用"
    排序反转   -> "跨中心时模型排序系统性反转、预测劣于随机" ——
                 即在这类数据上训练的模型部署到新实验室会**把资源导向错误候选**。
  后者更强也更有用。
  还要报: 反转是**全部**留出中心都出现, 还是仅折 2/折 3 —— 表述精度取决于此。

分级推进的继续条件要求 "L1 cross-center lift 不低于 L0", 所以必须同折对比 —— 
先前 L0 跑的是 lab_split (留出 CSGID+EFI), 与这里的 center_folds 不是同一个测试集, 不可比。

随机对照是**必须的**: L1 的 cross-center lift 中位 0.78 (低于 1), 这既可能是真实发现
(模型学到中心惯例, 到新中心上方向反转), 也可能是评估口径出错。
随机打分在同一折上应给出 lift ≈ 1.0; 若不是, 说明是我的评估有问题而不是模型的问题。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval.caliber import caliber_dir, caliber_meta  # noqa: E402
from eval.metrics import evaluate  # noqa: E402
from models.baseline_gbdt import AA20, TAG_PATTERNS  # noqa: E402

EMB = pathlib.Path("data/interim/esm_embeddings")
FOLDS = pathlib.Path("data/processed/splits/center_folds.parquet")
RECORDS = pathlib.Path("data/processed/records.parquet")
CONFIG = pathlib.Path("configs/stage3_train.yaml")
K_IN = 3


def composition(seqs: list[str]) -> np.ndarray:
    """默认口径的 L0 特征: **只有**氨基酸组成, 不含长度 (见 configs/baseline_gbdt.yaml)。"""
    X = np.zeros((len(seqs), len(AA20)), dtype=np.float32)
    for i, s in enumerate(seqs):
        if not s:
            continue
        for j, a in enumerate(AA20):
            X[i, j] = s.count(a)
        X[i] /= len(s)
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="express")
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    seed, ks = cfg["seed"], tuple(cfg["eval"]["ks"])

    ep = EMB / f"{args.task}_esm2_650M_frozen.parquet"
    t = pq.read_table(ep)
    rids = t.column("record_id").to_pylist()
    Xemb = np.array(t.column("emb").to_pylist(), dtype=np.float32)
    y = (np.array(t.column("label").to_pylist()) == 0).astype(np.int8)
    centers = np.array(t.column("center").to_pylist())
    groups = np.array(t.column("split_group").to_pylist())

    # 取同一批 record 的序列, 用**同样的剥标签预处理**算 L0 特征
    import re
    rt = pq.read_table(RECORDS, columns=["record_id", "sequence"])
    seq_of = dict(zip(rt.column("record_id").to_pylist(),
                      rt.column("sequence").to_pylist()))
    rxs = [re.compile(p) for p, _ in TAG_PATTERNS]
    seqs = []
    for r in rids:
        s = seq_of[r]
        cur = s
        for rx in rxs:
            cur = rx.sub("", cur)
        seqs.append(cur if len(cur) >= 20 else s)
    Xl0 = composition(seqs)
    print(f"[cmp] {args.task}: {len(y):,} 条  L0 特征 {Xl0.shape[1]} 维 (仅组成)  "
          f"L1 embedding {Xemb.shape[1]} 维  阴性 {y.mean()*100:.1f}%")

    ft = pq.read_table(FOLDS)
    fold_of = dict(zip(ft.column("center").to_pylist(), ft.column("center_fold").to_pylist()))
    m = cfg["levels"]["L1"]
    rng = np.random.default_rng(seed)
    rows = []
    preds: list[dict] = []

    print(f"\n{'折.内':>7}{'test':>7}{'base':>7}{'随机':>8}{'L0':>10}{'L0取反':>9}"
          f"{'L1':>10}{'L1取反':>9}{'L1 within':>11}")
    for j in range(cfg["eval"]["n_center_folds"]):
        held = [c for c, f in fold_of.items() if f == j]
        te_m = np.isin(centers, held)
        if te_m.sum() == 0 or y[te_m].sum() < cfg["eval"]["min_test_positives"]:
            continue
        touched = set(groups[te_m])
        cross_m = ~te_m & ~np.isin(groups, list(touched))
        gi = {g: (hash((int(g), seed)) % K_IN) for g in sorted(touched)}
        inner = np.array([gi.get(g, -1) for g in groups])

        for i in range(K_IN):
            te_i = te_m & (inner == i)
            if te_i.sum() < cfg["eval"]["min_test_n"] or \
                    y[te_i].sum() < cfg["eval"]["min_test_positives"]:
                continue
            wi = te_m & (inner != i)
            out = {"fold": j, "inner": i, "held": held, "n_test": int(te_i.sum()),
                   "base_rate": float(y[te_i].mean())}

            # 随机对照 —— 校准检查
            out["random"] = evaluate(y[te_i], rng.random(int(te_i.sum())), ks=ks)

            def gbdt(trm, Xf):
                clf = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                    l2_regularization=1.0, early_stopping=True,
                    validation_fraction=0.1, random_state=seed)
                clf.fit(Xf[trm], y[trm])
                return clf.predict_proba(Xf[te_i])[:, 1]

            sc_l0 = gbdt(cross_m, Xl0)
            sc_l1 = gbdt(cross_m, Xemb)
            out["L0_cross"] = evaluate(y[te_i], sc_l0, ks=ks)
            out["L1_cross"] = evaluate(y[te_i], sc_l1, ks=ks)
            # 取反验证: 同一预测取负重算。lift>1 说明排序方向是反的, 不是没有信息。
            out["L0_cross_inverted"] = evaluate(y[te_i], -sc_l0, ks=ks)
            out["L1_cross_inverted"] = evaluate(y[te_i], -sc_l1, ks=ks)
            if wi.sum() >= 200 and y[wi].sum() >= 30:
                out["L0_within"] = evaluate(y[te_i], gbdt(wi, Xl0), ks=ks)
                out["L1_within"] = evaluate(y[te_i], gbdt(wi, Xemb), ks=ks)
            preds.append({"fold": j, "inner": i,
                          "record_id": [rids[x] for x in np.where(te_i)[0]],
                          "center": centers[te_i].tolist(),
                          "y_fail": y[te_i].tolist(),
                          "L0_cross_score": sc_l0.tolist(),
                          "L1_cross_score": sc_l1.tolist()})
            rows.append(out)
            print(f"{j}.{i:<5}{out['n_test']:>7,}{out['base_rate']:>7.3f}"
                  f"{out['random']['lift@100']:>8.2f}{out['L0_cross']['lift@100']:>10.2f}"
                  f"{out['L0_cross_inverted']['lift@100']:>9.2f}"
                  f"{out['L1_cross']['lift@100']:>10.2f}"
                  f"{out['L1_cross_inverted']['lift@100']:>9.2f}"
                  f"{out.get('L1_within',{}).get('lift@100',float('nan')):>11.2f}")

    def med(key, sub="lift@100"):
        v = [r[key][sub] for r in rows if key in r and not np.isnan(r[key][sub])]
        return float(np.median(v)) if v else float("nan")

    print(f"\n{'':<14}{'lift@100 中位':>14}{'PR-AUC/base 中位':>18}")
    for k, zh in [("random", "随机对照"), ("L0_cross", "L0 cross"),
                  ("L0_cross_inverted", "L0 cross 取反"), ("L1_cross", "L1 cross"),
                  ("L1_cross_inverted", "L1 cross 取反"),
                  ("L0_within", "L0 within"), ("L1_within", "L1 within")]:
        print(f"{zh:<18}{med(k):>14.2f}{med(k,'pr_auc_over_base'):>18.2f}")

    # 反转是全部留出中心都出现, 还是仅某些折 —— 表述精度取决于此
    print(f"\n{'折':>4}{'留出中心':<40}{'L0 正':>7}{'L0 反':>7}{'L1 正':>7}{'L1 反':>7}  反转?")
    per_fold: dict[int, list] = {}
    for r in rows:
        per_fold.setdefault(r["fold"], []).append(r)
    inv_folds, noninv_folds = [], []
    for jj, rs in sorted(per_fold.items()):
        a0 = float(np.median([x["L0_cross"]["lift@100"] for x in rs]))
        b0 = float(np.median([x["L0_cross_inverted"]["lift@100"] for x in rs]))
        a1 = float(np.median([x["L1_cross"]["lift@100"] for x in rs]))
        b1 = float(np.median([x["L1_cross_inverted"]["lift@100"] for x in rs]))
        inverted = (b0 > 1.0 and b0 > a0) and (b1 > 1.0 and b1 > a1)
        (inv_folds if inverted else noninv_folds).append(jj)
        print(f"{jj:>4}{','.join(rs[0]['held'])[:38]:<40}{a0:>7.2f}{b0:>7.2f}"
              f"{a1:>7.2f}{b1:>7.2f}  {'**是**' if inverted else '否'}")
    print(f"\n出现反转的折: {inv_folds}   未出现: {noninv_folds}")
    if len(inv_folds) == len(per_fold):
        print("  -> **全部**留出中心都出现反转, 可写成普适现象")
    elif inv_folds:
        print(f"  -> 仅部分折出现, 正文必须写成'{len(inv_folds)}/{len(per_fold)} 个留出中心组'"
              f", 不可写成普适")

    cmeta = caliber_meta(True, False)
    outp = caliber_dir(True, False) / f"compare_levels_{args.task}.json"
    outp.write_text(json.dumps({"caliber": cmeta, "task": args.task,
                                "note": "同一 center_folds、同一测试折, 只换特征/训练来源",
                                "medians_lift100": {k: med(k) for k in
                                            ("random", "L0_cross", "L0_cross_inverted",
                                             "L1_cross", "L1_cross_inverted",
                                             "L0_within", "L1_within")},
                                # 主数字是 PR-AUC/base (见 configs/stage3_train.yaml changelog)
                                "medians_pr_auc_over_base": {
                                    k: med(k, "pr_auc_over_base") for k in
                                    ("random", "L0_cross", "L0_cross_inverted",
                                     "L1_cross", "L1_cross_inverted",
                                     "L0_within", "L1_within")},
                                "inverted_folds": inv_folds,
                                "non_inverted_folds": noninv_folds,
                                "folds": rows}, indent=2, ensure_ascii=False, default=float))
    # 预测落盘, 之后做任何后验分析都不必重跑模型
    import pyarrow as pa
    flat = {"fold": [], "inner": [], "record_id": [], "center": [], "y_fail": [],
            "L0_cross_score": [], "L1_cross_score": []}
    for b in preds:
        n = len(b["record_id"])
        flat["fold"] += [b["fold"]] * n
        flat["inner"] += [b["inner"]] * n
        for k in ("record_id", "center", "y_fail", "L0_cross_score", "L1_cross_score"):
            flat[k] += b[k]
    pq.write_table(pa.table(flat),
                   caliber_dir(True, False) / f"compare_levels_{args.task}_preds.parquet")
    print(f"\nwrote {outp} 及同名 _preds.parquet (预测已落盘)")


if __name__ == "__main__":
    main()
