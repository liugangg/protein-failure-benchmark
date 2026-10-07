"""按簇分层的自助法置信区间 —— 让每个情形判定都有区间支撑。

刘刚刚 2026-10-01 的硬要求: **按聚类 (split_group) 分层重采样, 不按单条序列。**
  原因: 同源序列在数据里成簇出现 (30% 相似度的传递闭包分量)。按单条重采样会把同一簇的
  成员反复抽到, 等于假装样本之间独立, **压窄区间、造出假显著**。
  按簇重采样把簇当成重采样单元, 区间才反映真实的有效样本量。

直接动因: L2 在 express 折 2 上的 PR-AUC/base = 1.15, 而判定门槛是 1.10 —— 只差 0.05,
  且该折基础失败率仅 3.2%。这种情况不能靠判断, 必须用区间定:
    区间跨 1.0 -> 按"与随机不可区分"定稿
    区间不跨   -> 如实改为"L2 在低基础率折上出现弱真信号",
                  并注明两个口径 (lift@100 与 PR-AUC/base) 方向相反
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval.metrics import pr_auc, precision_at_k  # noqa: E402

D = pathlib.Path("reports/default")
N_BOOT = 2000
LOW_BASE = 0.10


def cluster_bootstrap(y: np.ndarray, s: np.ndarray, g: np.ndarray,
                      n_boot: int, seed: int) -> dict:
    """按簇重采样。返回 PR-AUC/base 与 lift@100 的分布。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    # 预先把每个簇的下标存好, 避免每轮重新 mask
    idx_of = {u: np.where(g == u)[0] for u in uniq}
    pab, l100 = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_of[u] for u in pick])
        yy, ss = y[sel], s[sel]
        base = yy.mean()
        if base <= 0 or base >= 1:
            continue
        ap = pr_auc(yy, ss)
        if not np.isnan(ap):
            pab.append(ap / base)
        if len(yy) >= 100:
            p = precision_at_k(yy, ss, 100)
            if not np.isnan(p):
                l100.append(p / base)
    return {"pr_auc_over_base": np.array(pab), "lift@100": np.array(l100)}


def ci(a: np.ndarray, lo=2.5, hi=97.5) -> tuple[float, float]:
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi))) if len(a) else (float("nan"),) * 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="reports/default 下的 *_preds.parquet")
    ap.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--score-col", default="score",
                    help="分数列名。compare_levels 的产物一个文件里存了多个模型 "
                         "(L0_cross_score / L1_cross_score), 用这个选。")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="种子稳健性检验的重复次数。实查 (2026-10-02): L2 "
                         "soluble_expression 折 3 的区间下界是 1.0010, 换个种子就变 "
                         "0.9995 —— 判定会随种子翻转。所以方向判定必须**所有种子一致**, "
                         "否则按保守方向 (不可区分) 定。加大 n_boot 不解决: "
                         "20,000 次仍是 1.0007, 不确定性在数据里不在重采样次数里。")
    args = ap.parse_args()

    out = {}
    for fname in args.preds:
        p = D / fname
        if not p.exists():
            print(f"[skip] {p} 不存在")
            continue
        t = pq.read_table(p)
        d = {c: np.array(t.column(c).to_pylist()) for c in t.column_names}
        scol = args.score_col
        if scol not in d:
            print(f"[skip] {p} 没有列 {scol!r} (有: {', '.join(d)})")
            continue
        if "split_group" not in d:          # compare_levels 的产物没存这一列
            rt = pq.read_table("data/processed/records.parquet",
                               columns=["record_id", "split_group"])
            gmap = dict(zip(rt.column("record_id").to_pylist(),
                            rt.column("split_group").to_pylist()))
            miss = [r for r in d["record_id"] if r not in gmap]
            if miss:
                raise SystemExit(f"{p}: {len(miss)} 条 record_id 在 records.parquet 里"
                                 f"找不到 split_group, 拒绝按单条重采样")
            d["split_group"] = np.array([gmap[r] for r in d["record_id"]])
            print(f"  (split_group 由 records.parquet 按 record_id 回填)")
        tag = fname.replace("_preds.parquet", "")
        if scol != "score":
            tag += ":" + scol.replace("_cross_score", "")
        print(f"\n{'='*78}\n{tag}")
        print(f"{'折':>3}{'n':>7}{'簇数':>7}{'base':>7}  "
              f"{'PR-AUC/base [95% CI]':<30}{'lift@100 [95% CI]':<28}判定")
        out[tag] = {}
        for fo in args.folds:
            m = d["fold"] == fo
            if m.sum() == 0:
                continue
            y, s, g = d["y_fail"][m].astype(np.int8), d[scol][m], d["split_group"][m]
            base = float(y.mean())
            n_clu = len(np.unique(g))
            b = cluster_bootstrap(y, s, g, args.n_boot, args.seed + fo)
            pab_pt = pr_auc(y, s) / base
            # ── 种子稳健性: 同一数据重复 n_seeds 次独立自助法, 看方向判定是否一致 ──
            seed_dirs = []
            for si in range(args.n_seeds):
                bi = (b if si == 0 else
                      cluster_bootstrap(y, s, g, args.n_boot,
                                        args.seed + fo + 10007 * si))
                lo_i, hi_i = ci(bi["pr_auc_over_base"])
                seed_dirs.append("cross" if lo_i <= 1.0 <= hi_i
                                 else ("pos" if lo_i > 1.0 else "neg"))
            seed_stable = len(set(seed_dirs)) == 1
            l_pt = precision_at_k(y, s, 100) / base
            pab_lo, pab_hi = ci(b["pr_auc_over_base"])
            l_lo, l_hi = ci(b["lift@100"])
            # 主数字对**所有**折都是 PR-AUC/base (configs/stage3_train.yaml 的 changelog
            # 记录了这个改动的时间点与理由)。lift@100 仍算出来并报告, 但不用于判定 ——
            # 它在 n~2,500 的折上只用 top-100 = 4% 的样本, CI 宽 0.63, 支撑不了折级判定。
            primary = "pr_auc_over_base"
            plo, phi = pab_lo, pab_hi
            crosses1 = plo <= 1.0 <= phi
            verdict = ("与随机不可区分 (区间跨 1.0)" if crosses1
                       else ("真信号" if plo > 1.0 else "反向"))
            # 区间不跨 1.0 只说明"方向可分辨", 不说明"幅度够用"。n=7,399 的折上
            # 1.05 [1.03, 1.07] 在统计上显著、在实践上只是 +5%。所以另记一个
            # 带效应量的四分类, 判定依据写明, 不用区间显著性冒充效应量。
            if not seed_stable:
                # 方向本身随种子翻转 -> 这条数据不支持任何方向性结论, 按保守方向定。
                cls = "不可区分"
            elif crosses1:
                cls = "不可区分"
            elif phi < 1.0:
                cls = "反向"
            elif pab_pt >= 1.10:
                cls = "真信号"
            else:
                cls = "弱但可分辨"
            out[tag][fo] = {
                "n": int(m.sum()), "n_clusters": n_clu, "base_rate": base,
                "primary_metric": primary,
                "pr_auc_over_base": {"point": pab_pt, "ci95": [pab_lo, pab_hi]},
                "lift@100": {"point": l_pt, "ci95": [l_lo, l_hi]},
                "primary_ci_crosses_1": bool(crosses1), "verdict": verdict,
                "effect_size_class": cls,
                "seed_stability": {"n_seeds": args.n_seeds,
                                   "directions": seed_dirs,
                                   "stable": bool(seed_stable)},
                "n_boot": args.n_boot,
                "resample_unit": "split_group (按簇分层, 非单条序列)",
            }
            star = " *" if base < LOW_BASE else "  "
            print(f"{fo:>3}{int(m.sum()):>7,}{n_clu:>7,}{base:>7.3f}  "
                  f"{pab_pt:.2f} [{pab_lo:.2f}, {pab_hi:.2f}]{star:<14}"
                  f"{l_pt:.2f} [{l_lo:.2f}, {l_hi:.2f}]      {cls}"
                  + ("" if seed_stable else
                     f"  ⚠️ 方向随种子翻转 {seed_dirs} -> 按保守方向定"))
        print("  判定一律用 PR-AUC/base (见 configs changelog); * = 该折基础率 <10%\n"
              "  四分类: 区间跨 1.0 -> 不可区分; 整体 <1.0 -> 反向; "
              "不跨且点估计 \u22651.10 -> 真信号; 不跨但 <1.10 -> 弱但可分辨\n"
              f"  另: 方向判定须 {args.n_seeds} 个自助法种子全部一致, "
              "否则按保守方向 (不可区分) 定")

    _f = D / "bootstrap_ci.json"
    if _f.exists():                      # 合并写, 不覆盖 (覆盖写会静默丢掉上一次的折)
        prev = json.loads(_f.read_text()).get("results", {})
        for k, v in prev.items():
            out.setdefault(k, {}).update({kk: vv for kk, vv in v.items()
                                          if kk not in out.get(k, {})})
    _f.write_text(json.dumps(
        {"n_boot": args.n_boot, "resample_unit": "split_group",
         "rationale": ("按簇重采样而非按单条: 同源序列成簇出现, 按单条会把同簇成员反复抽到, "
                       "等于假装样本独立, 压窄区间造出假显著。"),
         "results": out}, indent=2, ensure_ascii=False, default=float))
    print(f"\nwrote {_f}  ({sum(len(v) for v in out.values())} 个折级区间)")


if __name__ == "__main__":
    main()
