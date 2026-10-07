"""His 标签混杂的中心内分层检验 (刘刚刚 2026-09-30 指定, 最高优先)。

背景: 先前发现 DS1 里 18.3% 的序列含 HHHHHH, 且 His6 率几乎是 center 级的构建习惯
(SSGCID 91.8% vs CESG 0.2%), 同时 His6 与各阶段失败率强相关。
但那个关联是**跨中心**算的, 无法区分两种解释:
  (A) 标签只是中心身份的代理 —— 真正驱动关联的是"哪个中心做的"
  (B) 标签本身与结局有生物学关联 (如 His 标签影响可溶性/纯化)
中心内分层是分开这两者的标准做法: 在标签使用混合的中心**内部**, 如果关联消失,
支持 (A); 如果关联保留, 支持 (B) 或存在中心内的其它混杂。

三项 (都只做分析, 不改任何产物):
  1a 中心内 His6 与各阶段结局的关联
  1b 中心内 长度 的预测力
  1c 各中心的记录长度分布 —— 判断是否存在"记录构建体 vs 记录 ORF"两种模式
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl

RECORDS = pathlib.Path("data/processed/records.parquet")
OUT = pathlib.Path("reports/tag_confound_analysis.json")
STAGES = ["clone", "express", "soluble", "purify", "stable"]
MIN_N = 300          # 分层后每格至少这么多条才算, 否则是噪声


def rank_auc(pos_vals: np.ndarray, neg_vals: np.ndarray) -> float:
    """P(随机取的正样本值 > 随机取的负样本值) —— Mann-Whitney U / (n1*n2)。
    0.5 = 无预测力。用秩而非均值差, 对长尾稳健。"""
    if len(pos_vals) == 0 or len(neg_vals) == 0:
        return float("nan")
    allv = np.concatenate([pos_vals, neg_vals])
    order = allv.argsort(kind="stable")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 处理并列: 取平均秩
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq)); np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    r1 = ranks[: len(pos_vals)].sum()
    n1, n2 = len(pos_vals), len(neg_vals)
    u1 = r1 - n1 * (n1 + 1) / 2
    return float(u1 / (n1 * n2))


def main() -> None:
    rec = pl.read_parquet(RECORDS).filter(pl.col("source") == "ds1_targettrack")
    d = rec.with_columns([
        pl.col("sequence").str.contains("H{6,}").alias("has_his"),
        pl.col("sequence").str.head(30).str.contains("H{6,}").alias("his_nterm"),
        pl.col("sequence").str.tail(30).str.contains("H{6,}").alias("his_cterm"),
        pl.col("sequence").str.starts_with("M").alias("starts_M"),
    ])
    out: dict = {"n_ds1": d.height}

    # ---------- 各中心的标签使用率 ----------
    cc = (d.group_by("center").agg(
            pl.len().alias("n"),
            pl.col("has_his").mean().alias("his_rate"),
            pl.col("his_nterm").mean().alias("nterm_rate"),
            pl.col("his_cterm").mean().alias("cterm_rate"),
            pl.col("starts_M").mean().alias("startsM_rate"),
            pl.col("seq_len").median().alias("len_med"))
          .filter(pl.col("n") >= 1000).sort("n", descending=True))
    out["centers"] = cc.to_dicts()
    # 标签使用"混合"的中心: 使用率在 5%-95% 之间, 才有中心内对比的可能
    mixed = cc.filter((pl.col("his_rate") > 0.05) & (pl.col("his_rate") < 0.95))
    out["mixed_centers"] = mixed["center"].to_list()
    print(f"标签使用混合的中心 (5%<his_rate<95%): {out['mixed_centers']}")
    print()

    # ---------- 1a 中心内 His6 与结局的关联 ----------
    print("=== 1a 中心内 His6 与各阶段失败率 ===")
    rows_1a = []
    for ctr in out["mixed_centers"]:
        sub = d.filter(pl.col("center") == ctr)
        for s in STAGES:
            o = sub.filter(pl.col(f"label_{s}") != -1)
            a = o.filter(pl.col("has_his")); b = o.filter(~pl.col("has_his"))
            if a.height < MIN_N or b.height < MIN_N:
                continue
            fa = float((a[f"label_{s}"] == 0).mean())
            fb = float((b[f"label_{s}"] == 0).mean())
            # 风险差与风险比
            rows_1a.append({
                "center": ctr, "stage": s,
                "n_his": a.height, "fail_rate_his": round(fa, 4),
                "n_nohis": b.height, "fail_rate_nohis": round(fb, 4),
                "risk_diff": round(fa - fb, 4),
                "risk_ratio": round(fa / fb, 3) if fb > 0 else None,
            })
    out["within_center_his_association"] = rows_1a
    print(f"{'center':<9}{'stage':<9}{'n_his':>8}{'fail_his':>10}{'n_nohis':>9}{'fail_nohis':>12}{'RR':>8}")
    for r in rows_1a:
        rr = "—" if r["risk_ratio"] is None else f"{r['risk_ratio']:.2f}"
        print(f"{r['center']:<9}{r['stage']:<9}{r['n_his']:>8,}{r['fail_rate_his']*100:>9.1f}%"
              f"{r['n_nohis']:>9,}{r['fail_rate_nohis']*100:>11.1f}%{rr:>8}")
    # 对照: 跨中心 (未分层) 的同一关联
    print()
    print("对照 —— 跨中心未分层:")
    rows_pool = []
    for s in STAGES:
        o = d.filter(pl.col(f"label_{s}") != -1)
        a = o.filter(pl.col("has_his")); b = o.filter(~pl.col("has_his"))
        if a.height < MIN_N or b.height < MIN_N:
            continue
        fa = float((a[f"label_{s}"] == 0).mean()); fb = float((b[f"label_{s}"] == 0).mean())
        rows_pool.append({"stage": s, "fail_rate_his": round(fa, 4),
                          "fail_rate_nohis": round(fb, 4),
                          "risk_ratio": round(fa / fb, 3) if fb > 0 else None})
        print(f"  {s:<9} 含His {fa*100:5.1f}%  不含 {fb*100:5.1f}%  "
              f"RR={'—' if fb==0 else f'{fa/fb:.2f}'}")
    out["pooled_his_association"] = rows_pool

    # ---------- 1b 中心内 长度 的预测力 ----------
    print()
    print("=== 1b 长度对失败的预测力 (rank AUC; 0.5 = 无预测力) ===")
    rows_1b = []
    for s in STAGES:
        o = d.filter(pl.col(f"label_{s}") != -1)
        pos = o.filter(pl.col(f"label_{s}") == 0)["seq_len"].to_numpy()
        neg = o.filter(pl.col(f"label_{s}") == 1)["seq_len"].to_numpy()
        pooled = rank_auc(pos, neg) if (len(pos) >= MIN_N and len(neg) >= MIN_N) else float("nan")
        per_center = {}
        for ctr in cc["center"].to_list():
            oc = o.filter(pl.col("center") == ctr)
            p2 = oc.filter(pl.col(f"label_{s}") == 0)["seq_len"].to_numpy()
            n2 = oc.filter(pl.col(f"label_{s}") == 1)["seq_len"].to_numpy()
            if len(p2) >= MIN_N and len(n2) >= MIN_N:
                per_center[ctr] = round(rank_auc(p2, n2), 4)
        if np.isnan(pooled) and not per_center:
            continue
        vals = list(per_center.values())
        rows_1b.append({
            "stage": s,
            "pooled_rank_auc": None if np.isnan(pooled) else round(pooled, 4),
            "per_center_rank_auc": per_center,
            "n_centers": len(vals),
            "within_center_median": round(float(np.median(vals)), 4) if vals else None,
            "within_center_range": [round(min(vals), 4), round(max(vals), 4)] if vals else None,
        })
        pm = "—" if np.isnan(pooled) else f"{pooled:.3f}"
        wm = "—" if not vals else f"{np.median(vals):.3f}"
        rng = "—" if not vals else f"[{min(vals):.3f},{max(vals):.3f}]"
        print(f"  {s:<9} 跨中心 {pm}   中心内中位 {wm}  范围 {rng}  ({len(vals)} 个中心)")
    out["length_predictive_power"] = rows_1b

    # ---------- 1c 记录长度分布: 构建体 vs ORF ----------
    print()
    print("=== 1c 记录的是构建体还是 ORF ===")
    print(f"{'center':<9}{'n':>8}{'his%':>7}{'N端%':>7}{'C端%':>7}{'startM%':>9}"
          f"{'len中位':>9}{'含His长':>9}{'无His长':>9}{'差':>7}")
    rows_1c = []
    for r in cc.iter_rows(named=True):
        sub = d.filter(pl.col("center") == r["center"])
        a = sub.filter(pl.col("has_his")); b = sub.filter(~pl.col("has_his"))
        la = float(a["seq_len"].median()) if a.height else float("nan")
        lb = float(b["seq_len"].median()) if b.height else float("nan")
        rows_1c.append({
            "center": r["center"], "n": r["n"], "his_rate": round(r["his_rate"], 4),
            "nterm_share_of_tagged": round(r["nterm_rate"] / r["his_rate"], 3) if r["his_rate"] else None,
            "cterm_share_of_tagged": round(r["cterm_rate"] / r["his_rate"], 3) if r["his_rate"] else None,
            "starts_M_rate": round(r["startsM_rate"], 4),
            "len_median_all": r["len_med"],
            "len_median_tagged": None if np.isnan(la) else la,
            "len_median_untagged": None if np.isnan(lb) else lb,
            "len_delta_tagged_minus_untagged": None if (np.isnan(la) or np.isnan(lb)) else la - lb,
        })
        print(f"{r['center']:<9}{r['n']:>8,}{r['his_rate']*100:>6.1f}%"
              f"{(r['nterm_rate']/r['his_rate']*100 if r['his_rate'] else float('nan')):>6.0f}%"
              f"{(r['cterm_rate']/r['his_rate']*100 if r['his_rate'] else float('nan')):>6.0f}%"
              f"{r['startsM_rate']*100:>8.0f}%{r['len_med']:>9.0f}"
              f"{la:>9.0f}{lb:>9.0f}{(la-lb) if not (np.isnan(la) or np.isnan(lb)) else float('nan'):>7.0f}")
    out["record_mode"] = rows_1c

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
