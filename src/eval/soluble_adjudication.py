"""soluble 阶段定案检查 (刘刚刚 2026-09-30 指定)。

要回答的: 中心内仍然保留的 His6↔soluble 关联 (NESG RR=7.00, SGPP RR=2.29),
是**生物学**(His 标签真的降低可溶性) 还是**记录习惯**(带标签的构建更容易被记成停在 express 之后)?

判据 (刘刚刚预先定义的, 不是事后挑的):
  只在**显式** soluble 阴性子集上重算 His6 关联, 与**推断**阴性子集对比。
    - 若关联只在推断子集存在 -> 判为记录习惯
    - 若两个子集都存在      -> 支持生物学
    - 样本不足以定论        -> 如实标注, soluble 在正文里单列为"未解决"

两类阴性的来源:
  显式 (evidence_tier = explicit): stopStatus 直接给出 membrane protein solubilization failed
  推断 (evidence_tier = inferred_next_step): v2 §2.4 规则, work stopped 且已达到 express
"""
from __future__ import annotations

import json
import pathlib

import polars as pl

RECORDS = pathlib.Path("data/processed/records.parquet")
OUT = pathlib.Path("reports/soluble_adjudication.json")
MIN_FOR_VERDICT = 100      # 每格至少这么多条才敢下判断


def main() -> None:
    rec = pl.read_parquet(RECORDS).filter(pl.col("source") == "ds1_targettrack")
    d = rec.with_columns(pl.col("sequence").str.contains("H{6,}").alias("has_his"))
    obs = d.filter(pl.col("label_soluble") != -1)
    neg = obs.filter(pl.col("label_soluble") == 0)
    pos = obs.filter(pl.col("label_soluble") == 1)

    out: dict = {
        "n_soluble_observed": obs.height,
        "n_soluble_negative": neg.height,
        "n_soluble_positive": pos.height,
    }
    by_tier = (neg.group_by("evidence_tier").agg(pl.len().alias("n"))
                  .sort("n", descending=True))
    out["negatives_by_evidence_tier"] = by_tier.to_dicts()
    print(f"soluble 有观测 {obs.height:,} 条 (阴性 {neg.height:,} / 阳性 {pos.height:,})")
    print("阴性按证据等级:")
    for r, n in by_tier.iter_rows():
        print(f"   {r:<22} {n:>7,}")
    print()

    # ---- 分别在两个子集上算 His6 关联 ----
    # 分母口径: 该子集的阴性 + 全部阳性 (阳性没有证据等级之分, 都是直接观测到 soluble)
    results = []
    for tier in [r for r, _ in by_tier.iter_rows()]:
        sub_neg = neg.filter(pl.col("evidence_tier") == tier)
        # 阴性侧按 tier 取, 阳性侧全取 —— 阳性不分 tier
        universe = pl.concat([sub_neg, pos], how="vertical_relaxed")
        a = universe.filter(pl.col("has_his"))
        b = universe.filter(~pl.col("has_his"))
        fa = float((a["label_soluble"] == 0).mean()) if a.height else float("nan")
        fb = float((b["label_soluble"] == 0).mean()) if b.height else float("nan")
        enough = (a.height >= MIN_FOR_VERDICT and b.height >= MIN_FOR_VERDICT
                  and int((a["label_soluble"] == 0).sum()) >= 20
                  and int((b["label_soluble"] == 0).sum()) >= 20)
        row = {
            "evidence_tier": tier,
            "n_negatives_in_tier": sub_neg.height,
            "n_his_total": a.height, "n_his_fail": int((a["label_soluble"] == 0).sum()),
            "fail_rate_his": None if a.height == 0 else round(fa, 5),
            "n_nohis_total": b.height, "n_nohis_fail": int((b["label_soluble"] == 0).sum()),
            "fail_rate_nohis": None if b.height == 0 else round(fb, 5),
            "risk_ratio": round(fa / fb, 3) if (fb and fb > 0) else None,
            "sufficient_for_verdict": bool(enough),
        }
        results.append(row)
        rr = "—" if row["risk_ratio"] is None else f"{row['risk_ratio']:.2f}"
        print(f"[{tier}] 阴性 {sub_neg.height:,}")
        print(f"   含His6  n={a.height:>7,}  失败 {row['n_his_fail']:>6,}  "
              f"失败率 {'—' if row['fail_rate_his'] is None else f'{fa*100:.3f}%'}")
        print(f"   不含    n={b.height:>7,}  失败 {row['n_nohis_fail']:>6,}  "
              f"失败率 {'—' if row['fail_rate_nohis'] is None else f'{fb*100:.3f}%'}")
        print(f"   RR = {rr}   够不够下判断: {'够' if enough else '**不够**'}")
        print()
    out["by_tier"] = results

    # ---- 显式阴性太少时, 把它们逐条列出来 (n 小到可以人工看) ----
    expl = neg.filter(pl.col("evidence_tier") == "explicit")
    if expl.height <= 50:
        out["explicit_negatives_listed"] = expl.select(
            "record_id", "center", "seq_len", "has_his", "evidence").to_dicts()
        print(f"显式 soluble 阴性只有 {expl.height} 条, 逐条列出:")
        for r in expl.iter_rows(named=True):
            print(f"   {r['record_id']:<42} {r['center']:<9} len={r['seq_len']:>5} "
                  f"His6={'是' if r['has_his'] else '否'}")
        print()

    # ---- 中心内 (推断子集) 对照, 说明方向不一致 ----
    print("推断子集内、按中心分层的 His6 关联 (说明方向在中心之间不一致):")
    infr = pl.concat([neg.filter(pl.col("evidence_tier") == "inferred_next_step"), pos],
                     how="vertical_relaxed")
    rows_c = []
    for ctr in infr["center"].unique().to_list():
        s = infr.filter(pl.col("center") == ctr)
        a = s.filter(pl.col("has_his")); b = s.filter(~pl.col("has_his"))
        if a.height < 300 or b.height < 300:
            continue
        fa = float((a["label_soluble"] == 0).mean()); fb = float((b["label_soluble"] == 0).mean())
        rows_c.append({"center": ctr, "n_his": a.height, "fail_rate_his": round(fa, 5),
                       "n_nohis": b.height, "fail_rate_nohis": round(fb, 5),
                       "risk_ratio": round(fa / fb, 3) if fb > 0 else None})
        rr = "—" if fb == 0 else f"{fa/fb:.2f}"
        print(f"   {ctr:<9} 含His {fa*100:6.2f}% (n={a.height:>6,})  "
              f"不含 {fb*100:6.2f}% (n={b.height:>6,})  RR={rr}")
    out["inferred_by_center"] = rows_c

    # ---- 判定 ----
    expl_row = next((r for r in results if r["evidence_tier"] == "explicit"), None)
    verdict = {
        "explicit_subset_testable": bool(expl_row and expl_row["sufficient_for_verdict"]),
        "verdict": None, "reason": None,
    }
    if not verdict["explicit_subset_testable"]:
        verdict["verdict"] = "UNRESOLVED"
        verdict["reason"] = (
            f"显式 soluble 阴性只有 {expl.height} 条 (全库唯一来源是 stopStatus 的 "
            f"membrane protein solubilization failed), 远低于下判断所需的 {MIN_FOR_VERDICT} 条。"
            "因此**无法**用刘刚刚预先定义的判据区分'生物学'与'记录习惯' —— "
            "不是关联不存在, 是这批数据里不存在可用的对照子集。"
            "按预先约定, soluble 在正文里单列为**未解决**, 不并入任一方向。"
        )
    out["verdict"] = verdict
    print()
    print("=== 判定 ===")
    print(f"  {verdict['verdict']}")
    print(f"  {verdict['reason']}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
