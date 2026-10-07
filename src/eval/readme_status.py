"""把 README.md 的「当前进度」数字段落改为生成式。

为什么: 2026-10-02 实查发现 README 的进度数字是 DNA bug 修复前、泄漏修补前的旧值
(记录 2,380,309 vs 实际 2,380,297, express 阴性簇 17,699 vs 22,771,
序列切分 test 114,508 vs 73,219 ...)。手写的数字一定会过期。
本脚本只改 <!-- STATUS:BEGIN --> 与 <!-- STATUS:END --> 之间的区块, 其余正文不动。
"""
from __future__ import annotations

import json
import pathlib

import polars as pl

README = pathlib.Path("README.md")
BEGIN, END = "<!-- STATUS:BEGIN -->", "<!-- STATUS:END -->"
R = pathlib.Path("reports")


def jl(p):
    f = pathlib.Path(p)
    return json.loads(f.read_text()) if f.exists() else None


def main() -> None:
    g1 = jl(R / "gate1_summary.json")
    mf = jl("data/processed/splits/MANIFEST.json")
    leak = {o["split"]: o for o in (jl(R / "leakage_check.json") or [])}
    gb = jl(R / "default" / "baseline_gbdt.json")
    sol = jl(R / "baseline_soluprot.json")
    bs = jl(R / "default" / "bootstrap_ci.json")
    rec = pl.read_parquet("data/processed/records.parquet",
                          columns=["label_express", "label_clone", "label_soluble",
                                   "label_purify", "label_stable", "label_bind",
                                   "label_soluble_expression", "cluster_rep"])

    L = [BEGIN, "",
         "> 本区块由 `src/eval/readme_status.py` 从冻结产物生成, **不要手改** —— "
         "手写的数字会过期 (2026-10-02 就抓到一批 DNA bug 修复前的旧值)。", ""]
    A = L.append

    A(f"入库: **{g1['records']:,} 条记录 · {g1['unique_sequences']:,} 条独立序列 · "
      f"30% 相似度下 {g1['clusters_30pct']:,} 个簇**。")
    A("")
    A("### 去冗余后各阶段阴性簇数")
    A("")
    A("| 阶段 | 阴性记录 | 阴性簇 | ≥10,000? |")
    A("|---|---|---|---|")
    for st in ("clone", "express", "soluble", "purify", "stable", "bind"):
        n = rec.filter(pl.col(f"label_{st}") == 0).height
        c = g1["negatives_clusters_per_stage"][st]
        A(f"| {st} | {n:,} | **{c:,}** | {'✅' if c >= 10000 else '❌'} |")
    _se = rec.filter(pl.col("label_soluble_expression") == 0)
    _sec = g1["negatives_clusters_per_stage"].get("soluble_expression")
    if _sec:
        A(f"| *(派生) soluble_expression* | *{_se.height:,}* | *{_sec:,}* | "
          f"*{'✅' if _sec >= 10000 else '❌'} 但非阶段, 仅为对齐 SoluProt* |")
    A("")
    A(f"**GATE 1 不通过**: 需要 3 个阶段的阴性簇 ≥10,000, 实际只有 "
      f"{sum(1 for st in ('clone','express','soluble','purify','stable','bind') if g1['negatives_clusters_per_stage'][st] >= 10000)} 个。")
    A("")
    A("### 四套冻结切分 (全部通过泄漏检查)")
    A("")
    A("| 切分 | 检验什么 | test 记录 | 跨侧 >30% 相似对 |")
    A("|---|---|---|---|")
    zh = {"sequence": "序列 (30% 相似度传递闭包整组划分)",
          "lab": "实验室 (留出 CSGID+EFI)",
          "time": "时间 (DS1 2015+)",
          "bind_target": "bind 跨靶点 (留出 IL7Ra/TrkA/Mdm2)"}
    what = {"sequence": "基本泛化", "lab": "是否学到实验室偏差",
            "time": "时代漂移", "bind_target": "面对新靶点能否预测 binder"}
    for k in ("sequence", "lab", "time", "bind_target"):
        s = mf["splits"][k]["sides"]["test"]
        lk = leak.get(k, {})
        A(f"| {zh[k]} | {what[k]} | {s['records']:,} | "
          f"**{lk.get('cross_pairs_over_30pct', '?')}** ({lk.get('verdict','?')}) |")
    A("")
    A("### 基线 (主数字一律 `PR-AUC/base`; 正类 = 失败)")
    A("")
    if gb and sol:
        gi = {(x["split"], x["stage"]): x["overall"]["pr_auc_over_base"]
              for x in gb["results"] if "overall" in x}
        # SoluProt 的产物是平铺的 (指标直接在行上, 没有 overall 嵌套)
        si = {(x["split"], x["variant"]): x.get("pr_auc_over_base")
              for x in sol["results"]
              if x.get("task") == "soluble_expression" and "pr_auc_over_base" in x}
        A("对齐 SoluProt 的复合口径 `soluble_expression`:")
        A("")
        A("| 方法 | 序列切分 | 实验室切分 |")
        A("|---|---|---|")
        for nm, a, b in (("SoluProt (全集)", si.get(("sequence", "full")), si.get(("lab", "full"))),
                         ("SoluProt (去污染子集)", si.get(("sequence", "clean")), si.get(("lab", "clean"))),
                         ("**氨基酸组成 GBDT (无长度)**", gi.get(("sequence", "soluble_expression")),
                          gi.get(("lab", "soluble_expression")))):
            f = lambda v: f"{v:.2f}" if v is not None else "—"   # noqa: E731
            A(f"| {nm} | {f(a)} | {f(b)} |")
        A("")
    A("### 阶段三 (L0/L1/L2) — `express` 逐折情形")
    A("")
    if bs:
        A("| 折 | L0 GBDT | L1 冻结 | L2 LoRA |")
        A("|---|---|---|---|")

        def cell(tag, f):
            c = (bs["results"].get(tag) or {}).get(str(f))
            if not c:
                return "未跑"
            adv = (bs["results"].get(tag + "_adv") or {}).get(str(f))
            k = c["effect_size_class"]
            if k == "真信号" and adv and adv["primary_ci_crosses_1"]:
                k = "边界情形"
            q = c["pr_auc_over_base"]
            return f"{q['point']:.2f} [{q['ci95'][0]:.2f}, {q['ci95'][1]:.2f}] {k}"
        for f in (1, 2, 3, 4):
            if not (bs["results"].get("compare_levels_express:L0") or {}).get(str(f)):
                continue
            A(f"| {f} | {cell('compare_levels_express:L0', f)} | "
              f"{cell('L1_esm2_650M_frozen_express', f)} | "
              f"{cell('L2_esm2_650M_lora_express', f)} |")
        A("")
        A("判定规则 (区间定方向、点估计定幅度 + 对抗稳健性降级) 见 "
          "`configs/stage3_train.yaml` 的 `changelog [2026-10-02]`。"
          "**平凡基线 L0 在唯一有稳定真信号的折上不低于 PLM** —— 这是如实报告的负面结果。")
        A("")
    A(END)

    txt = README.read_text(encoding="utf8")
    if BEGIN in txt and END in txt:
        pre = txt[:txt.index(BEGIN)]
        post = txt[txt.index(END) + len(END):]
        README.write_text(pre + "\n".join(L) + post, encoding="utf8")
        print(f"更新 README.md 的 STATUS 区块 ({len(L)} 行)")
    else:
        pathlib.Path("README_STATUS_BLOCK.md").write_text("\n".join(L), encoding="utf8")
        print("README.md 里没有 STATUS 标记 —— 区块写到 README_STATUS_BLOCK.md, "
              "请把它粘到「当前进度」下并加上标记")


if __name__ == "__main__":
    main()
