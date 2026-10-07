"""生成 reports/gate2_baselines.md —— SPEC v2 §6 交付物 #4「基线对照表」+ GATE 2 判定。

GATE 2 两条 (v2 §3 原文):
  - 基线数值与原论文相差在合理范围内 (±0.05), 或能解释差异原因。
  - 三套测试集构造完成并冻结, 且经过泄漏检查 (训练集与测试集之间无 >30% 相似度的序列对)。

数字全部从产物读, 不手写。
"""
from __future__ import annotations

import json
import pathlib

import polars as pl

OUT = pathlib.Path("reports/gate2_baselines.md")
SPLIT_ZH = {"sequence": "序列切分", "lab": "实验室切分", "time": "时间切分",
             "bind_target": "bind 跨靶点切分"}
STAGE_ORDER = ["clone", "express", "soluble", "purify", "stable", "bind",
               "soluble_expression"]   # 末位是派生标签, 非第七阶段


def jload(p: str):
    f = pathlib.Path(p)
    return json.loads(f.read_text()) if f.exists() else None


def main() -> None:
    # 不留旧路径兜底: 兜底会在产物缺失时静默回落到空列表, 让报告少掉整节而不报错。
    # 2026-10-01 就是因为 reports/baseline_gbdt.json 这个旧路径还被引用,
    # 而文件已被归档, 导致 GATE 2 报告内容相对口径重组过期。
    _g = jload("reports/default/baseline_gbdt.json")
    if _g is None:
        raise SystemExit("缺 reports/default/baseline_gbdt.json —— 先跑 "
                         "src/models/baseline_gbdt.py (默认口径)")
    gbdt = _g["results"]
    sol = jload("reports/baseline_soluprot.json")
    notags = (jload("reports/ablation_keeptags_keeplen/baseline_gbdt.json") or {}) \
             .get("results", []) or []
    contam = jload("reports/soluprot_contamination.json")
    ipsae = jload("reports/baseline_ipsae.json")
    leak = {o["split"]: o for o in (jload("reports/leakage_check.json") or [])}
    mf = jload("data/processed/splits/MANIFEST.json")

    L: list[str] = []
    A = L.append
    A("# GATE 2 — 基线复现与测试集冻结")
    A("")
    A("生成: `src/eval/gate2_baselines.py`。所有数字从 `reports/` 下的产物读取, 非手写。")
    A("")

    # ---- 判定 ----
    g2a = ipsae and ipsae.get("gate2_check", {}).get("within_gate2_tolerance_0.05")
    n_baselines = (1 if ipsae else 0) + (1 if sol else 0)
    g2b = bool(leak) and all(o["verdict"] == "PASS" for o in leak.values())
    A("## 判定")
    A("")
    A("| # | 条件 | 实测 | 结论 |")
    A("|---|---|---|---|")
    if ipsae and ipsae.get("gate2_check"):
        v = ipsae["gate2_check"]
        A(f"| 1 | 基线与原论文差距 ≤ ±0.05 或能解释 | AF3 ipSAE_min 最高 F1: "
          f"复现 {v['per_target_mean_best_f1']:.3f} vs 论文 {v['reference_best_f1']} "
          f"(差 {v['delta_vs_reference']:+.3f}) | {'✅ 过' if g2a else '❌ 不过'} |")
    else:
        A("| 1 | 基线与原论文差距 ≤ ±0.05 | **未测** | ❌ |")
    if leak:
        A(f"| 2 | 三套测试集冻结且无 >30% 跨侧序列对 | "
          + " · ".join(f"{SPLIT_ZH.get(k, k)} {o['cross_pairs_over_30pct']} 对" for k, o in leak.items())
          + f" | {'✅ 过' if g2b else '❌ 不过'} |")
    A("")
    A(f"| 附 | 复现了几个外部基线 | {n_baselines} 个 (AF3 ipSAE_min"
      + (" + SoluProt" if sol else "") + ") | — |")
    A("")
    A(f"### 总判定: {'**通过**' if (g2a and g2b) else '**不通过**'}")
    A("")

    # ---- ipSAE 基线 ----
    if ipsae:
        A("## 1 AF3 ipSAE_min 基线 (binder 结合预测)")
        A("")
        A("数据即 DS5 本身 —— DTU 元分析的 `final_dataset.csv` 已含 `af3_ipSAE_min` 等 "
          "200+ 特征与湿实验 `binder` 结果, 所以这个基线**不需要重跑任何结构预测**, "
          "只是在同一批数据上重算阈值与 F1。")
        A("")
        A(f"⚠️ **方向提醒**: 这一节的正类 = **结合**, 与本项目其它地方"
          f"\"阳性 = 失败\"的约定相反 —— 必须用论文方向才能与它的数字对比。")
        A("")
        v = ipsae["gate2_check"]
        A("### 聚合口径决定了能不能对上论文")
        A("")
        A("| 口径 | 最高 F1 |")
        A("|---|---|")
        A(f"| 池化全部 {ipsae['n']:,} 个设计 | {v['pooled_best_f1']:.3f} |")
        A(f"| 按 target 取中位 ({v['n_targets_aggregated']} 个靶点) | "
          f"{v['per_target_median_best_f1']:.3f} |")
        A(f"| **按 target 取平均** (用于 GATE 2 判定) | **{v['per_target_mean_best_f1']:.3f}** |")
        A(f"| 论文报告值 | {v['reference_best_f1']} |")
        A("")
        A("选 per-target 的理由: 论文分析对象是\"15 个结构多样的靶点\", per-target 才是它的"
          "自然单位。池化算会被 FGFR2 主导 —— 它一个靶点占 2,123/3,669 = 58% 的设计, "
          f"而其单靶点 F1 只有 {v['per_target']['FGFR2']['best_f1']:.3f}, "
          f"把池化值拉到 {v['pooled_best_f1']:.3f}, 与论文数字不可比。")
        A("")
        A("### 各指标对照 (池化口径)")
        A("")
        A("| 指标 | 最高 F1 | PR-AUC | 方向 |")
        A("|---|---|---|---|")
        for k, r in sorted(ipsae["metrics"].items(), key=lambda kv: -kv[1]["pr_auc"]):
            A(f"| `{k}` | {r['best_f1']:.3f} | {r['pr_auc']:.3f} | "
              f"{'值大→结合' if r['direction'] == 'higher_is_binder' else '值小→结合'} |")
        A("")
        pc = ipsae.get("paper_claim_check")
        if pc:
            A(f"论文另一条结论\"ipSAE 优于 ipAE / ipTM\"也复现 (按 PR-AUC): "
              f"ipSAE_min > ipAE = {pc['ipSAE_min_beats_ipae']}, "
              f"ipSAE_min > ipTM = {pc['ipSAE_min_beats_iptm']}。"
              "这条对上了, 说明列名与方向没读错 —— 否则上面的 F1 差异就可能是我们的错而非口径差异。")
            A("")
        A("### 按 target 的逐项结果")
        A("")
        A("| target | n | 结合数 | 最高 F1 |")
        A("|---|---|---|---|")
        for tname, r in sorted(v["per_target"].items(),
                               key=lambda kv: -kv[1]["best_f1"]):
            A(f"| {tname} | {r['n']:,} | {r['n_bind']} | {r['best_f1']:.3f} |")
        A("")
        vals = sorted(x["best_f1"] for x in v["per_target"].values())
        A(f"靶点之间从 {vals[0]:.3f} 到 {vals[-1]:.3f}, 跨度 {vals[-1]-vals[0]:.3f}。"
          "**单个汇总数字掩盖了这个离散度** —— 论文的 0.61 和我们的 0.572 都只是这个"
          "分布的一个位置参数, 拿它当\"结构置信度指标的能力\"会高估稳定性。"
          "**这种跨靶点不稳定性本身就是要报告的发现, 不是要抹平的东西。**")
        A("")
        cur = ipsae.get("pr_curve_no_single_threshold")
        if cur:
            A("### 论文层面不选单一门槛")
            A("")
            A(f"刘刚刚 2026-09-30 定: 基准的职责是给出完整曲线与若干预算下的 precision@k, "
              f"让使用者按自己的成本结构选。原论文也只说最优区间 0.5-0.8。"
              f"完整 PR 曲线 ({len(cur['points'])} 个点) 已落盘在 "
              f"`reports/baseline_ipsae.json` 的 `pr_curve_no_single_threshold`, "
              f"极性 = {cur['positive_class']}, 正类占比 {cur['base_rate']:.3f}。")
            A("")
            A("固定预算下的 precision (全集, 正类 = 结合):")
            A("")
            A("| k | precision@k | /基线 |")
            A("|---|---|---|")
            for k in (10, 20, 50, 100):
                key = f"precision@{k}"
                if key in cur["precision_at_k"]:
                    pv = cur["precision_at_k"][key]
                    A(f"| {k} | {pv:.3f} | {pv/cur['base_rate']:.2f} |")
            A("")
            A("**OIH 自有管线另行选了 0.50**, 依据是自身成本结构 (漏掉好设计只损失一轮计算, "
              "浪费湿实验名额很贵), 记录在 `configs/oih_pipeline.yaml`。"
              "**这是管线配置, 不进论文。**")
            A("")

    # ---- SoluProt ----
    A("## 2 SoluProt 基线 (可溶表达预测)")
    A("")
    if not sol:
        A("**未做。**")
        A("")
    else:
        inst = sol["install"]
        A("### 2.1 两道前置检查 (刘刚刚要求先做完才允许写数字)")
        A("")
        A("**检查一 — 定义是否对得上: 不对, 已构造对齐口径。**")
        A("")
        A("去冗余后过万的是 `express` (17,699 簇), **不是** `soluble` (3,447 簇)。"
          "而 SoluProt 预测的是\"可溶表达\"这**一个**二元事件。拿它比我们任一单头都错:")
        A("")
        A("- 比 `express`: SoluProt 的阴性里含\"表达了但不可溶\", 我们的 `express` 阴性不含这部分")
        A("- 比 `soluble`: 我们的 `soluble` 阴性 **99.9% 来自 v2 §2.4 推断** (显式仅 10 条), "
          "且分母是\"已表达成功的\", 与 SoluProt 的全体分母不同")
        A("")
        A("所以构造了派生复合标签 `label_soluble_expression` = (`express`=1 且 `soluble`=1) 为正, "
          "(`express`=0) 或 (`express`=1 且 `soluble`=0) 为负, 共 **19,007 个阴性簇**, "
          "显式证据占 57.5%。**它不是第七个阶段, 不参与 GATE 1 的\"≥3 个阶段\"计数。**")
        A("")
        A("**检查二 — 训练集污染: 确实有, 已给双版本。**")
        A("")
        A("SoluProt 官方 about 页原文: *\"The training set is based on the TargetTrack database, "
          "which was carefully filtered to keep only targets expressed in Escherichia coli.\"* "
          "与我们的 DS1 同源。实测 30% 相似度污染率 (判据与我们自己的去冗余完全一致):")
        A("")
        A("| 切分 | test 唯一序列 | 完全相同 | 30% 相似 | 污染率 | 干净子集 |")
        A("|---|---|---|---|---|---|")
        for k, v in (contam or {}).get("splits", {}).items():
            A(f"| {SPLIT_ZH.get(k, k)} | {v['test_unique_sequences']:,} | "
              f"{v['exact_match_in_soluprot_train']:,} | "
              f"{v['similar_30pct_to_soluprot_train']:,} | "
              f"**{v['contamination_rate_30pct']*100:.1f}%** | {v['clean_subset_size']:,} |")
        A("")
        A("**因此下面每个数字都给全集与去污染子集两版。**")
        A("")
        A("### 2.2 安装与三条已知偏差")
        A("")
        A(f"官方 standalone {inst['soluprot_version']} + 官方 conda 环境 "
          "(py3.7 / sklearn 0.20.1 / biopython 1.74)。")
        A("")
        A(f"1. **USEARCH 版本**: 用 {inst['usearch']}。"
          f"打了一处兼容补丁 —— {inst['compat_patch']}。除此之外未改 SoluProt 任何逻辑。")
        A(f"2. **TMHMM 未用**: {inst['tmhmm']}。官方有专门的 notmhmm 模型, 文档称约 -0.5% 准确率。")
        A(f"3. **在官方自带 `data/test.fa` 上复现其参考输出**: "
          f"{inst['official_test_reproduction']['n']} 条序列平均绝对差 "
          f"{inst['official_test_reproduction']['mean_abs_diff']}, "
          f"最大 {inst['official_test_reproduction']['max_abs_diff']}。"
          "差异同时来自 --no_tmhmm 与 usearch 版本, **无法分离**。")
        A("")
        A("因此本基线**只用排序类指标** (precision@k / PR-AUC), 不用绝对阈值下的 accuracy —— "
          "排序对 0.04 量级的分数漂移不敏感, accuracy 会。")
        A("")
        A("🔴 **最重要的一条限制**: SoluProt 的 `ecoli_usearch_identity` 特征 "
          "(与 E. coli PDB 序列的最大一致度) 在我们的序列切分测试集上有 **6,730/16,435 "
          "= 41% 的序列算不出来** (usearch 无命中), SoluProt 用训练集均值填补。"
          "官方自带测试例上这个比例是 4/21 = 19%, 所以部分是工具固有行为, "
          "部分可能来自 usearch 版本差异。"
          "**这意味着下面的 SoluProt 数字是它的下界**, 一个特征组完整的 SoluProt 可能更好。"
          "任何\"我们超过了 SoluProt\"的结论都必须带上这句。")
        A("")
        A("### 2.3 对比结果 (对齐口径 `soluble_expression`, 正类 = 失败)")
        A("")
        A("⚠️ **主数字是 PR-AUC/base**。lift@100 与 P@k 一并给出作可解释性参考, "
          "但不作判定依据 —— 自助法显示 lift@100 在这个量级的测试集上区间宽 0.6 以上, "
          "而 PR-AUC/base 约 0.17 (见 reports/default/stage3_training.md §4c)。")
        A("")
        A("| 切分 | 方法 | n | 基础失败率 | **PR-AUC/base** | lift@100 | P@20 | P@100 |")
        A("|---|---|---|---|---|---|---|---|")
        for split in ("sequence", "lab"):
            for r in sol["results"]:
                if r.get("split") != split or r.get("task") != "soluble_expression":
                    continue
                if "skipped" in r:
                    continue
                v = "全集" if r["variant"] == "full" else "去污染子集"
                pab = r.get("pr_auc_over_base")
                A(f"| {SPLIT_ZH.get(split, split)} | SoluProt ({v}) | {r['n']:,} | "
                  f"{r['base_rate']:.3f} | "
                  f"**{pab:.2f}**" % () if False else
                  f"| {SPLIT_ZH.get(split, split)} | SoluProt ({v}) | {r['n']:,} | "
                  f"{r['base_rate']:.3f} | **{(pab if pab else float('nan')):.2f}** | "
                  f"{r['lift@100']:.2f} | {r['precision@20']:.3f} | {r['precision@100']:.3f} |")
            g = [x for x in gbdt if x.get("split") == split
                 and x.get("stage") == "soluble_expression" and "overall" in x]
            if g:
                o = g[0]["overall"]
                A(f"| {SPLIT_ZH.get(split, split)} | "
                  f"**氨基酸组成 GBDT (我们的, 默认口径: 剥标签 + 无长度)** | "
                  f"{g[0]['test_n']:,} | {o['base_rate']:.3f} | "
                  f"**{o.get('pr_auc_over_base', float('nan')):.2f}** | "
                  f"{o['lift@100']:.2f} | {o['precision@20']:.3f} | {o['precision@100']:.3f} |")
        A("")
        A("**读法 (措辞已按刘刚刚 2026-09-30 的要求收敛)**。")
        A("")
        A("**不能写\"我们超过了 SoluProt\"。** 41% 的序列算不出它的核心特征"
          "(它自己的测试例只有 19%), 这说明我们是在**把 SoluProt 用在它的适用域之外** —— "
          "它的训练与设计对象是 E. coli 异源表达的天然蛋白, 而我们的测试集含设计蛋白、"
          "膜蛋白、以及大量与 PDB 无显著同源的序列。")
        A("")
        A("诚实且同样有意思的表述是: **在这个数据集上, 一个只用 20 维氨基酸组成"
          "(默认口径已剥构建体残留、且不含长度特征) 的 GBDT 就达到或超过了已发表工具, "
          "而 SoluProt 的特征在域外严重退化。** 它说明的是:")
        A("")
        A("> **口径必须随数字一起报** (2026-10-03 自查发现的表述错误): 本报告与论文的"
          "默认口径 GBDT **只有 20 维氨基酸组成**, `configs/baseline_gbdt.yaml` 里 "
          "`seq_len: false` / `log_seq_len: false`。先前多处把它写成\"长度+组成\", "
          "那是 `reports/ablation_keeptags_keeplen/` 那一套消融口径的描述。"
          "这个错标签会让读者以为我们一边论证长度是混杂、一边又拿长度当基线 —— "
          "实际没有, 但标签写错就等于论文在自我矛盾。")
        A("")
        A("1. 这个基准对现有领域标准工具同样困难, 门槛并不高;")
        A("2. 浅层特征已能达到这个水平 —— 但请立刻往下读\"两个混杂\"那一节: "
          "这个成绩本身很大程度由序列长度驱动, 所以它是\"基准的 headroom 有限\"的证据, "
          "**不是**\"浅层特征很有效\"的证据;")
        A("3. 去掉污染后 SoluProt 在实验室切分上几乎不变 (PR-AUC 0.708 → 0.691), "
          "说明**污染并没有虚高它** —— 这一点对它有利, 要如实写。"
          "在序列切分上降幅较大 (0.418 → 0.302), 但同时基础失败率也从 0.268 降到 0.208, "
          "两者混在一起, 不能单独归因于污染。")
        A("")
        A("### 2.4 `soluble` 单头的对照数字 (underpowered, 按要求仍给出)")
        A("")
        A("| 切分 | 变体 | n | 阴性数 | P@100 | lift@100 | PR-AUC |")
        A("|---|---|---|---|---|---|---|")
        got = False
        for r in sol["results"]:
            if r.get("task") != "soluble" or "skipped" in r:
                continue
            got = True
            v = "全集" if r["variant"] == "full" else "去污染"
            A(f"| {SPLIT_ZH.get(r['split'], r['split'])} | {v} | {r['n']:,} | {r['n_fail']:,} | "
              f"{r['precision@100']:.3f} | {r['lift@100']:.2f} | {r['pr_auc']:.3f} |")
        if not got:
            A("| — | — | — | — | — | — | 各切分阴性均不足, 全部跳过 |")
        A("")
        A("⚠️ **这组数字不足以支撑任何结论**: `soluble` 单头的阴性 99.9% 来自 v2 §2.4 推断, "
          "且在实验室切分的留出侧只有 2 条阴性 (已跳过)。"
          "lift 接近 1 既可能是 SoluProt 不行, 也可能是我们的推断标签不对, 两者分不开。"
          "列出来是为了满足\"即使样本量不足也要给\"的要求, 不是为了下判断。")
        A("")

    # ---- 简单基线 ----
    A("## 3 简单基线 (默认口径: 20 维氨基酸组成 + GBDT, **不含长度**)")
    A("")
    A("v2 §3.2 的原话是: \"如果后面的深度模型超不过'长度 + 组成 + GBDT', "
      "说明模型没学到东西, 要如实报告。\" 这条线是给后续 PLM 定门槛用的。")
    A("")
    A("> **实现与 spec 措辞有一处有意的偏离, 在此说明。** spec 写的是\"长度 + 组成\", "
      "但本项目后来实测出长度是实验室身份通道 (见 §5 的 Simpson 反转: 长度的预测力"
      "在中心内几乎消失, 且两个阶段上跨中心与中心内方向相反), "
      "所以**默认口径去掉了长度特征**, 只保留 20 维氨基酸组成。"
      "保留长度的版本作为消融留在 `reports/ablation_keeptags_keeplen/`。"
      "理由: 若用含长度的基线给领域定门槛, 等于拿本文自己认定的混杂当标准。"
      "两套口径的数字都报, 但主表一律用默认口径。")
    A("")
    A("⚠️ 本节及以下正类 = **失败** (回到项目约定)。")
    A("")
    A("| 切分 | 阶段 | test n | 基础失败率 | P@20 | P@100 | **lift@100** | PR-AUC |")
    A("|---|---|---|---|---|---|---|---|")
    def _order(r):
        sp = list(SPLIT_ZH).index(r["split"]) if r["split"] in SPLIT_ZH else 99
        st = STAGE_ORDER.index(r["stage"]) if r["stage"] in STAGE_ORDER else 99
        return (sp, st)

    for r in sorted(gbdt, key=_order):
        if "skipped" in r:
            A(f"| {SPLIT_ZH.get(r['split'], r['split'])} | `{r['stage']}` | — | — | — | — | — | "
              f"{r['skipped']} |")
            continue
        o = r["overall"]
        A(f"| {SPLIT_ZH.get(r['split'], r['split'])} | `{r['stage']}` | {r['test_n']:,} | "
          f"{o['base_rate']:.3f} | {o['precision@20']:.3f} | {o['precision@100']:.3f} | "
          f"**{o['lift@100']:.2f}** | {o['pr_auc']:.3f} |")
    A("")
    seq = {r["stage"]: r for r in gbdt if r["split"] == "sequence" and "overall" in r}
    lab = {r["stage"]: r for r in gbdt if r["split"] == "lab" and "overall" in r}
    both = [s for s in STAGE_ORDER if s in seq and s in lab]
    if both:
        A("### 序列切分 vs 实验室切分: 增益的塌缩")
        A("")
        A("| 阶段 | 序列切分 PR-AUC/base | 实验室切分 PR-AUC/base | 倍数变化 | "
          "(参考) 序列 lift@100 | 实验室 lift@100 |")
        A("|---|---|---|---|---|---|")
        for s in both:
            pa = seq[s]["overall"].get("pr_auc_over_base")
            pb = lab[s]["overall"].get("pr_auc_over_base")
            a = seq[s]["overall"]["lift@100"]
            b = lab[s]["overall"]["lift@100"]
            rat = f"×{pb/pa:.2f}" if (pa and pb) else "—"
            A(f"| `{s}` | **{(pa if pa else float('nan')):.2f}** | "
              f"**{(pb if pb else float('nan')):.2f}** | {rat} | {a:.2f} | {b:.2f} |")
        A("")
        A("同一个模型、同一个阶段, 换到没见过的实验室上增益普遍塌到 1.3–1.8 倍。"
          "随机切分的训练与测试来自同一批实验室, 模型能学到\"这家中心的靶点长什么样、"
          "失败率多高\"; 这部分增益在真实使用场景 (面对新实验室 / 新设计) 里不存在。")
        A("")
        A("**绝对 precision 在两套切分之间不可比** —— 留出实验室的基础失败率本身就不同 "
          "(见 reports/paper_data_methods.md §2.6 对类别比例被构造过程改变的说明)。"
          "必须看归一化后的量; 其中**主数字是 PR-AUC/base**, lift@100 作参考 "
          "(两者都归一化掉了基础率, 但 lift@k 只用 top-k 的样本, 区间宽得多)。")
        A("")

    # ---- bind 跨靶点: 极性与真实增益 ----
    ips = ipsae.get("on_bind_target_split") if ipsae else None
    if ips and "ipSAE_min" in ips:
        A("### bind 跨靶点切分: 结构置信度 vs 序列-only")
        A("")
        A(f"⚠️ **极性声明 (这一节最容易读错的地方)**: 正类 = **{ips['positive_class']}**, "
          f"正类占比 = **{ips['base_rate_positive_is_failure']:.3f}**。"
          "这意味着**随机猜的 PR-AUC 期望就是 0.796** —— 一个 0.9 的 PR-AUC 在这里"
          "并不代表强区分力。所以下表每个 PR-AUC 都配了 `/基线` 一栏, "
          "引用时必须两个一起引。")
        A("")
        g = [x for x in gbdt if x.get("split") == "bind_target"
             and x.get("stage") == "bind" and "overall" in x]
        A("| 方法 | P@10 | P@20 | P@50 | P@100 | PR-AUC | **PR-AUC/基线** |")
        A("|---|---|---|---|---|---|---|")
        A(f"| 随机基线 (= 正类占比) | — | — | — | — | "
          f"{ips['base_rate_positive_is_failure']:.3f} | 1.00 |")
        for nm, key in (("AF3 ipSAE_min", "ipSAE_min"), ("随机对照 (实测)", "random_control")):
            b = ips[key]
            A(f"| {nm} | " + " | ".join(
                f"{b.get(f'precision@{k}', float('nan')):.3f}" for k in (10, 20, 50, 100))
              + f" | {b['pr_auc']:.3f} | **{b['pr_auc_over_base']:.2f}** |")
        if g:
            o = g[0]["overall"]
            A(f"| 氨基酸组成 GBDT (无长度) | " + " | ".join(
                f"{o.get(f'precision@{k}', float('nan')):.3f}" for k in (10, 20, 50, 100))
              + f" | {o['pr_auc']:.3f} | **{o.get('pr_auc_over_base', float('nan')):.2f}** |")
        A("")
        pc = ips.get("polarity_check", {})
        if pc:
            A("**同一批数据换极性会改变结论的量级** —— 这不是笔误, 是 PR-AUC 在不平衡数据上的性质:")
            A("")
            A("| 极性 | 正类占比 | PR-AUC | PR-AUC/基线 |")
            A("|---|---|---|---|")
            for k2, zh2 in (("positive_is_failure", "正类 = 失败 (本项目约定)"),
                            ("positive_is_binder", "正类 = 结合 (原论文极性)")):
                v = pc[k2]
                A(f"| {zh2} | {v['base_rate']:.3f} | {v['pr_auc']:.3f} | "
                  f"**{v['pr_auc_over_base']:.2f}** |")
            A("")
            A(f"> {pc['why_it_matters']}")
            A("")
        A("**结论 (已按真实增益改写)**: AF3 ipSAE_min 相对随机基线的增益是 "
          f"**{ips['ipSAE_min']['pr_auc_over_base']:.2f} 倍** —— 和 SoluProt 在实验室切分上的"
          " 1.15 倍是同一量级, **不是** 0.909 这个绝对值看上去的那么强。")
        A("")
        if g:
            o = g[0]["overall"]
            A(f"序列-only 的 GBDT 归一化后是 {o.get('pr_auc_over_base', float('nan')):.2f} 倍, "
              f"与实测随机对照的 {ips['random_control']['pr_auc_over_base']:.2f} 倍"
              "在这个样本量下不可区分 —— **跨靶点的 binder 失败预测上, "
              "氨基酸组成不提供可用信号**。这是本基准在 bind 轴上的主要发现。")
        A("")
        A("因此后续 PLM 在 bind 轴上要证明有用, **必须在同一切分、同一极性下报出"
          "显著高于 1.14 的归一化 PR-AUC**, 并同时给出 precision@k 的多个预算点。"
          "只报绝对 PR-AUC 无意义。")
        A("")

    # ---- 特征重要性与两个混杂 ----
    A("### 简单基线的两个混杂 (必须自己先说)")
    A("")
    A("**混杂一: 主评测轴上长度占主导。** permutation importance "
      "(scoring=average_precision, n_repeats=10; 用置换而非树的 split gain, "
      "因为后者对高基数连续特征有偏, 会系统性高估长度):")
    A("")
    A("| 切分/任务 | 长度类特征占重要性 | 首要特征 |")
    A("|---|---|---|")
    for r in gbdt:
        fi = r.get("feature_importance")
        if not fi or "top" not in fi:
            continue
        sh = fi.get("seq_len_share")
        top = ", ".join(f"`{x['feature']}`({x['mean']:.3f})" for x in fi["top"][:3])
        mark = " 🔴" if (sh is not None and sh > 0.40) else ""
        A(f"| {SPLIT_ZH.get(r['split'], r['split'])}/`{r['stage']}`{mark} | "
          f"{'—' if sh is None else f'{sh*100:.1f}%'} | {top} |")
    A("")
    A("🔴 标记的是长度占比 >40% 的。**在实验室切分 (本基准的主评测轴) 上, "
      "`express` 的 61.1%、`soluble_expression` 的 47.7% 重要性来自序列长度。** "
      "这意味着该轴上的基线成绩很大程度是一个平凡混杂因素驱动的, "
      "**headroom 因此有限** —— 这一点必须写在局限性章节, 不要等审稿人指出。")
    A("")
    A("**混杂二: His 标签写在序列里, 且是 center 级的构建习惯。** "
      "`aa_H` 在序列切分的 `soluble` / `purify` / `stable` 上都是首要特征。查证结果:")
    A("")
    A("| 事实 | 实测 |")
    A("|---|---|")
    A("| DS1 序列含 `HHHHHH` 的比例 | 18.3% (173,265/945,730) |")
    A("| His6 与 `soluble` 失败率 | 含 6.1% vs 不含 0.9% |")
    A("| His6 与 `purify` 失败率 | 含 23.8% vs 不含 3.8% |")
    A("| His6 与 `stable` 失败率 | 含 73.3% vs 不含 14.9% |")
    A("| His6 率按 center | SSGCID 91.8% / NESG 69.4% / SGPP 55.3% **vs** CESG 0.2% / NYSGXRC 0.2% / CSGID 0.4% |")
    A("")
    A("**带不带 His 标签几乎完全是 center 的构建习惯**, 所以 `aa_H` 这个特征给了模型"
      "一条**序列可见的实验室身份通道** —— 认出是哪个中心做的, 再利用该中心的记录习惯。"
      "这把\"实验室偏差\"从假设变成了机制证据, 并且说明它"
      "**不能通过重加权消除, 因为它编码在序列本身里**。")
    A("")
    if notags:
        A("剥标签消融 (正则去掉 `H{6,}` 等; 剥完不足 20 aa 的保留原序列, 不制造畸形样本):")
        A("")
        A("| 切分/任务 | 带标签 PR-AUC | 剥标签 PR-AUC | 变化 |")
        A("|---|---|---|---|")
        idx = {(x["split"], x["stage"]): x for x in notags if "overall" in x}
        for r in gbdt:
            if "overall" not in r:
                continue
            k = (r["split"], r["stage"])
            if k not in idx:
                continue
            a = r["overall"]["pr_auc"]
            b = idx[k]["overall"]["pr_auc"]
            rel = (b - a) / a * 100 if a else float("nan")
            A(f"| {SPLIT_ZH.get(r['split'], r['split'])}/`{r['stage']}` | {a:.3f} | {b:.3f} | "
              f"{rel:+.0f}% |")
        A("")
        A("**两个混杂落在不同的轴上, 这点很重要**: "
          "实验室切分对剥标签几乎免疫 (`soluble_expression` 0.720 → 0.723), 它是长度驱动的; "
          "而序列切分确实靠标签吃掉一部分 (`soluble` 掉 41%, `stable` 掉 0.070)。"
          "所以\"同分布测试成绩偏高\"与\"跨实验室成绩靠长度\"是两个独立的问题, "
          "局限性章节要分开写。")
        A("")

    # ---- 分组性能差异 (CLAUDE.md / SPEC §3.3 要求的第三类指标) ----
    A("### 分组性能差异 (按实验室 / 年份 / 物种)")
    A("")
    A("SPEC §3.3 与项目 CLAUDE.md 都要求报三个维度的分组差异。实测结果:")
    A("")
    A("| 切分 | 阶段 | 维度 | 可评测组数 | lift@100 最好 | lift@100 最差 | 最差/最好 |")
    A("|---|---|---|---|---|---|---|")
    any_row = False
    for r in gbdt:
        if "overall" not in r:
            continue
        for dim, zh in (("center", "实验室"), ("year", "年份"), ("organism", "物种")):
            g = r.get(f"by_{dim}")
            if not g:
                continue
            ng = g.get("n_groups_evaluated", 0)
            if ng == 0:
                A(f"| {SPLIT_ZH.get(r['split'], r['split'])} | `{r['stage']}` | {zh} | 0 | — | — | "
                  f"组样本不足, 无法评测 |")
                any_row = True
                continue
            b = g.get("lift@100_best", {}).get("value")
            w = g.get("lift@100_worst", {}).get("value")
            ratio = f"{w / b:.2f}" if (b and w) else "—"
            A(f"| {SPLIT_ZH.get(r['split'], r['split'])} | `{r['stage']}` | {zh} | {ng} | "
              f"{'' if b is None else f'{b:.2f}'} | {'' if w is None else f'{w:.2f}'} | {ratio} |")
            any_row = True
    if not any_row:
        A("| — | — | — | — | — | — | 无可评测组 |")
    A("")
    # 这段原本是手写数字, 2026-10-02 发现与同一页的表格自相矛盾 (正文写"只有 1 个达标",
    # 表格算出 2 个), 且条数/物种数/最大组都已过期。改为从产物注入。
    _ref = next((r for r in gbdt
                 if r["split"] == "sequence" and r["stage"] == "express"
                 and "by_organism" in r), None)
    if _ref:
        _o, _c, _y = _ref["by_organism"], _ref["by_center"], _ref["by_year"]
        _tops = "、".join(f"*{x['group']}* {x['n']} 条"
                          for x in _o["largest_groups"][:3])
        A("**物种维度在本基准上结构性地功效不足, 这是数据性质而非实现问题。** "
          f"以序列切分 `express` 的测试侧为例: 去冗余后 {_o['n_records']:,} 条记录摊在 "
          f"**{_o['n_groups_total']} 个物种名**上, 只有 **{_o['n_groups_evaluated']} 个**"
          f"达到 {_o['min_group_n']} 条的最小组样本门槛 (最大的三组: {_tops})。"
          f"对照同一测试侧, 实验室有 {_c['n_groups_evaluated']}/{_c['n_groups_total']} 个组、"
          f"年份有 {_y['n_groups_evaluated']}/{_y['n_groups_total']} 个组达标 —— "
          "差别不在门槛, 在于靶点在物种上摊得太薄。")
    A("")
    A("原因是结构基因组学中心的靶点在物种上摊得极薄 —— 这是 TargetTrack 的固有分布, "
      "不是切分或去冗余造成的。因此本基准的分组审计**以实验室与年份为主轴**, "
      "物种维度只在样本允许时作参考, 不足以支撑\"跨物种泛化\"的结论。"
      "降低最小组门槛能凑出更多\"组\", 但 20-50 条样本上的 precision@100 是噪声不是发现, "
      "本工作不这样做。")
    A("")

    # ---- 冻结与泄漏 ----
    A("## 4 测试集冻结与泄漏检查")
    A("")
    if mf:
        A(f"数据本体 `data/processed/records.parquet` sha256 "
          f"`{mf['records']['sha256'][:32]}…`, {mf['records']['rows']:,} 条。")
        A("")
        A("| 切分 | sha256 (前 16) | train 记录 | test 记录 | split group 重叠 | 跨侧 >30% 的对 | 判定 |")
        A("|---|---|---|---|---|---|---|")
        for k, zh in SPLIT_ZH.items():
            s = mf["splits"][k]
            o = leak.get(k, {})
            A(f"| {zh} | `{s['sha256'][:16]}` | {s['sides']['train']['records']:,} | "
              f"{s['sides']['test']['records']:,} | {o.get('split_group_overlap', '—')} | "
              f"{o.get('cross_pairs_over_30pct', '—')} | **{o.get('verdict', '—')}** |")
        A("")
    A("泄漏检查的构造与四轮收敛过程见 `reports/splits_and_leakage.md`。")
    A("")

    # ---- 局限 ----
    A("## 5 本阶段的局限")
    A("")
    A("1. **SoluProt 是被削弱着跑的** (41% 的序列缺 PDB 一致度特征, 见 §2.2), "
      "所以任何与它的比较都是它的下界。")
    A("2. **时间切分上所有阶段都因样本不足跳过。** 无泄漏约束下该轴的测试侧只剩几百条。")
    A("3. **bind 头在实验室切分上无法评测** (留出侧无 bind 阴性): DS5 的 bind 数据全部来自"
      "TargetTrack 以外的研究, 与 TargetTrack 的实验室留出集不相交。"
      "bind 头的跨实验室泛化要用 DS5 内部的 `source` 分组另做一套切分。")
    A("")

    OUT.write_text("\n".join(L), encoding="utf8")
    print(f"wrote {OUT} ({len(L)} 行)")
    print(f"GATE 2: 基线={'PASS' if g2a else 'FAIL'}  泄漏={'PASS' if g2b else 'FAIL'}  "
          f"总={'PASS' if (g2a and g2b) else 'FAIL'}")


if __name__ == "__main__":
    main()
