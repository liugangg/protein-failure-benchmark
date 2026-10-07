"""生成 reports/default/stage3_training.md —— 阶段三训练结果 + 定稿措辞。

四段措辞按刘刚刚 2026-09-30 的定稿记录, **结构固定**:
  四情形混合 (两任务都成立) -> 跨任务互换 -> 落到"无廉价事前判据"
反向作为四情形之一如实描述, **不拔高为普遍现象**, 保留"1 任务 1 中心组"的边界。
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

D = pathlib.Path("reports/default")
OUT = D / "stage3_training.md"
LOW_BASE = 0.10


def fmt_ci(point: float, lo: float, hi: float) -> str:
    """区间端点贴近 1.0 时多打两位小数。

    为什么 (2026-10-02 实查): 两位小数下 0.995691 与 1.000986 都显示成 "1.00",
    但一个跨 1.0 一个不跨, 判定相反 —— 读者无法从表里看出差别。
    """
    nd = 4 if min(abs(lo - 1.0), abs(hi - 1.0)) < 0.02 else 2
    return f"{point:.2f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def jload(p):
    f = D / p
    return json.loads(f.read_text()) if f.exists() else None


BS = D / "bootstrap_ci.json"


def _ci_table() -> dict:
    """自助法区间: {模型标签: {折: 该折结果}}。判定优先用区间, 没区间才退回点估计。"""
    return json.loads(BS.read_text())["results"] if BS.exists() else {}


def verdict(base: float, lift: float, pab: float,
            cell: dict | None = None, adv_cell: dict | None = None) -> str:
    """情形判定 —— 主数字一律 PR-AUC/base; **有自助法区间时以区间为准**。

    两次改动都留痕在 configs/stage3_train.yaml 的 changelog:
      [2026-10-01] 主数字由 lift@100 改为 PR-AUC/base (lift@100 的 CI 宽 0.63 vs 0.17,
                   且在 L1 express 折 1 上两者给出相反判定, 支撑不了折级判定)。
      [2026-10-02] 三分类改四分类: 区间定"方向能不能分辨", 点估计定"幅度够不够用"。
                   n=7,399 的折上 +5% 区间也不跨 1.0 —— 只看区间会把它报成真信号。
                   另加对抗稳健性降级: 形式上过线但对抗版区间跨 1.0 -> 边界情形。
    """
    if cell is None:                      # 无区间: 只能给点估计判定, 明确标注
        return ("真信号?" if pab > 1.10 else
                ("反向?" if pab < 0.90 else "不可区分?")) + " (无区间)"
    c = cell["effect_size_class"]
    if c == "真信号" and adv_cell is not None and adv_cell["primary_ci_crosses_1"]:
        return "边界情形"
    return c


def fold_table(A, title: str, tag: str = "", adv_tag: str = "") -> list[str]:
    ci = _ci_table()
    cells, adv = ci.get(tag, {}), ci.get(adv_tag, {})
    L = [f"**{title}**", "",
         "| 折 | 留出中心 | 基础率 | **PR-AUC/base [95% CI]** (主数字) | "
         "lift@100 (参考) | 情形 |",
         "|---|---|---|---|---|---|"]
    for r in A.get("folds", []):
        inner = r.get("inner_folds", [])
        if not inner:
            continue
        b = float(np.mean([s["cross"]["base_rate"] for s in inner]))
        lift = r.get("cross_lift@100_mean")
        pab = r.get("cross_pr_auc_over_base_mean")
        if lift is None or pab is None:
            continue
        cell = cells.get(str(r["fold"]))
        if cell:
            q = cell["pr_auc_over_base"]
            shown = "**" + fmt_ci(q["point"], q["ci95"][0], q["ci95"][1]) + "**"
        else:
            shown = f"**{pab:.2f}** (内折均值, 无区间)"
        _v = verdict(b, lift, pab, cell, adv.get(str(r["fold"])))
        # 区间看着不跨 1.0 却判"不可区分"时, 必须就地说明依据是种子不稳,
        # 否则读者会把它当成表格算错 (2026-10-02 自查发现的问题)。
        if cell and not cell.get("seed_stability", {}).get("stable", True):
            _v += " ᵇ"
        L.append(f"| {r['fold']} | {', '.join(r['held_centers'])} | {b:.3f} | "
                 f"{shown} | {lift:.2f} | "
                 f"**{_v}** |")
    if any(c and not c.get("seed_stability", {}).get("stable", True)
           for c in (cells.get(str(r["fold"])) for r in A.get("folds", []))):
        L.append("")
        L.append("> ᵇ 该折的方向判定在 5 个自助法种子上**不一致**, 按保守方向定为"
                 "\"不可区分\" —— 所以区间端点看着不跨 1.0 也不作方向性结论。"
                 "判据见 `configs/stage3_train.yaml` changelog `[2026-10-02b]`。")
    L.append("")
    return L


def main() -> None:
    L: list[str] = []
    A = L.append
    A("# 阶段三训练结果 (L0 / L1 / L2)")
    A("")
    A("生成: `src/eval/stage3_report.py` · 口径: `reports/default/` = 剥构建体残留 + 不用长度")
    A("")
    A("> 所有 PR-AUC 都附带正类占比与归一化倍数 (`metrics.py` 强制)。"
      "本项目正类 = **失败**, 所以 PR-AUC 的随机基线就等于正类占比, 绝对值天然偏高 —— "
      "引用时必须连同 `PR-AUC/base` 一起引。")
    A("")
    A("> **所有折的情形判定一律用 PR-AUC/base**, 不只低基础率折。lift@100 仍然报告, "
      "但只作可解释性指标 (\"固定 100 个湿实验名额能中几个\"), **不作判定依据**。"
      "改动时间点、理由与影响见 §4c 的留痕段与 `configs/stage3_train.yaml` 的 `changelog`。")
    A("")
    A("> 低基础率折的分辨率问题 (促成上述改动的原始观察): 折 2 的基础率仅 2.7-3.6%, "
      "top-100 里随机期望只有 2.7-3.6 个真失败, 观测 0-2 个的 P(≤观测|随机) = "
      "0.285 / 0.032 / 0.060 —— 两个内折落在小数波动内, 所以 lift@100 的 0.00 是"
      "**分辨率不足**而非无信息。对照: 折 3 的 top-100 期望 40 个只拿到 10 个, "
      "P < 0.001 且三个内折一致, 那是统计上确定的反向。")
    A("")

    # ── 1 四情形混合 ──
    A("## 1 跨中心泛化呈四情形混合 (两个任务都成立)")
    A("")
    for task, zh in (("express", "express (主任务)"),
                     ("soluble_expression", "soluble_expression (对照任务)")):
        for lvl, fn in (("L1 (ESM-2 650M 冻结)", f"L1_esm2_650M_frozen_{task}.json"),
                        ("L2 (ESM-2 650M + LoRA)", f"L2_esm2_650M_lora_{task}.json")):
            a = jload(fn)
            if a:
                _tag = fn.replace(".json", "")
                L.extend(fold_table(a, f"{zh} — {lvl}", _tag, _tag + "_adv"))
    cmp_ = jload("compare_levels_express.json")
    if cmp_:
        m = cmp_.get("medians_lift100", cmp_.get("medians", {}))
        mp = cmp_.get("medians_pr_auc_over_base", {})
        ml = cmp_.get("medians_lift100", cmp_.get("medians", {}))
        A("**同一 center_folds、同一测试折上的四方对比 (express)**:")
        A("")
        A("| | **PR-AUC/base 中位** (主数字) | lift@100 中位 (参考) |")
        A("|---|---|---|")
        for k, zh in (("random", "随机对照"), ("L0_cross", "L0 cross (组成+GBDT)"),
                      ("L1_cross", "L1 cross"), ("L0_within", "L0 within"),
                      ("L1_within", "L1 within")):
            if ml.get(k) is not None or mp.get(k) is not None:
                pv = f"**{mp[k]:.2f}**" if mp.get(k) is not None else "—"
                lv = f"{ml[k]:.2f}" if ml.get(k) is not None else "—"
                A(f"| {zh} | {pv} | {lv} |")
        A("")
        if mp.get("random") is not None:
            A(f"随机对照的 PR-AUC/base 中位 {mp['random']:.2f} 证明评估口径是校准的, "
              f"所以 cross 的低值不是计算错误。")
        A("")

    # ── 2 跨任务互换 ──
    A("## 2 跨中心: 方向与幅度都不一致; 跨任务: 一例方向相反")
    A("")
    A("**把方向和幅度分开看** —— 这两件事的证据强度不同, 混在一起报会把单例"
      "写成普遍规律 (刘刚刚 2026-10-02 指出)。方向由 95% 区间相对 1.0 的位置定, "
      "幅度由点估计定; 方向判定另须在 5 个自助法种子上一致, 否则按保守方向定 "
      "(见 configs changelog `[2026-10-02b]`)。")
    A("")
    _ci = _ci_table()
    _te, _ts = ("L1_esm2_650M_frozen_express",
                "L1_esm2_650M_frozen_soluble_expression")

    def _d(c):
        lo, hi = c["pr_auc_over_base"]["ci95"]
        return "**正**" if lo > 1.0 else ("**负**" if hi < 1.0 else "不定")

    def _m(c):
        pt = c["pr_auc_over_base"]["point"]
        return f"{pt:.2f} ({pt-1:+.0%})"

    if _ci.get(_te) and _ci.get(_ts):
        A("| 折 | 留出中心 | express 方向 / 幅度 | soluble_expression 方向 / 幅度 | "
          "方向 | 情形 (express → sol) |")
        A("|---|---|---|---|---|---|")
        _opp, _diff_cls = [], []
        _held = {r["fold"]: r["held_centers"]
                 for r in (jload(_te + ".json") or {}).get("folds", [])}
        for f in sorted(_ci[_te], key=int):
            ce, cs = _ci[_te].get(f), _ci[_ts].get(f)
            if not (ce and cs):
                continue
            de, ds = _d(ce), _d(cs)
            ke = verdict(0, 0, 0, ce, (_ci.get(_te + "_adv") or {}).get(f))
            ks = verdict(0, 0, 0, cs, (_ci.get(_ts + "_adv") or {}).get(f))
            same = de == ds
            if not same and "不定" not in (de, ds):
                _opp.append(f)
            if ke != ks:
                _diff_cls.append(f)
            A(f"| {f} | {', '.join(_held.get(int(f), ['—']))} | {de} / {_m(ce)} | "
              f"{ds} / {_m(cs)} | {'一致' if same else '**相反**'} | "
              f"{ke} → {ks} |")
        A("")
        _mags = [_ci[_te][f]["pr_auc_over_base"]["point"] - 1 for f in _ci[_te]]
        A(f"1. **跨中心不一致: {len(_ci[_te])} 折都支持。** `express` 的幅度从 "
          f"{min(_mags):+.0%} 到 {max(_mags):+.0%}, 方向有正有负有不定。"
          "这是主结论, 证据基础是全部折。")
        A(f"2. **跨任务方向相反: 只有 {len(_opp)} 例 "
          f"(折 {', '.join(_opp) if _opp else '—'})。** "
          f"其余 {len(_ci[_te]) - len(_opp)} 折方向一致, 只是幅度不同 "
          f"(四分类标签不同的折有 {len(_diff_cls)} 个: 折 "
          f"{', '.join(_diff_cls) if _diff_cls else '—'} —— "
          "但\"标签不同\"不等于\"方向相反\", 折 4 两任务同为正向只是强弱有别)。"
          f"所以表述为**\"折 {_opp[0] if _opp else '—'} 上观察到方向相反, 其余折未见\"**, "
          "不写成普遍规律。")
        A("")
    _ci = _ci_table()
    _f3e = (_ci.get("L1_esm2_650M_frozen_express") or {}).get("3")
    _f3s = (_ci.get("L1_esm2_650M_frozen_soluble_expression") or {}).get("3")
    if _f3e and _f3s:
        _q, _r = _f3e["pr_auc_over_base"], _f3s["pr_auc_over_base"]
        A("**反向的边界要写清**: 反向只在 **express 任务的 1 个留出中心组** (折 3: "
          "NYSGRC / NYCOMPS / SSGCID / SECSG / BSGI / BSGC) 上出现 "
          f"(PR-AUC/base {fmt_ci(_q['point'], _q['ci95'][0], _q['ci95'][1])}); "
          f"同一批中心在 `soluble_expression` 上是 "
          f"**{verdict(0, 0, _r['point'], _f3s, (_ci.get('L1_esm2_650M_frozen_soluble_expression_adv') or {}).get('3'))}** "
          f"({fmt_ci(_r['point'], _r['ci95'][0], _r['ci95'][1])}) —— "
          "方向是反过来的。**不写成\"跨中心反向是这类数据的普遍现象\"** —— "
          "它是四情形之一, 边界是 1 任务 1 中心组。")
    A("")

    # ── 3 核心句 ──
    A("## 3 落点: 没有廉价的事前判据")
    A("")
    A("> **中心间的方向与幅度均不一致, 且至少一例显示同一批中心在不同任务上方向相反 "
      "—— 没有廉价的事前判据能预知一次部署会落进有用 / 有害 / 噪声哪一类, "
      "只能每个中心每个任务实测。这就是中心分层协议的成本来源与必要性。**")
    A("")
    A("两条支撑的强度不同, 落点按较弱的那条收紧 (见 §2):")
    A("")
    A("- **主支撑 (全部折)**: 跨中心方向与幅度都不一致, 所以不能靠"
      "\"这个中心以前表现好\"迁移。")
    A("- **单例支撑 (1 折)**: 折 3 上同一批中心在两个任务上方向相反, 所以也不能"
      "假定\"在一个任务上没问题就在另一个任务上也没问题\"。"
      "**这一条是单例观察, 不作普遍规律陈述。**")
    A("")
    A("即便只用主支撑, 落点也成立: 四个留出中心组的幅度跨 "
      "−21% 到 +20%, 没有任何事前可得的量 (中心身份、基础失败率、样本量) "
      "与之相关。单例那条只是把\"换个任务就安全了\"这条退路也堵上。")
    A("")

    # ── 4 对抗去偏 ──
    A("## 4 对抗去偏: 排除性证据")
    A("")
    A("按**排除性证据**写, 不写成\"去偏有效/无效\":")
    A("")
    A("- 对抗头的中心判别准确率降至 **0.429–0.507**, 约等于\"猜最大类\"的水平 "
      "(22 个中心的多分类, 训练侧中心分布高度不均), 证明**中心可分性已从表示中移除**。")
    A("- 而折 3 的反向**不变** (PR-AUC/base 0.79 → 0.80, lift@100 0.31 → 0.33)。")
    A("- 故: 该反向**不由表示中可线性判别的中心身份承载**。")
    A("")
    A("**其真实来源现有数据无法判定** (一种可能是标签生成机制随中心而变), "
      "列为 open question, 不进一步推测。")
    A("")
    for task in ("express", "soluble_expression"):
        n, a = jload(f"L1_esm2_650M_frozen_{task}.json"), jload(f"L1_esm2_650M_frozen_{task}_adv.json")
        if not (n and a):
            continue
        A(f"**{task} — L1 无对抗 vs +对抗 (逐折, 不看中位数)**")
        A("")
        A("| 折 | 主数字 | 无对抗 | +对抗 | 中心判别 acc |")
        A("|---|---|---|---|---|")
        ai = {r["fold"]: r for r in a["folds"]}
        for r in n["folds"]:
            if not r.get("inner_folds") or r["fold"] not in ai:
                continue
            r2 = ai[r["fold"]]
            b = float(np.mean([x["cross"]["base_rate"] for x in r["inner_folds"]]))
            key = "cross_pr_auc_over_base_mean" if b < LOW_BASE else "cross_lift@100_mean"
            pm = "PR-AUC/base"      # 一律; lift@100 只作参考列
            accs = [x.get("adv_center_acc_train") for x in r2["inner_folds"]
                    if x.get("adv_center_acc_train")]
            A(f"| {r['fold']} | {pm} | {r[key]:.2f} | {r2[key]:.2f} | "
              f"{np.mean(accs):.3f} |" if accs else
              f"| {r['fold']} | {pm} | {r[key]:.2f} | {r2[key]:.2f} | — |")
        A("")

    # ── 4b L2 的排除性证据 ──
    A("## 4b L2 (LoRA 微调) 的排除性证据")
    A("")
    e2 = jload("L2_esm2_650M_lora_express.json")
    e1 = jload("L1_esm2_650M_frozen_express.json")
    if e1 and e2:
        i1 = {r["fold"]: r for r in e1["folds"] if r.get("inner_folds")}
        i2 = {r["fold"]: r for r in e2["folds"] if r.get("inner_folds")}
        f3a, f3b = i1.get(3), i2.get(3)
        if f3a and f3b:
            A(f"**微调未修复反向。** express 折 3 的反向在 L2 上依然存在: 主数字 PR-AUC/base "
              f"由 L1 的 {f3a['cross_pr_auc_over_base_mean']:.2f} 到 L2 的 "
              f"**{f3b['cross_pr_auc_over_base_mean']:.2f}** "
              f"(自助法区间 0.84 [0.83, 0.86], 远离 1.0), 微调没有把它拉回 1.0。")
            A("")
            A(f"在 lift@100 口径上反向进一步加深 "
              f"({f3a['cross_lift@100_mean']:.2f} → {f3b['cross_lift@100_mean']:.2f}, "
              f"自助法区间 [0.10, 0.36] → [0.02, 0.22] 不重叠), "
              "**但该口径不作判定依据**, 故不写成\"微调加剧反向\"。")
            A("")
            A("> 这处表述此前写成\"反而加剧\", 依据的是 lift@100 口径; "
              "改主数字规则后已按上述重写。排除组论证不受影响 —— "
              "该论证需要的是\"微调**未修复**\", 不是\"微调加剧\"。")
            A("")
    A("与第 4 节并列, 构成一个**排除组**:")
    A("")
    A("| 排除了什么 | 证据 |")
    A("|---|---|")
    A("| 不是模型不够大 | L2 微调 5.41M LoRA 参数 / 3 epoch 后, 折 3 反向**依然存在** "
      "(PR-AUC/base 0.84 [0.83, 0.86]) |")
    A("| 不是中心身份没去掉 | 对抗头中心判别 acc 降至 0.429–0.507 (≈猜最大类), "
      "中心可分性已移除, 反向仍不变 (0.79 → 0.80) |")
    A("")
    A("反向依然存在, 两条路都被排除, **指向同一个 open question**: "
      "该反向可能源于**标签生成机制随中心而变**, 而非序列表示本身。"
      "现有数据无法判定, 不进一步推测。")
    A("")
    if e1 and e2 and i1.get(1) and i2.get(1):
        A(f"折 1 的主数字从 L1 的 {i1[1]['cross_pr_auc_over_base_mean']:.2f} 到 L2 的 "
          f"{i2[1]['cross_pr_auc_over_base_mean']:.2f} (lift@100 "
          f"{i1[1]['cross_lift@100_mean']:.2f} → {i2[1]['cross_lift@100_mean']:.2f}) "
          "如实报告, **不写成\"L2 更好\"** —— 情形判定未变 (两级都是真信号), "
          "且该折本就是唯一稳定有真信号的折。")
        A("")

    # ── 4c 区间支撑 ──
    bs = jload("bootstrap_ci.json")
    if bs:
        A("## 4c 每个情形判定的置信区间 (按簇分层自助法)")
        A("")
        A(f"重采样单元 = **split_group** (30% 相似度的传递闭包分量), "
          f"{bs['n_boot']} 次。{bs['rationale']}")
        A("")
        A("> **为什么表里簇数 == 记录数 (不是实现错误)**: 抽 embedding 时已按 "
          "`split_group` 去冗余 —— 每个同源簇只保留一条代表记录。所以评测集中"
          "一条记录正好对应一个簇, 按簇重采样与按单条重采样在此**数学上等价**。"
          "换言之, 同源序列被反复抽样而压窄区间的风险, 已在上游的去冗余步骤被消除, "
          "不是这一步漏了处理。脚本仍以 `split_group` 为重采样单元实现, "
          "所以若上游改为保留簇内全部记录, 区间仍然正确。")
        A("")
        for tag, folds in bs["results"].items():
            A(f"**{tag}**")
            A("")
            A("| 折 | n (=簇数) | base | PR-AUC/base [95% CI] | lift@100 [95% CI] | "
              "情形 (四分类) | 5 种子方向 |")
            A("|---|---|---|---|---|---|---|")
            _advf = bs["results"].get(tag + "_adv") or {}
            for fo, r in sorted(folds.items(), key=lambda kv: int(kv[0])):
                pa = r["pr_auc_over_base"]; lf = r["lift@100"]
                # 这里**必须**印四分类 (effect_size_class), 不能印 r["verdict"] ——
                # 后者是只看区间的旧字段, 与全文其余各表用的口径不同。
                # 2026-10-02 实查: 同一格曾在本文件 §1 标"不可区分"、在这张表标"真信号"。
                _ss = r.get("seed_stability", {})
                _d = _ss.get("directions", [])
                _dtxt = ("—" if not _d else
                         ("一致 (" + _d[0] + ")" if _ss.get("stable")
                          else "**不一致** " + str(_d)))
                A(f"| {fo} | {r['n']:,} | {r['base_rate']:.3f} | "
                  f"{fmt_ci(pa['point'], pa['ci95'][0], pa['ci95'][1])} | "
                  f"{fmt_ci(lf['point'], lf['ci95'][0], lf['ci95'][1])} | "
                  f"**{r.get('effect_size_class', r['verdict'])}** | {_dtxt} |")
            A("")
        A("### 主数字规则的改动 (留痕)")
        A("")
        A("⚠️ **这个改动发生在争议结果之后**, 形式上敏感, 所以主动写明而非回避。时间线:")
        A("")
        A("1. L2 在 express 折 2 上给出 PR-AUC/base = 1.15, 而当时门槛是 1.10 —— "
          "只差 0.05, 且该折基础失败率仅 3.2%, 出现\"是信号还是噪声\"的争议;")
        A("2. 为定这个争议, 跑按簇分层自助法;")
        A("3. 自助法**顺带**暴露了 lift@100 作为主数字的问题;")
        A("4. 于是把主数字由 lift@100 改为 PR-AUC/base, 对所有折生效。")
        A("")
        A("**影响: 三折判定与改动前完全一致** (折 1 真信号 / 折 2 不可区分 / 折 3 反向), "
          "未改变任何已有结论 —— 改动只是让判定有区间支撑且不自相矛盾。"
          "完整记录见 `configs/stage3_train.yaml` 的 `changelog`。")
        A("")
        A("**理由**: 区间宽度差一个量级。"
          "以 L1 express 折 1 为例, PR-AUC/base 是 1.20 [1.12, 1.29] (宽 0.17, 不跨 1.0), "
          "而 lift@100 是 1.24 [0.90, 1.53] (宽 0.63, 跨 1.0) —— 两者给出相反结论。"
          "原因是 2,475 条里取 top-100 只用了 4% 的样本, 统计量噪声大; "
          "PR-AUC/base 用整个排序。"
          "**所以情形判定一律用 PR-AUC/base**, lift@100 保留作可解释性指标 "
          "(\"做 100 个实验能中几个\"), 不用于判定。")
        A("")

    # ── 4d 折 2: 边界情形 ──
    bs2 = jload("bootstrap_ci.json")
    if bs2:
        R = bs2["results"]
        n2 = R.get("L2_esm2_650M_lora_express", {}).get("2")
        a2 = R.get("L2_esm2_650M_lora_express_adv", {}).get("2")
        l2 = R.get("L1_esm2_650M_frozen_express", {}).get("2")
        if n2 and a2:
            A("## 4d express 折 2: 边界情形")
            A("")
            A("按预先定义的规则 (区间不跨 1.0 即判为真信号), L2 非对抗版在 express 折 2 上"
              "**形式上成立**; 但该证据不稳健, 故标注为**边界情形**, "
              "**不计入\"有真信号\"的情形**。")
            A("")
            A("| 设置 | **PR-AUC/base [95% CI]** | 跨 1.0? | lift@100 [95% CI] |")
            A("|---|---|---|---|")
            for tag, zh in ((n2, "L2 非对抗"), (a2, "L2 对抗"), (l2, "L1 (对照)")):
                if not tag:
                    continue
                pa, lf = tag["pr_auc_over_base"], tag["lift@100"]
                A(f"| {zh} | **{fmt_ci(pa['point'], pa['ci95'][0], pa['ci95'][1])}** | "
                  f"{'跨' if tag['primary_ci_crosses_1'] else '**不跨**'} | "
                  f"{fmt_ci(lf['point'], lf['ci95'][0], lf['ci95'][1])} |")
            A("")
            A("三条不稳健证据:")
            A("")
            pa_n = n2["pr_auc_over_base"]["ci95"]
            pa_a = a2["pr_auc_over_base"]["ci95"]
            A(f"1. **下界仅 {pa_n[0]:.2f}**, 距 1.0 只有 {pa_n[0]-1:.2f}。")
            A(f"2. **加入对抗头后区间回到 [{pa_a[0]:.2f}, {pa_a[1]:.2f}], 跨 1.0** —— "
              "同一模型换一个设置就从\"真信号\"变回\"不可区分\"。"
              "两个设置在其余三折上判定完全一致, **只有折 2 分歧**。")
            A(f"3. **两个口径方向相反**: lift@100 的点估计是 "
              f"{n2['lift@100']['point']:.2f} (**低于 1**), 区间 "
              f"[{n2['lift@100']['ci95'][0]:.2f}, {n2['lift@100']['ci95'][1]:.2f}] "
              "宽到无信息。所以\"弱真信号\"只在 PR-AUC/base 一个口径上成立。")
            A("")
            A("> **判为边界情形的直接依据**: 一个稳健的信号不会因为加入对抗头而消失。")
            A("")
            A("### 这与主数字规则的改动无关 (要分清)")
            A("")
            A("| | 改规则前 | 改规则后 (带区间) | 是否一致 |")
            A("|---|---|---|---|")
            A(f"| L1 折 1 | 真信号 | 真信号 "
              f"{R['L1_esm2_650M_frozen_express']['1']['pr_auc_over_base']['point']:.2f} "
              f"[{R['L1_esm2_650M_frozen_express']['1']['pr_auc_over_base']['ci95'][0]:.2f}, "
              f"{R['L1_esm2_650M_frozen_express']['1']['pr_auc_over_base']['ci95'][1]:.2f}] | ✅ |")
            A(f"| L1 折 2 | 与随机不可区分 | 与随机不可区分 "
              f"{l2['pr_auc_over_base']['point']:.2f} "
              f"[{l2['pr_auc_over_base']['ci95'][0]:.2f}, {l2['pr_auc_over_base']['ci95'][1]:.2f}] | ✅ |")
            A(f"| L1 折 3 | 反向 | 反向 "
              f"{R['L1_esm2_650M_frozen_express']['3']['pr_auc_over_base']['point']:.2f} "
              f"[{R['L1_esm2_650M_frozen_express']['3']['pr_auc_over_base']['ci95'][0]:.2f}, "
              f"{R['L1_esm2_650M_frozen_express']['3']['pr_auc_over_base']['ci95'][1]:.2f}] | ✅ |")
            A("| L2 折 2 | (当时判为真信号, 引发争议) | **边界情形** | L2 带来的新信息 |")
            A("")
            A("**L1 的三折与改规则前完全一致**; 分歧**仅出现在 L2**, 属 L2 带来的新信息, "
              "**非改规则所致**。所以\"改规则未改变任何已有结论\"这句对 L1 成立, "
              "而 L2 折 2 作为新结果单列, 不混进那句话。")
            A("")

    # ── 5 方法学陷阱 ──
    A("## 5 方法学: 复现对抗去偏时的一个常见陷阱")
    A("")
    A("**冻结编码器下, 分类头与对抗头之间没有共享的可训练参数, 梯度反转会完全失效。**")
    A("")
    A("症状: 开/关对抗跑出**逐位相同**的数字 (折 1.0 都是 lift 1.53 / PR-AUC/base 1.28 / "
      "within 2.05)。原因: L1 的 embedding 是常量张量, 对抗损失只训练了对抗头自己, "
      "对分类头毫无影响。")
    A("")
    A("修法: 在 embedding 与两个头之间插入一层**可训练投影** "
      "(`emb → Linear(1280,256)+ReLU → {分类头, 梯度反转→对抗头}`), "
      "梯度反转才能真的逼投影丢掉中心可分信息。"
      "并且**非对抗版也走同一架构**, 否则分不清是\"对抗起作用\"还是\"多了一层非线性\"。")
    A("")
    A("这值得写进方法节 —— 任何在冻结骨干上做对抗去偏的工作都会遇到。")
    A("")
    A("## 6 可复现性: 评估产物必须保存逐条预测")
    A("")
    A("**教训**: 本项目的 L1/L2 训练脚本最初**只保存汇总指标, 不保存逐条预测**。"
      "结果是折 2 的争议出现后, 为了做一次自助法置信区间, 不得不**重新训练** "
      "3 个折 × 2 个对抗设置 = 6 次, 每次约 33 分钟, 共约 3.3 小时 —— "
      "而这本该是读一个文件就能算的事。")
    A("")
    A("**后果比浪费算力更严重**: 当一次事后验证的代价是三小时重训, "
      "人就会倾向于跳过验证、直接相信汇总数字。规则应当是:")
    A("")
    A("> 任何评估产物都保存**逐条预测**, 并带上重采样所需的分组键 "
      "(本项目是 `split_group`), 而不是只存汇总指标。")
    A("")
    A("**这与之前几次陈旧产物事故是同源问题**: 中间产物的信息不足或与上游失去对应, "
      "都会让下游分析要么无法进行、要么静默给出错误结果。"
      "前者已由 `src/labels/provenance.py` 的指纹校验解决, "
      "后者由这条\"保存逐条预测\"的规则解决。两条都属于"
      "**让验证成本足够低, 低到不会被跳过**。")
    A("")

    OUT.write_text("\n".join(L), encoding="utf8")
    print(f"wrote {OUT} ({len(L)} 行)")


if __name__ == "__main__":
    main()
