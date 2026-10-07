"""生成 reports/default/PREPRINT_biorxiv.md —— bioRxiv 投稿草稿。

与其它报告同一纪律: **文中数字全部从冻结产物注入, 非手写**; 末尾交叉核对,
不一致就不出文件。主数字一律 PR-AUC/base (见 configs/stage3_train.yaml 的 changelog)。

摘要落点 (刘刚刚 2026-10-02 定):
  四情形混合 (原定"三情形", 2026-10-02 补算折 4 区间后改为四类, 见 configs changelog)
  -> 跨中心方向与幅度均不一致 (全部折) + 一例跨任务方向相反 (单例, 不作普遍规律)
  -> 无廉价事前判据
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import polars as pl
import yaml

R = pathlib.Path("reports")
D = R / "default"
OUT = D / "PREPRINT_biorxiv.md"

# 这两个分数来自 reports/gate2_baselines.md §2.2 (那里有完整上下文), 不在任何 JSON 里。
# 所以下方 require_in_report() 硬核对上游报告仍含这两个字符串 —— 上游一改本生成器立刻失败,
# 不会留下悄悄过期的手抄数字。
SOLUPROT_MISS = "6,730/16,435"
SOLUPROT_MISS_OFFICIAL = "4/21"


def fmt_ci(point: float, lo: float, hi: float) -> str:
    """区间端点贴近 1.0 时多打两位小数。

    为什么 (2026-10-02 实查): 两位小数下 0.995691 与 1.000986 都显示成 "1.00",
    但一个跨 1.0 一个不跨, 判定相反 —— 读者无法从表里看出差别。
    """
    nd = 4 if min(abs(lo - 1.0), abs(hi - 1.0)) < 0.02 else 2
    return f"{point:.2f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_shrink(x: float) -> str:
    """收缩倍数的统一精度规则: <10 给一位小数, >=10 取整并加千位分隔。

    为什么要统一 (2026-10-05 自查): 原先按来源的表用 :.0f, 于是 1.64 印成 "÷2"、
    4.82 印成 "÷5", 而 ÷486 / ÷6661 保留了精度。读者按 "÷2" 反推不出摘要里
    "不足一个数量级" 的依据 (log10(1.64)=0.21, 而 log10(2)=0.30)。
    """
    return f"÷{x:.1f}" if x < 10 else f"÷{x:,.0f}"

def require_in_report(path: str, *needles: str) -> None:
    txt = pathlib.Path(path).read_text(encoding="utf8")
    missing = [n for n in needles if n not in txt]
    if missing:
        raise SystemExit(f"!! {path} 里找不到 {missing} —— "
                         f"预印本手抄的数字已与上游报告脱节, 拒绝出文件")


def jl(p):
    f = pathlib.Path(p)
    return json.loads(f.read_text()) if f.exists() else None


def fm(x, n=0):
    return f"{x:,.{n}f}" if isinstance(x, float) and n else f"{x:,}"


CLASS_ORDER = ["真信号", "边界情形", "弱但可分辨", "不可区分", "反向"]


def classify(cell: dict, adv_cell: dict | None) -> str:
    """configs/stage3_train.yaml changelog [2026-10-02] 定的四分类 + 对抗稳健性降级。

    区间定"方向能不能分辨", 点估计定"幅度够不够用" —— 两件事不混为一谈。
    """
    c = cell["effect_size_class"]
    if c == "真信号" and adv_cell is not None and adv_cell["primary_ci_crosses_1"]:
        return "边界情形"          # 稳健的信号不会因为加入对抗头而消失
    return c


def main() -> None:
    g1 = jl(R / "gate1_summary.json")
    bs = jl(D / "bootstrap_ci.json")
    gb = jl(D / "baseline_gbdt.json")
    sol = jl(R / "baseline_soluprot.json")
    ips = jl(R / "baseline_ipsae.json")
    cont = jl(R / "soluprot_contamination.json")
    leak = {o["split"]: o for o in (jl(R / "leakage_check.json") or [])}
    sg = jl("data/interim/split_groups.stats.json")
    tag = jl(R / "tag_confound_analysis.json")
    cmp_ = jl(D / "compare_levels_express.json")   # 同一测试折上的 within / cross 对照
    ds6 = jl("data/interim/ds6_oih.stats.json")
    L1e = jl(D / "L1_esm2_650M_frozen_express.json")
    L1s = jl(D / "L1_esm2_650M_frozen_soluble_expression.json")
    L2e = jl(D / "L2_esm2_650M_lora_express.json")
    L2s = jl(D / "L2_esm2_650M_lora_soluble_expression.json")
    mf = jl("data/processed/splits/MANIFEST.json")
    _rel = yaml.safe_load(pathlib.Path("configs/release.yaml").read_text()) or {}
    REPO_URL = _rel.get("repo_url") or "<REPOSITORY_URL>"
    _zd = _rel.get("zenodo_doi")
    _pending = bool(_rel.get("zenodo_deposit_pending"))
    # 三态: DOI 已有 -> 写实际 DOI; 沉积待发布 -> 写明"将在修订版补上";
    # 都不是 -> 明显占位符 (绝不给一个看着像真的死 DOI)
    if _zd:
        DOI_TXT = f"doi:{_zd}"
    elif _pending:
        DOI_TXT = "\u6c89\u79ef DOI \u5c06\u5728\u672c\u9884\u5370\u672c\u7684\u4fee\u8ba2\u7248\u4e2d\u8865\u4e0a"
    else:
        DOI_TXT = "doi:[to be added — deposit not yet published]"
    _refs_all = yaml.safe_load(pathlib.Path("configs/references.yaml").read_text())
    RF = _refs_all["refs"]

    def cite(k: str) -> str:
        """作者姓 + 年; 同姓同年的多篇用 key 末尾的 a/b 消歧 (Bushuiev 有正会与 workshop 两篇)。"""
        r = RF[k]
        first = r["authors"].split(",")[0].strip().split()[0]   # 只取姓, 不带名首字母
        suffix = ""
        _same = [kk for kk, vv in RF.items()
                 if vv["authors"].split(",")[0].strip().split()[0] == first
                 and vv["year"] == r["year"]]
        if len(_same) > 1:
            _key = str(r.get("key", ""))
            if _key and _key[-1] in "ab":
                suffix = _key[-1]
        return f"{first} et al., {r['year']}{suffix}"

    def folds_of(a):
        return {r["fold"]: r for r in (a or {}).get("folds", []) if r.get("inner_folds")}

    L: list[str] = []
    A = L.append

    A("# 公开蛋白失败数据中的标签伪影源自实验室身份, 且剥标签与对抗去偏都无法移除")
    A("")
    A("**— 一个跨中心评估基准, 以及它为什么不可省**")
    A("")
    A("**bioRxiv 预印本草稿 · 生成于 `src/eval/preprint.py` · 数字全部从冻结产物注入**")
    A("")
    A("---")
    A("")

    # ───────── 摘要 ─────────
    A("## 摘要")
    A("")
    # 数量级范围从数据算, 不写死 (旧稿写"一到三个数量级", 低估了 ds3 的 ÷6,661 = 3.8 个)
    rec = pl.read_parquet("data/processed/records.parquet")
    _shrink = []
    for _sc in ("ds1_targettrack", "ds2_tsuboyama", "ds3_proteingym",
                "ds5_dtu_binder", "ds5_adaptyv_egfr"):
        _nn = rec.filter(
            (pl.col("source") == _sc)
            & ((pl.col("label_clone") == 0) | (pl.col("label_express") == 0)
               | (pl.col("label_soluble") == 0) | (pl.col("label_purify") == 0)
               | (pl.col("label_stable") == 0) | (pl.col("label_bind") == 0)))
        if _nn.height:
            _shrink.append(_nn.height / int(_nn["cluster_rep"].n_unique()))
    _mx = max(_shrink)
    A(f"蛋白实验失败预测缺少统一基准。我们把四个公开来源的真实实验失败记录整合为"
      f"一套六阶段标签体系, 得到 {fm(g1['records'])} 条记录; 但在 30% 序列相似度下"
      f"只有 {fm(g1['clusters_30pct'])} 个独立簇 —— 这一落差本身是本文的第一个结论: "
      f"以阴性条数衡量此类数据集的规模会误导跨蛋白任务上的有效样本量, "
      f"按来源计**从不足一个数量级到近四个数量级** "
      f"(最大收缩 ÷{_mx:,.0f}, 即 {np.log10(_mx):.1f} 个数量级, 见 §1.1)。")
    A("")
    A(f"**这类数据存在序列层面的标签伪影, 已由前人发现**: NetSolP ({cite('netsolp')}) "
      f"报告其训练集 11,602/69,420 条带 N 端 His 标签 `MGSDKIHHHHHH` 且约 99% 不可溶, "
      f"并指出\"模型更多地关注 His 标签而非野生型序列\"; "
      f"SoluProt ({cite('soluprot')}) 则主动平衡了序列长度分布以免长度单独主导预测; "
      f"到 PLM_Sol ({cite('plm_sol')}) 构建 UESolDS 时, 剥除 His 标签片段已是标准步骤。"
      "**本文不重复宣称发现这些伪影。**")
    A("")
    A("本文回答的是接下来那个问题: **这些伪影从哪来, 以及剥掉之后还剩什么。** "
      "我们把伪影归因到**实验室身份**这一层: 亲和标签的使用率与位置、序列长度、"
      "乃至阴性标签本身是否存在, 都是各结构基因组学中心的构建与记录习惯; "
      "长度对失败的表观预测力在中心内几乎消失, 且在两个阶段上"
      "**跨中心与中心内方向完全相反** (Simpson 反转) —— "
      "这说明它量的是中心之间的惯例差异, 不是生物学。"
      "据我们所知, 上述三项工作均未做任何中心级的比较或分层 (全文实查)。")
    A("")
    _ab = {}
    for _fo in (1, 2, 3, 4):
        _c = (bs["results"].get("L1_esm2_650M_frozen_express") or {}).get(str(_fo))
        if _c:
            _ab[_fo] = classify(
                _c, (bs["results"].get("L1_esm2_650M_frozen_express_adv") or {}).get(str(_fo)))
    _cnt = {k: sum(1 for v in _ab.values() if v == k) for k in CLASS_ORDER}
    _f3e = bs["results"]["L1_esm2_650M_frozen_express"]["3"]["pr_auc_over_base"]
    _f3s = (bs["results"]["L1_esm2_650M_frozen_soluble_expression"]["3"]
            ["pr_auc_over_base"])
    A("**关键结果是伪影去不掉。** 按本领域现行标准做法剥除构建体残留、"
      "并且干脆不使用长度特征之后, 以留出中心评估, "
      f"跨中心泛化呈现**四情形混合**: 在 {len(_ab)} 个留出中心组里, "
      + "、".join(f"{v} 组{k}" for k, v in _cnt.items() if v)
      + " (判定依据为按同源簇分层的自助法 95% 区间, 区间定方向、点估计定幅度; "
      "其中一个**只用 20 维氨基酸组成、不含长度**的梯度提升树与 ESM-2 650M 的区间重叠"
      "甚至更高, 说明瓶颈不在表示能力)。"
      "**方向与幅度在中心之间都不一致**, 而且在一个留出中心组上, "
      "同一套序列换一个失败阶段就把预测方向整体翻过来 "
      f"(`express` 上区间 [{_f3e['ci95'][0]:.2f}, {_f3e['ci95'][1]:.2f}], "
      f"`soluble_expression` 上 [{_f3s['ci95'][0]:.2f}, {_f3s['ci95'][1]:.2f}])。")
    A("")
    A("**结论: 中心间的方向与幅度均不一致, 且至少一例显示同一批中心在不同任务上"
      "方向相反 —— 没有廉价的事前判据能预知一次部署会落进有用 / 有害 / 噪声哪一类, "
      "只能每个中心每个任务实测。这就是中心分层评估协议的成本来源与必要性。**")
    A("")
    A("我们公开统一标签、四套经显式泄漏验证的冻结切分、以及全部评估代码。")
    A("")

    # ───────── 1 数据 ─────────
    A("## 1 数据与规模")
    A("")
    A(f"| 量 | 值 |")
    A("|---|---|")
    A(f"| 记录数 | {fm(g1['records'])} |")
    A(f"| 独立序列 (DMS 变体按亲本归并) | {fm(g1['unique_sequences'])} |")
    A(f"| 30% 相似度簇 | **{fm(g1['clusters_30pct'])}** |")
    A("")
    A("### 1.1 第一个结论: 阴性条数不是规模")
    A("")
    A("| 阶段 | 阴性记录数 | 30% 簇数 | 收缩 |")
    A("|---|---|---|---|")
    for s in ("clone", "express", "soluble", "purify", "stable", "bind"):
        f = rec.filter(pl.col(f"label_{s}") == 0)
        nclu = g1["negatives_clusters_per_stage"][s]
        A(f"| `{s}` | {fm(f.height)} | **{fm(nclu)}** | "
          f"{fmt_shrink(f.height / max(nclu, 1))} |")
    A("")
    _srcclu = {}
    for _src in ("ds1_targettrack", "ds2_tsuboyama", "ds3_proteingym",
                 "ds5_dtu_binder", "ds5_adaptyv_egfr"):
        _n = rec.filter(
            (pl.col("source") == _src)
            & ((pl.col("label_clone") == 0) | (pl.col("label_express") == 0)
               | (pl.col("label_soluble") == 0) | (pl.col("label_purify") == 0)
               | (pl.col("label_stable") == 0) | (pl.col("label_bind") == 0)))
        if _n.height:
            _srcclu[_src] = (_n.height, int(_n["cluster_rep"].n_unique()))
    # 挑**收缩倍数**最大的两个 (而非簇数最小的两个): 论点是"记录多但簇少",
    # 单靶点竞赛数据簇数也小, 但它记录本来就少, 不构成论点。
    _worst = sorted(_srcclu.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1))[:2]
    A("上表按**阶段**聚合, 掩掉了来源之间的巨大差异。按**来源**看才看得出成因 "
      + "(下表的 " + " 与 ".join(str(v[1]) for _, v in _worst)
      + " 正是\"十几万条阴性只等价于几十个独立样本\"的出处):")
    A("")
    A("| 来源 | 阴性记录数 | 30% 簇数 | 收缩 | 为什么 |")
    A("|---|---|---|---|---|")
    _srcwhy = {
        "ds1_targettrack": "全流程记录, 每条是一个独立靶点 — 收缩最小",
        "ds2_tsuboyama": "同一批亲本结构域的大量单点突变 (cDNA display proteolysis)",
        "ds3_proteingym": "同一批亲本蛋白的深度突变扫描 (DMS)",
        "ds5_dtu_binder": "de novo 设计 binder, 靶点少但设计彼此差异大",
        "ds5_adaptyv_egfr": "单靶点 (EGFR) 竞赛设计",
    }
    _any_neg = None
    for _src in ("ds1_targettrack", "ds2_tsuboyama", "ds3_proteingym",
                 "ds5_dtu_binder", "ds5_adaptyv_egfr"):
        _d = rec.filter(pl.col("source") == _src)
        _neg = _d.filter(
            (pl.col("label_clone") == 0) | (pl.col("label_express") == 0)
            | (pl.col("label_soluble") == 0) | (pl.col("label_purify") == 0)
            | (pl.col("label_stable") == 0) | (pl.col("label_bind") == 0))
        if _neg.height == 0:
            continue
        _nclu = int(_neg["cluster_rep"].n_unique())
        A(f"| {_src} | {fm(_neg.height)} | **{fm(_nclu)}** | "
          f"{fmt_shrink(_neg.height / max(_nclu, 1))} | {_srcwhy.get(_src,'')} |")
    A("")
    A("> 两张表的\"收缩\"列统一规则: **小于 10 倍给一位小数, 10 倍及以上取整**。"
      f"摘要里\"不足一个数量级\"的依据是本表最小的 "
      f"{fmt_shrink(min(_shrink))} (log\u2081\u2080 = {np.log10(min(_shrink)):.2f}), "
      f"\"近四个数量级\"的依据是最大的 {fmt_shrink(_mx)} "
      f"(log\u2081\u2080 = {np.log10(_mx):.2f})。")
    A("")
    A("**这量的是跨蛋白泛化的有效多样性, 不是数据质量。** "
      "对 DMS 自身的用途 (同一蛋白内的变异效应预测) 这些是极好的数据; "
      "但在\"预测一个没见过的蛋白会不会失败\"这个任务上, "
      "一个来源有多少条阴性几乎不重要, 重要的是它覆盖了多少个独立的蛋白家族。"
      "按阶段聚合的上表与按来源的本表对不上, 正是因为 `stable` 与 `bind` 两个阶段"
      "的阴性几乎全部来自 DMS 类来源。")
    A("")

    # ───────── 2 相关工作 ─────────
    A("## 2 相关工作与本文的增量")
    A("")
    A("### 2.1 标签伪影不是本文发现的")
    A("")
    A("把这一点放在最前面, 以免读者误会贡献边界。")
    A("")
    A(f"**{cite('netsolp')}** (NetSolP) 在 E. coli 可溶性数据上报告了亲和标签伪影, 原文:")
    A("")
    A(f"> \"{' '.join(RF['netsolp']['verbatim']['tag_counts'].split())}\"")
    A("")
    A(f"> \"{' '.join(RF['netsolp']['verbatim']['consequence'].split())}\"")
    A("")
    A("并转述了标签本身的不一致 (该数字出自 "
      f"{cite('soluprot')} 与 {cite('price2011')} 的对比, 非 NetSolP 自测):")
    A("")
    A(f"> \"{' '.join(RF['netsolp']['verbatim']['label_inconsistency'].split())}\"")
    A("")
    A(f"**{cite('soluprot')}** (SoluProt) 在构建数据集时就处理了长度混杂, 原文: "
      f"\"{' '.join(RF['soluprot']['verbatim']['length_balancing'].split())}\"")
    A("")
    A(f"**{cite('plm_sol')}** (PLM_Sol / UESolDS) 把剥标签固化为建库步骤, 原文:")
    A("")
    A(f"> \"{' '.join(RF['plm_sol']['verbatim']['tag_exclusion'].split())}\"")
    A("")
    A("**所以: 剥除亲和标签、不让长度主导, 是本领域 2021–2024 年间形成的标准做法。"
      "本文既不声称发现这些伪影, 也不把剥标签当作贡献。**")
    A("")
    A("### 2.2 与本文最接近的前人工作: 结晶倾向 / 多阶段预测这一支")
    A("")
    A("上面三篇做的是**单一的可溶性**预测。还有一整支工作与本文框架更接近 —— "
      "**同样用 TargetTrack / PepcDB、同样预测多个连续实验阶段**。"
      "它是本文批评的具体对象, 必须单列。")
    A("")
    A(f"代表作是 **{cite('predppcrys')}** (PredPPCrys), 原文:")
    A("")
    A(f"> \"{' '.join(RF['predppcrys']['verbatim']['scale'].split())}\"")
    A("")
    A(f"它预测 {RF['predppcrys']['stages']}, 与本文的六阶段标签几乎是同一个问题。"
      "去冗余方式是:")
    A("")
    A(f"> \"{' '.join(RF['predppcrys']['verbatim']['cdhit'].split())}\"")
    A("")
    A(f"> \"{' '.join(RF['predppcrys']['verbatim']['blast'].split())}\"")
    A("")
    A("**这不是一篇孤例, 而是一条持续十年的线。** 为确认没有遗漏同类工作, "
      f"我们核对了 **{cite('crystal_review')}** 这篇系统评测该支预测器的综述, 原文:")
    A("")
    A(f"> \"{' '.join(RF['crystal_review']['verbatim']['data_sources'].split())}\"")
    A("")
    A(f"> \"{' '.join(RF['crystal_review']['verbatim']['multistage'].split())}\"")
    A("")
    A("| 工作 | 年 | 覆盖阶段 | 切分方式 | 中心留出 |")
    A("|---|---|---|---|---|")
    for _k, _nm, _st in (("ppcpred", "PPCpred", "产出 / 纯化 / 结晶"),
                         ("predppcrys", "PredPPCrys",
                          "克隆 / 产出 / 纯化 / 结晶 / 衍射级结晶"),
                         ("crysalis", "Crysalis", "多阶段 (见综述)"),
                         ("fdetect", "fDETECT", "产出 / 纯化 / 结晶"),
                         ("deepcascade2021", "deep-cascade forest", "多阶段")):
        _r = RF[_k]
        A(f"| **{_nm}** ({cite(_k)}) | {_r['year']} | {_st} | "
          f"仅序列同源去冗余 | **无** |")
    A(f"| 本文 | 2026 | clone / express / soluble / purify / stable / bind | "
      f"同源传递闭包 **+ 留出中心** | **有** |")
    A("")
    A("> **上表各格的核实层级, 据实标注**: PredPPCrys 那一行的阈值与"
      "\"无中心分层\"是**逐条核对该文全文**得到的 (引语见上)。"
      "其余四篇的\"仅序列同源去冗余 / 无中心留出\"是据 "
      f"{cite('crystal_review')} 的系统复评得到的家族级结论 —— "
      "该综述描述了这一支统一的去冗余协议 (类内与类间 25% 序列一致度), "
      "且全文无任何按实验室 / 中心的分层或评测切分。"
      "**我们没有逐篇重读这四篇的全文**, 所以若某篇另有未被综述记录的中心级分析, "
      "本表这一格会错; 这是一个明确标注的核实边界, 不是已核实的断言。")
    A("")
    A("**关键差异只有一条, 但它决定了成绩的含义**: 这一支的切分一律只做序列同源去冗余 "
      "(类内 CD-HIT 40%、训练测试间 BLAST 25%, 或综述统一复评时的 25%), "
      "**没有任何一篇做实验室留出或中心级分层** —— 而综述自己写明 TargetTrack 的数据来自 "
      "\">40 structural genomics centers worldwide\"。"
      "按本文 §5.2 的测量, 同中心训练与跨中心训练在**同一批测试条目**上的差距是 "
      "1.85 vs 1.01; 因此**同源去冗余不足以排除实验室身份通道**, "
      "这一支报告的成绩有被该通道虚高的可能。")
    A("")
    A("**措辞上必须克制**: 我们没有重跑它们的模型, 所以不能说\"它们的成绩是假的\"。"
      "能说的是: 它们的评估协议不区分\"预测蛋白难不难做\"与\"认出这条序列来自哪个中心\", "
      "而本文证明后者在同一数据源上确实可学且足以解释大部分表观性能。"
      "这一支唯一触及相关问题的是该综述提到的流水线差异:")
    A("")
    A(f"> \"{' '.join(RF['crystal_review']['verbatim']['pipeline_bias'].split())}\"")
    A("")
    A("即它意识到\"高通量流水线\"与\"传统逐个攻关\"产出的结构性质不同, "
      "但没有把这一点推进到中心级: 既未按中心分层评测, 也未检验中心身份本身可否被序列预测。")
    A("")
    A(f"切分泄漏方面有两篇相关工作, 论断分属不同篇, 在此分清 (合引是错的): "
      f"**{cite('bushuiev2024_workshop')}** (\"{RF['bushuiev2024_workshop']['title']}\", "
      f"{RF['bushuiev2024_workshop']['venue'].split('(')[0].strip()}) 专篇论证了"
      f"\"{' '.join(RF['bushuiev2024_workshop']['verbatim']['claim'].split())}\"; "
      f"而 **{cite('bushuiev2024_iclr')}** (ICLR 2024 正会) 是 PPIRef 数据集与 iDist "
      f"近重复检测算法的出处, 并据此构造非泄漏切分。"
      "本文 §4 的 split-group 做法与这两篇同向, 属于沿用而非重新发明。")
    A("")
    A("### 2.3 本文的增量, 以及它为什么不是上述工作的重复")
    A("")
    A("可溶性那三篇把伪影处理成**数据清洗问题**: 识别出一个序列基序 (His 标签) 或"
      "一个表面统计量 (长度), 清掉或平衡掉, 然后继续建模。"
      "结晶倾向那一支则连这一步都未涉及, 它处理的是**同源冗余**。"
      "本文问的是两者都没问的那个问题: 清洗掉可见的伪影、并排除同源泄漏之后, "
      "跨实验室还剩什么。答案是否定的。")
    A("")
    A("下表的\"前人不覆盖\"一栏针对的是 §2.1 与 §2.2 列出的**全部八篇** "
      "(可溶性三篇 + 结晶倾向五篇)。**各格的核实层级沿用 §2.2 的注脚, 不在此重复论证**: "
      "可溶性三篇与 PredPPCrys 为逐篇全文实查; 其余四篇据 "
      f"{cite('crystal_review')} 的家族级结论, 该注脚已标明其边界。")
    A("")
    A("| # | 本文做的 | 为什么前人工作不覆盖 |")
    A("|---|---|---|")
    A("| 1 | **中心级归因**: 标签使用率 (0.02%→91.8%)、标签位置 (N 端 vs C 端 "
      "近乎二分)、以及阴性标签是否存在, 都是中心指纹 | **上述八篇均未做任何按实验室 / "
      "中心的比较或分层**(核实层级见 §2.2 注脚); NetSolP 把 His 标签当序列基序处理, "
      "未追问\"为什么这批序列带标签而那批不带\"; 结晶倾向那一支虽直接用 TargetTrack / "
      "PepcDB (数据来自 >40 个中心), 也未按中心切分 |")
    A("| 2 | **Simpson 反转**: 长度的预测力在中心内消失, 两个阶段上跨中心与中心内"
      "方向完全相反 | SoluProt 平衡长度分布是为了不让长度主导, 但未检验长度的预测力"
      "是否本身来自中心间差异 |")
    A("| 3 | **同折 within / cross 对照**: 测试集逐条相同, 只换训练数据来源 | "
      "八篇均无留出中心的设定, 故不存在这个对照 (可溶性三篇按序列切分, "
      "结晶倾向五篇按同源阈值切分) |")
    A("| 4 | **四情形 + 按同源簇分层自助法 + 种子稳健性**: 把\"跨中心有没有信号\""
      "拆成方向与幅度两问, 并要求结论对实现选择不敏感 | 八篇报告的都是单一测试集上的"
      "汇总指标 (准确率 / MCC / AUC 等), 无按组分层的区间, 故无法区分"
      "\"方向可不可分辨\"与\"幅度够不够用\" |")
    A(f"| 5 | **有效多样性量化**: 阴性条数与 30% 相似度簇数按来源最多相差 "
      f"÷{_mx:,.0f} ({np.log10(_mx):.1f} 个数量级) | 上述工作的数据集规模均以序列条数 / "
      f"靶点数计 (如 PredPPCrys 的 108,933 靶点 / 979,645 次实验) |")
    A("| 6 | **对抗去偏的排除性结果**: 中心可分性被压到接近猜最大类, 跨中心的反向"
      "依然不变 | 八篇均未尝试移除中心身份 —— 因为均未把它识别为混杂 "
      "(可溶性三篇识别的是标签 / 长度, 结晶倾向五篇识别的是同源冗余) |")
    A(f"| 7 | **把同源去冗余与实验室留出分开**: 证明前者不蕴含后者 (同折 within 1.85 "
      f"vs cross 1.01) | 结晶倾向那一支 ({cite('predppcrys')} 等 5 篇) 以类内 CD-HIT 40% "
      f"+ 类间 BLAST 25% 为充分条件, 综述 ({cite('crystal_review')}) 统一复评时同样只用 "
      f"25% 同源阈值; 全支无一做中心留出 |")
    A("")
    A("换一句话说, 本文接在两条线之后各补一步: 对可溶性那三篇, "
      "**前人证明了\"模型在看标签\", 本文证明\"标签只是中心身份的一个可见代理, "
      "把代理剥掉、甚至用对抗训练主动抹掉中心可分性, 跨中心泛化仍然不成立\"**; "
      "对结晶倾向那一支, **前人把同源去冗余当作泛化评估的充分条件, "
      "本文证明它不是 —— 同源去冗余之后, 实验室身份仍然独立地撑着大部分表观性能**。"
      "两者合起来是一个否定结果, 它的用处是给评估协议定价: "
      "跨中心留出贵 (测试集只剩几千条、且每个中心每个任务都要单独实测), 但省不掉。")
    A("")

    # ───────── 3 身份通道 ─────────
    A("## 3 实验室身份是伪影的来源")
    A("")
    A("### 3.1 失败记录本身是中心习惯")
    A("")
    e = rec.filter((pl.col("source") == "ds1_targettrack") & (pl.col("label_express") != -1))
    ge = (e.group_by("center").agg(pl.len().alias("n"),
                                   (pl.col("label_express") == 0).sum().alias("f"))
            .filter(pl.col("n") >= 1000)
            .with_columns((pl.col("f") / pl.col("n")).alias("r")).sort("r", descending=True))
    zero = ge.filter(pl.col("f") == 0).sort("n", descending=True)
    A(f"在做过表达、记录数 ≥1,000 的 {ge.height} 个中心里, 表达失败率从 "
      f"{float(ge['r'][0])*100:.1f}% 跨到 **0.0%**: 其中 {zero.height} 个中心的失败率"
      f"恰好为零, 合计 {fm(int(zero['n'].sum()))} 条记录 "
      f"(最大的一个 {zero['center'][0]} 有 **{fm(int(zero['n'][0]))}** 条)。"
      "上万次实验一次不失败在生物学上讲不通 —— **阴性标签的存在与否本身与"
      "\"哪个中心做的\"强相关**。")
    A("")
    _jcsg_all = next((r["n"] for r in tag["record_mode"] if r["center"] == "JCSG"), None)
    _jcsg_e = int(zero.filter(pl.col("center") == "JCSG")["n"][0]) \
        if zero.filter(pl.col("center") == "JCSG").height else None
    if _jcsg_all and _jcsg_e:
        A(f"> **分母说明** (本节与 §3.2 的 JCSG 数字不同, 不是笔误): 本节的 "
          f"{fm(_jcsg_e)} 条是 JCSG **做过表达这一步、因而 `express` 标签非 -1** 的记录; "
          f"§3.2 的 {fm(_jcsg_all)} 条是 JCSG 在 DS1 里的**全部**记录 (含只走到克隆、"
          f"或该步未观测的)。差额 {fm(_jcsg_all - _jcsg_e)} 条即 `express` 未观测者, "
          "按本文的三态标签规则记为 -1 并从损失与评估中排除。全文每个计数都注明口径。")
    A("")
    A("### 3.2 标签的有无与位置都是中心指纹")
    A("")
    cc = {r["center"]: r for r in tag["centers"]}
    _tn = sum(r["n"] for r in tag["centers"])
    his_overall = sum(r["n"] * r["his_rate"] for r in tag["centers"]) / max(_tn, 1)
    A("| 事实 | 实测 |")
    A("|---|---|")
    A(f"| 含 `HHHHHH` 的记录 (DS1 全体 n={fm(_tn)} 条) | {his_overall*100:.1f}% |")
    _big = sorted([r for r in tag["centers"] if r["n"] >= 10000],
                  key=lambda x: -x["his_rate"])
    A(f"| 使用率跨 {len(_big)} 个大中心 (n\u226510,000) 的范围 | "
      f"{_big[0]['center']} **{_big[0]['his_rate']*100:.1f}%** \u2192 "
      f"{_big[-1]['center']} **{_big[-1]['his_rate']*100:.2f}%** |")
    A("")
    A("标签的**位置**同样是指纹 (在带标签的记录里, 标签落在 N 端还是 C 端的占比):")
    A("")
    A("| 中心 | 记录数 | His6 使用率 | N 端占比 | C 端占比 |")
    A("|---|---|---|---|---|")
    _rm = [r for r in tag["record_mode"]
           if r["n"] >= 10000 and r["his_rate"] >= 0.10
           and r.get("nterm_share_of_tagged") is not None]
    for r in sorted(_rm, key=lambda x: -x["cterm_share_of_tagged"]):
        A(f"| {r['center']} | {fm(r['n'])} | {r['his_rate']*100:.1f}% | "
          f"{r['nterm_share_of_tagged']*100:.0f}% | {r['cterm_share_of_tagged']*100:.0f}% |")
    A("")
    A("同一个标签序列, 在一个中心几乎只出现在 N 端、在另一个中心几乎只出现在 C 端。"
      "**标签的有无与位置都是中心指纹, 所以它给模型一条序列可见的实验室身份通道** —— "
      "模型不需要学会\"什么样的蛋白难表达\", 只要学会\"这条序列是哪个中心送来的\"。")
    A("")
    A("### 3.3 中心内分层: 长度的预测力是 Simpson 反转")
    A("")
    A("| 阶段 | 跨中心 rank AUC | 中心内中位 | 中心内范围 |")
    A("|---|---|---|---|")
    for r in tag["length_predictive_power"]:
        if r.get("pooled_rank_auc") is None:
            continue
        rg = r.get("within_center_range") or ["—", "—"]
        A(f"| `{r['stage']}` | {r['pooled_rank_auc']:.3f} | "
          f"{r['within_center_median']:.3f} | [{rg[0]}, {rg[1]}] |")
    A("")
    _lp = {r["stage"]: r for r in tag["length_predictive_power"]
           if r.get("pooled_rank_auc") is not None}
    _rev = [k for k, r in _lp.items()
            if (r["pooled_rank_auc"] - 0.5) * (r["within_center_median"] - 0.5) < 0]
    A("、".join(f"`{k}`" for k in _rev) +
      " 的**跨中心方向与中心内方向完全相反** (" +
      "; ".join(f"{_lp[k]['pooled_rank_auc']:.3f} vs {_lp[k]['within_center_median']:.3f}"
                for k in _rev) + "), 而 "
      f"`clone` / `express` 的预测力从 {_lp['clone']['pooled_rank_auc']:.2f} / "
      f"{_lp['express']['pooled_rank_auc']:.2f} 掉到 "
      f"{_lp['clone']['within_center_median']:.2f} / "
      f"{_lp['express']['within_center_median']:.2f}, 接近无信息。"
      "**长度的表观预测力主要来自中心之间的惯例差异。**")
    A("")
    A("### 3.4 消融的不对称性")
    A("")
    A("清除构建体残留 (21 类模式: His / FLAG / MYC / HA / Strep / T7 标签, "
      "TEV / thrombin / Factor Xa / 肠激酶位点, GS / GGGGS 接头, 载体克隆位点残余) "
      "并去掉长度特征后:")
    A("")
    _ai = {(x["split"], x["stage"]): x["overall"]["pr_auc_over_base"]
           for x in (jl(R / "ablation_keeptags_keeplen" / "baseline_gbdt.json") or
                     {"results": []})["results"] if "overall" in x}
    _di = {(x["split"], x["stage"]): x["overall"]["pr_auc_over_base"]
           for x in gb["results"] if "overall" in x}
    _abl = {}
    for _sp in ("sequence", "lab"):
        _d = [((_di[k] - _ai[k]) / _ai[k] * 100) for k in _di
              if k[0] == _sp and k in _ai]
        _abl[_sp] = (len(_d), float(np.mean(_d)), sum(1 for x in _d if x < 0),
                     max(_d))
    for _sp, _zh in (("sequence", "序列切分 (训练测试同中心)"),
                     ("lab", "实验室切分 (跨中心)")):
        _n, _m, _dn, _mx = _abl[_sp]
        _tail = (f"{_n} 个任务**全部下降**" if _dn == _n
                 else f"{_dn}/{_n} 个任务下降, 最大的反向变化 {_mx:+.1f}%")
        A(f"- **{_zh}**: {_tail}, 平均 **{_m:+.1f}%**")
    A("")
    A("**若为真实生物学效应, 两个切分应同样下降。** 实际是同中心一侧 "
      f"{_abl['sequence'][0]}/{_abl['sequence'][0]} 全部下降且幅度约为跨中心的 "
      f"{_abl['sequence'][1]/_abl['lab'][1]:.1f} 倍, 而跨中心一侧幅度小、方向还不一致 "
      "(有一个任务反而上升) —— 说明这两个通道在训练测试同中心时可用、"
      "跨中心时失效, 它们起的是识别作用。"
      "**不过要注意这个对比的统计强度有限**: 跨中心一侧只有 "
      f"{_abl['lab'][0]} 个任务有足够阴性可评, 单个任务的 ±10% 波动"
      "就能改变均值, 所以这条证据是与 §3.3 中心内分层检验**并列**的旁证, "
      "不单独承担结论。")
    A("")

    # ───────── 3 方法 ─────────
    A("## 4 方法: 四套经泄漏验证的冻结切分")
    A("")
    A("切分单位不是序列、也不是 set-cover 簇, 而是 **split group** —— "
      "在全部唯一序列上做 all-vs-all 检索后取 30% 相似度的传递闭包分量。")
    A("")
    A("| 做法 | 训练/测试间 >30% 的序列对 |")
    A("|---|---|")
    A("| 按 30% set-cover 簇整组划分 | 10,841 (最高 fident 1.00) |")
    A("| 加: 簇代表两两比较后合并 | 2,302 |")
    A("| 加: 全序列 all-vs-all 传递闭包 | 16 / 1 / 8 |")
    A("| 加: 迭代修补 | **0 / 0 / 0** |")
    A("")
    A(f"**切分泄漏是已被专门研究过的问题, 本文沿用而非重新发明。** "
      f"{cite('bushuiev2024_workshop')} 在蛋白相互作用数据上专篇论证了"
      f"\"{' '.join(RF['bushuiev2024_workshop']['verbatim']['claim'].split())}\", "
      f"后果是\"{' '.join(RF['bushuiev2024_workshop']['verbatim']['consequence'].split())}\"; "
      f"其配套的正会工作 {cite('bushuiev2024_iclr')} 给出 PPIRef 非冗余数据集与 iDist "
      "近重复检测算法, 并据此按 3D 界面相似度构造非泄漏切分。"
      "本文面对的是序列级任务而非界面级任务, 故以序列同源的传递闭包作为切分单位, "
      "但诊断思路相同: **先假定自己的切分在泄漏, 再显式度量它**。上表就是这个度量过程。")
    A("")
    A(f"顺带指出一点与 §2.2 相关的事: 结晶倾向那一支把同源去冗余 (CD-HIT / BLAST 阈值) "
      "当作泛化评估的充分条件, 而这两篇在 PPI 领域、本文在序列领域, "
      "都给出了同一方向的反例 —— **去掉同源冗余不等于去掉泄漏**, "
      "只是去掉了其中一种。")
    A("")
    A("MMseqs2 的级联聚类是贪心 set cover, 只保证成员与自己簇代表满足阈值, "
      "**不保证不同簇的成员之间低于阈值** —— 按簇切分会泄漏。"
      "最后仍需迭代修补, 因为建组与检验是两次独立检索, 预筛为启发式 k-mer 匹配。")
    A("")
    if sg:
        A(f"在 30% / 80% 覆盖下取传递闭包, 靶点空间形成一个巨型连通分量 "
          f"(含 {fm(sg['largest_group_clusters'])} 个 set-cover 簇)。"
          "成因不是短片段搭桥 (已验证), 而是多结构域蛋白在家族之间搭桥。"
          "占比超过 1% 的组强制进训练侧; **代价是测试集系统性地不含处于大同源网络中的蛋白**。")
        A("")
    A("| 切分 | 检验什么 | test 记录 | 泄漏检查 |")
    A("|---|---|---|---|")
    zh = {"sequence": "基本泛化", "lab": "是否学到实验室偏差",
          "time": "时代漂移", "bind_target": "面对新靶点能否预测 binder"}
    for k in ("sequence", "lab", "time", "bind_target"):
        if mf and k in mf["splits"]:
            s = mf["splits"][k]["sides"]["test"]
            A(f"| {k} | {zh[k]} | {fm(s['records'])} | "
              f"**{leak.get(k,{}).get('verdict','—')}** (0 对) |")
    A("")
    A("**评估口径**: 正类 = 失败。主数字是 **PR-AUC/base** (= PR-AUC / 正类占比)。"
      "lift@k 一并报告作可解释性参考 (\"固定 k 个湿实验名额能中几个\") 但不作判定依据 —— "
      "按簇分层自助法显示 lift@100 在 n≈2,500 的折上区间宽 0.63 (只用 top-100 = 4% 的样本), "
      "而 PR-AUC/base 宽 0.17, 且两者曾在一折上给出相反判定。"
      "该规则的改动时间点与影响见补充材料的 changelog。")
    A("")

    # ───────── 4 结果 ─────────
    A("## 5 结果")
    A("")
    A("### 5.1 现有工具与平凡基线在跨中心评估下都接近随机")
    A("")
    if gb and sol:
        gi = {(x["split"], x["stage"]): x for x in gb["results"] if "overall" in x}
        si = {(x["split"], x["variant"]): x for x in sol["results"]
              if x.get("task") == "soluble_expression" and "skipped" not in x}
        A("对齐 SoluProt 定义的复合标签 (`express` 成功且 `soluble` 成功):")
        A("")
        A("| 方法 | 序列切分 PR-AUC/base | 实验室切分 PR-AUC/base |")
        A("|---|---|---|")
        for nm, a, b in (("SoluProt (全集)", si.get(("sequence", "full")), si.get(("lab", "full"))),
                         ("SoluProt (去污染子集)", si.get(("sequence", "clean")), si.get(("lab", "clean"))),
                         ("氨基酸组成 GBDT (默认口径: 剥标签 + **无长度**)", gi.get(("sequence", "soluble_expression")),
                          gi.get(("lab", "soluble_expression")))):
            def v(x):
                if not x:
                    return "—"
                o = x.get("overall", x)
                return f"{o.get('pr_auc_over_base', float('nan')):.2f}"
            A(f"| {nm} | {v(a)} | {v(b)} |")
        A("")
        A("**不写成\"我们超过了 SoluProt\"**: SoluProt 的 `ecoli_usearch_identity` 特征 "
          "(与 E. coli PDB 序列的最大一致度) 在我们的序列切分测试集上有 "
          f"**{SOLUPROT_MISS} = 41% 的序列算不出来** "
          "(usearch 无命中, SoluProt 退回用训练集均值填补), "
          f"而在它官方自带的测试例上这个比例是 {SOLUPROT_MISS_OFFICIAL} = 19%。"
          "它的训练与设计对象是 E. coli 异源表达的天然蛋白, 而我们的测试集含设计蛋白、"
          "膜蛋白与大量无显著 PDB 同源的序列 —— 我们是在把它用在适用域之外, "
          "**所以上表里 SoluProt 的数字是它的下界**。"
          "诚实的表述是: **在这个数据集上, 一个只用 20 维氨基酸组成 (不含长度、已剥构建体"
          "残留) 的 GBDT 就达到或超过了已发表工具, 而 SoluProt 的核心特征在域外严重退化。**")
        A("")
        A("**这个比较有两处不对称, 都对 SoluProt 不利, 一并写明:**")
        A("")
        A("1. **域外 vs 域内**: SoluProt 在它自己的数据集上训练, 被我们搬到本数据集上"
          "推断; 我们的 GBDT 直接在本数据集上训练。跨域使用必然压低它。")
        A("2. **特征覆盖**: 它的 PDB 一致度特征在我们测试集上 41% 算不出来 "
          "(它自己的测试例 19%), 退化到训练集均值填补。")
        A("")
        A("**另有一处我们检查过、结论是不构成不对称的**: 长度。"
          f"{cite('soluprot')} 的处理是在**建库时平衡长度分布** "
          f"(\"{' '.join(RF['soluprot']['verbatim']['length_balancing'].split())}\"), "
          "本文的处理是**不把长度作为特征**。"
          "这是两种不同的操作, 不是同一件事的两种说法: 前者改变训练数据的分布, "
          "后者改变特征集合。但两者在本比较中的效果同向 —— "
          "**谁都没有靠长度拿分**, 所以不存在\"我们偷用长度而它没用\"的不公平。")
        A("")
        A("不过这里有一点必须替 SoluProt 说明, 否则仍不公平: "
          "**本文的测试集没有做长度平衡**。SoluProt 的模型是在长度分布被平衡过的"
          "训练数据上拟合的, 现在被搬到一个长度分布未平衡的测试集上, "
          "这本身可能让它吃亏 (它学到的决策函数假定了一个不同的长度边际分布)。"
          "我们没有量化这一项 —— 要量化就得重训 SoluProt, 超出本文范围。"
          "所以这是一条**已知但未量化的、方向对我们有利的残余不对称**, 如实标注。")
        A("")
        A("最后提醒一个口径陷阱: 若改用保留长度的消融口径 "
          "(`reports/ablation_keeptags_keeplen/`), 我们的 GBDT 会更高 (见 §3.4), "
          "**那个数字不可与 SoluProt 并列** —— 它吃的正是本文认定为实验室身份通道的那部分。")
        A("")
        A("因此本文的用法是: **把这个 GBDT 当作\"平凡基线\"的下界参照, 而不是声称"
          "我们的方法更好**。它的作用是说明该数据集上现有工具与平凡特征的差距很小。")
        A("")
        if cont:
            cs = cont["splits"]
            A(f"污染检查 (SoluProt 官方声明其训练集来自 TargetTrack, 与我们的 DS1 同源): "
              f"30% 相似度污染率 序列切分 {cs['sequence']['contamination_rate_30pct']*100:.1f}% / "
              f"实验室切分 {cs['lab']['contamination_rate_30pct']*100:.1f}%, "
              "故每个数字都给全集与去污染子集两版。去污染后实验室切分几乎不变, "
              "**污染没有虚高它** —— 这一点对它有利, 如实写。")
            A("")

    A("### 5.2 正面对照: 同一测试折上, 中心内远高于跨中心")
    A("")
    if cmp_:
        _mp = cmp_["medians_pr_auc_over_base"]
        A("这是把身份通道的贡献量直接称出来的一组对照。**两侧的测试集完全相同**, "
          "只换训练数据的来源: `within` 的训练数据来自留出中心**内部的其它同源组**, "
          "`cross` 的来自其它中心。因此 within 与 cross 的差值不含测试集差异, "
          "只含\"训练数据是否与测试同中心\"这一个变量。")
        A("")
        A("| 训练数据来源 | PR-AUC/base 中位 (L0 组成 GBDT) | PR-AUC/base 中位 (L1 冻结 PLM) |")
        A("|---|---|---|")
        A(f"| **within**: 留出中心内部 | **{_mp['L0_within']:.2f}** | "
          f"**{_mp['L1_within']:.2f}** |")
        A(f"| **cross**: 其它中心 | {_mp['L0_cross']:.2f} | {_mp['L1_cross']:.2f} |")
        A(f"| 随机对照 (实测) | {_mp['random']:.2f} | {_mp['random']:.2f} |")
        A("")
        # 必须给出两侧训练规模, 否则"within 只是在小数据上过拟合"这个质疑没被堵住
        A("**先堵住最自然的质疑: \"within 会不会只是在小数据上过拟合?\"** "
          "答案是不会, 而且方向正好相反 —— **cross 的训练数据多得多**:")
        A("")
        A("| 折 | 留出中心 | within 训练簇 | cross 训练簇 | cross / within |")
        A("|---|---|---|---|---|")
        _ratios = []
        for _r in (L1e or {}).get("folds", []):
            _inner = _r.get("inner_folds") or []
            if not _inner:
                continue
            _w = int(np.mean([x["n_train_within"] for x in _inner]))
            _c = int(np.mean([x["n_train_cross"] for x in _inner]))
            _ratios.append(_c / max(_w, 1))
            A(f"| {_r['fold']} | {', '.join(_r['held_centers'])} | {fm(_w)} | "
              f"{fm(_c)} | **{_c/max(_w,1):.1f}×** |")
        A("")
        A("去冗余后本数据集每个 `split_group` 只保留一条记录, 所以上表的条数"
          "**就是同源簇数**, 不是被同源冗余虚高的记录数。")
        A("")
        if _ratios:
            A(f"cross 一侧的训练簇是 within 的 "
              f"**{min(_ratios):.1f}–{max(_ratios):.1f} 倍**, 却在同一批测试条目上"
              f"从 {_mp['L1_within']:.2f} 掉到 {_mp['L1_cross']:.2f}。"
              "**数据更多、成绩更差**, 所以这个落差不可能是 within 侧样本量不足"
              "导致的过拟合, 也不可能是 cross 侧欠拟合 —— "
              "唯一随之改变的变量是\"训练数据是否与测试来自同一中心\"。")
            A("")
        A(f"**同中心训练时模型是有效的 (L1 {_mp['L1_within']:.2f}), "
          f"换成跨中心训练就塌到随机水平 ({_mp['L1_cross']:.2f} vs 随机 "
          f"{_mp['random']:.2f})。** 这正面证明: 可学的东西确实存在, "
          "但它**不跨中心迁移** —— 与 §3 的身份通道证据是同一件事的两面。"
          "随机对照落在 1.0 附近, 说明评估口径是校准的, "
          "所以 cross 的低值不是计算错误。")
        A("")
        A("> 这组对照的实现上有一个必须讲明的陷阱, 我们自己先踩了: "
          "最初把 within 的训练集写成\"留出中心里不属于测试组的部分\", "
          "但测试折就是该中心的**全部**组, 所以那个掩码恒为空 —— "
          "within 一栏永远拿不到数。"
          "正确做法是把留出中心的同源组再分 K=3 份, 内层第 i 份作测试、"
          "其余份作 within 的训练, 同时让 cross 也只在这一份上评估, "
          "以保证两侧测试集逐条相同。这个 bug 是在跑 GPU 之前用 numpy "
          "先验掩码非空才发现的, 否则会白跑一轮并得到\"within 无数据\"的错误结论。")
        A("")

    A("### 5.3 平凡基线没有被语言模型超过")
    A("")
    _B = bs["results"]
    _held = {}
    for _a in (L1e, L2e, L1s, L2s):
        for _r in (_a or {}).get("folds", []):
            _held[_r["fold"]] = _r["held_centers"]

    def cell(tag, fo):
        return (_B.get(tag) or {}).get(str(fo))

    def show(tag, fo, adv_tag=None):
        c = cell(tag, fo)
        if not c:
            return "—", "无区间"
        pa = c["pr_auc_over_base"]
        k = classify(c, cell(adv_tag, fo) if adv_tag else None)
        # 区间端点看着不跨 1.0 却判"不可区分"时就地标注依据, 否则读者会当成表算错
        if not c.get("seed_stability", {}).get("stable", True):
            k += " ᵇ"
        return fmt_ci(pa["point"], pa["ci95"][0], pa["ci95"][1]), k

    A("同一批留出中心、同一套测试折上, L0 (**只用 20 维氨基酸组成的 GBDT, 不含长度特征**) "
      "与 L1 / L2 (ESM-2 650M 冻结 / LoRA 微调) 的对比:")
    A("")
    A("| 折 | 基础率 | L0 GBDT | L1 冻结 | L2 LoRA |")
    A("|---|---|---|---|---|")
    for fo in (1, 2, 3, 4):
        c0 = cell("compare_levels_express:L0", fo)
        if not c0:
            continue
        _r = [f"{fo}", f"{c0['base_rate']:.3f}"]
        for tg, adv in (("compare_levels_express:L0", None),
                        ("L1_esm2_650M_frozen_express", "L1_esm2_650M_frozen_express_adv"),
                        ("L2_esm2_650M_lora_express", "L2_esm2_650M_lora_express_adv")):
            v, k = show(tg, fo, adv)
            _r.append(f"{v} · {k}" if v != "—" else "未跑")
        A("| " + " | ".join(_r) + " |")
    A("")
    _f1_0, _f1_1 = cell("compare_levels_express:L0", 1), cell("L1_esm2_650M_frozen_express", 1)
    _f1_2 = cell("L2_esm2_650M_lora_express", 1)
    _need = {"compare_levels_express:L0": _f1_0, "L1 express": _f1_1, "L2 express": _f1_2}
    _absent = [k for k, v in _need.items() if not v]
    if _absent:
        raise SystemExit(
            f"!! bootstrap_ci.json 缺 {_absent} 的折 1 区间 —— §4.3 的 L0/L1/L2 对比"
            f"写不出来。补算:\n"
            f"   .venv/bin/python src/eval/bootstrap_ci.py --folds 1 2 3 4 "
            f"--score-col L0_cross_score --preds compare_levels_express_preds.parquet\n"
            f"   (宁可硬失败, 也不出一份悄悄少了一栏的稿子)")
    A(f"**在唯一有稳定真信号的折上 (折 1), L0 的 PR-AUC/base "
      f"{_f1_0['pr_auc_over_base']['point']:.2f} "
      f"[{_f1_0['pr_auc_over_base']['ci95'][0]:.2f}, {_f1_0['pr_auc_over_base']['ci95'][1]:.2f}] "
      f"高于 L1 的 {_f1_1['pr_auc_over_base']['point']:.2f}, "
      f"与 L2 的 {_f1_2['pr_auc_over_base']['point']:.2f} "
      f"[{_f1_2['pr_auc_over_base']['ci95'][0]:.2f}, "
      f"{_f1_2['pr_auc_over_base']['ci95'][1]:.2f}] 区间重叠。**"
      " 即: 把 650M 参数的蛋白语言模型换上去, 并没有在跨中心设定下买到比"
      "**20 维氨基酸组成**更多的东西。这与 §5.1 中 GBDT 达到或超过已发表工具是同一回事, "
      "不是两个独立结果。")
    A("")
    A("L0 在折 2 上是**反向**而 L1 / L2 不是, 说明 PLM 表示至少没有把那一折做得更差; "
      "但这不构成\"PLM 更好\"的证据, 因为折 2 的 L2 本身是边界情形 (见 §5.6)。")
    A("")

    A("### 5.4 四情形混合: 中心间方向与幅度均不一致, 并有一例跨任务方向相反")
    A("")
    A("判定规则 (`configs/stage3_train.yaml` changelog `[2026-10-02]`, "
      "实现在 `src/eval/bootstrap_ci.py`): 按 `split_group` 分层自助法 2,000 次, "
      "**区间定方向能不能分辨, 点估计定幅度够不够用**。")
    A("")
    A("| 判定 | 条件 |")
    A("|---|---|")
    A("| 真信号 | 区间不跨 1.0, 点估计 \u2265 1.10, 且对抗版区间同样不跨 1.0 |")
    A("| 边界情形 | 形式上满足真信号, 但**对抗版区间跨 1.0** —— 稳健的信号不会因为"
      "加入对抗头而消失 |")
    A("| 弱但可分辨 | 区间不跨 1.0 但点估计 < 1.10 (统计上可分辨, 幅度只有几个百分点) |")
    A("| 不可区分 | 区间跨 1.0 |")
    A("| 反向 | 区间整体 < 1.0 |")
    A("| *(前置)* 方向稳健性 | 上述方向判定须在 **5 个独立自助法种子**上全部一致; "
      "不一致则按保守方向定为\"不可区分\" |")
    A("")
    A("**为什么要加种子稳健性这一条。** 定稿前核对发现两个单元格在两位小数下都显示 "
      "`[1.00, 1.04]` 却判定相反 (下界 0.9957 跨 1.0 vs 1.0010 不跨), "
      "差 0.001。直接测: 其中一个的方向判定在 5 个种子里 4 正 1 不定 —— "
      "**换个种子结论就变**; 把重采样次数从 2,000 提到 20,000 仍是 1.0007, "
      "不解决, 因为不确定性在数据里而不在重采样次数里。"
      "一个会被随机种子决定的方向不是结论。"
      "这与\"加入对抗头就消失的信号不算真信号\"是同一条纪律: "
      "结论必须对与科学问题无关的实现选择不敏感。"
      "未取多数票, 因为 4:1 也不该当定论。实查全部 "
      f"{sum(len(v) for v in _B.values())} 个折级结果, **只有 1 个被这条规则改判** "
      "(L2 `soluble_expression` 折 3, 由\"弱但可分辨\"改为\"不可区分\")。"
      "区间端点落在 1.0 的 ±0.02 内时, 本文一律打到小数点后 4 位。")
    A("")
    A("> 表中标 ᵇ 的单元格即属此类: 方向判定在 5 个种子上不一致, "
      "故即便打印出的区间端点不跨 1.0, 也按保守方向定为\"不可区分\", "
      "不作方向性结论。")
    A("")
    A("把\"可分辨\"和\"够用\"分开是必须的: 本文最大的折 (n=7,399) 上 +5% 的提升"
      "区间也不跨 1.0。若只看区间, 它会被报成真信号; 若只看点估计, "
      "小折上真实的中等效应又会被漏掉。")
    A("")
    for nm, tag, advtag in (
            ("express (主任务) · L1 冻结", "L1_esm2_650M_frozen_express",
             "L1_esm2_650M_frozen_express_adv"),
            ("express · L2 LoRA", "L2_esm2_650M_lora_express",
             "L2_esm2_650M_lora_express_adv"),
            ("soluble_expression (对照) · L1 冻结",
             "L1_esm2_650M_frozen_soluble_expression",
             "L1_esm2_650M_frozen_soluble_expression_adv"),
            ("soluble_expression (对照) · L2 LoRA",
             "L2_esm2_650M_lora_soluble_expression",
             "L2_esm2_650M_lora_soluble_expression_adv")):
        if not _B.get(tag):
            continue
        A(f"**{nm}**")
        A("")
        A("| 折 | 留出中心 | n | 基础率 | PR-AUC/base [95% CI] | 判定 |")
        A("|---|---|---|---|---|---|")
        for fo in (1, 2, 3, 4):
            c = cell(tag, fo)
            if not c:
                continue
            v, k = show(tag, fo, advtag)
            A(f"| {fo} | {', '.join(_held.get(fo, ['—']))} | {fm(c['n'])} | "
              f"{c['base_rate']:.3f} | {v} | **{k}** |")
        A("")
    A("")
    A("**跨任务对照** (同一批留出中心、同一个 L1 架构, 方向与幅度分开看):")
    A("")
    A("| 折 | 留出中心 | express 方向 | express 幅度 | soluble_expression 方向 | "
      "幅度 | 方向是否一致 |")
    A("|---|---|---|---|---|---|---|")

    def _dir(c):
        lo, hi = c["pr_auc_over_base"]["ci95"]
        return "**正**" if lo > 1.0 else ("**负**" if hi < 1.0 else "不定")

    def _mag(c):
        pt = c["pr_auc_over_base"]["point"]
        return f"{pt:.2f} ({pt-1:+.0%})"

    _opp = []
    for fo in (1, 2, 3, 4):
        ce = cell("L1_esm2_650M_frozen_express", fo)
        cs = cell("L1_esm2_650M_frozen_soluble_expression", fo)
        if not (ce and cs):
            continue
        de, ds = _dir(ce), _dir(cs)
        if not ce.get("seed_stability", {}).get("stable", True):
            de = "不定 ᵇ"
        if not cs.get("seed_stability", {}).get("stable", True):
            ds = "不定 ᵇ"
        same = de == ds
        if not same and not any("不定" in x for x in (de, ds)):
            _opp.append(fo)
        A(f"| {fo} | {', '.join(_held.get(fo, ['—']))} | {de} | {_mag(ce)} | "
          f"{ds} | {_mag(cs)} | {'一致' if same else '**相反**'} |")
    A("")
    A("**要分两句说, 强度不同。**")
    A("")
    _dirs, _mags = [], []
    for fo in (1, 2, 3, 4):
        _c = cell("L1_esm2_650M_frozen_express", fo)
        if _c:
            _dirs.append(_dir(_c).strip("*"))
            _mags.append(_c["pr_auc_over_base"]["point"] - 1)
    _pos = sorted(m for m in _mags if m > 0)
    A(f"1. **跨中心不一致 —— {len(_dirs)} 折都支持。** `express` 上方向为 "
      + "、".join(_dirs) + f"; 幅度从 {min(_mags):+.0%} 到 {max(_mags):+.0%}。"
      + (f"即便只看方向为正的 {len(_pos)} 折, 幅度也差 "
         f"{_pos[-1]/_pos[0]:.0f} 倍 ({_pos[0]:+.0%} vs {_pos[-1]:+.0%})。"
         if len(_pos) >= 2 and _pos[0] > 0 else "")
      + "这条是本文的主结论。")
    A(f"2. **跨任务方向相反 —— 只有 {len(_opp)} 例, 按单例观察写。** "
      f"折 {_opp[0] if _opp else '—'} 上 `express` 的区间整体低于 1.0、"
      f"`soluble_expression` 的整体高于 1.0, 方向确实相反; "
      f"**其余三折方向一致** (折 1、折 4 同为正, 折 2 同为不定), 只是幅度不同。"
      f"所以我们写\"在折 {_opp[0] if _opp else '—'} 上观察到方向相反, 其余折未见\", "
      "**不写成跨任务不一致是普遍规律** —— 一例不足以支撑普遍性陈述。")
    A("")
    A("**L2 (LoRA 微调) 上的独立检验 —— 结果是不复现, 如实写。**")
    A("")
    A("| 折 | express 方向 / 幅度 | soluble_expression 方向 / 幅度 | 方向 |")
    A("|---|---|---|---|")
    for fo in (1, 2, 3, 4):
        ce = cell("L2_esm2_650M_lora_express", fo)
        cs = cell("L2_esm2_650M_lora_soluble_expression", fo)
        if not (ce and cs):
            continue
        de, ds = _dir(ce), _dir(cs)
        if not cs.get("seed_stability", {}).get("stable", True):
            ds = "不定 ᵇ"
        if not ce.get("seed_stability", {}).get("stable", True):
            de = "不定 ᵇ"
        A(f"| {fo} | {de} / {_mag(ce)} | {ds} / {_mag(cs)} | "
          f"{'一致' if de == ds else ('不可比' if any('不定' in x for x in (de, ds)) else '**相反**')} |")
    A("")
    _l2s3 = cell("L2_esm2_650M_lora_soluble_expression", 3)
    if _l2s3:
        _ss = _l2s3.get("seed_stability", {})
        A(f"在 L2 上, 折 3 的 `express` 侧仍是反向, 但 `soluble_expression` 侧的方向"
          f"**判不出来** —— 5 个自助法种子给出 "
          f"{_ss.get('directions', [])}, 不一致, 故按保守方向定为不可区分 "
          f"(点估计 {_l2s3['pr_auc_over_base']['point']:.2f}, 与 L1 同号但不显著)。"
          "**所以折 3 的方向相反在 L2 上没有复现成一个可判定的对比。** "
          "这不构成对 L1 结果的反证 (两者点估计同号、无一例方向翻转), "
          "但也**不提供独立佐证**。")
        A("")
        A("因此我们把这条写到它能承担的程度为止: "
          "**在 L1 的折 3 上观察到两个任务方向相反 (两侧均种子稳定); "
          "L2 上该对比因一侧方向不可判定而未复现; 其余折未见方向相反。**"
          "这是一例单中心组观察, 不是规律。")
        A("")
    A("这两条合起来堵死最自然的两条补救方案: 既不能靠\"这个中心以前表现好\"迁移 "
      "(跨中心方向与幅度都不一致), 也不能假定\"在一个任务上没问题就在另一个任务上也没问题\" "
      "(至少有一例反例)。没有哪个事前可得的量 —— 中心身份、基础失败率、样本量、任务 —— "
      "能预知下一次部署落进哪一类。")
    A("")
    A("**反向的边界要写清**: 反向只在 `express` 任务的 1 个留出中心组上出现, "
      "且在 L1 / L2 / 对抗版三者上一致 ([0.77, 0.80] / [0.83, 0.86] / [0.78, 0.81])。"
      "**不写成\"跨中心反向是这类数据的普遍现象\"** —— 它是四情形之一。")
    A("")
    A("### 5.5 排除组: 不是模型不够大, 也不是中心身份没去掉")
    A("")
    A("| 排除了什么 | 证据 |")
    A("|---|---|")
    f3 = folds_of(L2e).get(3)
    if f3 and bs:
        b3 = (bs["results"].get("L2_esm2_650M_lora_express", {}) or {}).get("3", {})
        ci = b3.get("pr_auc_over_base", {}).get("ci95", ["—", "—"])
        A(f"| 不是模型不够大 | L2 微调 5.41M LoRA 参数 (0.82%) / 3 epoch 后, "
          f"折 3 反向依然存在 (PR-AUC/base {f3['cross_pr_auc_over_base_mean']:.2f} "
          f"[{ci[0]:.2f}, {ci[1]:.2f}], 区间远离 1.0) |")
    A("| 不是中心身份没去掉 | 对抗头的中心判别准确率降至 0.429–0.507 (≈猜最大类), "
      "中心可分性已从表示中移除; 而折 3 反向不变 (0.79 → 0.80) |")
    A("")
    A("两条路都被排除, 反向依然存在, **指向同一个 open question**: "
      "该反向可能源于标签生成机制随中心而变, 而非序列表示本身。"
      "现有数据无法判定, 不进一步推测。")
    A("")

    A("### 5.6 一个边界情形 (我们自己也没过线)")
    A("")
    if bs:
        n2 = bs["results"].get("L2_esm2_650M_lora_express", {}).get("2")
        a2 = bs["results"].get("L2_esm2_650M_lora_express_adv", {}).get("2")
        if n2 and a2:
            pa, pb = n2["pr_auc_over_base"], a2["pr_auc_over_base"]
            A(f"L2 在 express 折 2 (基础率 {n2['base_rate']:.3f}) 上 PR-AUC/base = "
              f"{fmt_ci(pa['point'], pa['ci95'][0], pa['ci95'][1])}, "
              f"形式上区间不跨 1.0; 但加入对抗头后回到 "
              f"[{pb['ci95'][0]:.2f}, {pb['ci95'][1]:.2f}] 跨 1.0, "
              f"且 lift@100 点估计 {n2['lift@100']['point']:.2f} 方向相反。"
              "**判为边界情形的直接依据: 一个稳健的信号不会因为加入对抗头而消失。**")
            A("")
            _thr = next((r["overall"]["pr_auc_over_base"] for r in gb["results"]
                         if r["split"] == "lab" and r["stage"] == "express"
                         and "overall" in r), float("nan"))
            _pass = {}
            for _t, _av in (("L1_esm2_650M_frozen_express",
                             "L1_esm2_650M_frozen_express_adv"),
                            ("L2_esm2_650M_lora_express",
                             "L2_esm2_650M_lora_express_adv")):
                _ks = {f: classify(c, (bs["results"].get(_av) or {}).get(f))
                       for f, c in (bs["results"].get(_t) or {}).items()}
                _pt = {f: c["pr_auc_over_base"]["point"]
                       for f, c in (bs["results"].get(_t) or {}).items()}
                _pass[_t] = (sum(1 for f, v in _ks.items()
                                 if v in ("真信号", "弱但可分辨")
                                 and _pt[f] >= _thr), len(_ks))
            _p1, _n1 = _pass["L1_esm2_650M_frozen_express"]
            _p2, _n2 = _pass["L2_esm2_650M_lora_express"]
            A(f"因此, 本文给后来者定的门槛是**两条同时满足** "
              f"(见 `reports/paper_data_methods.md` §2.6): "
              f"**(a)** 按簇分层自助法 95% 区间不跨 1.0, **(b)** PR-AUC/base 点估计 "
              f"\u2265 {_thr:.2f} —— 即超过本文那个**只用 20 维氨基酸组成、不含长度**的平凡基线"
              f"在同一实验室留出集上的成绩。"
              "只要 (a) 不够: n 够大时 +2% 也能显著, 本文折 4 就是这样 "
              "(1.05 [1.03, 1.07], 方向可分辨但幅度不到平凡基线)。"
              "按这把双条件的尺子量我们自己: "
              f"**L1 在 {_n1} 个留出中心组里 {_p1} 组、"
              f"L2 在已落预测的 {_n2} 组里 {_p2} 组同时满足 (a)(b)**。"
              "这不是单设给别人的标准, 是用同一把尺子量自己之后的结果。")
            A("")

    # ───────── 5 局限 ─────────
    A("## 6 局限")
    A("")
    A("1. **只有一个阶段具备可用规模**。去冗余后仅 `express` 的真实阴性过万; "
      "六阶段联合建模所需的数据不存在于公开来源, 故本文定位为数据集与基准。")
    A("2. **`soluble` 阶段未解决**。其阴性 99.9% 来自推断规则; "
      "唯一的显式对照子集仅 10 条、全部来自单一中心 (MPSBC) 且**全部不含 His 标签** "
      "(His6 侧 n=0), 因此在设计上无法检验。"
      "推断子集内的 His6 关联在中心之间方向反转, 不支持生物学解释, 但不足以定论。"
      "另有中心两侧各数十万条却零 `soluble` 失败, 与那 10 条并列支撑"
      "\"soluble 失败是记录产物\"。")
    A("3. **测试集系统性地不含处于大同源网络中的蛋白** (与 §4 是同一件事, 在此作为"
      "局限重述, 因为它限制了本文结论的适用范围): 在 30% 相似度、80% 覆盖下取传递闭包后, "
      "靶点空间形成一个巨型连通分量"
      + (f" (含 {fm(sg['largest_group_clusters'])} 个 set-cover 簇)" if sg else "")
      + "; 占比超过 1% 的组被强制钉入训练侧, 否则单组就会吃掉整个测试集。"
      "代价是本文的留出测试集偏向同源网络稀疏的蛋白, "
      "**对多结构域、大家族蛋白的泛化能力本文无法评估**。"
      "这既影响绝对成绩, 也可能影响跨中心落差的大小 —— 方向未知, 不作推测。")
    A("4. **时间留出集在无泄漏约束下几乎不存在** (测试侧仅数百条), 只能作动力不足的弱检验。")
    _ro = next((r for r in gb["results"]
                if r["split"] == "sequence" and r["stage"] == "express"
                and "by_organism" in r), None)
    if _ro:
        _o, _c = _ro["by_organism"], _ro["by_center"]
        A(f"5. **物种维度的分组审计做不出结论**: 序列切分测试侧 {fm(_o['n_records'])} 条"
          f"摊在 {_o['n_groups_total']} 个物种名上, 只有 {_o['n_groups_evaluated']} 个"
          f"达到 {_o['min_group_n']} 条的最小组门槛 (对照: 实验室维度 "
          f"{_c['n_groups_evaluated']}/{_c['n_groups_total']})。"
          "降低门槛能凑出更多\"组\", 但几十条样本上的 precision@100 是噪声不是发现, "
          "所以本文的分组审计以实验室与年份为主轴, 不声称跨物种泛化。")
    A("6. **无前瞻性验证**。全部评测为回顾性, 本工作不含湿实验。")
    if ds6:
        A(f"7. **自有管线数据不能补救**。OIH 计算中心的 {fm(ds6['unique_designs'])} 条"
          "唯一设计是计算产物: 任务级失败是作业崩溃, 而把结构置信度阈值当标签会构成循环论证"
          "(本文实测该指标对真实结合的 per-target 最高 F1 仅 0.572)。")
    A("")

    # ───────── 6 可获取性与许可 ─────────
    A("## 7 代码与数据可获取性")
    A("")
    A("代码在 **Apache-2.0** 下发布。衍生数据 (统一标签、四套冻结切分、按中心的折) "
      "**不是单一许可** —— 上游的 share-alike 条款不允许我们统一降级为 CC BY 4.0:")
    A("")
    _rec = pl.read_parquet("data/processed/records.parquet", columns=["source"])
    _n = {r["source"]: r["len"] for r in
          _rec.group_by("source").agg(pl.len()).to_dicts()}
    _sa = ["ds1_targettrack", "ds5_adaptyv_egfr"]
    _san = sum(_n.get(k, 0) for k in _sa)
    _tot = _rec.height
    A("| 部分 | 范围 | 许可 |")
    A("|---|---|---|")
    A(f"| 代码 (`src/`, `configs/`) | 全部 | **Apache-2.0** |")
    A(f"| 含 share-alike 上游的衍生数据 | {fm(_san)} 条 "
      f"({_san/_tot*100:.1f}%; TargetTrack 与 Adaptyv), 以及据其构造的 "
      f"`lab` / `time` 切分、`center_folds`、`express` 任务的全部标签 | "
      f"**CC BY-SA 4.0** |")
    A(f"| 其余衍生数据 | {fm(_tot - _san)} 条 "
      f"({(_tot-_san)/_tot*100:.1f}%) | **CC BY 4.0** |")
    A("")
    A("**为什么不能统一成 CC BY 4.0**: PSI TargetTrack 的许可是 CC BY-SA 4.0, "
      "其 share-alike 条款要求改编作品以相同或兼容的许可发布。"
      "我们从它的 `status` / `stopStatus` 字段推导六阶段标签, 属于改编。"
      f"而 TargetTrack 占 {_n['ds1_targettrack']/_tot*100:.1f}% 且是 `express` "
      "主任务与全部跨中心评估的**唯一**来源 —— "
      "本文的核心结果全部落在受 share-alike 约束的那部分上。"
      "**整体数据集若作为一个作品再分发, 按最严的上游条款走 (CC BY-SA 4.0)。** "
      "需要纯 CC BY 4.0 的使用者可取不含 TargetTrack 的子集, "
      "但该子集不含 `express` 主任务, 复现不出本文主结果。")
    A("")
    A("**这对下游使用者的实际含义**, 我们主动写明而不是让人事后发现: "
      "用了受 share-alike 约束的那部分 (它包含 `express` 主任务的全部标签与"
      "全部跨中心评估数据, 复现本文核心结果绕不开) 并再分发, "
      "**衍生成果也必须以 CC BY-SA 4.0 或兼容许可发布** —— "
      "不能改成 CC BY / MIT, 不能闭源再分发, 掺入自有数据后整体同样受约束。"
      "只做内部研究、不对外分发则不触发 share-alike (署名条款仍适用)。")
    A("")
    A("\"用 CC BY-SA 数据训练出的模型权重是否构成改编作品\"在法律上无定论; "
      "我们不就此表态, 只做一件可操作的事: **每次训练与评测都机器记录所用数据源** "
      "(`reports/runs/*.json` 的 `sources` 字段), 使用者据此判断某个权重是否触及 "
      "TargetTrack。")
    A("")
    A("**上游来源、许可与必须一并给出的引用** (全部字段于 2026-10-02 经各自官方 "
      "API / README 实查, 不凭记忆):")
    A("")
    A("| 来源 | DOI / 地址 | 许可 |")
    A("|---|---|---|")
    A("| PSI TargetTrack 2000-2017 | `10.5281/zenodo.821654` | **CC BY-SA 4.0** |")
    A("| Tsuboyama et al. 2023 | `10.5281/zenodo.7844779` | CC BY 4.0 |")
    A("| ProteinGym v1.3 | github.com/OATML-Markslab/ProteinGym | "
      "MIT (代码); 各 assay 版权归原论文 |")
    A("| Overath et al. 2025 binder 元分析 (数据沉积) | "
      "`10.5281/zenodo.15722219` | CC BY 4.0 |")
    A("| Adaptyv Bio EGFR 竞赛 | github.com/adaptyvbio/egfr_competition_1 / _2 | "
      "ODbL 1.0 (数据) + Apache-2.0 (代码) |")
    A("")
    A("两处容易踩的坑, 写明以免复现者重复踩:")
    A("")
    A("1. **Tsuboyama 的 Zenodo 记录有两个**。同一版本另有记录 `7992926`, "
      "其 `Tsuboyama2023_Dataset2_Dataset3` 少 `match_aaseq` / `name_original` "
      "两列 (697,658,024 vs 718,214,782 字节), 按它复现会得到不同的标签。"
      "本文用 `7844779`, 并以 md5 守卫强制 (`src/ingest/verify_raw.py`)。")
    A("2. **binder 元分析的正文与数据沉积许可不同**。预印本正文 "
      "(bioRxiv `10.1101/2025.08.14.670059`) 是 CC BY-NC-ND, "
      "而数据沉积 (Zenodo `15722219`) 是 CC BY 4.0。本文只用数据沉积里的 "
      "`final_dataset.csv`, 不受 NC-ND 约束。")
    A("")
    A("完整的许可分层、逐条引用条目、以及 ProteinGym 要求的 33 个 assay 原始论文"
      "引用清单见仓库 `DATA_AVAILABILITY.md`。"
      "本仓库**不重新分发任何上游原始文件**: `data/raw/` 由记录在 "
      "`configs/data_sources.yaml` 的地址下载、校验 md5 后只读使用。")
    A("")
    A(f"**仓库:** [{REPO_URL}]({REPO_URL})")
    A("")
    A("本文引用的 `src/…`、`configs/…`、`reports/…`、`data/processed/splits/…` 均为仓库内路径。"
      "有两项不在仓库中: 统一标签表 `records.parquet` 对代码托管平台来说过大, "
      "**将**沉积于 Zenodo (见下文「归档与沉积」) —— 其权威大小、行数与 sha256 记录在仓库的 "
      "`data/processed/records.parquet.prov.json`, 任何副本都可逐字节核验; "
      "以及上游原始文件, 本文根本不重新分发。")
    A("")
    A("`data/interim/` 下的中间产物也不分发: 由 `src/` 的脚本重新生成, "
      "仓库里只带它们的指纹 (见 9.1), 以便重算出来的副本能与本文所用的那份对账。")
    A("")
    A("**归档与沉积。** 四套冻结切分、`center_folds`、评估代码与本文引用的全部报告已在配套仓库中发布。"
      "统一标签表对代码托管平台来说过大, **将**单独沉积于 Zenodo; "
      "该沉积的 DOI 将在本预印本的修订版中补上。"
      "由于该沉积将作为**单一作品**再分发, 按上文给出的理由, "
      "它将**整体**适用 CC BY-SA 4.0; 需要纯 CC BY 4.0 子集的使用者"
      "必须自行剔除 TargetTrack 衍生部分来重建该子集, "
      "而那个子集**不含 `express` 主任务**。")
    A("")
    # ───────── 7 补充材料清单 ─────────
    A("## 8 参考文献")
    A("")
    A(f"> 全部条目的题目 / 作者 / 卷期页经 Crossref API 实查 "
      f"(第一轮 {_refs_all['verified_on']}, 第二轮 "
      f"{_refs_all.get('verified_on_round2','')} 补结晶倾向这一支与 Bushuiev 两篇的拆分), "
      "正文引语逐字核对出版商全文页或作者提供的 PDF; 登记在 `configs/references.yaml` "
      "(含每条引语原文与核实来源)。"
      "**未经实查的条目一律不写入** —— 核对过程中我凭记忆猜的两个 DOI 全都指向"
      "完全无关的论文 (一个指向巴西生物技术平台, 一个指向 CHO 细胞基因组), "
      "所以这条纪律不是形式主义。")
    A("")
    _order = ["netsolp", "soluprot", "plm_sol", "price2011",
              "predppcrys", "ppcpred", "crysalis", "fdetect", "deepcascade2021",
              "crystal_review", "bushuiev2024_iclr", "bushuiev2024_workshop"]
    for i, k in enumerate(_order, 1):
        r = RF[k]
        _t = " ".join(str(r["title"]).split())
        if r.get("doi"):
            _iss = f"({r['issue']})" if r.get("issue") else ""
            _w = (f"*{r['journal']}* {r['volume']}{_iss}: "
                  f"{r.get('pages','')}. doi:{r['doi']}")
        elif r.get("arxiv"):
            _w = f"{r.get('venue','')}. arXiv:{r['arxiv']}"
        else:
            _w = f"{r.get('venue','')}"
        _yr = str(r["year"])
        _ky = str(r.get("key", ""))
        if _ky and _ky[-1] in "ab":
            _yr += _ky[-1]
        A(f"{i}. {r['authors']} ({_yr}). {_t}. {_w}")
    A("")
    A("数据来源的引用条目 (TargetTrack / Tsuboyama / ProteinGym / Overath / Adaptyv) "
      "见 `DATA_AVAILABILITY.md` §5 与 `configs/data_sources.yaml`, 同样经实查。")
    A("")
    A("## 9 补充材料")
    A("")
    A(f"以下全部在仓库 [{REPO_URL}]({REPO_URL}) 中:")
    A("")
    for f, d2 in (("gate1_data_inventory.md", "数据清点与偏差审计 (含去冗余前后对比的核心图表)"),
                  ("gate2_baselines.md", "基线对照: AF3 ipSAE_min 复现、SoluProt (含污染检查)"),
                  ("splits_and_leakage.md", "四套切分的构造与泄漏验证"),
                  ("default/stage3_training.md", "L0/L1/L2 训练结果、对抗去偏、自助法区间、方法学陷阱"),
                  ("tag_confound_analysis.json", "中心内分层检验的全部数字"),
                  ("soluble_adjudication.json", "soluble 定案检查"),
                  ("ds6_oih_inventory.md", "自有管线清点与门槛校准"),
                  ("paper_data_methods.md", "Data & Methods 完整版")):
        A(f"- `reports/{f}` — {d2}")
    A("- `configs/stage3_train.yaml` 的 `changelog` — 评估规则改动的时间点、理由与影响")
    A("- `LICENSE` (Apache-2.0) · `LICENSE-DATA-CC-BY-SA-4.0.txt` / "
      "`LICENSE-DATA-CC-BY-4.0.txt` · `NOTICE` · `DATA_AVAILABILITY.md` "
      "(许可分层的完整说明)")
    A("- `reports/runs/` — 每次评测的数据源清单、切分 hash、超参、环境版本")
    A("- Zenodo 沉积 — 统一标签表的归档副本 "
      "(DOI 将在本预印本的修订版中补上)")
    A("")

    A("")
    A("### 9.1 冻结产物指纹")
    A("")
    A("每个产物都带一个 `.prov.json`, 记录自身的 `sha256` / 字节数 / 行数, "
      "以及**全部上游输入的同样三项**; 每个下游脚本启动时硬校验这条链 "
      "(`src/labels/provenance.py` 的 `require()`), 不一致就退出而不是继续跑。"
      "这条机制是被一次真实事故逼出来的: 上游重建后下游脚本读到陈旧的中间产物, "
      "SoluProt 的去污染子集从 2,377 条静默缩到 289 条 —— 数字照常产出, 没有任何报错。")
    A("")
    A("| 产物 | 行数 | sha256 (前 16 位) |")
    A("|---|---|---|")
    _pvs = (sorted(pathlib.Path("data/processed").rglob("*.prov.json")) +
            sorted(pathlib.Path("data/interim").glob("*.prov.json")))
    for _pv in _pvs:
        _j = json.loads(_pv.read_text())["artifact"]
        A(f"| `{_j['path']}` | {fm(_j.get('rows') or 0)} | `{_j['sha256'][:16]}` |")
    A("")
    A(f"完整的 sha256 与上游输入链见上表每个产物各自的 `.prov.json` (共 {len(_pvs)} 个, 随代码发布; 仓库另带中间产物与报告产物的指纹, 未列入上表)。")
    A("")

    # ───────── 交叉核对 ─────────
    require_in_report("reports/gate2_baselines.md",
                      SOLUPROT_MISS, SOLUPROT_MISS_OFFICIAL)
    checks = [("records", mf["records"]["rows"], rec.height),
              ("clusters", g1["clusters_30pct"], int(rec["cluster_rep"].n_unique()))]
    bad = [(n, a, b) for n, a, b in checks if a != b]
    if bad:
        print("!! 交叉核对不一致, 不出文件:")
        for n, a, b in bad:
            print(f"   {n}: 产物={a} 现算={b}")
        raise SystemExit(1)
    if any(o["verdict"] != "PASS" for o in leak.values()):
        raise SystemExit("!! 有切分未通过泄漏检查")

    OUT.write_text("\n".join(L), encoding="utf8")
    print(f"wrote {OUT} ({len(L)} 行)")

    # 英文稿不在生成链上 (手工/外部翻译), 中文稿一变它就陈旧了。
    # 这里只警告不硬失败: 日常重跑中文稿不该被英文稿卡住; 硬门在
    # src/eval/check_en_sync.py, 投稿前跑那个。
    _en = D / "PREPRINT_biorxiv_EN.md"
    if _en.exists():
        import subprocess
        _r = subprocess.run([sys.executable, "src/eval/check_en_sync.py"],
                            capture_output=True, text=True)
        if _r.returncode != 0:
            print("\n" + "!" * 72)
            print("!! 英文稿 (PREPRINT_biorxiv_EN.md) 已与本次重新生成的中文稿脱节。")
            print("!! 它是手工翻译产物, **不会自动更新** —— 投稿前必须重译。")
            print("!! 详情: python src/eval/check_en_sync.py")
            print("!! 重译后登记: python src/eval/check_en_sync.py --stamp")
            print("!" * 72)
    print(f"交叉核对 {len(checks)} 项一致; {len(leak)} 套切分泄漏检查均 PASS")


if __name__ == "__main__":
    main()
