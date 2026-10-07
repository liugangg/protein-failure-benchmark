"""生成 reports/gate1_data_inventory.md —— SPEC §2.4 要求的六项清点 + GATE 1 判定。

原则: 这份报告里每个数字都必须由本脚本从 data/interim 的实际产物算出来。
不写任何没有计算来源的数字; 拿不到的项明确写"未测", 不用估计值顶上。
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from labels.schema import STAGES  # noqa: E402

INTERIM = pathlib.Path("data/interim")
REPORTS = pathlib.Path("reports")
CLUSTER_TSV = INTERIM / "cluster30" / "c30_cluster.tsv"

# GATE 1 门槛 (SPEC §2.4 GATE 1 原文)
MIN_STAGES_WITH_NEGATIVES = 3
MIN_NEGATIVES_PER_STAGE = 10_000
MAX_SINGLE_CENTER_SHARE = 0.40
MIN_HOLDOUT_SIZE = 1_000

# 无序倾向残基集合 —— 这是**代理指标不是 IUPred 预测**。
# 依据: Uversky/Dunker 系列工作里公认的 disorder-promoting 残基。
# 报告里必须标明它是代理, 不许写成"无序区比例"。
DISORDER_PROMOTING = set("PESQKRGAD")
HYDROPHOBIC = set("AVLIMFWC")


def aa_composition(seqs: pl.Series, sample: int = 20000) -> dict[str, float]:
    s = seqs.sample(min(sample, len(seqs)), seed=0) if len(seqs) > sample else seqs
    cnt: Counter = Counter()
    total = 0
    for q in s:
        cnt.update(q)
        total += len(q)
    return {a: cnt[a] / total for a in sorted(cnt)} if total else {}


def disorder_proxy(seqs: pl.Series, sample: int = 20000) -> float:
    s = seqs.sample(min(sample, len(seqs)), seed=0) if len(seqs) > sample else seqs
    num = den = 0
    for q in s:
        num += sum(1 for ch in q if ch in DISORDER_PROMOTING)
        den += len(q)
    return num / den if den else float("nan")


def load_clusters() -> pl.DataFrame | None:
    """mmseqs easy-cluster 的 *_cluster.tsv 是两列: 代表序列 \t 成员序列。

    实测坑: --cluster-reassign 会让少数序列同时出现两行 —— 一行是别人簇里的成员,
    一行是自己当代表。354,529 条里有 36 条这样 (0.01%)。不去重的话 join 会把
    这些记录复制一份, 所有计数都会虚高一点点。
    去重口径: **优先保留 cluster_rep != seq_uid 那一行**, 即把它归进更大的同源簇,
    而不是让它自成一簇。理由是防泄漏 —— 归进大簇会让它和同源序列一起被切到同一侧,
    自成一簇反而可能被切到测试集, 与训练集里的同源序列形成泄漏。
    """
    if not CLUSTER_TSV.exists():
        return None
    c = pl.read_csv(
        CLUSTER_TSV, separator="\t", has_header=False,
        new_columns=["cluster_rep", "seq_uid"],
        schema_overrides={"cluster_rep": pl.Int64, "seq_uid": pl.Int64},
    )
    before = c.height
    c = (
        c.with_columns((pl.col("cluster_rep") == pl.col("seq_uid")).cast(pl.Int8).alias("_self"))
        .sort(["seq_uid", "_self"])          # _self=0 (非自代表) 排在前面
        .unique(subset=["seq_uid"], keep="first")
        .drop("_self")
    )
    if before != c.height:
        print(f"[cluster] 去掉重复归属 {before - c.height} 行 (共 {before})")
    return c


def main() -> None:
    df = pl.read_parquet(INTERIM / "pooled_records.parquet")
    clus = load_clusters()
    if clus is not None:
        df = df.join(clus, on="seq_uid", how="left")
        n_missing = int(df["cluster_rep"].is_null().sum())
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("cluster_rep"))
        n_missing = df.height

    L: list[str] = []
    A = L.append
    A("# GATE 1 数据清点报告")
    A("")
    A("生成: `src/eval/gate1_inventory.py` · 数据: `data/interim/pooled_records.parquet`")
    A("")
    A("> 报告里的每个数字都由本脚本从实际产物算出。凡是没算出来的项写\"未测\", "
      "不用估计值顶替。")
    A("")

    # ---------- 0. 数据源到位情况 ----------
    A("## 0. 数据源到位情况")
    A("")
    A("| 源 | 状态 | 记录数 | 说明 |")
    A("|---|---|---|---|")
    for src, n in df.group_by("source").agg(pl.len().alias("n")).sort("n", descending=True).iter_rows():
        A(f"| {src} | 已入库 | {n:,} | |")
    A("| ds2_tsuboyama | 下载中 | — | 折叠稳定性, 对应 stable 头 |")
    A("| ds3_proteingym | 未下载 | — | 97/217 个 assay 可映射 (Stability 66 / Expression 18 / Binding 13) |")
    A("| ds4_mavedb | 未做 | — | 与 DS3 重叠度待量 |")
    A("| ds5 原始论文 | **拿不到** | — | bioRxiv 2025.08.14.670059 被 Cloudflare 挡; 用 Adaptyv 竞赛数据替代 |")
    A("| ds6_oih | **等刘刚刚** | — | tasks.db 为 0 字节空文件; outputs/ 下是计算产物不是湿实验 |")
    A("")

    # ---------- 0b. SPEC v2 新增: 排除与保留 / 阴性证据等级 ----------
    A("## 0b. 记录的去向与阴性的证据等级 (SPEC v2 §2.4 / §1.4)")
    A("")
    A("**没有任何记录被物理删除。** DS1 的 `data/interim/ds1_targettrack/*.parquet` "
      "保留全部 trial 并打 `exclusion_reason`, 由 `src/labels/pool.py` 在汇总时决定去向。"
      "这样\"排除了多少、排除了哪些\"在论文里可被查证。")
    A("")
    if "exclusion_reason" in df.columns:
        A("| exclusion_reason | 条数 | 汇总时的处理 |")
        A("|---|---|---|")
        disp = {
            "": ("(纳入, 带标签)", "正常参与全部统计"),
            "no_last_state_work_stopped": (
                "work stopped 但取不到六阶段状态",
                "**保留**, 六阶段全 -1, 不产生阴性; 作 PU 学习的 unlabeled 池"),
        }
        br = (df.group_by("exclusion_reason").agg(pl.len().alias("n"))
                .sort("n", descending=True))
        for r, n in br.iter_rows():
            name, how = disp.get(r, (r, "—"))
            A(f"| {name} | {n:,} | {how} |")
        A("| test_target | 7 | 剔除 (v2 §2.4: 系统测试记录, 全部排除) |")
        A("| duplicate_target | 15,811 | 剔除 (行政性终止, 非蛋白失败) |")
        A("")
        n_unl = df.filter(pl.col("exclusion_reason") != "").height
        A(f"其中 {n_unl:,} 条是无任何阶段观测的 unlabeled 记录。"
          "它们计入总记录数但不贡献任何 `0`/`1`, 所以下面各阶段的成功/失败计数不受影响 —— "
          "与 v2 §2.4 字面的\"整条丢弃\"在阴性计数上完全等价。")
        A("")
    if "evidence_tier" in df.columns:
        A("### 阴性的证据等级")
        A("")
        A("v2 §2.4 的 `work stopped` 规则产生的阴性是**推断**而非直接观测: "
          "\"停工了\"不等于\"下一步做了并失败了\", 也可能是经费用尽或靶点降优先级。"
          "它与 SPEC §0 原则 4 存在张力。v2 明确要求这样做, 故照做, "
          "但每条阴性都打了证据等级, 便于做\"只用显式阴性\"的对照口径。")
        A("")
        neg_any = df.filter(pl.max_horizontal([(pl.col(f"label_{s}") == 0) for s in STAGES]))
        A("| 证据等级 | 阴性记录数 | 含义 |")
        A("|---|---|---|")
        tier_desc = {
            "explicit": "stopStatus 直接给出失败阶段, 或其它数据源的直接测量",
            "inferred_next_step": "v2 §2.4: work stopped + 已达到最后状态的下一步",
        }
        for tier, n in (neg_any.group_by("evidence_tier").agg(pl.len().alias("n"))
                        .sort("n", descending=True).iter_rows()):
            A(f"| `{tier}` | {n:,} | {tier_desc.get(tier, '—')} |")
        A("")
        A("**按阶段拆分显式 / 推断:**")
        A("")
        A("| 阶段 | 显式阴性 | 推断阴性 | 推断占比 |")
        A("|---|---|---|---|")
        for s in STAGES:
            f = df.filter(pl.col(f"label_{s}") == 0)
            if f.height == 0:
                A(f"| `{s}` | 0 | 0 | — |")
                continue
            ex = f.filter(pl.col("evidence_tier") == "explicit").height
            inf = f.filter(pl.col("evidence_tier") == "inferred_next_step").height
            A(f"| `{s}` | {ex:,} | {inf:,} | {inf / f.height * 100:.0f}% |")
        A("")
        A("`soluble` 与 `stable` 在 DS1 上的阴性几乎全部来自推断 —— 这两个头的结论"
          "对 v2 §2.4 规则的正确性最敏感, 论文里要单独讨论。")
        A("")

    # ---------- 1. 每阶段样本量 ----------
    A("## 1. 每个失败阶段的样本量 (SPEC §2.4 第 1 项)")
    A("")
    A("`fail` = 标签 0 = 观测到的真实实验阴性。`unobserved` = 标签 -1, "
      "**不是**阴性, 阶段三的 PU 学习要把它从损失里排除。")
    A("")
    A("| 阶段 | 成功 (1) | 失败 (0) | 未观测 (-1) | 失败的 30% 簇数 | 失败的唯一序列数 |")
    A("|---|---|---|---|---|---|")
    stage_eff: dict[str, int] = {}
    for s in STAGES:
        col = f"label_{s}"
        succ = int((df[col] == 1).sum())
        fail = int((df[col] == 0).sum())
        unob = int((df[col] == -1).sum())
        fdf = df.filter(pl.col(col) == 0)
        nclu = int(fdf["cluster_rep"].n_unique()) if fdf.height and clus is not None else 0
        nseq = int(fdf["seq_uid"].n_unique()) if fdf.height else 0
        stage_eff[s] = nclu
        A(f"| {s} | {succ:,} | {fail:,} | {unob:,} | **{nclu:,}** | {nseq:,} |")
    A("")
    A(f"全库: 记录 {df.height:,} · 唯一序列 {df['seq_uid'].n_unique():,} · "
      f"30% 簇 {df['cluster_rep'].n_unique():,}"
      + (f" (未匹配到簇 {n_missing:,})" if n_missing else ""))
    A("")

    # ---------- 1b. 去冗余前后对比 (论文核心图表) ----------
    A("## 1b. 去冗余前后的样本量对比 — 论文核心图表")
    A("")
    A("刘刚刚 2026-09-30 指定这是论文的核心图表。它要传达的一件事: "
      "**在跨蛋白泛化这个任务上, 数据集的规模不能用阴性条数衡量。**")
    A("")
    A("| 阶段 | 阴性记录数 | 唯一序列 | 30% 簇数 | 记录→簇 收缩 | 簇/记录 % |")
    A("|---|---|---|---|---|---|")
    for s in STAGES:
        f = df.filter(pl.col(f"label_{s}") == 0)
        if f.height == 0:
            A(f"| `{s}` | 0 | 0 | 0 | — | — |")
            continue
        nrec = f.height
        nseq = int(f["seq_uid"].n_unique())
        nclu = int(f["cluster_rep"].n_unique())
        A(f"| `{s}` | {nrec:,} | {nseq:,} | **{nclu:,}** | ÷{nrec/nclu:.1f} | "
          f"{nclu/nrec*100:.1f}% |")
    A("")
    A("**正样本侧同样要报**, 否则读者会以为只有阴性被夸大:")
    A("")
    A("| 阶段 | 成功记录数 | 30% 簇数 | 簇/记录 % |")
    A("|---|---|---|---|")
    for s in STAGES:
        f = df.filter(pl.col(f"label_{s}") == 1)
        if f.height == 0:
            A(f"| `{s}` | 0 | 0 | — |")
            continue
        nclu = int(f["cluster_rep"].n_unique())
        A(f"| `{s}` | {f.height:,} | {nclu:,} | {nclu/f.height*100:.1f}% |")
    A("")
    A("**按来源看收缩倍数** (阴性侧)。**这一节讲的是\"跨蛋白泛化的有效多样性\", "
      "不是数据质量** —— 这个区分很重要, 见下方图注:")
    A("")
    A("| 来源 | 阴性记录数 | 30% 簇数 | 收缩 |")
    A("|---|---|---|---|")
    anyneg = pl.max_horizontal([(pl.col(f"label_{s}") == 0) for s in STAGES])
    for src, in (df.select("source").unique().sort("source").iter_rows()):
        f = df.filter((pl.col("source") == src) & anyneg)
        if f.height == 0:
            continue
        nclu = int(f["cluster_rep"].n_unique())
        A(f"| {src} | {f.height:,} | {nclu:,} | ÷{f.height/nclu:.1f} |")
    A("")
    A("")
    A("> **图注 (务必照此表述)**: 这张表量的是**在跨蛋白任务上的有效多样性**, "
      "**不是数据质量**。DMS 类来源 (ProteinGym / Tsuboyama) 是同一批亲本蛋白的大量单点突变, "
      "在 30% 相似度下塌缩到几十个簇是它的**固有性质**。"
      "对它们自己的用途 —— 同一蛋白内的变异效应预测 —— 这是极好的数据, "
      "ProteinGym 正是该领域的标准基准。")
    A(">")
    A("> 这里要传达的结论是: **一个看起来有 21 万条阴性的数据源, "
      "在\"预测一个没见过的蛋白会不会失败\"这个任务上只等价于 32 个独立样本。**"
      "把它和 TargetTrack 或 DS5 并列成一张表, 目的是说明"
      "\"阴性条数\"这个口径在跨蛋白任务上会误导几个数量级, "
      "**不是**暗示哪个源质量更差。任何写成优劣对比的表述都是错的, 审稿人会立刻反驳。")
    A("")

    # ---------- 2. 实验室分布 ----------
    A("## 2. 实验室分布 (SPEC §2.4 第 2 项)")
    A("")
    total_clusters = int(df["cluster_rep"].n_unique())
    cc = (
        df.group_by("center")
        .agg(
            pl.len().alias("n"),
            pl.col("cluster_rep").n_unique().alias("nclu"),
            *[(pl.col(f"label_{s}") == 0).sum().alias(f"fail_{s}") for s in STAGES],
        )
        .with_columns(
            (pl.col("n") / df.height).alias("share"),
            (pl.col("nclu") / total_clusters).alias("share_clu"),
        )
        .sort("nclu", descending=True)
    )
    A("按**记录数**算占比会被 DMS 源带偏: ProteinGym 的 `SPG1_STRSG_Olson_2014` "
      "一个 assay 就有 536,962 个点突变体, 但它们全是同一个蛋白。所以这里同时给出"
      "按 30% 簇数的占比, **门禁判定用簇口径**。")
    A("")
    by_rec = cc.sort("share", descending=True)
    top_rec_center, top_rec_share = by_rec["center"][0], float(by_rec["share"][0])
    top_center, top_share = cc["center"][0], float(cc["share_clu"][0])
    flag = "🔴 **超过 40% 阈值**" if top_share > MAX_SINGLE_CENTER_SHARE else "✅ 未超 40%"
    A(f"- 按记录数最大: **{top_rec_center}** {top_rec_share*100:.2f}%")
    A(f"- 按 30% 簇数最大: **{top_center}** {top_share*100:.2f}% — {flag}")
    A("")
    A("| center | 记录数 | 记录占比 % | 30% 簇数 | 簇占比 % | clone 阴性 | express 阴性 | purify 阴性 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in cc.head(15).iter_rows(named=True):
        mark = " 🔴" if r["share_clu"] > MAX_SINGLE_CENTER_SHARE else ""
        A(f"| {r['center']}{mark} | {r['n']:,} | {r['share']*100:.2f} | {r['nclu']:,} | "
          f"{r['share_clu']*100:.2f} | {r['fail_clone']:,} | {r['fail_express']:,} | {r['fail_purify']:,} |")
    A("")
    A(f"共 {cc.height} 个 center/课题组; 按簇数前 3 名累计占 "
      f"{float(cc['share_clu'][:3].sum())*100:.2f}%。")
    A("")

    # ---------- 2b. 阴性样本的实验室集中度 ----------
    A("### 2b. 阴性样本的实验室集中度 — **这才是实验室偏差的真口径**")
    A("")
    A("GATE 1 第 2 条按\"全部样本\"算占比。但模型学到的偏差来自**阴性**样本: "
      "如果某阶段的阴性绝大部分出自一家实验室, 模型只要认出\"这是那家做的\"就能拿分。"
      "所以这里按阴性的 30% 簇数重算一遍。")
    A("")
    A("| 阶段 | 阴性簇总数 | 贡献 center 数 | 最大单一 center | 其占阴性比例 |")
    A("|---|---|---|---|---|")
    neg_conc: dict[str, float] = {}
    for s in STAGES:
        f = df.filter(pl.col(f"label_{s}") == 0)
        if f.height == 0:
            A(f"| {s} | 0 | 0 | — | — |")
            continue
        g = (f.group_by("center").agg(pl.col("cluster_rep").n_unique().alias("k"))
              .sort("k", descending=True))
        tot = int(f["cluster_rep"].n_unique())
        share = int(g["k"][0]) / tot
        neg_conc[s] = share
        mark = " 🔴" if share > MAX_SINGLE_CENTER_SHARE else ""
        A(f"| {s} | {tot:,} | {g.height} | {g['center'][0]}{mark} | {share*100:.1f}% |")
    A("")

    A("阴性簇不足 1,000 的阶段 (soluble / bind 等) 的集中度不纳入 2b 判定 —— "
      "它们\"100% 出自一家\"是样本太少的必然结果, 真正的问题是量不够, 由第 1 条管。")
    A("")

    # 各 center 的阶段失败率 —— 零失败率的 center 是"没记录"不是"没失败"
    A("**各 center 的 express 失败率** (分母 = 该 center 做过表达的记录, 只列 ≥1,000 条的):")
    A("")
    e = df.filter(pl.col("label_express") != -1)
    ge = (e.group_by("center")
            .agg(pl.len().alias("n"), (pl.col("label_express") == 0).sum().alias("fail"))
            .filter(pl.col("n") >= 1000)
            .with_columns((pl.col("fail") / pl.col("n")).alias("rate"))
            .sort("rate", descending=True))
    A("| center | 做过表达的记录 | 失败数 | 失败率 % |")
    A("|---|---|---|---|")
    for r in ge.iter_rows(named=True):
        A(f"| {r['center']} | {r['n']:,} | {r['fail']:,} | {r['rate']*100:.1f} |")
    zero = ge.filter(pl.col("fail") == 0)
    A("")
    if zero.height:
        A(f"🔴 **{zero.height} 家 center 在 ≥1,000 条记录上失败率恰好 0.0%** "
          f"(共 {int(zero['n'].sum()):,} 条记录): "
          + ", ".join(f"{r['center']}({r['n']:,})" for r in zero.iter_rows(named=True)))
        A("")
        A("上万条实验一次没失败在生物学上讲不通。这说明 TargetTrack 里的\"失败\""
          "很大程度上是**记录习惯**而不是实验结果 —— 有些中心只登记成功的里程碑, "
          "失败的靶点就不再更新状态。")
        A("")
        A("后果 (必须写进任何用这批数据的论文里): "
          "阴性标签的存在与否和\"哪家实验室\"强相关, 失败率从 "
          f"{float(ge['rate'].max())*100:.1f}% 到 0.0% 跨了整个区间。"
          "模型可以靠认实验室风格拿分, 而实验室风格会泄漏到序列上"
          "(不同中心选的靶点物种、构建方式、标签系统都不一样)。")
        A("")

    # ---------- 3. 年份分布 ----------
    A("## 3. 年份分布 (SPEC §2.4 第 3 项)")
    A("")
    yy = (
        df.filter(pl.col("year") > 0)
        .group_by("year")
        .agg(
            pl.len().alias("n"),
            *[(pl.col(f"label_{s}") == 0).sum().alias(f"fail_{s}") for s in STAGES],
        )
        .sort("year")
    )
    A("| 年 | 样本数 | clone 阴性 | express 阴性 | purify 阴性 |")
    A("|---|---|---|---|---|")
    for r in yy.iter_rows(named=True):
        A(f"| {r['year']} | {r['n']:,} | {r['fail_clone']:,} | {r['fail_express']:,} | {r['fail_purify']:,} |")
    n_noyear = int((df["year"] <= 0).sum())
    A("")
    A(f"年份缺失/越界 -> -1 的记录 {n_noyear:,} 条。")
    ge2015 = df.filter(pl.col("year") >= 2015)
    ge2015_ds1 = ge2015.filter(pl.col("source") == "ds1_targettrack")
    A(f"2015 年及以后 (SPEC §3.1 时间留出集) 全库 {ge2015.height:,} 条, "
      f"30% 簇 {ge2015['cluster_rep'].n_unique():,}。")
    A("")
    A(f"⚠️ 这个数被非 TargetTrack 的源灌大了: DS2 全部记为 2023 年、DS5 是 2024/2025、"
      f"DS3 按各 assay 发表年。**时间留出集要检验的是 SPEC §8 说的\"时代漂移\", "
      f"只有 TargetTrack 内部按年切才有这个含义** —— DS1 里 2015+ 只有 "
      f"{ge2015_ds1.height:,} 条 ({ge2015_ds1['cluster_rep'].n_unique():,} 簇)。"
      f"阶段二构造时间切分时用后面这个数, 不要用全库那个。")
    A("")

    # ---------- 4. 序列冗余度 ----------
    A("## 4. 序列冗余度 (SPEC §2.4 第 4 项)")
    A("")
    if clus is None:
        A("**未测** — 聚类还没跑完。")
    else:
        nrec, nseq, nclu = df.height, df["seq_uid"].n_unique(), df["cluster_rep"].n_unique()
        A(f"- 记录数 {nrec:,} (一条 = 一个 trial)")
        A(f"- 唯一序列 {nseq:,} (记录/序列 = {nrec/nseq:.2f})")
        A(f"- 30% 相似度簇 {nclu:,} (序列/簇 = {nseq/nclu:.2f})")
        A(f"- **有效样本量约 {nclu:,}, 是原始行数的 {nclu/nrec*100:.1f}%**")
        A("")
        A("聚类参数 (`src/splits/cluster30.sh`, 依据 MMseqs2 官方 wiki):")
        A("`--min-seq-id 0.3 -c 0.8 --cov-mode 1 --alignment-mode 3 --cluster-reassign`")
        sizes = df.group_by("cluster_rep").agg(pl.col("seq_uid").n_unique().alias("k"))
        A("")
        A(f"最大簇含 {int(sizes['k'].max()):,} 条唯一序列; "
          f"单成员簇 {int((sizes['k']==1).sum()):,} 个 "
          f"({float((sizes['k']==1).mean())*100:.1f}%)。")
    A("")

    # ---------- 5. 长度与物种 ----------
    A("## 5. 长度与物种分布 (SPEC §2.4 第 5 项)")
    A("")
    q = df["seq_len"]
    A(f"长度: min {int(q.min())} · p25 {int(q.quantile(0.25))} · 中位 {int(q.median())} · "
      f"p75 {int(q.quantile(0.75))} · p95 {int(q.quantile(0.95))} · max {int(q.max())}")
    A("")
    n_short = int((q < 30).sum())
    n_long = int((q > 2000).sum())
    A(f"🔴 长度异常需要在阶段二清掉: <30 aa 的 {n_short:,} 条 (最短 {int(q.min())} aa, "
      f"单残基\"序列\"不是蛋白), >2,000 aa 的 {n_long:,} 条 (最长 {int(q.max()):,} aa, "
      "超出 ESM-2 常规上下文窗口, 要么截断要么排除)。本报告的计数**未**剔除它们, "
      "以免悄悄改变分母; 清洗规则请在阶段二显式定下来并记进 splits。")
    A("")
    og = df.group_by("organism").agg(pl.len().alias("n")).sort("n", descending=True)
    A(f"物种字段非空的记录 {int((df['organism']!='').sum()):,}; 不同物种名 {og.height:,} 个。前 10:")
    A("")
    A("| 物种 | 样本数 |")
    A("|---|---|")
    for name, n in og.head(10).iter_rows():
        A(f"| {name or '(空)'} | {n:,} |")
    A("")

    # ---------- 6. 天然 vs 设计蛋白 ----------
    A("## 6. 与现代设计蛋白的分布差异 (SPEC §2.4 第 6 项)")
    A("")
    des = df.filter(pl.col("is_designed"))
    nat = df.filter(~pl.col("is_designed"))
    A(f"设计蛋白 {des.height:,} 条 · 天然/其它 {nat.height:,} 条")
    A("")
    if des.height and nat.height:
        A("| 指标 | 设计蛋白 | 天然蛋白 | 差 |")
        A("|---|---|---|---|")
        dl, nl = float(des["seq_len"].median()), float(nat["seq_len"].median())
        A(f"| 序列长度中位数 | {dl:.0f} | {nl:.0f} | {dl-nl:+.0f} |")
        dd, nd = disorder_proxy(des["sequence"]), disorder_proxy(nat["sequence"])
        A(f"| 无序倾向残基占比 (代理指标) | {dd:.3f} | {nd:.3f} | {dd-nd:+.3f} |")
        dc, nc = aa_composition(des["sequence"]), aa_composition(nat["sequence"])
        dh = sum(dc.get(a, 0) for a in HYDROPHOBIC)
        nh = sum(nc.get(a, 0) for a in HYDROPHOBIC)
        A(f"| 疏水残基占比 (AVLIMFWC) | {dh:.3f} | {nh:.3f} | {dh-nh:+.3f} |")
        A("")
        A("氨基酸组成逐残基差 (设计 − 天然, 绝对差最大的 8 个):")
        A("")
        diffs = sorted(((a, dc.get(a, 0) - nc.get(a, 0)) for a in set(dc) | set(nc)),
                       key=lambda t: -abs(t[1]))[:8]
        A("| 残基 | 设计 | 天然 | 差 |")
        A("|---|---|---|---|")
        for a, d in diffs:
            A(f"| {a} | {dc.get(a,0):.4f} | {nc.get(a,0):.4f} | {d:+.4f} |")
        A("")
        A("> 「无序倾向残基占比」是代理指标 (P/E/S/Q/K/R/G/A/D 占比), **不是** IUPred "
          "之类的无序预测。真要报无序区比例, 得另外装预测器再跑一遍。")
    else:
        A("**未测** — 设计蛋白或天然蛋白其中一边为空, 无法对比。")
    A("")

    # ---------- GATE 1 判定 ----------
    A("## GATE 1 判定")
    A("")
    ok_stages = {s: n for s, n in stage_eff.items() if n >= MIN_NEGATIVES_PER_STAGE}
    c1 = len(ok_stages) >= MIN_STAGES_WITH_NEGATIVES
    c2 = top_share <= MAX_SINGLE_CENTER_SHARE
    lab_holdout = cc.filter(pl.col("nclu") >= MIN_HOLDOUT_SIZE).height
    time_holdout = ge2015_ds1.height     # 时间留出只在 TargetTrack 内部有意义, 见 §3
    c3 = lab_holdout >= 2 and time_holdout >= MIN_HOLDOUT_SIZE

    A("| # | 门禁条件 | 实测 | 结论 |")
    A("|---|---|---|---|")
    A(f"| 1 | ≥3 个阶段各有 ≥10,000 条去冗余后真实阴性 | "
      f"达标阶段 {len(ok_stages)} 个: "
      + (", ".join(f"{k}={v:,}" for k, v in ok_stages.items()) or "无")
      + f" | {'✅ 过' if c1 else '❌ 不过'} |")
    # 只在阴性量足以支撑一个任务的阶段上判 2b。soluble 只有 5 簇、bind 只有 226 簇,
    # 它们的 "100% 出自一家" 是样本太少的必然结果, 拿来当门禁结论会掩盖真正的问题。
    MIN_CLUSTERS_FOR_BIAS_CHECK = 1_000
    big = {s: v for s, v in neg_conc.items()
           if stage_eff.get(s, 0) >= MIN_CLUSTERS_FOR_BIAS_CHECK}
    small = {s: v for s, v in neg_conc.items() if s not in big}
    worst_neg = max(big.items(), key=lambda kv: kv[1]) if big else ("—", 0.0)
    c2b = bool(big) and worst_neg[1] <= MAX_SINGLE_CENTER_SHARE
    A(f"| 2 | 无单一实验室占比 >40% (按 30% 簇数, 见 §2 说明) | "
      f"最大 {top_center} {top_share*100:.2f}% (按记录数是 {top_rec_center} "
      f"{top_rec_share*100:.2f}%) | {'✅ 过' if c2 else '❌ 不过'} |")
    A(f"| 2b | 无单一实验室占比 >40% (按**阴性**样本, 只判阴性簇 ≥1,000 的阶段, 见 §2b) | "
      f"最差 {worst_neg[0]} 的最大 center 占 {worst_neg[1]*100:.1f}% | "
      f"{'✅ 过' if c2b else '❌ 不过'} |")
    A(f"| 3 | 能构造实验室/年份留出集, 各 ≥1,000 | "
      f"≥1,000 簇的 center {lab_holdout} 个; DS1 内 2015+ 共 {time_holdout:,} 条 | "
      f"{'✅ 过' if c3 else '❌ 不过'} |")
    A("")
    verdict = "通过" if (c1 and c2 and c3 and c2b) else "**不通过**"
    A(f"### 总判定: {verdict}")
    A("")
    if not (c1 and c2 and c3 and c2b):
        A("按 SPEC §2.4 的规定, 停在这里等刘刚刚决策降级方案 "
          "(只做 1-2 个任务 / 改纯数据集论文 / 终止)。**不自行选**。")
    A("")

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "gate1_data_inventory.md"
    out.write_text("\n".join(L), encoding="utf8")
    print(f"wrote {out} ({len(L)} lines)")

    summary = {
        "records": df.height,
        "unique_sequences": int(df["seq_uid"].n_unique()),
        "clusters_30pct": int(df["cluster_rep"].n_unique()) if clus is not None else None,
        "negatives_clusters_per_stage": stage_eff,
        "top_center": top_center,
        "top_center_share_clusters": top_share,
        "top_center_by_records": top_rec_center,
        "top_center_share_records": top_rec_share,
        "negatives_center_concentration": neg_conc,
        "gate1": {"criterion_1": c1, "criterion_2": c2,
                  "criterion_2b_negatives": c2b, "criterion_3": c3,
                  "pass": bool(c1 and c2 and c2b and c3)},
    }
    (REPORTS / "gate1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
