"""把各 ingest 的 interim 产物汇成一张统一表, 并导出去冗余用的 FASTA。

去冗余的口径 (SPEC §2.4 第 4 项): 有效样本量看的是 30% 相似度下的**簇数**,
不是行数。同一条序列被同一个 center 反复试很多次, 行数会虚高。
"""
from __future__ import annotations

import pathlib
import sys

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from labels import provenance
from labels.schema import RECORD_COLUMNS, STAGES  # noqa: E402

INTERIM = pathlib.Path("data/interim")
BASE_COLS = [c for c in RECORD_COLUMNS]


def dedup_bind_against_dtu(df: pl.DataFrame) -> pl.DataFrame:
    """DTU 元分析已含 Adaptyv 两轮竞赛, 去掉 Adaptyv 侧重复的 bind 观测。

    为什么要做: SPEC v2 §2.2 指定的 DS5 (Zenodo 15722219) 里 `source` 包含
    "Adaptyv binder comp round 1/2" 共 302 行, 与我们先前从 Adaptyv GitHub 仓库
    抓的 ds5_adaptyv_egfr 重叠。不去重的话同一个设计的结合结果被计两次,
    bind 头的阴性数与簇数都会虚高。

    规则: **DTU 为 bind 的权威来源** (它经过统一的复现流程与特征计算)。
    Adaptyv 侧 binder 序列在 DTU 中出现过的行, 其 label_bind 降为 -1 (未观测);
    但保留 Adaptyv 独有的 clone / express 观测 —— DTU 数据集没有表达量记录。
    """
    if "ds5_dtu_binder" not in df["source"].unique().to_list():
        return df
    if "ds5_adaptyv_egfr" not in df["source"].unique().to_list():
        return df

    dtu_seqs = (df.filter(pl.col("source") == "ds5_dtu_binder")
                  .select("sequence").unique())
    dtu_list = dtu_seqs["sequence"].implode()
    overlap = (df.filter((pl.col("source") == "ds5_adaptyv_egfr")
                         & pl.col("sequence").is_in(dtu_list))).height
    df = df.with_columns(
        pl.when((pl.col("source") == "ds5_adaptyv_egfr")
                & pl.col("sequence").is_in(dtu_list))
        .then(pl.lit(-1, dtype=pl.Int8))
        .otherwise(pl.col("label_bind"))
        .alias("label_bind")
    )
    print(f"[pool] DS5 去重: Adaptyv 中有 {overlap} 条序列已在 DTU 元分析内, "
          f"其 bind 标签降为未观测 (DTU 为权威来源)")
    return df


def load_all() -> pl.DataFrame:
    parts = []
    def _norm(d: pl.DataFrame) -> pl.DataFrame:
        cols = list(BASE_COLS)
        if "cluster_key_seq" in d.columns:
            cols.append("cluster_key_seq")
        if "exclusion_reason" in d.columns:
            cols.append("exclusion_reason")
        if "evidence_tier" in d.columns:
            cols.append("evidence_tier")
        d = d.select(cols)
        if "cluster_key_seq" not in d.columns:
            d = d.with_columns(pl.col("sequence").alias("cluster_key_seq"))
        if "exclusion_reason" not in d.columns:
            d = d.with_columns(pl.lit("").alias("exclusion_reason"))
        # evidence_tier 只有 DS1 有 (显式 stopStatus 阴性 vs v2 §2.4 推断阴性)。
        # 其余源的阴性都是直接测量, 统一记 "explicit"。
        if "evidence_tier" not in d.columns:
            d = d.with_columns(
                pl.when(pl.max_horizontal(
                    [(pl.col(f"label_{s}") == 0) for s in STAGES]))
                .then(pl.lit("explicit")).otherwise(pl.lit("none")).alias("evidence_tier")
            )
        # 列顺序必须统一后再 concat: 各源补列的先后不同, 顺序不一致会让
        # pl.concat 报 "schema names differ" (实测踩到)。
        return d.select(list(BASE_COLS)
                        + ["cluster_key_seq", "exclusion_reason", "evidence_tier"])



    d1 = sorted((INTERIM / "ds1_targettrack").glob("*.parquet"))
    if d1:
        raw1 = pl.read_parquet(d1)
        # DS1 的 parquet 保留了**全部** trial 并打了 exclusion_reason (不物理删除)。
        # 在这里按 reason 决定去向:
        #   test_target      -> 剔除。v2 §2.4 "系统测试记录, 不是真实验, 全部排除"。
        #   duplicate_target -> 剔除。"发现重复所以停了"不是蛋白失败; 留着还会污染去冗余统计。
        #   no_last_state_work_stopped -> **保留**, 六阶段全 -1。
        #       刘刚刚 2026-09-30 定: 不算阴性, 但不要物理删除。
        #       它们有真实序列、只缺失败观测, 正是 PU 学习 (SPEC §4.3) 的 unlabeled 数据。
        if "exclusion_reason" in raw1.columns:
            br = (raw1.group_by("exclusion_reason").agg(pl.len().alias("n"))
                      .sort("n", descending=True))
            print("[pool] DS1 exclusion_reason 分布:")
            for r, n in br.iter_rows():
                print(f"        {r or '(纳入)':<32} {n:>8,}")
            drop_reasons = ["test_target", "duplicate_target"]
            before = raw1.height
            raw1 = raw1.filter(~pl.col("exclusion_reason").is_in(drop_reasons))
            print(f"[pool] 剔除 test_target + duplicate_target: {before - raw1.height:,} 条 "
                  f"(原始行仍在 data/interim/ds1_targettrack/ 内, 未物理删除)")
        parts.append(_norm(raw1))
    for p in [INTERIM / "ds2_tsuboyama.parquet",
              INTERIM / "ds3_proteingym.parquet",
              INTERIM / "ds5_dtu_binder.parquet",
              INTERIM / "ds5_adaptyv_egfr.parquet"]:
        if p.exists():
            parts.append(_norm(pl.read_parquet(p)))
    if not parts:
        raise SystemExit("data/interim 下没有任何 parquet, 先跑 ingest")
    df = pl.concat(parts, how="vertical_relaxed")
    df = dedup_bind_against_dtu(df)

    # 派生标签: 对齐 SoluProt 的「可溶表达」(见 labels.schema.soluble_expression_label)。
    # 不是第七个阶段, 不参与 GATE 1 计数, 只供基线对照。
    df = df.with_columns(
        pl.when((pl.col("label_express") == 1) & (pl.col("label_soluble") == 1)).then(1)
        .when(pl.col("label_express") == 0).then(0)
        .when((pl.col("label_express") == 1) & (pl.col("label_soluble") == 0)).then(0)
        .otherwise(-1).cast(pl.Int8).alias("label_soluble_expression")
    )

    # 年份清洗: 1999-2017 之外的是录入错误 (实测有 year=2104 一条)。
    # 不猜它本来是哪一年, 直接置 -1 当未知, 并在报告里记数。
    bad = df.filter((pl.col("year") != -1) & ~pl.col("year").is_between(1999, 2026)).height
    df = df.with_columns(
        pl.when((pl.col("year") != -1) & ~pl.col("year").is_between(1999, 2026))
        .then(-1)
        .otherwise(pl.col("year"))
        .alias("year")
    )
    print(f"[pool] 年份越界置为 -1 的记录数: {bad}")

    # 聚类键: 默认就是序列本身; DMS 类源 (DS3) 用亲本野生型序列。
    # 依据: 30% 去冗余要回答的是"有多少个**不同的蛋白**", 而点突变体之间差 1-2 个残基,
    # 逐条参与聚类既必然同簇又会把 all-vs-all 比对炸掉 (实测 mmseqs 卡死在 align)。
    if "cluster_key_seq" not in df.columns:
        df = df.with_columns(pl.col("sequence").alias("cluster_key_seq"))
    else:
        df = df.with_columns(
            pl.when(pl.col("cluster_key_seq").is_null() | (pl.col("cluster_key_seq") == ""))
            .then(pl.col("sequence"))
            .otherwise(pl.col("cluster_key_seq"))
            .alias("cluster_key_seq")
        )

    # seq_uid 必须**确定性**: 它是 pooled_unique_seqs.fasta 的 id, 下游 mmseqs 产物
    # (cluster30 / split_groups) 全靠它回连。
    # 踩过的坑 (2026-09-30): 原来写的是 .unique().with_row_index(), unique 不保证顺序,
    # 于是每次重跑 pool.py 都会重新编号, 而 c30_cluster.tsv 里还是旧编号 ——
    # join 上去不报错, 但序列和簇的对应**静默错位**, 所有簇级统计全失真
    # (clone 阴性簇数从 5,463 变成 9,260 才暴露出来)。
    # 解法: 先按序列排序再编号, 同样的输入永远得到同样的 seq_uid。
    seq2id = (
        df.select("cluster_key_seq").unique().sort("cluster_key_seq")
        .with_row_index("seq_uid")
        .with_columns(pl.col("seq_uid").cast(pl.Int64))
    )
    df = df.join(seq2id, on="cluster_key_seq", how="left")
    print(f"[pool] 记录 {df.height}, 唯一原始序列 {df['sequence'].n_unique()}, "
          f"唯一聚类键 {seq2id.height}")
    return df


def main() -> None:
    df = load_all()
    out = INTERIM / "pooled_records.parquet"
    df.write_parquet(out)
    # 盖章: 记录本产物的全部上游 (各源 ingest 的 parquet)
    ups = sorted((INTERIM / "ds1_targettrack").glob("*.parquet"))
    ups += [p for p in (INTERIM / "ds2_tsuboyama.parquet",
                        INTERIM / "ds3_proteingym.parquet",
                        INTERIM / "ds5_dtu_binder.parquet",
                        INTERIM / "ds5_adaptyv_egfr.parquet") if p.exists()]
    provenance.stamp(out, ups, produced_by="src/labels/pool.py")

    fa = INTERIM / "pooled_unique_seqs.fasta"
    uniq = df.select("seq_uid", "cluster_key_seq").rename({"cluster_key_seq": "sequence"}) \
             .unique(subset=["seq_uid"]).sort("seq_uid")
    with fa.open("w") as fh:
        for uid, seq in uniq.iter_rows():
            fh.write(f">{uid}\n{seq}\n")
    provenance.stamp(fa, [out], produced_by="src/labels/pool.py")
    print(f"[pool] wrote {out} 和 {fa} ({uniq.height} 条唯一序列)")

    n_unl = df.filter(pl.col("exclusion_reason") != "").height
    print(f"[pool] 其中无任何阶段观测的 unlabeled 记录 {n_unl:,} 条 "
          f"(PU 学习的 unlabeled 池; 不贡献任何 label 0/1)")
    print("[pool] 阴性的证据等级分布:")
    for tier, n in (df.filter(pl.max_horizontal([(pl.col(f"label_{s}") == 0) for s in STAGES]))
                      .group_by("evidence_tier").agg(pl.len().alias("n"))
                      .sort("n", descending=True).iter_rows()):
        print(f"        {tier:<22} {n:>9,}")
    for s in STAGES:
        c = df[f"label_{s}"]
        print(f"  {s:8s} success={int((c==1).sum()):7d} fail={int((c==0).sum()):7d}")
    c = df["label_soluble_expression"]
    print(f"  [派生] soluble_expression success={int((c==1).sum()):7d} "
          f"fail={int((c==0).sum()):7d}  <- 对齐 SoluProt 的口径, 非第七阶段")


if __name__ == "__main__":
    main()
