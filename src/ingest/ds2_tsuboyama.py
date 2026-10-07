"""DS2: Tsuboyama 2023 折叠稳定性 (Nature) -> stable 阶段标签。

用哪个文件, 为什么:
  Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv (776,298 行)
  它带 aa_seq / mut_type / WT_name / WT_cluster / deltaG 及 95% 置信区间上下界。
  Dataset1 (1,841,285 行) 只有 dna_seq 没有 aa_seq, 拿来还要自己翻译, 不用。

标签规则 (只用测量本身的不确定度定, 不自己拍阈值):
  deltaG 是折叠自由能 (kcal/mol), 正值 = 折叠稳定。
  - 95% 置信区间**上界 < 0**  -> 有把握地没折叠 -> stable = 0  (真实测量到的阴性)
  - 95% 置信区间**下界 > 0**  -> 有把握地折叠了 -> stable = 1
  - 区间跨 0                  -> 测不准 -> stable = -1 (未观测, 不当阴性)
  实测三档: 106,887 / 651,415 / 17,996。
  用区间而不是 "deltaG < 0" 这种点估计比较, 是因为近 0 的那一批本来就分不出来,
  硬判会把噪声写成阴性 —— 正是 SPEC §0 原则 4 要防的事。

  dG_ML 列里有 '<-1' / '>5' / '-' 这种越界截尾值 (100,831 条非数值), 本脚本不用它,
  只用 deltaG 及其区间 (全为数值, 无缺失)。

其它阶段一律 -1: 蛋白酶解稳定性实验只测折叠稳定性, 不涉及克隆/表达/纯化/结合。

center 字段: 全库同一个实验室 (Rocklin lab / Tsuboyama et al.), 记作 tsuboyama_2023。
  它在 GATE 1 的实验室集中度里就是一个单点 —— 这是事实, 不要拆成假的多实验室。

去冗余键 (cluster_key_seq): 该变体所属亲本的野生型序列。
  479 个 WT_name / 133 个 WT_cluster —— 77 万行数据在蛋白层面只有几百个不同蛋白,
  这是 DS2 的根本局限, 必须让 GATE 1 的簇口径如实反映出来。
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from labels.schema import (  # noqa: E402
    OBSERVED_FAIL,
    OBSERVED_SUCCESS,
    STAGES,
    UNOBSERVED,
    validate_label,
)

# SPEC v2 §2.1 点名 Zenodo 10.5281/zenodo.7844779。
# 踩过的坑 (2026-09-30): 先前按关键词搜到的是同一版本的另一个 Zenodo 记录 (7992926),
# 它的 Tsuboyama2023_Dataset2_Dataset3 只有 697,658,024 字节, 而 v2 指定的 7844779
# 是 718,214_782 字节 —— 行数相同 (776,298) 但**少两列** match_aaseq / name_original。
# 两个 zip 的其余 7 个成员 CRC 完全一致。论文的 data availability 必须写 v2 指定的 DOI,
# 所以这里只认 7844779 解出来的路径。
SRC = pathlib.Path(
    "data/interim/extracted/ds2_7844779/"
    "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
)
ZENODO_RECORD = "10.5281/zenodo.7844779"
ZIP_MD5 = "27a0936c8f80e5b18ed330ed6b98a3a6"   # Processed_K50_dG_datasets.zip, 实测与 Zenodo 一致
OUT = pathlib.Path("data/interim/ds2_tsuboyama.parquet")

# de novo 设计蛋白的命名家族 (Rocklin/Baker 系的拓扑名与 run 编号)。
# 注意**设计模式优先于 PDB 模式**: EEHEE_rd3_0657.pdb 尾部的 "0657.pdb" 会被
# PDB 正则误命中, 实测 479 个 WT_name 里有 128 个这样两头都像。
DESIGNED_RE = re.compile(r"(EHEE|HEEH|HHH|EEHH|EEHEE|_rd\d|\|run\d|TrROS|rd\d_\d)", re.I)
PDB_RE = re.compile(r"(^|[|_])[0-9][A-Za-z0-9]{3}\.pdb")


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} 不存在, 先解压 Processed_K50_dG_datasets.zip")

    df = pl.read_csv(
        SRC,
        columns=["name", "aa_seq", "mut_type", "WT_name", "WT_cluster",
                 "deltaG", "deltaG_95CI_high", "deltaG_95CI_low",
                 "match_aaseq"],
        infer_schema_length=200_000,
    )
    print(f"[ds2] 读入 {df.height:,} 行  (Zenodo {ZENODO_RECORD})")

    # match_aaseq 是作者的序列一致性校验标志。**注意原始数据把 True 拼成了 "Ture"**
    # (566,178 行 Ture / 210,118 行 True / 2 行 False), 两种拼法都是通过。
    # 只有 2 行 False, 但序列对不上的行不该进训练集 —— 剔除并记数, 不静默带过。
    if "match_aaseq" in df.columns:
        vals = df["match_aaseq"].unique().to_list()
        unknown = [v for v in vals if v not in ("True", "Ture", "False", None)]
        if unknown:
            raise SystemExit(f"match_aaseq 出现未知取值 {unknown}, 停下来核对上游")
        bad = df.filter(pl.col("match_aaseq") == "False").height
        df = df.filter(pl.col("match_aaseq") != "False")
        print(f"[ds2] match_aaseq=False 剔除 {bad} 行 (序列与记录不一致)")
    for c in ["deltaG", "deltaG_95CI_high", "deltaG_95CI_low"]:
        if df[c].dtype != pl.Float64:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))
        n_null = int(df[c].null_count())
        if n_null:
            print(f"[ds2] 警告: {c} 有 {n_null:,} 个空值, 这些行的 stable 会记 -1")

    # 亲本野生型序列: 取该 WT_name 下 mut_type == 'wt' 的 aa_seq;
    # 没有 wt 行的 (实测存在) 退而用该 WT_name 下最短的 aa_seq。
    wt = (
        df.filter(pl.col("mut_type") == "wt")
        .group_by("WT_name").agg(pl.col("aa_seq").first().alias("wt_seq"))
    )
    fallback = (
        df.sort("aa_seq").group_by("WT_name")
        .agg(pl.col("aa_seq").first().alias("fb_seq"))
    )
    df = df.join(wt, on="WT_name", how="left").join(fallback, on="WT_name", how="left")
    n_fb = int(df["wt_seq"].is_null().sum())
    if n_fb:
        print(f"[ds2] {n_fb:,} 行的 WT_name 没有 wt 记录, 用同亲本最短序列当聚类键")
    df = df.with_columns(
        pl.coalesce([pl.col("wt_seq"), pl.col("fb_seq"), pl.col("aa_seq")]).alias("cluster_key_seq")
    )

    designed = pl.col("WT_name").str.contains(DESIGNED_RE.pattern)
    stable_label = (
        pl.when(pl.col("deltaG_95CI_high") < 0).then(OBSERVED_FAIL)
        .when(pl.col("deltaG_95CI_low") > 0).then(OBSERVED_SUCCESS)
        .otherwise(UNOBSERVED)
        .cast(pl.Int8)
    )

    # DS2 的 name 列**不唯一**: 实测 275 个 name 各出现 2 次, 是同一变体的独立重复测量
    # (deltaG 不同)。两条都要保留 (它们是真实的重复), 但 record_id 必须唯一,
    # 否则下游按 record_id join 切分标签时一条记录会被复制成两条, 簇完整性断言失效。
    df = df.with_row_index("_row")
    out = df.select(
        (pl.lit("ds2:") + pl.col("name") + pl.lit("#") + pl.col("_row").cast(pl.Utf8))
            .alias("record_id"),
        pl.lit("ds2_tsuboyama").alias("source"),
        pl.col("aa_seq").str.to_uppercase().alias("sequence"),
        pl.col("aa_seq").str.len_chars().cast(pl.Int32).alias("seq_len"),
        pl.col("WT_name").alias("target_id"),
        pl.lit("tsuboyama_2023").alias("center"),
        pl.lit(2023).cast(pl.Int16).alias("year"),
        pl.when(designed).then(pl.lit("de_novo_design"))
          .otherwise(pl.lit("natural_domain")).alias("organism"),
        designed.alias("is_designed"),
        *[
            (stable_label if s == "stable" else pl.lit(UNOBSERVED, dtype=pl.Int8)).alias(f"label_{s}")
            for s in STAGES
        ],
        (pl.lit("deltaG=") + pl.col("deltaG").round(3).cast(pl.Utf8)
         + pl.lit(" 95CI=[") + pl.col("deltaG_95CI_low").round(3).cast(pl.Utf8)
         + pl.lit(",") + pl.col("deltaG_95CI_high").round(3).cast(pl.Utf8)
         + pl.lit("] mut=") + pl.col("mut_type")).alias("evidence"),
        pl.col("cluster_key_seq").str.to_uppercase().alias("cluster_key_seq"),
    )

    samp = out.sample(min(5000, out.height), seed=0)
    for r in samp.iter_rows(named=True):
        validate_label([r[f"label_{s}"] for s in STAGES])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)

    c = out["label_stable"]
    stats = {
        "zenodo_record": ZENODO_RECORD,
        "zip_md5": ZIP_MD5,
        "total": out.height,
        "stable": {"success": int((c == 1).sum()), "fail": int((c == 0).sum()),
                   "unobserved": int((c == -1).sum())},
        "distinct_WT_name": int(out["target_id"].n_unique()),
        "distinct_cluster_key_seq": int(out["cluster_key_seq"].n_unique()),
        "designed_records": int(out["is_designed"].sum()),
        "designed_WT_names": int(out.filter(pl.col("is_designed"))["target_id"].n_unique()),
        "negatives_distinct_parents": int(
            out.filter(pl.col("label_stable") == 0)["target_id"].n_unique()
        ),
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    pathlib.Path("data/interim/ds2_tsuboyama.stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
