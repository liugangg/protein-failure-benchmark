"""DS3: ProteinGym v1.3 DMS 替换基准 -> 统一记录。

两条硬规矩, 都是读了官方 reference 文件之后定的, 不是照抄 SPEC 的假设:

规矩 1 —— **只收 DMS_binarization_method == "manual" 的 assay。**
  ProteinGym 的 DMS_score_bin 有两种来源: "median" (126/217 个 assay) 就是
  拿该 assay 的**中位数一刀切**, 天生有 50% 的变体被标成 0。那不是"实验观测到失败",
  是排序位置。把它当 label 0 用, 等于 SPEC §0 原则 2 禁止的合成假阴性 ——
  只是这次假阴性是别人替我们造的。"manual" 是作者按实验语义定的阈值, 可以信。
  实测可映射的 97 个 assay 里 manual 只有 33 个 (stable 21 / express 8 / bind 4)。

规矩 2 —— **只收 coarse_selection_type 能映射到六阶段的 assay。**
  Activity (43 个) 和 OrganismalFitness (77 个) 不对应六阶段任何一步, 直接丢,
  不硬凑。映射表见 configs/data_sources.yaml。

必须一起报出去的两个结构性问题 (见 reports/):
  (a) DMS 是同一个亲本蛋白的点突变。行数上百万, 但**不同蛋白**只有几十到两百个。
      按 SPEC §2.4 第 4 项的 30% 簇口径, DS3 贡献的有效样本量是几十级, 不是十万级。
  (b) DS3 的 "Binding" 测的是"突变让已有结合变弱多少", 和我们 bind 头要答的
      "这个从头设计的 binder 会不会结合"不是同一个问题。合进同一个头之前要想清楚。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import polars as pl
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from labels.schema import (  # noqa: E402
    OBSERVED_FAIL,
    OBSERVED_SUCCESS,
    STAGES,
    UNOBSERVED,
    validate_label,
)

RAW = pathlib.Path("data/raw/ds3_proteingym")
ASSAY_DIR = pathlib.Path("data/interim/extracted/DMS_ProteinGym_substitutions")
REF = RAW / "DMS_substitutions_reference.csv"
OUT = pathlib.Path("data/interim/ds3_proteingym.parquet")
CONFIG = pathlib.Path("configs/data_sources.yaml")

TAXON_TO_ORGANISM_FALLBACK = "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-median", action="store_true",
                    help="也收 median 二值化的 assay。默认关闭 —— 打开等于引入假阴性, "
                         "只在明确知道自己在干什么时用, 并且要在报告里写明。")
    ap.add_argument("--max-rows-per-assay", type=int, default=0,
                    help="每个 assay 最多取多少行 (0 = 全取)。用于控制 bind 头被 4 个 "
                         "assay 的 54 万点突变淹掉。")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    stage_map = {
        k: v for k, v in cfg["ds3_proteingym"]["stage_mapping_by_coarse_selection_type"].items()
        if v != "DROP"
    }
    print(f"[ds3] 阶段映射: {stage_map}")

    ref = pl.read_csv(REF, infer_schema_length=10000)
    ref = ref.with_columns(
        pl.col("coarse_selection_type").replace_strict(stage_map, default=None).alias("stage")
    )
    sel = ref.filter(pl.col("stage").is_not_null())
    print(f"[ds3] 可映射 assay {sel.height} / {ref.height}")
    if not args.allow_median:
        before = sel.height
        sel = sel.filter(pl.col("DMS_binarization_method") == "manual")
        print(f"[ds3] 只保留 manual 二值化: {sel.height} / {before} "
              f"(丢掉 {before - sel.height} 个中位数劈开的 assay)")

    frames = []
    skipped = []
    for row in sel.iter_rows(named=True):
        f = ASSAY_DIR / row["DMS_filename"]
        if not f.exists():
            skipped.append(row["DMS_filename"])
            continue
        df = pl.read_csv(f, columns=["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"])
        if args.max_rows_per_assay and df.height > args.max_rows_per_assay:
            df = df.sample(args.max_rows_per_assay, seed=0)
        stage = row["stage"]
        label_cols = []
        for s in STAGES:
            if s == stage:
                # bin: 1 = fit, 0 = not fit (ProteinGym README 原话)
                label_cols.append(
                    pl.when(pl.col("DMS_score_bin") == 1).then(OBSERVED_SUCCESS)
                    .when(pl.col("DMS_score_bin") == 0).then(OBSERVED_FAIL)
                    .otherwise(UNOBSERVED).cast(pl.Int8).alias(f"label_{s}")
                )
            else:
                label_cols.append(pl.lit(UNOBSERVED, dtype=pl.Int8).alias(f"label_{s}"))

        frames.append(
            df.select(
                (pl.lit(f"ds3:{row['DMS_id']}:") + pl.col("mutant")).alias("record_id"),
                pl.lit("ds3_proteingym").alias("source"),
                pl.col("mutated_sequence").str.to_uppercase().alias("sequence"),
                pl.col("mutated_sequence").str.len_chars().cast(pl.Int32).alias("seq_len"),
                pl.lit(str(row["UniProt_ID"])).alias("target_id"),
                # center: DMS 没有"实验室"字段, 用 第一作者_年份 当代理 ——
                # 它确实就是"哪个课题组哪一年做的", 正好是实验室留出切分要的东西
                pl.lit(f"pg_{row['first_author']}_{row['year']}").alias("center"),
                pl.lit(int(row["year"]) if row["year"] else -1).cast(pl.Int16).alias("year"),
                pl.lit(str(row["source_organism"] or TAXON_TO_ORGANISM_FALLBACK)).alias("organism"),
                pl.lit(False).alias("is_designed"),
                *label_cols,
                (pl.lit(f"{row['selection_assay']}|bin={row['DMS_binarization_method']}|score=")
                 + pl.col("DMS_score").cast(pl.Utf8)).alias("evidence"),
                # 去冗余的正确单位是**亲本蛋白**, 不是每个点突变体。
                # 同一个 assay 的几十万个变体彼此只差 1-2 个残基, 逐条丢进 30% 聚类
                # 既无意义 (必然同簇) 又会把 all-vs-all 比对炸掉 —— 实测把 mmseqs
                # 卡在 align 阶段起不来 (SPG1_STRSG_Olson_2014 一个 assay 53.7 万条)。
                # 所以带上 reference 文件里的野生型 target_seq 作为聚类键。
                pl.lit(str(row["target_seq"]).upper()).alias("cluster_key_seq"),
            )
        )
        print(f"  {row['DMS_id']:50s} stage={stage:8s} rows={df.height:7d}")

    if skipped:
        raise SystemExit(f"缺 assay 文件 {len(skipped)} 个: {skipped[:5]} —— 先确认解压完整")
    if not frames:
        raise SystemExit("没有任何 assay 入选")

    out = pl.concat(frames, how="vertical_relaxed")
    # 结构校验: 抽样过一遍 validate_label, 全量过太慢但必须验
    samp = out.sample(min(5000, out.height), seed=0)
    for r in samp.iter_rows(named=True):
        validate_label([r[f"label_{s}"] for s in STAGES])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)

    stats = {"total": out.height, "assays": sel.height, "stages": {}}
    for s in STAGES:
        c = out[f"label_{s}"]
        stats["stages"][s] = {
            "success": int((c == 1).sum()),
            "fail": int((c == 0).sum()),
            "unobserved": int((c == -1).sum()),
        }
    # 蛋白层面的多样性 —— 这才是 30% 簇口径下真正贡献的量级
    stats["distinct_uniprot"] = int(out["target_id"].n_unique())
    stats["distinct_centers"] = int(out["center"].n_unique())
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    pathlib.Path("data/interim/ds3_proteingym.stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
