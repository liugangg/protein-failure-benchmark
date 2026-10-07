"""DS5: DTU binder 设计元分析 (Overath et al., bioRxiv 2025.08.14.670059) -> bind 阶段。

SPEC v2 §2.2 指定的获取方式 (已按原文照抄, 不自行改动):
  Zenodo 10.5281/zenodo.15722219, CC BY 4.0, **只下 final_dataset.csv (82 MB)**。
  9 GB 的结构预测输出 (AF3 / Boltz-1 / ColabFold) 先不下, 等确实需要结构特征时再取。

实测内容 (与论文标题声称的 3,766 个设计略有差异, 报实测值):
  3,676 行 / 312 列; `binder` 布尔列即湿实验结合结果:
    false 3,275 (真实 bind 阴性) / true 394 / null 7
  16 个 target_id (含 1 个 null), 21 个 source (研究来源)。
  `binder_chain` 全部为 "A", 所以 **A_seq 是 binder 序列**, B_seq 等是靶点链。

标签映射:
  bind = 0 / 1 / -1 直接来自 `binder` 列 (湿实验测得), 其余五个阶段一律 -1。
  不把"能测到结合"回填成"可溶/纯化成功" —— 那是推断不是观测。
  clone 也不回填: 本数据集没有构建成功与否的记录 (与 Adaptyv 仓库 README 不同)。

center 字段用 `source` (研究来源) 而非单一机构:
  这批数据横跨 21 个研究 (Cao et al. 各靶点、Adaptyv 两轮竞赛、Watson et al. 等),
  它们才是真正的"批次/实验室"单位, 也是实验室留出切分能用上的分组。
  这一点比旧的 Adaptyv 替代方案强很多: 后者只有 1 个靶点 1 个来源。

与 ds5_adaptyv_egfr 的重叠:
  本数据集已包含 "Adaptyv binder comp round 1/2" 共 302 行。
  去重规则: 以本数据集为 bind 的权威来源; Adaptyv ingest 中 binder 序列
  在本数据集出现过的行, 其 bind 标签降为 -1 (见 src/labels/pool.py 的 dedup_bind)。
  Adaptyv 仍保留其独有的 clone / express 观测 (本数据集没有表达量记录)。
"""
from __future__ import annotations

import json
import pathlib
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

SRC = pathlib.Path("data/raw/ds5_binder_negatives/dtu_meta_analysis/final_dataset.csv")
OUT = pathlib.Path("data/interim/ds5_dtu_binder.parquet")
AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} 不存在; 按 SPEC v2 §2.2 从 zenodo 15722219 下 final_dataset.csv")

    df = pl.read_csv(SRC, infer_schema_length=20000, ignore_errors=True,
                     columns=["binder_id", "target_id", "source", "binder",
                              "binder_chain", "A_seq", "A_length"])
    print(f"[ds5_dtu] 读入 {df.height:,} 行")

    # binder_chain 必须是 A, 否则 A_seq 不是 binder —— 硬失败而不是默默错位
    bad_chain = df.filter(pl.col("binder_chain").is_not_null()
                          & (pl.col("binder_chain") != "A")).height
    if bad_chain:
        raise SystemExit(f"有 {bad_chain} 行 binder_chain != 'A', A_seq 不一定是 binder, 停下来核对")

    before = df.height
    df = df.filter(pl.col("A_seq").is_not_null() & pl.col("binder").is_not_null())
    print(f"[ds5_dtu] 丢弃 A_seq 或 binder 为空的行: {before - df.height}")

    # 序列字符校验: 出现非标准字符就报出来, 不静默带进训练集
    df = df.with_columns(pl.col("A_seq").str.to_uppercase().alias("seq"))
    ok = df.filter(pl.col("seq").str.contains(r"^[ACDEFGHIKLMNPQRSTVWYXBZUO]+$"))
    if ok.height != df.height:
        print(f"[ds5_dtu] 警告: {df.height - ok.height} 行序列含非标准残基, 已丢弃")
    df = ok

    bind_label = (
        pl.when(pl.col("binder")).then(OBSERVED_SUCCESS)
        .otherwise(OBSERVED_FAIL).cast(pl.Int8)
    )
    out = df.select(
        (pl.lit("ds5dtu:") + pl.col("binder_id")).alias("record_id"),
        pl.lit("ds5_dtu_binder").alias("source"),
        pl.col("seq").alias("sequence"),
        pl.col("seq").str.len_chars().cast(pl.Int32).alias("seq_len"),
        pl.col("target_id").fill_null("unknown").alias("target_id"),
        # center = 研究来源, 这是真正的批次单位
        ("dtu_" + pl.col("source").fill_null("unknown")
         .str.replace_all(r"[^A-Za-z0-9]+", "_")).alias("center"),
        pl.lit(2025).cast(pl.Int16).alias("year"),
        pl.lit("de_novo_design").alias("organism"),
        pl.lit(True).alias("is_designed"),
        *[(bind_label if s == "bind" else pl.lit(UNOBSERVED, dtype=pl.Int8)).alias(f"label_{s}")
          for s in STAGES],
        (pl.lit("binder=") + pl.col("binder").cast(pl.Utf8)
         + pl.lit("|target=") + pl.col("target_id").fill_null("?")
         + pl.lit("|study=") + pl.col("source").fill_null("?")).alias("evidence"),
    )

    for r in out.sample(min(3000, out.height), seed=0).iter_rows(named=True):
        validate_label([r[f"label_{s}"] for s in STAGES])

    # record_id 唯一性 —— DS2 上踩过 name 重复的坑, 这里先验
    if out["record_id"].n_unique() != out.height:
        dup = out.height - out["record_id"].n_unique()
        print(f"[ds5_dtu] binder_id 有 {dup} 个重复, 追加行号保证唯一")
        out = out.with_row_index("_r").with_columns(
            (pl.col("record_id") + pl.lit("#") + pl.col("_r").cast(pl.Utf8)).alias("record_id")
        ).drop("_r")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)

    c = out["label_bind"]
    stats = {
        "total": out.height,
        "bind": {"success": int((c == 1).sum()), "fail": int((c == 0).sum()),
                 "unobserved": int((c == -1).sum())},
        "distinct_targets": int(out["target_id"].n_unique()),
        "distinct_studies": int(out["center"].n_unique()),
        "distinct_sequences": int(out["sequence"].n_unique()),
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    pathlib.Path("data/interim/ds5_dtu_binder.stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )
    print("\n各研究来源的结合失败率:")
    print(out.group_by("center").agg(
        pl.len().alias("n"), (pl.col("label_bind") == 0).sum().alias("fail")
    ).with_columns((pl.col("fail") / pl.col("n") * 100).round(1).alias("fail_%"))
     .sort("n", descending=True).head(12))


if __name__ == "__main__":
    main()
