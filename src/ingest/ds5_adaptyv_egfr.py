"""DS5 (部分): Adaptyv Bio EGFR 设计竞赛 round 1 / round 2 湿实验结果 -> 统一记录。

为什么用这个源: SPEC §2.2 指的 bioRxiv 2025.08.14.670059 (Overath et al., 3,330 个
非结合设计) 正文被 Cloudflare 挡住, 补充数据拿不到; 而 Adaptyv 自己把两轮竞赛的
BLI 实测结果开源在 GitHub 上, 是同一类东西 —— **de novo 设计蛋白的真实湿实验阴性**,
而且同时给了 expression 和 binding 两级, 正好落在我们的 express / bind 两个头上。
它还顺手解掉 SPEC §8 的"时代错配"风险: TargetTrack 全是天然蛋白, 这批全是设计蛋白。

许可 (必须知道): 数据 ODbL, 代码 Apache-2.0。ODbL 带 share-alike ——
衍生数据库对外发布时有同样的开放义务。商用要另找 Adaptyv 谈。见 reports/ 里的许可登记。

标签映射依据 (全部出自两个仓库 README 的 Experimental Workflow 段, 不是猜的):
  - "Suitable gene constructs were successfully generated for all submitted protein
     sequences." -> clone = 1 (全部)
  - 无细胞表达, "Sequences that yielded less than 0.02 µg/mL of protein were excluded
     from further experimental characterization." -> expression == "none" 即表达失败
     -> express = 0, 之后全部 -1 (它们的 binding 栏确实都是 unknown, 已核对)
  - expression in {low, medium, high} -> express = 1
  - binding == "false" -> bind = 0 ; "true" -> bind = 1 ; "unknown" -> -1
  - soluble / purify / stable 一律 -1: BLI 用 tag 化学把配体固定在探针上,
    没有独立的可溶性和纯化步骤, 也没测折叠稳定性。**不许因为"能测到结合就说明可溶"
    去回填 1** —— 那是推断不是观测。

多次重复的处理 (SPEC §8 "假阴性" 那一条):
  竞赛对每条序列做了 2-4 个表达批次、结合做 duplicate。只有**全部重复都失败**才记 0;
  任一重复成功就记 1。换条件能成的不算确定阴性。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from labels.schema import (  # noqa: E402
    OBSERVED_FAIL,
    OBSERVED_SUCCESS,
    RECORD_COLUMNS,
    STAGES,
    UNOBSERVED,
    validate_label,
)

RAW = pathlib.Path("data/raw/ds5_binder_negatives")
OUT = pathlib.Path("data/interim")

EXPRESSED = {"low", "medium", "high"}


def _agg_replicates(rep: pl.DataFrame, has_strength: bool) -> pl.DataFrame:
    """按 name 聚合重复实验: 任一重复成功即成功。"""
    exprs = [
        pl.len().alias("n_replicates"),
        # 任一重复表达出蛋白 -> 表达成功
        pl.col("expression").str.to_lowercase().is_in(list(EXPRESSED)).any().alias("any_expressed"),
        pl.col("expression").str.to_lowercase().alias("expression_values").unique().sort().alias("expression_values"),
        # 任一重复结合 -> 结合成功
        pl.col("binding").cast(pl.Utf8).str.to_lowercase().eq("true").any().alias("any_binding"),
        pl.col("binding").cast(pl.Utf8).str.to_lowercase().eq("false").any().alias("any_nonbinding"),
        pl.col("binding").cast(pl.Utf8).str.to_lowercase().alias("binding_values").unique().sort().alias("binding_values"),
        pl.col("kd").min().alias("kd_min"),
    ]
    return rep.group_by("name").agg(exprs)


def _label_row(any_expressed: bool, expr_values: list[str], any_binding: bool,
               any_nonbinding: bool) -> tuple[list[int], str]:
    label = [UNOBSERVED] * len(STAGES)
    # clone: README 明确说所有提交序列都成功造出构建体
    label[STAGES.index("clone")] = OBSERVED_SUCCESS

    if any_expressed:
        label[STAGES.index("express")] = OBSERVED_SUCCESS
    else:
        # 所有重复都 none -> 真实表达阴性, 后续阶段没做 -> 保持 -1
        label[STAGES.index("express")] = OBSERVED_FAIL
        validate_label(label)
        return label, f"expression={'/'.join(expr_values)}; binding not attempted"

    if any_binding:
        label[STAGES.index("bind")] = OBSERVED_SUCCESS
        b = "bound"
    elif any_nonbinding:
        label[STAGES.index("bind")] = OBSERVED_FAIL
        b = "no binding in any replicate"
    else:
        b = "binding unknown"
    validate_label(label)
    return label, f"expression={'/'.join(expr_values)}; {b}"


def ingest_round(round_dir: pathlib.Path, round_tag: str, year: int) -> pl.DataFrame:
    res = pl.read_csv(round_dir / "result_summary.csv", infer_schema_length=10000)
    rep = pl.read_csv(round_dir / "replicate_summary.csv", infer_schema_length=10000)

    if "binding" not in rep.columns or "expression" not in rep.columns:
        raise RuntimeError(
            f"{round_dir}/replicate_summary.csv 缺 binding/expression 列, "
            "上游格式变了, 停下来核对 README 而不要改这里的默认值"
        )

    agg = _agg_replicates(rep, has_strength="binding_strength" in rep.columns)
    df = res.join(agg, on="name", how="left")

    rows = []
    for r in df.iter_rows(named=True):
        seq = (r.get("sequence") or "").strip().upper()
        if not seq:
            continue  # 参考抗体等没有序列的行直接丢, 不编造
        if r["n_replicates"] is None:
            continue  # 没有重复记录 -> 没有观测, 不产标签
        label, evidence = _label_row(
            bool(r["any_expressed"]),
            list(r["expression_values"] or []),
            bool(r["any_binding"]),
            bool(r["any_nonbinding"]),
        )
        rec = {
            "record_id": f"ds5_adaptyv_{round_tag}:{r['name']}",
            "source": "ds5_adaptyv_egfr",
            "sequence": seq,
            "seq_len": len(seq),
            "target_id": "EGFR",
            "center": "adaptyv_bio",          # 单一实验室 —— GATE 1 的实验室集中度要算进去
            "year": year,
            "organism": "de_novo_design",
            "is_designed": True,
            **{f"label_{s}": v for s, v in zip(STAGES, label)},
            "evidence": evidence,
        }
        rows.append(rec)

    out = pl.DataFrame(rows, schema={k: getattr(pl, v.capitalize(), None) or pl.Utf8 for k, v in []} or None)
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "ds5_adaptyv_egfr.parquet"))
    args = ap.parse_args()

    parts = []
    # 年份取自竞赛举办时间: round 1 = 2024, round 2 = 2025 (Adaptyv 官网/仓库 README)
    for tag, year in (("r1", 2024), ("r2", 2025)):
        d = RAW / f"adaptyv_egfr_{tag}"
        if not d.exists():
            print(f"[skip] {d} 不存在")
            continue
        part = ingest_round(d, tag, year)
        print(f"[{tag}] {part.height} records")
        parts.append(part)

    if not parts:
        raise SystemExit("没有任何输入, 先跑下载")

    df = pl.concat(parts, how="vertical_relaxed")
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(outp)

    stats = {}
    for s in STAGES:
        c = df[f"label_{s}"]
        stats[s] = {
            "success": int((c == OBSERVED_SUCCESS).sum()),
            "fail": int((c == OBSERVED_FAIL).sum()),
            "unobserved": int((c == UNOBSERVED).sum()),
        }
    print(json.dumps({"total": df.height, "stages": stats}, indent=2, ensure_ascii=False))
    (OUT / "ds5_adaptyv_egfr.stats.json").write_text(
        json.dumps({"total": df.height, "stages": stats}, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
