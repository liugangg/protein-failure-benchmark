"""DS6: OIH 自有管线 (药物发现计算中心 / MDC·ADC 线) 的设计与验证记录。

**这批数据不产生任何 label 0 或 1。** 全部六个阶段标 -1, 作为 unlabeled 探针集入库。

为什么 (SPEC v2 §0 原则 2, 刘刚刚 2026-09-30 确认"不是湿实验, 是我跑的数据"):
  管线里的"失败"是**计算失败或预测分数不达阈值**, 不是蛋白在实验里失败。
  - 任务级失败 238/3,016 条, 实测错误全是 "AF3 fast inference failed (exit N)"、
    "GNINA failed"、"timed out"、"低复杂度序列 MSA 搜不到同源"、"文件路径不存在"
    这一类 —— 是作业崩了, 与蛋白性质无关。
  - MDC 管线 stage 5 的 `ipSAE >= 0.15` 是**预测阈值**。拿它当 bind 标签会构成循环论证:
    本基准在 DS5 上量到 AF3 ipSAE_min 对真实结合的最高 F1 只有 0.57 (per-target),
    用 ipSAE 派生的标签训练, 学到的是"模仿 ipSAE", 不是"预测真实结合"。
    这正是 SPEC §0 原则 2 要防的事, 只是这次假标签由我们自己的管线产生。

那 DS6 有什么用 (三件, 都不需要假标签):
  1. **GATE 4 回顾性验证集** (v2 §GATE 4 要求): 拿历史跑过的设计看模型能否提前标出
     被管线拒掉的那些。注意结论只能说"与管线判断的一致性", 不能说"预测了真实失败"。
  2. **适用域探针 (最有价值的一条)**: 这批是 OIH 真正要预测的对象 —— de novo 迷你 binder,
     中位 94 aa。把它们投到基准的训练分布上, 量 SPEC §5.2 的
     `n_similar_training_examples`, 直接回答"在这个基准上训出来的模型, 对我们自己的
     设计到底适不适用"。这是 SPEC §8 "时代错配"风险的定量版本。
  3. **管线各阶段流失统计**: 计算流水线自身的多阶段流失, 可作为论文里
     "in-silico 类比"报告, 但必须显式标注它不是实验流失。

数据来源 (实查, 2026-09-30):
  <OIH_OUTPUTS_DIR>/*/proteinmpnn/seqs/*.fa      773 个目录, 5,029 条设计 (唯一 4,762)
  <OIH_OUTPUTS_DIR>/**/summary_confidences.json  6,161 个 AF3 置信度文件
  <OIH_TASKS_DIR>/*.json                         3,016 条任务记录
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import statistics
import sys

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from labels.schema import STAGES, UNOBSERVED, validate_label  # noqa: E402

OUTPUTS = paths.get("oih_outputs")
TASKS = paths.get("oih_tasks")
OUT = pathlib.Path("data/interim/ds6_oih.parquet")
STATS = pathlib.Path("data/interim/ds6_oih.stats.json")
AA = set("ACDEFGHIKLMNPQRSTVWY")

MDC_ADC_RE = re.compile(r"(mdc|adc)", re.I)


def collect_designs() -> pl.DataFrame:
    rows: list[dict] = []
    seen: set[str] = set()
    for f in glob.glob(str(OUTPUTS / "*/proteinmpnn/seqs/*.fa")):
        job = pathlib.Path(f).parts[-4]
        try:
            lines = pathlib.Path(f).read_text().split("\n")
        except Exception:
            continue
        hdr = ""
        for ln in lines:
            if ln.startswith(">"):
                hdr = ln[1:].strip()
                continue
            s = ln.strip().upper()
            if not s or not set(s) <= AA:
                continue
            # MPNN 输出的第一条是输入骨架占位行 (常为全 G), 残基种类 <=2 的丢掉
            if len(set(s)) <= 2:
                continue
            if s in seen:
                continue
            seen.add(s)
            rows.append({"job": job, "header": hdr[:200], "sequence": s})
    return pl.DataFrame(rows)


def collect_af3_scores() -> dict:
    """AF3 置信度按 job 汇总。不与序列强行 join —— 文件里没有序列, 硬匹配会错配。"""
    per_job: dict[str, list[float]] = {}
    n_files = 0
    for f in glob.glob(str(OUTPUTS / "**/*summary_confidences.json"), recursive=True):
        n_files += 1
        try:
            d = json.loads(pathlib.Path(f).read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        ci = d.get("chain_iptm")
        val = None
        if isinstance(ci, list) and ci and all(isinstance(x, (int, float)) for x in ci):
            val = float(min(ci))          # 最弱的链间 iptm, 与 ipSAE_min 精神一致
        elif isinstance(d.get("iptm"), (int, float)):
            val = float(d["iptm"])
        if val is None:
            continue
        try:
            job = pathlib.Path(f).relative_to(OUTPUTS).parts[0]
        except Exception:
            job = "?"
        per_job.setdefault(job, []).append(val)
    return {"n_files": n_files, "per_job": per_job}


def task_stats() -> dict:
    st: dict[str, int] = {}
    tools: dict[str, dict[str, int]] = {}
    for f in glob.glob(str(TASKS / "*.json")):
        try:
            d = json.loads(pathlib.Path(f).read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        s = str(d.get("status", "?"))
        st[s] = st.get(s, 0) + 1
        t = str(d.get("tool", "?"))
        tools.setdefault(t, {}).setdefault(s, 0)
        tools[t][s] += 1
    return {"by_status": st, "by_tool_status": tools}


def main() -> None:
    if not OUTPUTS.exists():
        raise SystemExit(f"{OUTPUTS} 不存在")

    des = collect_designs()
    print(f"[ds6] 唯一设计序列 {des.height:,} 条, 来自 {des['job'].n_unique():,} 个输出目录")
    af3 = collect_af3_scores()
    print(f"[ds6] AF3 置信度文件 {af3['n_files']:,} 个, 覆盖 {len(af3['per_job']):,} 个 job")
    ts = task_stats()
    print(f"[ds6] 任务记录 {sum(ts['by_status'].values()):,} 条, status {ts['by_status']}")

    mdc = des.filter(pl.col("job").str.contains(MDC_ADC_RE.pattern))
    print(f"[ds6] 其中 MDC/ADC 线的设计 {mdc.height:,} 条 "
          f"({mdc['job'].n_unique()} 个目录)")

    # 全部六阶段 -1 —— 这是 unlabeled 探针集, 不是标注数据
    label = [UNOBSERVED] * len(STAGES)
    validate_label(label)
    out = des.with_row_index("_r").select(
        (pl.lit("ds6:") + pl.col("job") + pl.lit("#") + pl.col("_r").cast(pl.Utf8))
            .alias("record_id"),
        pl.lit("ds6_oih").alias("source"),
        pl.col("sequence"),
        pl.col("sequence").str.len_chars().cast(pl.Int32).alias("seq_len"),
        pl.col("job").alias("target_id"),
        pl.lit("oih_compute_center").alias("center"),
        pl.lit(2026).cast(pl.Int16).alias("year"),
        pl.lit("de_novo_design").alias("organism"),
        pl.lit(True).alias("is_designed"),
        *[pl.lit(UNOBSERVED, dtype=pl.Int8).alias(f"label_{s}") for s in STAGES],
        (pl.lit("oih_pipeline|job=") + pl.col("job")).alias("evidence"),
        pl.lit("unlabeled_probe").alias("evidence_tier"),
        pl.lit("ds6_in_silico_only").alias("exclusion_reason"),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)

    L = [len(s) for s in des["sequence"].to_list()]
    scores = [v for vs in af3["per_job"].values() for v in vs]
    stats = {
        "unique_designs": des.height,
        "output_dirs": int(des["job"].n_unique()),
        "mdc_adc_designs": mdc.height,
        "af3_confidence_files": af3["n_files"],
        "af3_jobs_with_scores": len(af3["per_job"]),
        "seq_len": {"median": statistics.median(L), "min": min(L), "max": max(L)},
        "af3_chain_iptm_min": ({
            "n": len(scores),
            "median": round(statistics.median(scores), 4),
            "frac_ge_0.15": round(sum(1 for v in scores if v >= 0.15) / len(scores), 4),
            "frac_ge_0.50": round(sum(1 for v in scores if v >= 0.50) / len(scores), 4),
        } if scores else None),
        "tasks": ts["by_status"],
        # 直接回答"DS6 能不能让 stable 或 bind 过一万"
        "can_reach_10000_real_negatives": False,
        "why_not": (
            f"唯一设计序列只有 {des.height:,} 条, 在任何去冗余之前就低于 10,000; "
            "且全部没有真实实验结局 —— 管线里的'失败'是计算失败或预测分数不达阈值。"
            "把 ipSAE>=0.15 当 bind 标签会构成循环论证 (本基准实测 ipSAE 对真实结合的"
            "per-target 最高 F1 仅 0.572)。"
        ),
        "legitimate_uses": [
            "GATE 4 回顾性验证集 (只能说与管线判断的一致性)",
            "适用域探针: 量 n_similar_training_examples, 回答模型对 OIH 自有设计是否适用",
            "管线各阶段流失统计 (须标注为 in-silico 类比)",
        ],
    }
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
