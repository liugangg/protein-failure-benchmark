"""生成 reports/ds6_oih_inventory.md —— DS6 清点 + 一条对 OIH 管线直接可用的校准结论。

两个问题要回答清楚:
  Q1 (刘刚刚 2026-09-30 问): DS6 能不能让 stable 或 bind 的真实阴性过一万?
  Q2 OIH 自己的管线门槛值多少? —— 用 DS5 的真实湿实验结局校准 MDC stage 5 的 ipSAE 门。
"""
from __future__ import annotations

import json
import pathlib

import sys

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

DS6 = pathlib.Path("data/interim/ds6_oih.stats.json")
DS5_CSV = pathlib.Path("data/raw/ds5_binder_negatives/dtu_meta_analysis/final_dataset.csv")
OUT = pathlib.Path("reports/ds6_oih_inventory.md")
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
MDC_GATE = 0.15          # MDC_WORKFLOW.md stage 5: "ipSAE >= 0.15 filter"


def calibrate() -> dict | None:
    if not DS5_CSV.exists():
        return None
    d = pl.read_csv(DS5_CSV, columns=["binder", "af3_ipSAE_min"],
                    infer_schema_length=20000, ignore_errors=True) \
          .filter(pl.col("binder").is_not_null())
    y = d["binder"].cast(pl.Int8).to_numpy()
    s = d["af3_ipSAE_min"].cast(pl.Float64, strict=False).to_numpy()
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    base = float(y.mean())
    rows = []
    for thr in THRESHOLDS:
        m = s >= thr
        if m.sum() == 0:
            continue
        rows.append({
            "threshold": thr,
            "n_pass": int(m.sum()),
            "pass_rate": float(m.mean()),
            "n_true_binder_in_pass": int(y[m].sum()),
            "precision": float(y[m].mean()),
            "recall": float(y[m].sum() / y.sum()),
            "lift": float(y[m].mean() / base),
            "true_binders_lost": int(y[~m].sum()),
            "recall_lost": float(y[~m].sum() / y.sum()),
        })
    return {"n": int(len(y)), "base_rate": base, "rows": rows}


def main() -> None:
    s6 = json.loads(DS6.read_text()) if DS6.exists() else {}
    cal = calibrate()

    L: list[str] = []
    A = L.append
    A("# DS6 (OIH 自有管线) 清点 + 管线门槛校准")
    A("")
    A("生成: `src/eval/ds6_report.py` · 清点脚本 `src/ingest/ds6_oih.py`")
    A("")
    A("数据来源 (2026-09-30 实查):")
    A("")
    A("| 位置 | 内容 |")
    A("|---|---|")
    A(f"| `{paths.shown('oih_outputs')}/*/proteinmpnn/seqs/*.fa` | "
      f"{s6.get('output_dirs', 0):,} 个输出目录, **{s6.get('unique_designs', 0):,} 条唯一设计序列** |")
    A(f"| `{paths.shown('oih_outputs')}/**/summary_confidences.json` | "
      f"{s6.get('af3_confidence_files', 0):,} 个 AF3 置信度文件, 覆盖 {s6.get('af3_jobs_with_scores', 0):,} 个 job |")
    t = s6.get("tasks", {})
    A(f"| `{paths.shown('oih_tasks')}/*.json` | "
      f"{sum(t.values()):,} 条任务记录 (completed {t.get('completed', 0):,} / "
      f"failed {t.get('failed', 0)} / cancelled {t.get('cancelled', 0)} / pending {t.get('pending', 0)}) |")
    sl = s6.get("seq_len", {})
    A("")
    A(f"设计序列长度: 中位 {sl.get('median', '?')} aa (min {sl.get('min', '?')} / "
      f"max {sl.get('max', '?')}) —— 正是 MDC 线的迷你 binder 尺度 (50-100 aa)。")
    A("")
    A("> 更正一处早期误判: 我最初只看了 `data/tasks/tasks.db` (0 字节空文件) 就判定"
      "\"没有任何任务历史\"。实际历史在同目录的 3,016 个 JSON 里。")
    A("")

    # ---- Q1 ----
    A("## Q1 DS6 能不能让 stable 或 bind 的真实阴性过一万?")
    A("")
    A("### 不能。两个层面都不行。")
    A("")
    A(f"**量不够**: 唯一设计序列只有 {s6.get('unique_designs', 0):,} 条, "
      "在任何 30% 去冗余**之前**就已低于 10,000。而且它们高度同源 —— "
      "同一 job 的 MPNN 设计是同一骨架的序列变体, 去冗余后只会剩几百个簇量级。")
    A("")
    A("**性质不对 (更根本)**: 这批是计算产物, 没有真实实验结局。刘刚刚已确认"
      "\"不是湿实验, 是我跑的数据\"。管线里的\"失败\"有两种, 都不能当六阶段的 `0`:")
    A("")
    A(f"1. **任务级失败** ({t.get('failed', 0)}/{sum(t.values()):,} 条)。实测错误全是 "
      "`AF3 fast inference failed (exit N)`、`GNINA failed (exit N)`、"
      "`oih-vina-gpu timed out`、`低复杂度序列 MSA 搜不到同源`、"
      "`Ligand looks like a file path but does not exist` 这一类 —— **作业崩了, 与蛋白性质无关**。"
      "把它当阴性等于教模型识别\"什么输入会让我们的集群崩\"。")
    A("2. **预测分数不达阈值**。MDC 管线 stage 5 的 `ipSAE >= 0.15` 是预测阈值。"
      "拿它当 `bind` 标签会构成**循环论证**: 本基准在 DS5 上实测 AF3 ipSAE_min 对真实结合的"
      "per-target 最高 F1 只有 0.572, 用 ipSAE 派生标签训练, 学到的是\"模仿 ipSAE\", "
      "不是\"预测真实结合\"。这正是 SPEC §0 原则 2 禁止的假阴性, 只是这次由我们自己的管线产生。")
    A("")
    A("**结论: 多任务方案不能靠 DS6 重启。** 六阶段联合建模所需的真实阴性, "
      "公开数据与自有管线加起来都不存在。")
    A("")
    A("### DS6 的三个正当用途 (都不需要假标签)")
    A("")
    A("已按 unlabeled 探针集入库 (`data/interim/ds6_oih.parquet`, 六阶段全 -1, "
      "`exclusion_reason = ds6_in_silico_only`):")
    A("")
    A("1. **GATE 4 回顾性验证集** (v2 §GATE 4 要求)。拿历史跑过的设计看模型能否提前标出"
      "被管线拒掉的那些。**措辞必须准确**: 只能说\"与管线判断的一致性\", "
      "不能说\"预测了真实失败\"。")
    A("2. **适用域探针 —— 最有价值的一条**。这批是 OIH 真正要预测的对象。"
      "把它们投到基准的训练分布上量 SPEC §5.2 的 `n_similar_training_examples`, "
      "直接回答\"在这个基准上训出来的模型, 对我们自己的设计到底适不适用\"。"
      "这是 SPEC §8「时代错配」风险的定量版本, 也是论文 Limitations 里最有说服力的一段。")
    A("3. **管线各阶段流失统计**, 作为 in-silico 类比报告, 须显式标注不是实验流失。")
    A("")

    # ---- Q2 ----
    if cal:
        A("## Q2 OIH 管线的 `ipSAE >= 0.15` 门值多少?")
        A("")
        A(f"用 DS5 的 **{cal['n']:,} 个带真实湿实验结局的设计**校准 "
          f"(实际结合率 {cal['base_rate']*100:.1f}%)。这是 DS6 与 DS5 合起来唯一能产出的"
          "直接可用结论, 也是本基准对 OIH 管线的即时价值。")
        A("")
        A("| ipSAE 阈值 | 放过数 | 放过率 | 其中真结合 | **precision** | recall | lift |")
        A("|---|---|---|---|---|---|---|")
        for r in cal["rows"]:
            mark = " ← **当前门**" if abs(r["threshold"] - MDC_GATE) < 1e-9 else ""
            A(f"| {r['threshold']:.2f}{mark} | {r['n_pass']:,} | {r['pass_rate']*100:.1f}% | "
              f"{r['n_true_binder_in_pass']:,} | **{r['precision']:.3f}** | "
              f"{r['recall']:.3f} | {r['lift']:.2f} |")
        A("")
        g = next(r for r in cal["rows"] if abs(r["threshold"] - MDC_GATE) < 1e-9)
        A(f"**当前门 (0.15) 的实际含义**: 放过 {g['pass_rate']*100:.1f}% 的设计, "
          f"其中只有 **{g['precision']*100:.1f}%** 真的结合 —— "
          f"相对不设门的 {cal['base_rate']*100:.1f}% 只有 **{g['lift']:.2f} 倍**增益。"
          f"同时它挡掉的那部分里有 {g['true_binders_lost']} 个真结合的, "
          f"即**漏掉 {g['recall_lost']*100:.1f}% 的真阳性**。")
        A("")
        hi = next(r for r in cal["rows"] if abs(r["threshold"] - 0.60) < 1e-9)
        mid = next(r for r in cal["rows"] if abs(r["threshold"] - 0.50) < 1e-9)
        A(f"**可操作的建议**: 阈值提高时 precision 单调上升 —— "
          f"0.50 给 {mid['precision']*100:.1f}% ({mid['lift']:.2f} 倍, recall {mid['recall']:.3f}), "
          f"0.60 给 {hi['precision']*100:.1f}% ({hi['lift']:.2f} 倍, recall {hi['recall']:.3f})。"
          "如果瓶颈是湿实验预算 (每轮只能做几十个), 把门提到 0.50-0.60 能让命中率接近翻倍; "
          "如果瓶颈是不想漏掉好设计, 当前的 0.15 是对的。**这个取舍取决于哪一边更贵, "
          "不是一个纯技术问题**, 需要刘刚刚定。")
        A("")
        A("⚠️ 两条使用限制:")
        A("")
        A("1. 校准用的是 DS5 的 15 个靶点 / 20 个研究, 与 OIH 的靶点不完全重叠。"
          "跨靶点的 ipSAE 判别力差异极大 (DS5 上逐靶点最高 F1 从 0.222 到 1.000, "
          "见 `reports/gate2_baselines.md`), 所以这张表是**总体期望**, "
          "不保证在某个特定新靶点上成立。")
        A("2. DS6 自己的分数文件记的是 `chain_iptm`, 与 DS5 的 `af3_ipSAE_min` 不是同一个量, "
          f"所以本表**不用** DS6 的分数, 只用 DS5 校准阈值本身。"
          f"(DS6 侧 {s6.get('af3_chain_iptm_min', {}).get('n', 0):,} 个 chain_iptm 读数的"
          f"中位是 {s6.get('af3_chain_iptm_min', {}).get('median', '?')}, "
          f"仅作规模参考。)")
        A("")

    OUT.write_text("\n".join(L), encoding="utf8")
    print(f"wrote {OUT} ({len(L)} 行)")


if __name__ == "__main__":
    main()
