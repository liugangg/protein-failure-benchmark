"""生成 DATA_AVAILABILITY.md —— 开源与数据声明, 计数从产物注入。

为什么要生成而不是手写: 各来源的条数/簇数是论文里的数字, 手写会和 records.parquet 脱节。
"""
from __future__ import annotations

import json
import pathlib
import sys

import polars as pl
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = pathlib.Path("DATA_AVAILABILITY.md")

# 上游来源的许可事实 (来自 configs/data_sources.yaml 与各来源的官方页面)
# records.parquet 的 source 值 -> configs/data_sources.yaml 里取 citation 的路径
CITATION_PATH = {
    "ds1_targettrack": ["ds1_targettrack", "citation", "dataset"],
    "ds2_tsuboyama": ["ds2_tsuboyama", "citation"],
    "ds3_proteingym": ["ds3_proteingym", "citation"],
    "ds5_dtu_binder": ["ds5_binder_negatives", "dataset_deposit", "citation"],
    "ds5_adaptyv_egfr": ["ds5_binder_negatives", "citation_adaptyv"],
}


def dig(d, path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


UPSTREAM = {
    "ds1_targettrack": dict(
        name="PSI TargetTrack / TargetDB",
        doi="10.5281/zenodo.821654",
        license="CC BY-SA 4.0",
        sa=True,
        note="六阶段标签的基石, 也是 `express` 主任务的**唯一**来源。"
             "share-alike 具传染性, 见下方 §3。"),
    "ds2_tsuboyama": dict(
        name="Tsuboyama et al. 2023 大规模折叠稳定性",
        doi="10.5281/zenodo.7844779",
        license="CC BY 4.0",
        sa=False,
        note="只用 `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` 的 ΔG 与 95% CI。"
             "**版本坑**: 同一数据在 Zenodo 上另有记录 7992926, 其核心 CSV 少两列, "
             "按它复现会得到不同的标签。使用登记表由项目方手动填写。"),
    "ds3_proteingym": dict(
        name="ProteinGym v1.3 (DMS substitutions)",
        doi="https://proteingym.org",
        license="MIT (代码); 各 DMS assay 的版权归原论文",
        sa=False,
        note="只采用**原论文自己给出过二元阈值**的 assay (33/217); "
             "其余不自行二分 —— 自定阈值会把分析选择伪装成标签。"),
    "ds5_dtu_binder": dict(
        name="DTU de novo binder 设计元分析",
        doi="10.5281/zenodo.15722219",
        license="CC BY 4.0",
        sa=False,
        note="只取配套**数据沉积**里的 `final_dataset.csv` (82MB), 不下载结构包。"
             "**许可要分清**: 预印本正文 (bioRxiv 10.1101/2025.08.14.670059) 是 "
             "CC BY-NC-ND, 而数据沉积 (Zenodo 15722219) 是 CC BY 4.0 "
             "(2026-10-02 经 Zenodo API 实查)。我们只用数据沉积, 不受 NC-ND 约束。"),
    "ds5_adaptyv_egfr": dict(
        name="Adaptyv Bio binder 设计竞赛 (EGFR)",
        doi="https://github.com/adaptyvbio/egfr_competition",
        license="ODbL 1.0 (数据) + Apache-2.0 (代码)",
        sa=True,
        note="ODbL 同样带 share-alike; 商用需另行取得许可。"),
}


def main() -> None:
    rec = pl.read_parquet("data/processed/records.parquet",
                          columns=["source", "cluster_rep"])
    tot = rec.height
    g = (rec.group_by("source")
         .agg(pl.len().alias("n"), pl.col("cluster_rep").n_unique().alias("clu"))
         .sort("n", descending=True))
    cnt = {r["source"]: (r["n"], r["clu"]) for r in g.to_dicts()}
    ds1_n = cnt["ds1_targettrack"][0]
    sa_sources = [k for k, v in UPSTREAM.items() if v["sa"]]
    sa_n = sum(cnt.get(k, (0, 0))[0] for k in sa_sources)

    cfg = yaml.safe_load(pathlib.Path("configs/data_sources.yaml").read_text())
    L: list[str] = []
    A = L.append

    A("# 代码与数据可获取性 / 开源声明")
    A("")
    A("> 本文件由 `src/eval/data_availability.py` 生成, 条数与簇数从 "
      "`data/processed/records.parquet` 注入, 不手写。")
    A("")
    A("## 1 代码")
    A("")
    A("`src/` 与 `configs/` 下的全部代码在 **Apache License 2.0** 下发布 (见 `LICENSE`)。")
    A("")
    A("## 2 衍生数据")
    A("")
    A("我们产出的衍生数据是: 统一的六阶段失败标签、四套冻结切分、按中心的交叉验证折、"
      "以及全部评估产物。**不包含任何上游原始文件的再分发** —— "
      "`data/raw/` 的内容由 `configs/data_sources.yaml` 记录的地址下载、校验 md5 后"
      "只读使用 (`src/ingest/verify_raw.py`), 复现者需自行从原始来源获取。")
    A("")
    A("| 来源 | 记录数 | 占比 | 30% 簇数 | 上游许可 |")
    A("|---|---|---|---|---|")
    for s, (n, c) in cnt.items():
        u = UPSTREAM.get(s, {})
        A(f"| {u.get('name', s)} | {n:,} | {n/tot*100:.1f}% | {c:,} | "
          f"{u.get('license', '—')} |")
    A(f"| **合计** | **{tot:,}** | 100% | — | — |")
    A("")

    A("## 3 ⚠️ share-alike 不能绕过 —— 这里有一个必须讲清的限制")
    A("")
    A("本项目原定的意向是衍生标签统一用 **CC BY 4.0**。"
      "**这一点对含 DS1 的部分做不到**, 原因如下:")
    A("")
    A(f"- PSI TargetTrack (DS1) 的许可是 **CC BY-SA 4.0**。"
      f"CC BY-SA 的 share-alike 条款要求**改编作品以相同或兼容的许可发布**; "
      f"我们从它的 `status` / `stopStatus` 字段推导六阶段标签, 属于改编, "
      f"因此无权把这部分降级为 CC BY 4.0。")
    A(f"- DS1 占 {ds1_n:,} 条 ({ds1_n/tot*100:.1f}%), 并且是 `express` "
      f"**主任务与全部跨中心评估的唯一来源** —— 论文的核心结果全部落在这部分上。")
    A(f"- Adaptyv (ODbL 1.0) 同样带 share-alike。两者合计 {sa_n:,} 条 "
      f"({sa_n/tot*100:.1f}%)。")
    A("")
    A("**因此采用分层许可, 而不是对外声称单一 CC BY 4.0**:")
    A("")
    A("| 部分 | 范围 | 许可 |")
    A("|---|---|---|")
    A(f"| 含 share-alike 上游的衍生数据 | `source \u2208 "
      f"{{{', '.join(sa_sources)}}}` 的 {sa_n:,} 条, 以及据其构造的 "
      f"`lab` / `time` 切分、`center_folds`、`express` 任务的全部标签 | "
      f"**CC BY-SA 4.0** (`LICENSE-DATA-CC-BY-SA-4.0.txt`) |")
    A(f"| 其余衍生数据 | 其余 {tot-sa_n:,} 条 | "
      f"**CC BY 4.0** (`LICENSE-DATA-CC-BY-4.0.txt`) |")
    A("")
    A("合并后的整体数据集若作为一个作品再分发, **按最严的上游条款走, 即 CC BY-SA 4.0**。"
      "想要纯 CC BY 4.0 的使用者应取不含 DS1 的子集 "
      "(`records.parquet` 里 `source != \"ds1_targettrack\"`), "
      "但须注意该子集**不包含 express 主任务**, 复现不出本文的主结果。")
    A("")
    A("### 3.1 ⚠️ 这对**你**(下游使用者)意味着什么 —— 请先读这一段")
    A("")
    A(f"share-alike 的义务是会传下去的。用了上面那 {sa_n:,} 条 "
      f"({sa_n/tot*100:.1f}%) 里的任何部分、或据其构造的切分与标签, "
      "**你再分发的衍生成果也必须以 CC BY-SA 4.0 (或兼容许可) 发布**, "
      "不能改成 CC BY、不能改成 MIT、也不能闭源再分发。具体地:")
    A("")
    A(f"- 这部分**包含 `express` 主任务的全部标签和全部跨中心评估数据** —— "
      f"换句话说, 想复现或扩展本文的核心结果, 就一定会用到受 share-alike "
      f"约束的数据, 没有绕开的路径。")
    A("- 混入自己的数据后再发布, 整体同样受 share-alike 约束 "
      "(CC BY-SA 的 \"Adapted Material\" 条款), 不会因为掺了别的数据而解除。")
    A("- 只做内部研究、不对外分发, 不触发 share-alike (但署名条款仍然适用)。")
    A("- Adaptyv 的 ODbL 另有一条: 公开发布基于它的衍生数据库时, "
      "**衍生数据库也要以 ODbL 提供**; 商用需直接联系 Adaptyv 另谈许可。")
    A("")
    A("我们把这一条放在显眼处主动说明, 而不是让使用者在发布前夜才发现。"
      "如果你的项目**必须**是 CC BY 或更宽松的许可, "
      f"请只取不含 share-alike 上游的那 {tot-sa_n:,} 条 "
      "(`source` 不在 `{" + ", ".join(sa_sources) + "}` 内), "
      "并注意该子集不含 `express` 主任务。")
    A("")
    A("**一个我们无法替使用者解答的问题**: \"用 CC BY-SA 数据训练出来的模型权重"
      "是否构成改编作品\"在法律上没有定论。我们不就此表态, 只做一件可操作的事 —— "
      "**每一次训练/评测运行都机器记录它用了哪些数据源** "
      "(`reports/runs/*.json` 的 `sources` 字段), "
      "使用者据此可以判断某个权重是否触及 DS1。")
    A("")

    A("## 4 上游来源逐条说明")
    A("")
    for s, u in UPSTREAM.items():
        n, c = cnt.get(s, (0, 0))
        A(f"### {u['name']}")
        A("")
        A(f"- 地址 / DOI: `{u['doi']}`")
        A(f"- 许可: **{u['license']}**" + (" (share-alike)" if u["sa"] else ""))
        A(f"- 本文用到: {n:,} 条 / {c:,} 个 30% 簇")
        A(f"- 说明: {u['note']}")
        A("")

    A("## 5 引用要求")
    A("")
    A("使用本数据集时, 除引用本文外, **必须一并引用上游原始数据的论文** —— "
      "CC BY / CC BY-SA 的署名条款要求如此, 且上游的工作量远大于我们的整合工作。"
      "各来源的引用条目见 `configs/data_sources.yaml` 的 `citation` 字段。")
    A("")
    cits = {s: dig(cfg, CITATION_PATH[s]) for s in cnt if s in CITATION_PATH}
    _miss = [s for s in cnt if not cits.get(s)]
    if _miss:
        raise SystemExit(f"!! 下列已使用的来源在 configs/data_sources.yaml 里缺 citation: "
                         f"{', '.join(_miss)} —— 投稿材料不能缺引用, 拒绝出文件")
    for s in cnt:
        A(f"**{UPSTREAM[s]['name']}**")
        A("")
        A(f"> {' '.join(str(cits[s]).split())}")
        A("")
    A("ProteinGym 的官方 README 另有要求: 还须引用各 DMS assay 的**原始实验论文** "
      "(仓库提供 `assays.bib`)。本工作用到 33 个 assay, 它们出自 **13 篇**原始论文 —— assay 数不等于论文数, 多篇论文各贡献多个 assay; 此前文档写成「33 篇原始论文」是把两者混了。逐条对应见 `reports/proteingym_assays_used.md`。")
    A("")

    A("## 6 不包含的内容")
    A("")
    A("- 无湿实验数据 (本工作全部为回顾性计算评测)")
    A("- 无患者数据、无个人可识别信息")
    A("- 无上游原始文件的再分发")
    A("- OIH 自有管线 (DS6) 的 4,762 条设计只用作**未标注探测集**, "
      "其序列与内部路径不随本仓库发布")
    A("")

    OUT.write_text("\n".join(L), encoding="utf8")
    print(f"wrote {OUT} ({len(L)} 行)")
    print(f"  share-alike 覆盖 {sa_n:,}/{tot:,} ({sa_n/tot*100:.1f}%) 条; "
          f"DS1 单独 {ds1_n/tot*100:.1f}%")
    if _miss:
        print(f"  ⚠️ 缺 citation 的来源: {', '.join(_miss)}")


if __name__ == "__main__":
    main()
