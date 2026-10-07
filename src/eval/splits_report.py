"""生成 reports/splits_and_leakage.md —— 三套留出集的定义、规模、以及泄漏检查结果。

所有数字都从 data/processed/splits/MANIFEST.json 和 reports/leakage_check.json 读,
不在这里重算, 避免报告和冻结产物对不上。
"""
from __future__ import annotations

import json
import pathlib

import yaml

MANIFEST = pathlib.Path("data/processed/splits/MANIFEST.json")
LEAK = pathlib.Path("reports/leakage_check.json")
GROUPS = pathlib.Path("data/interim/split_groups.stats.json")
OUT = pathlib.Path("reports/splits_and_leakage.md")
CFG = pathlib.Path("configs/splits.yaml")


def main() -> None:
    m = json.loads(MANIFEST.read_text())
    leaks = {o["split"]: o for o in json.loads(LEAK.read_text())} if LEAK.exists() else {}
    gstats = json.loads(GROUPS.read_text()) if GROUPS.exists() else {}
    cfg = yaml.safe_load(CFG.read_text())

    L: list[str] = []
    A = L.append
    A("# 三套留出测试集 + 泄漏检查")
    A("")
    A("生成: `src/eval/splits_report.py` · 冻结产物: `data/processed/splits/`")
    A("")
    A("> SPEC §3.1 原话: \"测试集一旦确定就冻结, 写入 data/processed/splits/ 并记录 hash。"
      "之后任何阶段都不允许因为分数不好而重新切分。\" MANIFEST.json 里的 sha256 就是凭据。")
    A("")

    # --- 切分单位 ---
    A("## 切分单位: split_group, 不是序列也不是 set-cover 簇")
    A("")
    A("这一节是踩出来的, 不是设计出来的, 值得完整记下来。")
    A("")
    A("1. **按序列切分不行** —— 同源序列会跨侧, 这是常识。")
    A("2. **按 30% set-cover 簇切分也不行。** MMseqs2 的级联聚类是贪心 set cover, "
      "官方 wiki 只承诺\"每个成员与自己簇的代表满足判据\", **不承诺不同簇的成员之间"
      "低于阈值**。实测按簇切分后, 序列切分的训练/测试之间仍有 **10,841 对** >30% "
      "的序列对, 最高一对 `fident = 1.00`（短序列被长序列完整包含, 贪心分簇时分到了两簇）。")
    A("3. **只把簇代表两两比一遍去合并, 还是不行。** 这样建出的组仍留下 **2,302 对** "
      "跨侧 >30% 的对。原因: A 簇的非代表成员可能和 B 簇的非代表成员高度相似, "
      "而两个代表彼此不相似。传递闭包必须建在**所有成员**上。")
    A("4. 最终做法 (`src/splits/split_groups.py`): 在**全部唯一序列**上做 all-vs-all "
      "检索 (`--min-seq-id 0.3 -c 0.8 --cov-mode 1 --alignment-mode 3 -s 7.5`), "
      "把有相似边的序列用并查集并成连通分量, 这个分量就是 `split_group`。")
    A("")
    if gstats:
        A("| 量 | 值 |")
        A("|---|---|")
        A(f"| 唯一序列 | {gstats.get('sequences', '—')} |")
        A(f"| 30% set-cover 簇 | {gstats.get('set_cover_clusters', 0):,} |")
        A(f"| **split_group (切分单位)** | **{gstats.get('split_groups', 0):,}** |")
        A(f"| 序列间 >30% 的边 | {gstats.get('inter_cluster_edges_over_30pct', 0):,} |")
        A(f"| 最大 split_group 含簇数 | {gstats.get('largest_group_clusters', 0):,} |")
        A(f"| 单簇 split_group 数 | {gstats.get('singleton_groups', 0):,} |")
        A("")
    A("代价要说清楚: 连通分量会链式合并, split_group 数少于 set-cover 簇数。"
      "**报告有效样本量时用 set-cover 簇** (与 SoluProt 一类文献可比), "
      "**做切分时用 split_group** (这才防得住泄漏)。两个数不要混用。")
    A("")

    # --- 三套切分 ---
    names = {"sequence": "序列切分", "lab": "实验室切分", "time": "时间切分",
             "bind_target": "bind 跨靶点切分"}
    for key, zh in names.items():
        s = m["splits"].get(key)
        if not s:
            continue
        A(f"## {zh} (`{s['file']}`)")
        A("")
        A(f"`sha256` = `{s['sha256']}`")
        A("")
        if key == "bind_target":
            hold = ", ".join(cfg["bind_target_split"]["holdout_targets"])
            A(f"规则: 留出靶点 **{hold}**, 只用 DS5 (真实 de novo binder 湿实验结合结果)。")
            A("")
            A("为什么需要第四套: SPEC 的三套里 bind 没有可评测的泛化切分 —— "
              "实验室留出集 (留出 TargetTrack 的 CSGID+EFI) 侧的 bind 阴性是 0, "
              "因为 bind 数据全部来自 TargetTrack 以外的研究。")
            A("")
            A("留出靶点的挑选规则 (实测 15 个靶点 / 1,325 个 split group 后定):"
              " (a) 最大的两个靶点必须留训练侧 (FGFR2 占 38.0%、EGFR 占 20.0%, 合计 58%);"
              " (b) 留出集合占 20-30% 的 group; "
              "(c) **必须含至少一个失败率明显低的靶点** —— 各靶点结合失败率多在 85-98%, "
              "只有 Mdm2 是 42.7%, 不含它测试集会几乎全是阴性, precision@k 失去区分力。")
            A("")
            A(f"混合组处理: 丢弃 {s.get('mixed_split_groups_dropped',0):,} 个跨留出/训练的 "
              f"split group ({s.get('records_dropped',0):,} 条记录)。"
              "**注意代价**: 跨靶点的 group 只占 1.7% (23/1,325), 但它们包含最大的三个骨架族 "
              "(1,233 + 334 + 274 条), 所以丢掉的记录占约一半。"
              "成因是不同靶点的 binder 设计共用骨架库 (Cao et al. 那批)。"
              "后果与实验室切分同理: 测试集是\"骨架族不跨靶点\"的那部分设计, 是有偏子集。")
        elif key == "sequence":
            fr = cfg["sequence_split"]["fractions"]
            A(f"规则: 按 split_group 哈希分配, train/val/test = "
              f"{fr['train']}/{fr['val']}/{fr['test']}, 全部数据源。")
            A("")
            A("用 `sha256(seed:split_group)` 而不是 shuffle —— 同一个 seed 在任何机器上"
              "结果一致, 且以后加新数据不会打乱已有组的归属。")
        elif key == "lab":
            A(f"规则: 留出 center = **{', '.join(s['holdout_centers'])}**, "
              f"只用 DS1 (TargetTrack, 唯一的多实验室源)。")
            A("")
            A("为什么是这两家: 留出方必须有足够阴性才能算 precision@k, 又不能是阴性的"
              "最大贡献方否则训练侧被抽干。NYSGRC 一家占 express 阴性 53.7%, 必须留在训练侧。")
            A("")
            A(f"混合组处理: **两边都不要**。丢弃 {s.get('mixed_split_groups_dropped',0):,} "
              f"个混合 split_group ({s.get('records_dropped',0):,} 条记录)。"
              "这样测试侧记录全部来自留出实验室, 训练侧不含与测试侧同源的序列。")
        else:
            A(f"规则: DS1 内 {s['test_from_year']} 年及以后的记录作测试。")
            A("")
            A("只在 DS1 上做: SPEC §8 的\"时代漂移\"指 TargetTrack 2000-2017 内部的年代差。"
              "DS2 全部记 2023 年、DS5 是 2024/2025, 算进来会让这个集子失去含义。")
            A("")
            A(f"混合组处理同上, 丢弃 {s.get('mixed_split_groups_dropped',0):,} 个 "
              f"({s.get('records_dropped',0):,} 条记录)。")
        A("")
        A("| 侧 | 记录数 | set-cover 簇 | split_group | center 数 |")
        A("|---|---|---|---|---|")
        for side, v in s["sides"].items():
            A(f"| {side} | {v['records']:,} | {v['clusters']:,} | "
              f"{v.get('split_groups', 0):,} | {v['centers']} |")
        A("")
        A("各阶段阴性簇数 (能不能在这个测试集上量出东西, 全看这张表):")
        A("")
        hdr = "| 侧 | " + " | ".join(
            k for k in next(iter(s["sides"].values()))["stages"]) + " |"
        A(hdr)
        A("|---" * (len(next(iter(s["sides"].values()))["stages"]) + 1) + "|")
        for side, v in s["sides"].items():
            A(f"| {side} | " + " | ".join(
                f"{d['neg_clusters']:,}" for d in v["stages"].values()) + " |")
        A("")
        lk = leaks.get(key)
        if lk:
            ok = lk.get("verdict") == "PASS"
            A(f"**泄漏检查: {'✅ PASS' if ok else '❌ FAIL'}** — "
              f"split_group 重叠 {lk.get('split_group_overlap')} 个, "
              f"跨侧 >30% 相似度的序列对 {lk.get('cross_pairs_over_30pct')} 对")
            if not ok and lk.get("worst_examples"):
                A("")
                A(f"最严重的一对 fident = {lk['worst_examples'][0][2]}")
            A("")

    # --- 已知局限 ---
    A("## 已知局限 (不要在论文里回避)")
    A("")
    tm = m["splits"].get("time", {}).get("sides", {}).get("test", {})
    if tm:
        A(f"1. **时间留出集太小, 严格无泄漏的版本几乎不存在。** 测试侧只有 "
          f"{tm['records']:,} 条记录 / {tm['clusters']:,} 个 set-cover 簇, "
          f"express 阴性只有 {tm['stages']['express']['neg_clusters']:,} 簇。"
          "原因是结构基因组学中心在 2015 年后接着做的靶点大多与早年的同源, "
          "一旦要求无泄漏就只剩下极少数真正\"新\"的簇。低于 SPEC GATE 2 要的 1,000, "
          "只能当**动力不足的弱检验**报告, 不要拿它下\"没有时代漂移\"的结论。")
    A("2. **soluble 头在任何一套切分里都没有可用的阴性** (全库 10 条)。这个头做不了。")
    A("3. **stable / bind 两个头的阴性簇数是两位数**, 因为它们的阴性全来自 DMS 类数据 "
      "(同一亲本的点突变)。记录数上十万但蛋白多样性只有几百, 切分后每侧几十簇。")
    A("4. **实验室切分只覆盖 DS1。** DS2/DS3/DS5 各自单一课题组, 跨实验室泛化在这些源上"
      "无法检验。")
    A("")

    OUT.write_text("\n".join(L), encoding="utf8")
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
