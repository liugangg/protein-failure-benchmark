# 代码与数据可获取性 / 开源声明

> 本文件由 `src/eval/data_availability.py` 生成, 条数与簇数从 `data/processed/records.parquet` 注入, 不手写。

## 1 代码

`src/` 与 `configs/` 下的全部代码在 **Apache License 2.0** 下发布 (见 `LICENSE`)。

## 2 衍生数据

我们产出的衍生数据是: 统一的六阶段失败标签、四套冻结切分、按中心的交叉验证折、以及全部评估产物。**不包含任何上游原始文件的再分发** —— `data/raw/` 的内容由 `configs/data_sources.yaml` 记录的地址下载、校验 md5 后只读使用 (`src/ingest/verify_raw.py`), 复现者需自行从原始来源获取。

| 来源 | 记录数 | 占比 | 30% 簇数 | 上游许可 |
|---|---|---|---|---|
| PSI TargetTrack / TargetDB | 945,718 | 39.7% | 96,266 | CC BY-SA 4.0 |
| Tsuboyama et al. 2023 大规模折叠稳定性 | 776,296 | 32.6% | 243 | CC BY 4.0 |
| ProteinGym v1.3 (DMS substitutions) | 654,010 | 27.5% | 32 | MIT (代码); 各 DMS assay 的版权归原论文 |
| DTU de novo binder 设计元分析 | 3,669 | 0.2% | 2,188 | CC BY 4.0 |
| Adaptyv Bio binder 设计竞赛 (EGFR) | 604 | 0.0% | 259 | ODbL 1.0 (数据) + Apache-2.0 (代码) |
| **合计** | **2,380,297** | 100% | — | — |

## 3 ⚠️ share-alike 不能绕过 —— 这里有一个必须讲清的限制

本项目原定的意向是衍生标签统一用 **CC BY 4.0**。**这一点对含 DS1 的部分做不到**, 原因如下:

- PSI TargetTrack (DS1) 的许可是 **CC BY-SA 4.0**。CC BY-SA 的 share-alike 条款要求**改编作品以相同或兼容的许可发布**; 我们从它的 `status` / `stopStatus` 字段推导六阶段标签, 属于改编, 因此无权把这部分降级为 CC BY 4.0。
- DS1 占 945,718 条 (39.7%), 并且是 `express` **主任务与全部跨中心评估的唯一来源** —— 论文的核心结果全部落在这部分上。
- Adaptyv (ODbL 1.0) 同样带 share-alike。两者合计 946,322 条 (39.8%)。

**因此采用分层许可, 而不是对外声称单一 CC BY 4.0**:

| 部分 | 范围 | 许可 |
|---|---|---|
| 含 share-alike 上游的衍生数据 | `source ∈ {ds1_targettrack, ds5_adaptyv_egfr}` 的 946,322 条, 以及据其构造的 `lab` / `time` 切分、`center_folds`、`express` 任务的全部标签 | **CC BY-SA 4.0** (`LICENSE-DATA-CC-BY-SA-4.0.txt`) |
| 其余衍生数据 | 其余 1,433,975 条 | **CC BY 4.0** (`LICENSE-DATA-CC-BY-4.0.txt`) |

合并后的整体数据集若作为一个作品再分发, **按最严的上游条款走, 即 CC BY-SA 4.0**。想要纯 CC BY 4.0 的使用者应取不含 DS1 的子集 (`records.parquet` 里 `source != "ds1_targettrack"`), 但须注意该子集**不包含 express 主任务**, 复现不出本文的主结果。

### 3.1 ⚠️ 这对**你**(下游使用者)意味着什么 —— 请先读这一段

share-alike 的义务是会传下去的。用了上面那 946,322 条 (39.8%) 里的任何部分、或据其构造的切分与标签, **你再分发的衍生成果也必须以 CC BY-SA 4.0 (或兼容许可) 发布**, 不能改成 CC BY、不能改成 MIT、也不能闭源再分发。具体地:

- 这部分**包含 `express` 主任务的全部标签和全部跨中心评估数据** —— 换句话说, 想复现或扩展本文的核心结果, 就一定会用到受 share-alike 约束的数据, 没有绕开的路径。
- 混入自己的数据后再发布, 整体同样受 share-alike 约束 (CC BY-SA 的 "Adapted Material" 条款), 不会因为掺了别的数据而解除。
- 只做内部研究、不对外分发, 不触发 share-alike (但署名条款仍然适用)。
- Adaptyv 的 ODbL 另有一条: 公开发布基于它的衍生数据库时, **衍生数据库也要以 ODbL 提供**; 商用需直接联系 Adaptyv 另谈许可。

我们把这一条放在显眼处主动说明, 而不是让使用者在发布前夜才发现。如果你的项目**必须**是 CC BY 或更宽松的许可, 请只取不含 share-alike 上游的那 1,433,975 条 (`source` 不在 `{ds1_targettrack, ds5_adaptyv_egfr}` 内), 并注意该子集不含 `express` 主任务。

**一个我们无法替使用者解答的问题**: "用 CC BY-SA 数据训练出来的模型权重是否构成改编作品"在法律上没有定论。我们不就此表态, 只做一件可操作的事 —— **每一次训练/评测运行都机器记录它用了哪些数据源** (`reports/runs/*.json` 的 `sources` 字段), 使用者据此可以判断某个权重是否触及 DS1。

## 4 上游来源逐条说明

### PSI TargetTrack / TargetDB

- 地址 / DOI: `10.5281/zenodo.821654`
- 许可: **CC BY-SA 4.0** (share-alike)
- 本文用到: 945,718 条 / 96,266 个 30% 簇
- 说明: 六阶段标签的基石, 也是 `express` 主任务的**唯一**来源。share-alike 具传染性, 见下方 §3。

### Tsuboyama et al. 2023 大规模折叠稳定性

- 地址 / DOI: `10.5281/zenodo.7844779`
- 许可: **CC BY 4.0**
- 本文用到: 776,296 条 / 243 个 30% 簇
- 说明: 只用 `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` 的 ΔG 与 95% CI。**版本坑**: 同一数据在 Zenodo 上另有记录 7992926, 其核心 CSV 少两列, 按它复现会得到不同的标签。使用登记表由项目方手动填写。

### ProteinGym v1.3 (DMS substitutions)

- 地址 / DOI: `https://proteingym.org`
- 许可: **MIT (代码); 各 DMS assay 的版权归原论文**
- 本文用到: 654,010 条 / 32 个 30% 簇
- 说明: 只采用**原论文自己给出过二元阈值**的 assay (33/217); 其余不自行二分 —— 自定阈值会把分析选择伪装成标签。

### DTU de novo binder 设计元分析

- 地址 / DOI: `10.5281/zenodo.15722219`
- 许可: **CC BY 4.0**
- 本文用到: 3,669 条 / 2,188 个 30% 簇
- 说明: 只取配套**数据沉积**里的 `final_dataset.csv` (82MB), 不下载结构包。**许可要分清**: 预印本正文 (bioRxiv 10.1101/2025.08.14.670059) 是 CC BY-NC-ND, 而数据沉积 (Zenodo 15722219) 是 CC BY 4.0 (2026-10-02 经 Zenodo API 实查)。我们只用数据沉积, 不受 NC-ND 约束。

### Adaptyv Bio binder 设计竞赛 (EGFR)

- 地址 / DOI: `https://github.com/adaptyvbio/egfr_competition`
- 许可: **ODbL 1.0 (数据) + Apache-2.0 (代码)** (share-alike)
- 本文用到: 604 条 / 259 个 30% 簇
- 说明: ODbL 同样带 share-alike; 商用需另行取得许可。

## 5 引用要求

使用本数据集时, 除引用本文外, **必须一并引用上游原始数据的论文** —— CC BY / CC BY-SA 的署名条款要求如此, 且上游的工作量远大于我们的整合工作。各来源的引用条目见 `configs/data_sources.yaml` 的 `citation` 字段。

**PSI TargetTrack / TargetDB**

> Berman, H. M., Gabanyi, M. J., Kouranov, A., Micallef, D. I., Westbrook, J., & Protein Structure Initiative network of investigators (2017). Protein Structure Initiative - TargetTrack 2000-2017 - all data files [Data set]. Zenodo. https://doi.org/10.5281/zenodo.821654

**Tsuboyama et al. 2023 大规模折叠稳定性**

> Tsuboyama, K., Dauparas, J., Chen, J., Laine, E., Mohseni Behbahani, Y., Weinstein, J. J., et al. (2023). Mega-scale experimental analysis of protein folding stability in biology and protein design. Zenodo. https://doi.org/10.5281/zenodo.7844779 (论文: Nature 620, 434-444)

**ProteinGym v1.3 (DMS substitutions)**

> Notin, P., Kollasch, A., Ritter, D., van Niekerk, L., Paul, S., Spinner, H., Rollins, N., Shaw, A., Orenbuch, R., Weitzman, R., Frazer, J., Dias, M., Franceschi, D., Gal, Y., & Marks, D. (2023). ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design. Advances in Neural Information Processing Systems 36, 64331-64379.

**DTU de novo binder 设计元分析**

> Overath, M. D., Rygaard, A., & Jenkins, T. P. (2025). Dataset for: Predicting Experimental Success in De Novo Binder Design: A Meta-Analysis of 3,766 Experimentally Characterised Binders [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15722219

**Adaptyv Bio binder 设计竞赛 (EGFR)**

> Adaptyv Bio (2024, 2025). EGFR Protein Design Competition, Rounds 1 and 2. https://github.com/adaptyvbio/egfr_competition_1 ; https://github.com/adaptyvbio/egfr_competition_2

ProteinGym 的官方 README 另有要求: 还须引用各 DMS assay 的**原始实验论文** (仓库提供 `assays.bib`)。本工作用到 33 个 assay, 它们出自 **13 篇**原始论文 —— assay 数不等于论文数, 多篇论文各贡献多个 assay; 此前文档写成「33 篇原始论文」是把两者混了。逐条对应见 `reports/proteingym_assays_used.md`。

## 6 不包含的内容

- 无湿实验数据 (本工作全部为回顾性计算评测)
- 无患者数据、无个人可识别信息
- 无上游原始文件的再分发
- OIH 自有管线 (DS6) 的 4,762 条设计只用作**未标注探测集**, 其序列与内部路径不随本仓库发布
