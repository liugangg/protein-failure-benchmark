# 公开蛋白失败数据中的标签伪影源自实验室身份, 且剥标签与对抗去偏都无法移除

**— 一个跨中心评估基准, 以及它为什么不可省**

**bioRxiv 预印本草稿 · 生成于 `src/eval/preprint.py` · 数字全部从冻结产物注入**

---

## 摘要

蛋白实验失败预测缺少统一基准。我们把四个公开来源的真实实验失败记录整合为一套六阶段标签体系, 得到 2,380,297 条记录; 但在 30% 序列相似度下只有 98,617 个独立簇 —— 这一落差本身是本文的第一个结论: 以阴性条数衡量此类数据集的规模会误导跨蛋白任务上的有效样本量, 按来源计**从不足一个数量级到近四个数量级** (最大收缩 ÷6,661, 即 3.8 个数量级, 见 §1.1)。

**这类数据存在序列层面的标签伪影, 已由前人发现**: NetSolP (Thumuluri et al., 2022) 报告其训练集 11,602/69,420 条带 N 端 His 标签 `MGSDKIHHHHHH` 且约 99% 不可溶, 并指出"模型更多地关注 His 标签而非野生型序列"; SoluProt (Hon et al., 2021) 则主动平衡了序列长度分布以免长度单独主导预测; 到 PLM_Sol (Zhang et al., 2024) 构建 UESolDS 时, 剥除 His 标签片段已是标准步骤。**本文不重复宣称发现这些伪影。**

本文回答的是接下来那个问题: **这些伪影从哪来, 以及剥掉之后还剩什么。** 我们把伪影归因到**实验室身份**这一层: 亲和标签的使用率与位置、序列长度、乃至阴性标签本身是否存在, 都是各结构基因组学中心的构建与记录习惯; 长度对失败的表观预测力在中心内几乎消失, 且在两个阶段上**跨中心与中心内方向完全相反** (Simpson 反转) —— 这说明它量的是中心之间的惯例差异, 不是生物学。据我们所知, 上述三项工作均未做任何中心级的比较或分层 (全文实查)。

**关键结果是伪影去不掉。** 按本领域现行标准做法剥除构建体残留、并且干脆不使用长度特征之后, 以留出中心评估, 跨中心泛化呈现**四情形混合**: 在 4 个留出中心组里, 1 组真信号、1 组弱但可分辨、1 组不可区分、1 组反向 (判定依据为按同源簇分层的自助法 95% 区间, 区间定方向、点估计定幅度; 其中一个**只用 20 维氨基酸组成、不含长度**的梯度提升树与 ESM-2 650M 的区间重叠甚至更高, 说明瓶颈不在表示能力)。**方向与幅度在中心之间都不一致**, 而且在一个留出中心组上, 同一套序列换一个失败阶段就把预测方向整体翻过来 (`express` 上区间 [0.77, 0.80], `soluble_expression` 上 [1.03, 1.07])。

**结论: 中心间的方向与幅度均不一致, 且至少一例显示同一批中心在不同任务上方向相反 —— 没有廉价的事前判据能预知一次部署会落进有用 / 有害 / 噪声哪一类, 只能每个中心每个任务实测。这就是中心分层评估协议的成本来源与必要性。**

我们公开统一标签、四套经显式泄漏验证的冻结切分、以及全部评估代码。

## 1 数据与规模

| 量 | 值 |
|---|---|
| 记录数 | 2,380,297 |
| 独立序列 (DMS 变体按亲本归并) | 354,019 |
| 30% 相似度簇 | **98,617** |

### 1.1 第一个结论: 阴性条数不是规模

| 阶段 | 阴性记录数 | 30% 簇数 | 收缩 |
|---|---|---|---|
| `clone` | 18,398 | **5,663** | ÷3.2 |
| `express` | 94,165 | **22,771** | ÷4.1 |
| `soluble` | 9,995 | **4,640** | ÷2.2 |
| `purify` | 29,854 | **9,001** | ÷3.3 |
| `stable` | 153,858 | **3,766** | ÷41 |
| `bind` | 163,950 | **2,161** | ÷76 |

上表按**阶段**聚合, 掩掉了来源之间的巨大差异。按**来源**看才看得出成因 (下表的 32 与 220 正是"十几万条阴性只等价于几十个独立样本"的出处):

| 来源 | 阴性记录数 | 30% 簇数 | 收缩 | 为什么 |
|---|---|---|---|---|
| ds1_targettrack | 146,678 | **30,438** | ÷4.8 | 全流程记录, 每条是一个独立靶点 — 收缩最小 |
| ds2_tsuboyama | 106,886 | **220** | ÷486 | 同一批亲本结构域的大量单点突变 (cDNA display proteolysis) |
| ds3_proteingym | 213,151 | **32** | ÷6,661 | 同一批亲本蛋白的深度突变扫描 (DMS) |
| ds5_dtu_binder | 3,275 | **1,998** | ÷1.6 | de novo 设计 binder, 靶点少但设计彼此差异大 |
| ds5_adaptyv_egfr | 230 | **40** | ÷5.8 | 单靶点 (EGFR) 竞赛设计 |

> 两张表的"收缩"列统一规则: **小于 10 倍给一位小数, 10 倍及以上取整**。摘要里"不足一个数量级"的依据是本表最小的 ÷1.6 (log₁₀ = 0.21), "近四个数量级"的依据是最大的 ÷6,661 (log₁₀ = 3.82)。

**这量的是跨蛋白泛化的有效多样性, 不是数据质量。** 对 DMS 自身的用途 (同一蛋白内的变异效应预测) 这些是极好的数据; 但在"预测一个没见过的蛋白会不会失败"这个任务上, 一个来源有多少条阴性几乎不重要, 重要的是它覆盖了多少个独立的蛋白家族。按阶段聚合的上表与按来源的本表对不上, 正是因为 `stable` 与 `bind` 两个阶段的阴性几乎全部来自 DMS 类来源。

## 2 相关工作与本文的增量

### 2.1 标签伪影不是本文发现的

把这一点放在最前面, 以免读者误会贡献边界。

**Thumuluri et al., 2022** (NetSolP) 在 E. coli 可溶性数据上报告了亲和标签伪影, 原文:

> "An example of this is that 11 602 out of 69 420 sequences of the training set and 344 out of 2001 sequences of the test set have the N-terminal His-tag 'MGSDKIHHHHHH' with ~99% and ~97% of them being insoluble, respectively."

> "The consequence of this is that the trained models focus more on the His-tag instead of the wild-type sequence."

并转述了标签本身的不一致 (该数字出自 Hon et al., 2021 与 Price et al., 2011 的对比, 非 NetSolP 自测):

> "[Hon et al. (2021)] compared the labels of sequences from this dataset with another dataset whose solubility was provided separately ([Price et al., 2011]) and found that around 18.6% of labels were different, even with 100% identical sequences."

**Hon et al., 2021** (SoluProt) 在构建数据集时就处理了长度混杂, 原文: "we balanced the sequence length distribution so that length alone would not play a dominant role in the predictions."

**Zhang et al., 2024** (PLM_Sol / UESolDS) 把剥标签固化为建库步骤, 原文:

> "The following His tag fragments in proteins were excluded due to an uneven distribution of these tags between Insol and Sol proteins revealed by NetSolP: "MGSDKIHHHHHH", "MGSSHHHHHH", "MHHHHHHS", "MRGSHHHHHH", "MAHHHHHH", "MGHHHHHH", "MGGSHHHHHH", "HHHHHHH" and "AHHHHHHH"."

**所以: 剥除亲和标签、不让长度主导, 是本领域 2021–2024 年间形成的标准做法。本文既不声称发现这些伪影, 也不把剥标签当作贡献。**

### 2.2 与本文最接近的前人工作: 结晶倾向 / 多阶段预测这一支

上面三篇做的是**单一的可溶性**预测。还有一整支工作与本文框架更接近 —— **同样用 TargetTrack / PepcDB、同样预测多个连续实验阶段**。它是本文批评的具体对象, 必须单列。

代表作是 **Wang et al., 2014** (PredPPCrys), 原文:

> "We downloaded the most recent datasets from the PepcDB database comprising 108,933 targets and 979,645 experimental trials."

它预测 克隆 / 产出 / 纯化 / 结晶 / 衍射级结晶 (五个连续阶段), 与本文的六阶段标签几乎是同一个问题。去冗余方式是:

> "We reduced sequence homology in the datasets by removing sequences with >=40% sequence identity using CD-HIT within each class."

> "We applied BLAST to further reduce the sequence redundancy between the training and independent test datasets using a cutoff of 25% sequence identity."

**这不是一篇孤例, 而是一条持续十年的线。** 为确认没有遗漏同类工作, 我们核对了 **Wang et al., 2018** 这篇系统评测该支预测器的综述, 原文:

> "The data sets used to develop these tools were derived from a number of relevant public databases, including TargetDB, PepcDB and TargetTrack."

> "Three tools (PPCPred, PredPPCrys and Crysalis) also predict the propensity for successfully completing some of the crucial steps during the crystallization process."

| 工作 | 年 | 覆盖阶段 | 切分方式 | 中心留出 |
|---|---|---|---|---|
| **PPCpred** (Mizianty et al., 2011) | 2011 | 产出 / 纯化 / 结晶 | 仅序列同源去冗余 | **无** |
| **PredPPCrys** (Wang et al., 2014) | 2014 | 克隆 / 产出 / 纯化 / 结晶 / 衍射级结晶 | 仅序列同源去冗余 | **无** |
| **Crysalis** (Wang et al., 2016) | 2016 | 多阶段 (见综述) | 仅序列同源去冗余 | **无** |
| **fDETECT** (Meng et al., 2017) | 2017 | 产出 / 纯化 / 结晶 | 仅序列同源去冗余 | **无** |
| **deep-cascade forest** (Zhu et al., 2021) | 2021 | 多阶段 | 仅序列同源去冗余 | **无** |
| 本文 | 2026 | clone / express / soluble / purify / stable / bind | 同源传递闭包 **+ 留出中心** | **有** |

> **上表各格的核实层级, 据实标注**: PredPPCrys 那一行的阈值与"无中心分层"是**逐条核对该文全文**得到的 (引语见上)。其余四篇的"仅序列同源去冗余 / 无中心留出"是据 Wang et al., 2018 的系统复评得到的家族级结论 —— 该综述描述了这一支统一的去冗余协议 (类内与类间 25% 序列一致度), 且全文无任何按实验室 / 中心的分层或评测切分。**我们没有逐篇重读这四篇的全文**, 所以若某篇另有未被综述记录的中心级分析, 本表这一格会错; 这是一个明确标注的核实边界, 不是已核实的断言。

**关键差异只有一条, 但它决定了成绩的含义**: 这一支的切分一律只做序列同源去冗余 (类内 CD-HIT 40%、训练测试间 BLAST 25%, 或综述统一复评时的 25%), **没有任何一篇做实验室留出或中心级分层** —— 而综述自己写明 TargetTrack 的数据来自 ">40 structural genomics centers worldwide"。按本文 §5.2 的测量, 同中心训练与跨中心训练在**同一批测试条目**上的差距是 1.85 vs 1.01; 因此**同源去冗余不足以排除实验室身份通道**, 这一支报告的成绩有被该通道虚高的可能。

**措辞上必须克制**: 我们没有重跑它们的模型, 所以不能说"它们的成绩是假的"。能说的是: 它们的评估协议不区分"预测蛋白难不难做"与"认出这条序列来自哪个中心", 而本文证明后者在同一数据源上确实可学且足以解释大部分表观性能。这一支唯一触及相关问题的是该综述提到的流水线差异:

> "a portion of crystal structures deposited in PDB was determined traditionally by structural biologists using specialized equipment, unique protocols and laborious trial-and-error efforts."

即它意识到"高通量流水线"与"传统逐个攻关"产出的结构性质不同, 但没有把这一点推进到中心级: 既未按中心分层评测, 也未检验中心身份本身可否被序列预测。

切分泄漏方面有两篇相关工作, 论断分属不同篇, 在此分清 (合引是错的): **Bushuiev et al., 2024b** ("Revealing Data Leakage in Protein Interaction Benchmarks", ICLR 2024 Workshop on Generative and Experimental Perspectives for Biomolecular Design) 专篇论证了"commonly used splitting strategies for protein complexes, based on protein sequence or metadata similarity, introduce major data leakage"; 而 **Bushuiev et al., 2024a** (ICLR 2024 正会) 是 PPIRef 数据集与 iDist 近重复检测算法的出处, 并据此构造非泄漏切分。本文 §4 的 split-group 做法与这两篇同向, 属于沿用而非重新发明。

### 2.3 本文的增量, 以及它为什么不是上述工作的重复

可溶性那三篇把伪影处理成**数据清洗问题**: 识别出一个序列基序 (His 标签) 或一个表面统计量 (长度), 清掉或平衡掉, 然后继续建模。结晶倾向那一支则连这一步都未涉及, 它处理的是**同源冗余**。本文问的是两者都没问的那个问题: 清洗掉可见的伪影、并排除同源泄漏之后, 跨实验室还剩什么。答案是否定的。

下表的"前人不覆盖"一栏针对的是 §2.1 与 §2.2 列出的**全部八篇** (可溶性三篇 + 结晶倾向五篇)。**各格的核实层级沿用 §2.2 的注脚, 不在此重复论证**: 可溶性三篇与 PredPPCrys 为逐篇全文实查; 其余四篇据 Wang et al., 2018 的家族级结论, 该注脚已标明其边界。

| # | 本文做的 | 为什么前人工作不覆盖 |
|---|---|---|
| 1 | **中心级归因**: 标签使用率 (0.02%→91.8%)、标签位置 (N 端 vs C 端 近乎二分)、以及阴性标签是否存在, 都是中心指纹 | **上述八篇均未做任何按实验室 / 中心的比较或分层**(核实层级见 §2.2 注脚); NetSolP 把 His 标签当序列基序处理, 未追问"为什么这批序列带标签而那批不带"; 结晶倾向那一支虽直接用 TargetTrack / PepcDB (数据来自 >40 个中心), 也未按中心切分 |
| 2 | **Simpson 反转**: 长度的预测力在中心内消失, 两个阶段上跨中心与中心内方向完全相反 | SoluProt 平衡长度分布是为了不让长度主导, 但未检验长度的预测力是否本身来自中心间差异 |
| 3 | **同折 within / cross 对照**: 测试集逐条相同, 只换训练数据来源 | 八篇均无留出中心的设定, 故不存在这个对照 (可溶性三篇按序列切分, 结晶倾向五篇按同源阈值切分) |
| 4 | **四情形 + 按同源簇分层自助法 + 种子稳健性**: 把"跨中心有没有信号"拆成方向与幅度两问, 并要求结论对实现选择不敏感 | 八篇报告的都是单一测试集上的汇总指标 (准确率 / MCC / AUC 等), 无按组分层的区间, 故无法区分"方向可不可分辨"与"幅度够不够用" |
| 5 | **有效多样性量化**: 阴性条数与 30% 相似度簇数按来源最多相差 ÷6,661 (3.8 个数量级) | 上述工作的数据集规模均以序列条数 / 靶点数计 (如 PredPPCrys 的 108,933 靶点 / 979,645 次实验) |
| 6 | **对抗去偏的排除性结果**: 中心可分性被压到接近猜最大类, 跨中心的反向依然不变 | 八篇均未尝试移除中心身份 —— 因为均未把它识别为混杂 (可溶性三篇识别的是标签 / 长度, 结晶倾向五篇识别的是同源冗余) |
| 7 | **把同源去冗余与实验室留出分开**: 证明前者不蕴含后者 (同折 within 1.85 vs cross 1.01) | 结晶倾向那一支 (Wang et al., 2014 等 5 篇) 以类内 CD-HIT 40% + 类间 BLAST 25% 为充分条件, 综述 (Wang et al., 2018) 统一复评时同样只用 25% 同源阈值; 全支无一做中心留出 |

换一句话说, 本文接在两条线之后各补一步: 对可溶性那三篇, **前人证明了"模型在看标签", 本文证明"标签只是中心身份的一个可见代理, 把代理剥掉、甚至用对抗训练主动抹掉中心可分性, 跨中心泛化仍然不成立"**; 对结晶倾向那一支, **前人把同源去冗余当作泛化评估的充分条件, 本文证明它不是 —— 同源去冗余之后, 实验室身份仍然独立地撑着大部分表观性能**。两者合起来是一个否定结果, 它的用处是给评估协议定价: 跨中心留出贵 (测试集只剩几千条、且每个中心每个任务都要单独实测), 但省不掉。

## 3 实验室身份是伪影的来源

### 3.1 失败记录本身是中心习惯

在做过表达、记录数 ≥1,000 的 17 个中心里, 表达失败率从 73.7% 跨到 **0.0%**: 其中 3 个中心的失败率恰好为零, 合计 367,452 条记录 (最大的一个 JCSG 有 **350,702** 条)。上万次实验一次不失败在生物学上讲不通 —— **阴性标签的存在与否本身与"哪个中心做的"强相关**。

> **分母说明** (本节与 §3.2 的 JCSG 数字不同, 不是笔误): 本节的 350,702 条是 JCSG **做过表达这一步、因而 `express` 标签非 -1** 的记录; §3.2 的 378,364 条是 JCSG 在 DS1 里的**全部**记录 (含只走到克隆、或该步未观测的)。差额 27,662 条即 `express` 未观测者, 按本文的三态标签规则记为 -1 并从损失与评估中排除。全文每个计数都注明口径。

### 3.2 标签的有无与位置都是中心指纹

| 事实 | 实测 |
|---|---|
| 含 `HHHHHH` 的记录 (DS1 全体 n=939,665 条) | 21.1% |
| 使用率跨 12 个大中心 (n≥10,000) 的范围 | SSGCID **91.8%** → NYCOMPS **0.02%** |

标签的**位置**同样是指纹 (在带标签的记录里, 标签落在 N 端还是 C 端的占比):

| 中心 | 记录数 | His6 使用率 | N 端占比 | C 端占比 |
|---|---|---|---|---|
| NYSGXRC | 44,727 | 54.5% | 2% | 98% |
| NESG | 133,249 | 69.4% | 41% | 59% |
| SGPP | 20,856 | 55.3% | 100% | 3% |
| SSGCID | 21,715 | 91.8% | 99% | 1% |
| SECSG | 14,816 | 16.5% | 99% | 0% |
| JCSG | 378,364 | 11.5% | 100% | 0% |

同一个标签序列, 在一个中心几乎只出现在 N 端、在另一个中心几乎只出现在 C 端。**标签的有无与位置都是中心指纹, 所以它给模型一条序列可见的实验室身份通道** —— 模型不需要学会"什么样的蛋白难表达", 只要学会"这条序列是哪个中心送来的"。

### 3.3 中心内分层: 长度的预测力是 Simpson 反转

| 阶段 | 跨中心 rank AUC | 中心内中位 | 中心内范围 |
|---|---|---|---|
| `clone` | 0.621 | 0.547 | [0.5158, 0.6608] |
| `express` | 0.577 | 0.533 | [0.4926, 0.6751] |
| `soluble` | 0.427 | 0.441 | [0.4209, 0.5706] |
| `purify` | 0.476 | 0.545 | [0.4703, 0.703] |
| `stable` | 0.412 | 0.548 | [0.5043, 0.5916] |

`purify`、`stable` 的**跨中心方向与中心内方向完全相反** (0.476 vs 0.545; 0.412 vs 0.548), 而 `clone` / `express` 的预测力从 0.62 / 0.58 掉到 0.55 / 0.53, 接近无信息。**长度的表观预测力主要来自中心之间的惯例差异。**

### 3.4 消融的不对称性

清除构建体残留 (21 类模式: His / FLAG / MYC / HA / Strep / T7 标签, TEV / thrombin / Factor Xa / 肠激酶位点, GS / GGGGS 接头, 载体克隆位点残余) 并去掉长度特征后:

- **序列切分 (训练测试同中心)**: 6 个任务**全部下降**, 平均 **-4.8%**
- **实验室切分 (跨中心)**: 3/4 个任务下降, 最大的反向变化 +9.8%, 平均 **-1.5%**

**若为真实生物学效应, 两个切分应同样下降。** 实际是同中心一侧 6/6 全部下降且幅度约为跨中心的 3.2 倍, 而跨中心一侧幅度小、方向还不一致 (有一个任务反而上升) —— 说明这两个通道在训练测试同中心时可用、跨中心时失效, 它们起的是识别作用。**不过要注意这个对比的统计强度有限**: 跨中心一侧只有 4 个任务有足够阴性可评, 单个任务的 ±10% 波动就能改变均值, 所以这条证据是与 §3.3 中心内分层检验**并列**的旁证, 不单独承担结论。

## 4 方法: 四套经泄漏验证的冻结切分

切分单位不是序列、也不是 set-cover 簇, 而是 **split group** —— 在全部唯一序列上做 all-vs-all 检索后取 30% 相似度的传递闭包分量。

| 做法 | 训练/测试间 >30% 的序列对 |
|---|---|
| 按 30% set-cover 簇整组划分 | 10,841 (最高 fident 1.00) |
| 加: 簇代表两两比较后合并 | 2,302 |
| 加: 全序列 all-vs-all 传递闭包 | 16 / 1 / 8 |
| 加: 迭代修补 | **0 / 0 / 0** |

**切分泄漏是已被专门研究过的问题, 本文沿用而非重新发明。** Bushuiev et al., 2024b 在蛋白相互作用数据上专篇论证了"commonly used splitting strategies for protein complexes, based on protein sequence or metadata similarity, introduce major data leakage", 后果是"may result in overoptimistic evaluation of generalization, as well as unfair benchmarking of the models, biased towards assessing their overfitting capacity rather than practical utility"; 其配套的正会工作 Bushuiev et al., 2024a 给出 PPIRef 非冗余数据集与 iDist 近重复检测算法, 并据此按 3D 界面相似度构造非泄漏切分。本文面对的是序列级任务而非界面级任务, 故以序列同源的传递闭包作为切分单位, 但诊断思路相同: **先假定自己的切分在泄漏, 再显式度量它**。上表就是这个度量过程。

顺带指出一点与 §2.2 相关的事: 结晶倾向那一支把同源去冗余 (CD-HIT / BLAST 阈值) 当作泛化评估的充分条件, 而这两篇在 PPI 领域、本文在序列领域, 都给出了同一方向的反例 —— **去掉同源冗余不等于去掉泄漏**, 只是去掉了其中一种。

MMseqs2 的级联聚类是贪心 set cover, 只保证成员与自己簇代表满足阈值, **不保证不同簇的成员之间低于阈值** —— 按簇切分会泄漏。最后仍需迭代修补, 因为建组与检验是两次独立检索, 预筛为启发式 k-mer 匹配。

在 30% / 80% 覆盖下取传递闭包, 靶点空间形成一个巨型连通分量 (含 28,646 个 set-cover 簇)。成因不是短片段搭桥 (已验证), 而是多结构域蛋白在家族之间搭桥。占比超过 1% 的组强制进训练侧; **代价是测试集系统性地不含处于大同源网络中的蛋白**。

| 切分 | 检验什么 | test 记录 | 泄漏检查 |
|---|---|---|---|
| sequence | 基本泛化 | 73,219 | **PASS** (0 对) |
| lab | 是否学到实验室偏差 | 5,843 | **PASS** (0 对) |
| time | 时代漂移 | 606 | **PASS** (0 对) |
| bind_target | 面对新靶点能否预测 binder | 300 | **PASS** (0 对) |

**评估口径**: 正类 = 失败。主数字是 **PR-AUC/base** (= PR-AUC / 正类占比)。lift@k 一并报告作可解释性参考 ("固定 k 个湿实验名额能中几个") 但不作判定依据 —— 按簇分层自助法显示 lift@100 在 n≈2,500 的折上区间宽 0.63 (只用 top-100 = 4% 的样本), 而 PR-AUC/base 宽 0.17, 且两者曾在一折上给出相反判定。该规则的改动时间点与影响见补充材料的 changelog。

## 5 结果

### 5.1 现有工具与平凡基线在跨中心评估下都接近随机

对齐 SoluProt 定义的复合标签 (`express` 成功且 `soluble` 成功):

| 方法 | 序列切分 PR-AUC/base | 实验室切分 PR-AUC/base |
|---|---|---|
| SoluProt (全集) | 1.47 | 1.16 |
| SoluProt (去污染子集) | 1.44 | 1.15 |
| 氨基酸组成 GBDT (默认口径: 剥标签 + **无长度**) | 1.60 | 1.17 |

**不写成"我们超过了 SoluProt"**: SoluProt 的 `ecoli_usearch_identity` 特征 (与 E. coli PDB 序列的最大一致度) 在我们的序列切分测试集上有 **6,730/16,435 = 41% 的序列算不出来** (usearch 无命中, SoluProt 退回用训练集均值填补), 而在它官方自带的测试例上这个比例是 4/21 = 19%。它的训练与设计对象是 E. coli 异源表达的天然蛋白, 而我们的测试集含设计蛋白、膜蛋白与大量无显著 PDB 同源的序列 —— 我们是在把它用在适用域之外, **所以上表里 SoluProt 的数字是它的下界**。诚实的表述是: **在这个数据集上, 一个只用 20 维氨基酸组成 (不含长度、已剥构建体残留) 的 GBDT 就达到或超过了已发表工具, 而 SoluProt 的核心特征在域外严重退化。**

**这个比较有两处不对称, 都对 SoluProt 不利, 一并写明:**

1. **域外 vs 域内**: SoluProt 在它自己的数据集上训练, 被我们搬到本数据集上推断; 我们的 GBDT 直接在本数据集上训练。跨域使用必然压低它。
2. **特征覆盖**: 它的 PDB 一致度特征在我们测试集上 41% 算不出来 (它自己的测试例 19%), 退化到训练集均值填补。

**另有一处我们检查过、结论是不构成不对称的**: 长度。Hon et al., 2021 的处理是在**建库时平衡长度分布** ("we balanced the sequence length distribution so that length alone would not play a dominant role in the predictions."), 本文的处理是**不把长度作为特征**。这是两种不同的操作, 不是同一件事的两种说法: 前者改变训练数据的分布, 后者改变特征集合。但两者在本比较中的效果同向 —— **谁都没有靠长度拿分**, 所以不存在"我们偷用长度而它没用"的不公平。

不过这里有一点必须替 SoluProt 说明, 否则仍不公平: **本文的测试集没有做长度平衡**。SoluProt 的模型是在长度分布被平衡过的训练数据上拟合的, 现在被搬到一个长度分布未平衡的测试集上, 这本身可能让它吃亏 (它学到的决策函数假定了一个不同的长度边际分布)。我们没有量化这一项 —— 要量化就得重训 SoluProt, 超出本文范围。所以这是一条**已知但未量化的、方向对我们有利的残余不对称**, 如实标注。

最后提醒一个口径陷阱: 若改用保留长度的消融口径 (`reports/ablation_keeptags_keeplen/`), 我们的 GBDT 会更高 (见 §3.4), **那个数字不可与 SoluProt 并列** —— 它吃的正是本文认定为实验室身份通道的那部分。

因此本文的用法是: **把这个 GBDT 当作"平凡基线"的下界参照, 而不是声称我们的方法更好**。它的作用是说明该数据集上现有工具与平凡特征的差距很小。

污染检查 (SoluProt 官方声明其训练集来自 TargetTrack, 与我们的 DS1 同源): 30% 相似度污染率 序列切分 18.7% / 实验室切分 9.9%, 故每个数字都给全集与去污染子集两版。去污染后实验室切分几乎不变, **污染没有虚高它** —— 这一点对它有利, 如实写。

### 5.2 正面对照: 同一测试折上, 中心内远高于跨中心

这是把身份通道的贡献量直接称出来的一组对照。**两侧的测试集完全相同**, 只换训练数据的来源: `within` 的训练数据来自留出中心**内部的其它同源组**, `cross` 的来自其它中心。因此 within 与 cross 的差值不含测试集差异, 只含"训练数据是否与测试同中心"这一个变量。

| 训练数据来源 | PR-AUC/base 中位 (L0 组成 GBDT) | PR-AUC/base 中位 (L1 冻结 PLM) |
|---|---|---|
| **within**: 留出中心内部 | **1.58** | **1.85** |
| **cross**: 其它中心 | 0.97 | 1.01 |
| 随机对照 (实测) | 1.03 | 1.03 |

**先堵住最自然的质疑: "within 会不会只是在小数据上过拟合?"** 答案是不会, 而且方向正好相反 —— **cross 的训练数据多得多**:

| 折 | 留出中心 | within 训练簇 | cross 训练簇 | cross / within |
|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | 1,650 | 29,284 | **17.7×** |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | 5,768 | 23,107 | **4.0×** |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | 4,472 | 25,051 | **5.6×** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | 4,932 | 24,360 | **4.9×** |

去冗余后本数据集每个 `split_group` 只保留一条记录, 所以上表的条数**就是同源簇数**, 不是被同源冗余虚高的记录数。

cross 一侧的训练簇是 within 的 **4.0–17.7 倍**, 却在同一批测试条目上从 1.85 掉到 1.01。**数据更多、成绩更差**, 所以这个落差不可能是 within 侧样本量不足导致的过拟合, 也不可能是 cross 侧欠拟合 —— 唯一随之改变的变量是"训练数据是否与测试来自同一中心"。

**同中心训练时模型是有效的 (L1 1.85), 换成跨中心训练就塌到随机水平 (1.01 vs 随机 1.03)。** 这正面证明: 可学的东西确实存在, 但它**不跨中心迁移** —— 与 §3 的身份通道证据是同一件事的两面。随机对照落在 1.0 附近, 说明评估口径是校准的, 所以 cross 的低值不是计算错误。

> 这组对照的实现上有一个必须讲明的陷阱, 我们自己先踩了: 最初把 within 的训练集写成"留出中心里不属于测试组的部分", 但测试折就是该中心的**全部**组, 所以那个掩码恒为空 —— within 一栏永远拿不到数。正确做法是把留出中心的同源组再分 K=3 份, 内层第 i 份作测试、其余份作 within 的训练, 同时让 cross 也只在这一份上评估, 以保证两侧测试集逐条相同。这个 bug 是在跑 GPU 之前用 numpy 先验掩码非空才发现的, 否则会白跑一轮并得到"within 无数据"的错误结论。

### 5.3 平凡基线没有被语言模型超过

同一批留出中心、同一套测试折上, L0 (**只用 20 维氨基酸组成的 GBDT, 不含长度特征**) 与 L1 / L2 (ESM-2 650M 冻结 / LoRA 微调) 的对比:

| 折 | 基础率 | L0 GBDT | L1 冻结 | L2 LoRA |
|---|---|---|---|---|
| 1 | 0.306 | 1.33 [1.25, 1.42] · 真信号 | 1.20 [1.13, 1.28] · 真信号 | 1.28 [1.21, 1.37] · 真信号 |
| 2 | 0.032 | 0.86 [0.81, 0.94] · 反向 | 0.97 [0.90, 1.07] · 不可区分 | 1.13 [1.0199, 1.3076] · 边界情形 |
| 3 | 0.413 | 0.77 [0.76, 0.78] · 反向 | 0.79 [0.77, 0.80] · 反向 | 0.84 [0.83, 0.86] · 反向 |
| 4 | 0.522 | 1.02 [0.9957, 1.0399] · 不可区分 | 1.05 [1.03, 1.07] · 弱但可分辨 | 1.04 [1.0139, 1.0582] · 弱但可分辨 |

**在唯一有稳定真信号的折上 (折 1), L0 的 PR-AUC/base 1.33 [1.25, 1.42] 高于 L1 的 1.20, 与 L2 的 1.28 [1.21, 1.37] 区间重叠。** 即: 把 650M 参数的蛋白语言模型换上去, 并没有在跨中心设定下买到比**20 维氨基酸组成**更多的东西。这与 §5.1 中 GBDT 达到或超过已发表工具是同一回事, 不是两个独立结果。

L0 在折 2 上是**反向**而 L1 / L2 不是, 说明 PLM 表示至少没有把那一折做得更差; 但这不构成"PLM 更好"的证据, 因为折 2 的 L2 本身是边界情形 (见 §5.6)。

### 5.4 四情形混合: 中心间方向与幅度均不一致, 并有一例跨任务方向相反

判定规则 (`configs/stage3_train.yaml` changelog `[2026-10-02]`, 实现在 `src/eval/bootstrap_ci.py`): 按 `split_group` 分层自助法 2,000 次, **区间定方向能不能分辨, 点估计定幅度够不够用**。

| 判定 | 条件 |
|---|---|
| 真信号 | 区间不跨 1.0, 点估计 ≥ 1.10, 且对抗版区间同样不跨 1.0 |
| 边界情形 | 形式上满足真信号, 但**对抗版区间跨 1.0** —— 稳健的信号不会因为加入对抗头而消失 |
| 弱但可分辨 | 区间不跨 1.0 但点估计 < 1.10 (统计上可分辨, 幅度只有几个百分点) |
| 不可区分 | 区间跨 1.0 |
| 反向 | 区间整体 < 1.0 |
| *(前置)* 方向稳健性 | 上述方向判定须在 **5 个独立自助法种子**上全部一致; 不一致则按保守方向定为"不可区分" |

**为什么要加种子稳健性这一条。** 定稿前核对发现两个单元格在两位小数下都显示 `[1.00, 1.04]` 却判定相反 (下界 0.9957 跨 1.0 vs 1.0010 不跨), 差 0.001。直接测: 其中一个的方向判定在 5 个种子里 4 正 1 不定 —— **换个种子结论就变**; 把重采样次数从 2,000 提到 20,000 仍是 1.0007, 不解决, 因为不确定性在数据里而不在重采样次数里。一个会被随机种子决定的方向不是结论。这与"加入对抗头就消失的信号不算真信号"是同一条纪律: 结论必须对与科学问题无关的实现选择不敏感。未取多数票, 因为 4:1 也不该当定论。实查全部 40 个折级结果, **只有 1 个被这条规则改判** (L2 `soluble_expression` 折 3, 由"弱但可分辨"改为"不可区分")。区间端点落在 1.0 的 ±0.02 内时, 本文一律打到小数点后 4 位。

> 表中标 ᵇ 的单元格即属此类: 方向判定在 5 个种子上不一致, 故即便打印出的区间端点不跨 1.0, 也按保守方向定为"不可区分", 不作方向性结论。

把"可分辨"和"够用"分开是必须的: 本文最大的折 (n=7,399) 上 +5% 的提升区间也不跨 1.0。若只看区间, 它会被报成真信号; 若只看点估计, 小折上真实的中等效应又会被漏掉。

**express (主任务) · L1 冻结**

| 折 | 留出中心 | n | 基础率 | PR-AUC/base [95% CI] | 判定 |
|---|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | 2,475 | 0.306 | 1.20 [1.13, 1.28] | **真信号** |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | 8,652 | 0.032 | 0.97 [0.90, 1.07] | **不可区分** |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | 6,708 | 0.413 | 0.79 [0.77, 0.80] | **反向** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | 7,399 | 0.522 | 1.05 [1.03, 1.07] | **弱但可分辨** |

**express · L2 LoRA**

| 折 | 留出中心 | n | 基础率 | PR-AUC/base [95% CI] | 判定 |
|---|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | 2,475 | 0.306 | 1.28 [1.21, 1.37] | **真信号** |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | 8,652 | 0.032 | 1.13 [1.0199, 1.3076] | **边界情形** |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | 6,708 | 0.413 | 0.84 [0.83, 0.86] | **反向** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | 7,399 | 0.522 | 1.04 [1.0139, 1.0582] | **弱但可分辨** |

**soluble_expression (对照) · L1 冻结**

| 折 | 留出中心 | n | 基础率 | PR-AUC/base [95% CI] | 判定 |
|---|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | 2,539 | 0.404 | 1.28 [1.22, 1.35] | **真信号** |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | 5,388 | 0.080 | 0.99 [0.94, 1.07] | **不可区分** |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | 4,693 | 0.640 | 1.05 [1.03, 1.07] | **弱但可分辨** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | 7,147 | 0.611 | 1.12 [1.10, 1.14] | **真信号** |

**soluble_expression (对照) · L2 LoRA**

| 折 | 留出中心 | n | 基础率 | PR-AUC/base [95% CI] | 判定 |
|---|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | 2,539 | 0.404 | 1.32 [1.26, 1.40] | **真信号** |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | 5,388 | 0.080 | 1.01 [0.95, 1.08] | **不可区分** |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | 4,693 | 0.640 | 1.02 [1.0010, 1.0415] | **不可区分 ᵇ** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | 7,147 | 0.611 | 1.13 [1.11, 1.15] | **真信号** |


**跨任务对照** (同一批留出中心、同一个 L1 架构, 方向与幅度分开看):

| 折 | 留出中心 | express 方向 | express 幅度 | soluble_expression 方向 | 幅度 | 方向是否一致 |
|---|---|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | **正** | 1.20 (+20%) | **正** | 1.28 (+28%) | 一致 |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | 不定 | 0.97 (-3%) | 不定 | 0.99 (-1%) | 一致 |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | **负** | 0.79 (-21%) | **正** | 1.05 (+5%) | **相反** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | **正** | 1.05 (+5%) | **正** | 1.12 (+12%) | 一致 |

**要分两句说, 强度不同。**

1. **跨中心不一致 —— 4 折都支持。** `express` 上方向为 正、不定、负、正; 幅度从 -21% 到 +20%。即便只看方向为正的 2 折, 幅度也差 4 倍 (+5% vs +20%)。这条是本文的主结论。
2. **跨任务方向相反 —— 只有 1 例, 按单例观察写。** 折 3 上 `express` 的区间整体低于 1.0、`soluble_expression` 的整体高于 1.0, 方向确实相反; **其余三折方向一致** (折 1、折 4 同为正, 折 2 同为不定), 只是幅度不同。所以我们写"在折 3 上观察到方向相反, 其余折未见", **不写成跨任务不一致是普遍规律** —— 一例不足以支撑普遍性陈述。

**L2 (LoRA 微调) 上的独立检验 —— 结果是不复现, 如实写。**

| 折 | express 方向 / 幅度 | soluble_expression 方向 / 幅度 | 方向 |
|---|---|---|---|
| 1 | **正** / 1.28 (+28%) | **正** / 1.32 (+32%) | 一致 |
| 2 | **正** / 1.13 (+13%) | 不定 / 1.01 (+1%) | 不可比 |
| 3 | **负** / 0.84 (-16%) | 不定 ᵇ / 1.02 (+2%) | 不可比 |
| 4 | **正** / 1.04 (+4%) | **正** / 1.13 (+13%) | 一致 |

在 L2 上, 折 3 的 `express` 侧仍是反向, 但 `soluble_expression` 侧的方向**判不出来** —— 5 个自助法种子给出 ['pos', 'cross', 'pos', 'pos', 'pos'], 不一致, 故按保守方向定为不可区分 (点估计 1.02, 与 L1 同号但不显著)。**所以折 3 的方向相反在 L2 上没有复现成一个可判定的对比。** 这不构成对 L1 结果的反证 (两者点估计同号、无一例方向翻转), 但也**不提供独立佐证**。

因此我们把这条写到它能承担的程度为止: **在 L1 的折 3 上观察到两个任务方向相反 (两侧均种子稳定); L2 上该对比因一侧方向不可判定而未复现; 其余折未见方向相反。**这是一例单中心组观察, 不是规律。

这两条合起来堵死最自然的两条补救方案: 既不能靠"这个中心以前表现好"迁移 (跨中心方向与幅度都不一致), 也不能假定"在一个任务上没问题就在另一个任务上也没问题" (至少有一例反例)。没有哪个事前可得的量 —— 中心身份、基础失败率、样本量、任务 —— 能预知下一次部署落进哪一类。

**反向的边界要写清**: 反向只在 `express` 任务的 1 个留出中心组上出现, 且在 L1 / L2 / 对抗版三者上一致 ([0.77, 0.80] / [0.83, 0.86] / [0.78, 0.81])。**不写成"跨中心反向是这类数据的普遍现象"** —— 它是四情形之一。

### 5.5 排除组: 不是模型不够大, 也不是中心身份没去掉

| 排除了什么 | 证据 |
|---|---|
| 不是模型不够大 | L2 微调 5.41M LoRA 参数 (0.82%) / 3 epoch 后, 折 3 反向依然存在 (PR-AUC/base 0.84 [0.83, 0.86], 区间远离 1.0) |
| 不是中心身份没去掉 | 对抗头的中心判别准确率降至 0.429–0.507 (≈猜最大类), 中心可分性已从表示中移除; 而折 3 反向不变 (0.79 → 0.80) |

两条路都被排除, 反向依然存在, **指向同一个 open question**: 该反向可能源于标签生成机制随中心而变, 而非序列表示本身。现有数据无法判定, 不进一步推测。

### 5.6 一个边界情形 (我们自己也没过线)

L2 在 express 折 2 (基础率 0.032) 上 PR-AUC/base = 1.13 [1.0199, 1.3076], 形式上区间不跨 1.0; 但加入对抗头后回到 [0.96, 1.22] 跨 1.0, 且 lift@100 点估计 0.62 方向相反。**判为边界情形的直接依据: 一个稳健的信号不会因为加入对抗头而消失。**

因此, 本文给后来者定的门槛是**两条同时满足** (见 `reports/paper_data_methods.md` §2.6): **(a)** 按簇分层自助法 95% 区间不跨 1.0, **(b)** PR-AUC/base 点估计 ≥ 1.17 —— 即超过本文那个**只用 20 维氨基酸组成、不含长度**的平凡基线在同一实验室留出集上的成绩。只要 (a) 不够: n 够大时 +2% 也能显著, 本文折 4 就是这样 (1.05 [1.03, 1.07], 方向可分辨但幅度不到平凡基线)。按这把双条件的尺子量我们自己: **L1 在 4 个留出中心组里 1 组、L2 在已落预测的 4 组里 1 组同时满足 (a)(b)**。这不是单设给别人的标准, 是用同一把尺子量自己之后的结果。

## 6 局限

1. **只有一个阶段具备可用规模**。去冗余后仅 `express` 的真实阴性过万; 六阶段联合建模所需的数据不存在于公开来源, 故本文定位为数据集与基准。
2. **`soluble` 阶段未解决**。其阴性 99.9% 来自推断规则; 唯一的显式对照子集仅 10 条、全部来自单一中心 (MPSBC) 且**全部不含 His 标签** (His6 侧 n=0), 因此在设计上无法检验。推断子集内的 His6 关联在中心之间方向反转, 不支持生物学解释, 但不足以定论。另有中心两侧各数十万条却零 `soluble` 失败, 与那 10 条并列支撑"soluble 失败是记录产物"。
3. **测试集系统性地不含处于大同源网络中的蛋白** (与 §4 是同一件事, 在此作为局限重述, 因为它限制了本文结论的适用范围): 在 30% 相似度、80% 覆盖下取传递闭包后, 靶点空间形成一个巨型连通分量 (含 28,646 个 set-cover 簇); 占比超过 1% 的组被强制钉入训练侧, 否则单组就会吃掉整个测试集。代价是本文的留出测试集偏向同源网络稀疏的蛋白, **对多结构域、大家族蛋白的泛化能力本文无法评估**。这既影响绝对成绩, 也可能影响跨中心落差的大小 —— 方向未知, 不作推测。
4. **时间留出集在无泄漏约束下几乎不存在** (测试侧仅数百条), 只能作动力不足的弱检验。
5. **物种维度的分组审计做不出结论**: 序列切分测试侧 3,194 条摊在 580 个物种名上, 只有 2 个达到 200 条的最小组门槛 (对照: 实验室维度 6/32)。降低门槛能凑出更多"组", 但几十条样本上的 precision@100 是噪声不是发现, 所以本文的分组审计以实验室与年份为主轴, 不声称跨物种泛化。
6. **无前瞻性验证**。全部评测为回顾性, 本工作不含湿实验。
7. **自有管线数据不能补救**。OIH 计算中心的 4,762 条唯一设计是计算产物: 任务级失败是作业崩溃, 而把结构置信度阈值当标签会构成循环论证(本文实测该指标对真实结合的 per-target 最高 F1 仅 0.572)。

## 7 代码与数据可获取性

代码在 **Apache-2.0** 下发布。衍生数据 (统一标签、四套冻结切分、按中心的折) **不是单一许可** —— 上游的 share-alike 条款不允许我们统一降级为 CC BY 4.0:

| 部分 | 范围 | 许可 |
|---|---|---|
| 代码 (`src/`, `configs/`) | 全部 | **Apache-2.0** |
| 含 share-alike 上游的衍生数据 | 946,322 条 (39.8%; TargetTrack 与 Adaptyv), 以及据其构造的 `lab` / `time` 切分、`center_folds`、`express` 任务的全部标签 | **CC BY-SA 4.0** |
| 其余衍生数据 | 1,433,975 条 (60.2%) | **CC BY 4.0** |

**为什么不能统一成 CC BY 4.0**: PSI TargetTrack 的许可是 CC BY-SA 4.0, 其 share-alike 条款要求改编作品以相同或兼容的许可发布。我们从它的 `status` / `stopStatus` 字段推导六阶段标签, 属于改编。而 TargetTrack 占 39.7% 且是 `express` 主任务与全部跨中心评估的**唯一**来源 —— 本文的核心结果全部落在受 share-alike 约束的那部分上。**整体数据集若作为一个作品再分发, 按最严的上游条款走 (CC BY-SA 4.0)。** 需要纯 CC BY 4.0 的使用者可取不含 TargetTrack 的子集, 但该子集不含 `express` 主任务, 复现不出本文主结果。

**这对下游使用者的实际含义**, 我们主动写明而不是让人事后发现: 用了受 share-alike 约束的那部分 (它包含 `express` 主任务的全部标签与全部跨中心评估数据, 复现本文核心结果绕不开) 并再分发, **衍生成果也必须以 CC BY-SA 4.0 或兼容许可发布** —— 不能改成 CC BY / MIT, 不能闭源再分发, 掺入自有数据后整体同样受约束。只做内部研究、不对外分发则不触发 share-alike (署名条款仍适用)。

"用 CC BY-SA 数据训练出的模型权重是否构成改编作品"在法律上无定论; 我们不就此表态, 只做一件可操作的事: **每次训练与评测都机器记录所用数据源** (`reports/runs/*.json` 的 `sources` 字段), 使用者据此判断某个权重是否触及 TargetTrack。

**上游来源、许可与必须一并给出的引用** (全部字段于 2026-10-02 经各自官方 API / README 实查, 不凭记忆):

| 来源 | DOI / 地址 | 许可 |
|---|---|---|
| PSI TargetTrack 2000-2017 | `10.5281/zenodo.821654` | **CC BY-SA 4.0** |
| Tsuboyama et al. 2023 | `10.5281/zenodo.7844779` | CC BY 4.0 |
| ProteinGym v1.3 | github.com/OATML-Markslab/ProteinGym | MIT (代码); 各 assay 版权归原论文 |
| Overath et al. 2025 binder 元分析 (数据沉积) | `10.5281/zenodo.15722219` | CC BY 4.0 |
| Adaptyv Bio EGFR 竞赛 | github.com/adaptyvbio/egfr_competition_1 / _2 | ODbL 1.0 (数据) + Apache-2.0 (代码) |

两处容易踩的坑, 写明以免复现者重复踩:

1. **Tsuboyama 的 Zenodo 记录有两个**。同一版本另有记录 `7992926`, 其 `Tsuboyama2023_Dataset2_Dataset3` 少 `match_aaseq` / `name_original` 两列 (697,658,024 vs 718,214,782 字节), 按它复现会得到不同的标签。本文用 `7844779`, 并以 md5 守卫强制 (`src/ingest/verify_raw.py`)。
2. **binder 元分析的正文与数据沉积许可不同**。预印本正文 (bioRxiv `10.1101/2025.08.14.670059`) 是 CC BY-NC-ND, 而数据沉积 (Zenodo `15722219`) 是 CC BY 4.0。本文只用数据沉积里的 `final_dataset.csv`, 不受 NC-ND 约束。

完整的许可分层、逐条引用条目、以及 ProteinGym 要求的 33 个 assay 原始论文引用清单见仓库 `DATA_AVAILABILITY.md`。本仓库**不重新分发任何上游原始文件**: `data/raw/` 由记录在 `configs/data_sources.yaml` 的地址下载、校验 md5 后只读使用。

**归档沉积。** 衍生数据与四套冻结切分归档于 Zenodo, doi:[to be added — deposit not yet published]。由于该沉积是作为**单一作品**再分发的, 按上文给出的理由, 它**整体**适用 CC BY-SA 4.0; 需要纯 CC BY 4.0 子集的使用者必须自行剔除 TargetTrack 衍生部分来重建该子集, 而那个子集**不含 `express` 主任务**。

## 8 参考文献

> 全部条目的题目 / 作者 / 卷期页经 Crossref API 实查 (第一轮 2026-10-02, 第二轮 2026-10-05 补结晶倾向这一支与 Bushuiev 两篇的拆分), 正文引语逐字核对出版商全文页或作者提供的 PDF; 登记在 `configs/references.yaml` (含每条引语原文与核实来源)。**未经实查的条目一律不写入** —— 核对过程中我凭记忆猜的两个 DOI 全都指向完全无关的论文 (一个指向巴西生物技术平台, 一个指向 CHO 细胞基因组), 所以这条纪律不是形式主义。

1. Thumuluri V, Martiny HM, Almagro Armenteros JJ, Salomon J, Nielsen H, Johansen AR (2022). NetSolP: predicting protein solubility in Escherichia coli using language models. *Bioinformatics* 38(4): 941-946. doi:10.1093/bioinformatics/btab801
2. Hon J, Marusiak M, Martinek T, Kunka A, Zendulka J, Bednar D, Damborsky J (2021). SoluProt: prediction of soluble protein expression in Escherichia coli. *Bioinformatics* 37(1): 23-28. doi:10.1093/bioinformatics/btaa1102
3. Zhang X, Hu X, Zhang T, Yang L, Liu C, Xu N, Wang H, Sun W (2024). PLM_Sol: predicting protein solubility by benchmarking multiple protein language models with the updated Escherichia coli protein solubility dataset. *Briefings in Bioinformatics* 25(5): bbae404. doi:10.1093/bib/bbae404
4. Price WN, Handelman SK, Everett JK, Tong SN, Bracic A, Luff JD, Naumov V, Acton T, Manor P, Xiao R, Rost B, Montelione GT, Hunt JF (2011). Large-scale experimental studies show unexpected amino acid effects on protein expression and solubility in vivo in E. coli. *Microbial Informatics and Experimentation* 1(1): 6. doi:10.1186/2042-5783-1-6
5. Wang H, Wang M, Tan H, Li Y, Zhang Z, Song J (2014). PredPPCrys: Accurate Prediction of Sequence Cloning, Protein Production, Purification and Crystallization Propensity from Protein Sequences Using Multi-Step Heterogeneous Feature Fusion and Selection. *PLoS ONE* 9(8): e105902. doi:10.1371/journal.pone.0105902
6. Mizianty MJ, Kurgan L (2011). Sequence-based prediction of protein crystallization, purification and production propensity. *Bioinformatics* 27(13): i24-i33. doi:10.1093/bioinformatics/btr229
7. Wang H, Feng L, Zhang Z, Webb GI, Lin D, Song J (2016). Crysalis: an integrated server for computational analysis and design of protein crystallization. *Scientific Reports* 6: 21383. doi:10.1038/srep21383
8. Meng F, Wang C, Kurgan L (2017). fDETECT webserver: fast predictor of propensity for protein production, purification, and crystallization. *BMC Bioinformatics* 18: 580. doi:10.1186/s12859-017-1995-z
9. Zhu Y, Hu J, Ge F, Li F, Song J, Zhang Y, Yu DJ (2021). Accurate multistage prediction of protein crystallization propensity using deep-cascade forest with sequence-based features. *Briefings in Bioinformatics* 22(3): bbaa076. doi:10.1093/bib/bbaa076
10. Wang H, Feng L, Webb GI, Kurgan L, Song J, Lin D (2018). Critical evaluation of bioinformatics tools for the prediction of protein crystallization propensity. *Briefings in Bioinformatics* 19(5): 838-852. doi:10.1093/bib/bbx018
11. Bushuiev A, Bushuiev R, Kouba P, Filkin A, Gabrielova M, Gabriel M, Sedlar J, Pluskal T, Damborsky J, Mazurenko S, Sivic J (2024a). Learning to design protein-protein interactions with enhanced generalization. International Conference on Learning Representations (ICLR) 2024. arXiv:2310.18515
12. Bushuiev A, Bushuiev R, Sedlar J, Pluskal T, Damborsky J, Mazurenko S, Sivic J (2024b). Revealing Data Leakage in Protein Interaction Benchmarks. ICLR 2024 Workshop on Generative and Experimental Perspectives for Biomolecular Design (GEM)

数据来源的引用条目 (TargetTrack / Tsuboyama / ProteinGym / Overath / Adaptyv) 见 `DATA_AVAILABILITY.md` §5 与 `configs/data_sources.yaml`, 同样经实查。

## 9 补充材料

- `reports/gate1_data_inventory.md` — 数据清点与偏差审计 (含去冗余前后对比的核心图表)
- `reports/gate2_baselines.md` — 基线对照: AF3 ipSAE_min 复现、SoluProt (含污染检查)
- `reports/splits_and_leakage.md` — 四套切分的构造与泄漏验证
- `reports/default/stage3_training.md` — L0/L1/L2 训练结果、对抗去偏、自助法区间、方法学陷阱
- `reports/tag_confound_analysis.json` — 中心内分层检验的全部数字
- `reports/soluble_adjudication.json` — soluble 定案检查
- `reports/ds6_oih_inventory.md` — 自有管线清点与门槛校准
- `reports/paper_data_methods.md` — Data & Methods 完整版
- `configs/stage3_train.yaml` 的 `changelog` — 评估规则改动的时间点、理由与影响
- `LICENSE` (Apache-2.0) · `LICENSE-DATA-CC-BY-SA-4.0.txt` / `LICENSE-DATA-CC-BY-4.0.txt` · `NOTICE` · `DATA_AVAILABILITY.md` (许可分层的完整说明)
- `reports/runs/` — 每次评测的数据源清单、切分 hash、超参、环境版本
- Zenodo 沉积 doi:[to be added — deposit not yet published] — 衍生数据与四套冻结切分的归档副本, 作为单一作品适用 CC BY-SA 4.0 (见 §7)


### 9.1 冻结产物指纹

每个产物都带一个 `.prov.json`, 记录自身的 `sha256` / 字节数 / 行数, 以及**全部上游输入的同样三项**; 每个下游脚本启动时硬校验这条链 (`src/labels/provenance.py` 的 `require()`), 不一致就退出而不是继续跑。这条机制是被一次真实事故逼出来的: 上游重建后下游脚本读到陈旧的中间产物, SoluProt 的去污染子集从 2,377 条静默缩到 289 条 —— 数字照常产出, 没有任何报错。

| 产物 | 行数 | sha256 (前 16 位) |
|---|---|---|
| `data/processed/records.parquet` | 2,380,297 | `e887059736b60a98` |
| `data/processed/splits/bind_target_split.parquet` | 3,922 | `5ac1e62546717aa6` |
| `data/processed/splits/center_folds.parquet` | 41 | `7c28d45ac7c2b73a` |
| `data/processed/splits/lab_split.parquet` | 945,718 | `8db25c131e451c88` |
| `data/processed/splits/sequence_split.parquet` | 2,380,297 | `1d5032c5d3f3d4c7` |
| `data/processed/splits/time_split.parquet` | 945,718 | `a08c217a6ed3cb57` |
| `data/interim/pooled_records.parquet` | 2,380,297 | `0056308e0ad5a484` |
| `data/interim/pooled_unique_seqs.fasta` | 708,038 | `d8fc464697129e26` |
| `data/interim/split_groups.parquet` | 354,019 | `ba4ddfbe54fdff87` |

完整的 sha256 与上游输入链见各 `.prov.json` (9 个文件, 随代码发布)。
