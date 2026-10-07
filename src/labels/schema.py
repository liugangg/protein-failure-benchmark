"""统一失败阶段标签体系 (SPEC §2.3)。

所有数据源最终都要映射到这里定义的 6 维标签向量。

标签取值语义 —— 这是全项目最容易搞错的地方:
    OBSERVED_SUCCESS =  1   该步做了, 成功了
    OBSERVED_FAIL    =  0   该步做了, 失败了  <- 真实实验阴性, 本项目的全部价值所在
    UNOBSERVED       = -1   该步没做到 / 没记录

关键规则 (SPEC §2.3 明确要求, 违反会直接污染 PU 学习):
    如果记录在第 k 步失败, 则第 k+1 步及之后全部是 -1 (未观测), **不是** 0。
    "没做到" 不等于 "做了但失败了"。第三阶段的 PU 学习 (SPEC §4.3) 依赖这个区分:
    -1 会被从损失里排除, 0 会作为确定负样本参与梯度。把 -1 写成 0 = 造假阴性。

同样地, 如果记录在第 k 步成功且流程继续, 前 k-1 步应回填为 1 (单调成功前缀),
因为要走到第 k 步必须先通过前面各步。见 fill_success_prefix()。
"""

from __future__ import annotations

from typing import Iterable, Sequence

# 流程顺序即列表顺序, 不要重排 —— 下游 fill/censor 逻辑依赖它
STAGES: list[str] = [
    "clone",      # 基因合成 / 克隆是否成功
    "express",    # 是否产出蛋白
    "soluble",    # 是否可溶 (未形成包涵体)
    "purify",     # 是否通过纯化
    "stable",     # 是否正确折叠且有稳定性
    "bind",       # 是否结合目标
]

N_STAGES = len(STAGES)
STAGE_INDEX: dict[str, int] = {s: i for i, s in enumerate(STAGES)}

OBSERVED_SUCCESS = 1
OBSERVED_FAIL = 0
UNOBSERVED = -1

VALID_LABELS = (OBSERVED_SUCCESS, OBSERVED_FAIL, UNOBSERVED)


def empty_label() -> list[int]:
    """全未观测的标签向量。所有映射都从这里出发, 只填有证据的位。"""
    return [UNOBSERVED] * N_STAGES


def make_label(
    *,
    reached: str | None = None,
    failed_at: str | None = None,
    extra: dict[str, int] | None = None,
) -> list[int]:
    """由 "走到了哪一步 / 在哪一步失败" 构造标签向量。

    reached:   最后一个**成功**通过的阶段名 (含该阶段)。None = 连第一步都没证据。
    failed_at: 观测到失败的阶段名。该位记 0, 其后全部留 -1。
    extra:     少数数据源能直接给某些位的观测值 (如 DS2 只给 stable),
               用 {stage: value} 覆盖, value 必须在 VALID_LABELS 内。

    reached 与 failed_at 可同时给出 (例如: express 成功、soluble 失败),
    此时 failed_at 必须严格排在 reached 之后, 否则抛错 —— 宁可让上游映射崩掉,
    也不要静默产出自相矛盾的标签。
    """
    label = empty_label()

    if reached is not None:
        if reached not in STAGE_INDEX:
            raise ValueError(f"未知阶段 reached={reached!r}")
        for i in range(STAGE_INDEX[reached] + 1):
            label[i] = OBSERVED_SUCCESS

    if failed_at is not None:
        if failed_at not in STAGE_INDEX:
            raise ValueError(f"未知阶段 failed_at={failed_at!r}")
        fi = STAGE_INDEX[failed_at]
        if reached is not None and fi <= STAGE_INDEX[reached]:
            raise ValueError(
                f"标签自相矛盾: failed_at={failed_at} 不晚于 reached={reached}"
            )
        label[fi] = OBSERVED_FAIL
        # fi 之后保持 -1: 失败之后的步骤是"未观测", 不是"失败"
        for i in range(fi + 1, N_STAGES):
            label[i] = UNOBSERVED

    if extra:
        for stage, value in extra.items():
            if stage not in STAGE_INDEX:
                raise ValueError(f"未知阶段 extra={stage!r}")
            if value not in VALID_LABELS:
                raise ValueError(f"非法标签值 {value!r} (只允许 {VALID_LABELS})")
            label[STAGE_INDEX[stage]] = value

    validate_label(label)
    return label


def validate_label(label: Sequence[int]) -> None:
    """结构校验。任何 ingest 脚本写盘前必须过一遍。"""
    if len(label) != N_STAGES:
        raise ValueError(f"标签长度应为 {N_STAGES}, 得到 {len(label)}")
    for v in label:
        if v not in VALID_LABELS:
            raise ValueError(f"非法标签值 {v!r} (只允许 {VALID_LABELS})")

    # 失败之后不允许再有观测值 (成功或失败)。出现即说明上游映射把
    # "未做到" 当成了观测, 或把多次独立试验混成了一条记录。
    for i, v in enumerate(label):
        if v == OBSERVED_FAIL:
            for j in range(i + 1, N_STAGES):
                if label[j] != UNOBSERVED:
                    raise ValueError(
                        f"{STAGES[i]} 失败后 {STAGES[j]} 仍有观测值 {label[j]}: "
                        "失败之后的阶段必须是 -1"
                    )
            break

    # 注意: **不**要求成功前缀连续。1 之前允许有 -1。
    # 理由 (2026-09-30 被 DS5 打出来的): 不同数据源观测的是不同阶段子集, 不是都跑
    # 完整六步流程。Adaptyv 的 BLI 直接把表达产物用 tag 化学固定到探针上测结合,
    # 根本没有独立的可溶/纯化/稳定性步骤, 合法标签就是 [1,1,-1,-1,-1,1]。
    # 若在这里强制回填, 等于把"没测"写成"测了且通过" —— 造假阳性, 比造假阴性一样糟。
    # 真正做完整顺序流程的源 (如 TargetTrack) 才显式调 fill_success_prefix()。


def fill_success_prefix(label: Sequence[int]) -> list[int]:
    """把最后一个成功位之前的 -1 回填为 1。

    依据: 要走到第 k 步, 必须先通过第 1..k-1 步。数据源常只记录最终状态,
    不逐步记录。**只回填成功, 绝不回填失败** —— 前面的步骤没记录失败就是没失败。

    **只对"真的按 clone->express->soluble->purify->stable->bind 顺序跑完整流程"的
    数据源调用它** (TargetTrack 是这种)。对只测某几个阶段的源 (BLI 结合实验、
    蛋白酶解稳定性实验) 调用它就是伪造观测, 见 validate_label() 里的说明。
    """
    out = list(label)
    last_success = -1
    for i, v in enumerate(out):
        if v == OBSERVED_SUCCESS:
            last_success = i
    for i in range(last_success):
        if out[i] == UNOBSERVED:
            out[i] = OBSERVED_SUCCESS
    validate_label(out)
    return out


def label_to_dict(label: Sequence[int]) -> dict[str, int]:
    validate_label(label)
    return dict(zip(STAGES, label))


def summarize(labels: Iterable[Sequence[int]]) -> dict[str, dict[str, int]]:
    """按阶段汇总 成功/失败/未观测 三态计数 —— GATE 1 报告的第 1 项。"""
    counts = {s: {"success": 0, "fail": 0, "unobserved": 0} for s in STAGES}
    for label in labels:
        validate_label(label)
        for stage, v in zip(STAGES, label):
            if v == OBSERVED_SUCCESS:
                counts[stage]["success"] += 1
            elif v == OBSERVED_FAIL:
                counts[stage]["fail"] += 1
            else:
                counts[stage]["unobserved"] += 1
    return counts


# 统一记录 schema: 所有 ingest 脚本输出这套列, 写 parquet 到 data/interim/
RECORD_COLUMNS: dict[str, str] = {
    "record_id": "str",        # 全局唯一, 形如 ds1:TT_123456
    "source": "str",           # ds1_targettrack / ds2_tsuboyama / ...
    "sequence": "str",         # 氨基酸序列, 大写单字母
    "seq_len": "int32",
    "target_id": "str",        # 源库内部 ID
    "center": "str",           # 实验室 / center, 无则 ""  (GATE 1 实验室分布要用)
    "year": "int16",           # 记录年份, 无则 -1        (GATE 1 年份分布 + 时间切分要用)
    "organism": "str",
    "is_designed": "bool",     # 是否 de novo 设计蛋白 (对比天然蛋白分布用)
    **{f"label_{s}": "int8" for s in STAGES},
    "evidence": "str",         # 原始状态字符串, 保留以便回溯映射对不对
}


# ---------------------------------------------------------------------------
# 派生标签 (不是第七个阶段, 只是为了和外部基线对齐定义而算出来的视图)
# ---------------------------------------------------------------------------
DERIVED_SOLUBLE_EXPRESSION = "label_soluble_expression"


def soluble_expression_label(label_express: int, label_soluble: int) -> int:
    """对齐 SoluProt 的「可溶表达」单一二元事件。

    为什么需要它 (2026-09-30, 刘刚刚要求先确认定义是否对得上):
      SoluProt 预测的是 "soluble protein expression in E. coli" —— **一个**二元结局。
      我们的 schema 把它拆成两步: express (有没有产出蛋白) 然后 soluble (是否不成包涵体),
      而且 soluble 只在 express 成功的条件下才被观测 (express=0 会把 soluble 截断成 -1,
      实测 530,019 条两阶段都有观测的记录里, 组合只有 (1,1) 和 (1,0))。
      所以拿 SoluProt 去比我们的 express 头或 soluble 头**都是在比两件不同的事**:
        - 比 express: SoluProt 的阴性里混了"表达了但不可溶", 我们的 express 阴性不含这部分
        - 比 soluble: 我们的 soluble 阴性 99.9% 来自 v2 §2.4 推断 (显式只有 10 条),
                     且它的分母是"已经表达成功的", 与 SoluProt 的全体分母不同
      正确的对齐是这个复合标签:
        1 = express 成功 且 soluble 成功
        0 = express 失败, 或 (express 成功 但 soluble 失败)
       -1 = 其余 (express 未观测, 或 express 成功但 soluble 未观测)

    **它不是第七个阶段, 不参与 GATE 1 的"≥3 个阶段"计数。** 只用于基线对照。
    """
    if label_express == OBSERVED_SUCCESS and label_soluble == OBSERVED_SUCCESS:
        return OBSERVED_SUCCESS
    if label_express == OBSERVED_FAIL:
        return OBSERVED_FAIL
    if label_express == OBSERVED_SUCCESS and label_soluble == OBSERVED_FAIL:
        return OBSERVED_FAIL
    return UNOBSERVED
