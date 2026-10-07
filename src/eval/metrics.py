"""评估指标 (SPEC §3.3)。只报这三类, **不报总体 accuracy**。

约定: 本项目里 "阳性" = **失败** (label 0 的那一类)。
  理由: 模型的用途是"从候选里挑出会失败的", 湿实验预算花在被挑中的那些上。
  所以 precision@k 里的 k 个就是"模型最认为会失败的 k 个", precision 就是其中
  真的失败了的比例。这个方向定错的话所有数都反了 —— 代码里统一用 y_fail 命名。

未观测 (-1) 的样本**不参与任何指标计算**, 和 PU 学习里从损失排除是一回事。

═══════════════════════════════════════════════════════════════════════
**强制规则: PR-AUC 不许单独出现, 必须同时带上正类占比与归一化倍数。**
═══════════════════════════════════════════════════════════════════════
这条是被真实错误逼出来的 (2026-09-30, 刘刚刚抓到):

  bind 跨靶点测试集上我报了 "ipSAE PR-AUC = 0.909, 随机对照 0.812", 并据此写下
  "后续 PLM 的门槛是 0.909"。**这句话是误导的。** 因为该测试集的正类 (失败) 占比
  就是 0.796 —— 随机猜的 PR-AUC 天然就接近 0.8。真实的相对增益只有
  0.909 / 0.796 = **1.14 倍**, 和 SoluProt 在实验室切分上的 1.17 是同一量级,
  远不是 0.909 这个数字看上去的那么强。

  更要命的是极性会改变结论的量级: 同一批数据同一个 ipSAE,
    正类 = 失败 (占比 0.796): PR-AUC 0.909, 归一化 1.14
    正类 = 结合 (占比 0.204): PR-AUC 0.353, 归一化 1.73
  把**多数类**当正类会让 PR-AUC 天然接近 1 并压缩相对增益。
  所以任何 PR-AUC 数字, 不声明极性与正类占比就是无意义的。

因此 evaluate() 强制输出:
  positive_class     正类是什么 (调用方必须显式传)
  base_rate          正类占比 = 随机基线的 PR-AUC 期望
  pr_auc             原始值
  pr_auc_over_base   pr_auc / base_rate  <- **要引用就引用这个, 或两个一起引**
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def precision_at_k(y_fail: np.ndarray, score: np.ndarray, k: int) -> float:
    """固定湿实验预算 k 时, 挑出来的 k 个里真正失败的比例。

    score 越大 = 模型越认为会失败。样本不足 k 个时返回 nan 而不是拿 n 充当 k ——
    用更小的分母会把 precision 抬高, 那是自欺。
    """
    n = len(y_fail)
    if n < k:
        return float("nan")
    idx = np.argsort(-score, kind="stable")[:k]
    return float(y_fail[idx].mean())


def recall_at_k(y_fail: np.ndarray, score: np.ndarray, k: int) -> float:
    n_pos = int(y_fail.sum())
    if n_pos == 0 or len(y_fail) < k:
        return float("nan")
    idx = np.argsort(-score, kind="stable")[:k]
    return float(y_fail[idx].sum() / n_pos)


def pr_auc(y_fail: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC (average precision)。SPEC 明确要求用它而不是 ROC-AUC, 因为类别极不平衡。"""
    if y_fail.sum() == 0 or y_fail.sum() == len(y_fail):
        return float("nan")
    return float(average_precision_score(y_fail, score))


def positive_rate(y_fail: np.ndarray) -> float:
    """失败率基线 —— precision@k 必须和它比才有意义。
    随机挑 k 个的期望 precision 就是这个数; 模型没超过它就是没学到东西。"""
    return float(y_fail.mean()) if len(y_fail) else float("nan")


def evaluate(y_fail: np.ndarray, score: np.ndarray, ks=(10, 20, 50, 100),
             positive_class: str = "failure") -> dict:
    """positive_class: 正类是什么, 会原样写进结果供报告引用。
    默认 "failure" (项目约定)。复现外部论文时若用它的极性, 必须显式传 "binder" 等。
    """
    base = positive_rate(y_fail)
    ap = pr_auc(y_fail, score)
    out = {
        # 极性与基线放在最前面 —— 报告里引用 pr_auc 必须同时引用这两个
        "positive_class": positive_class,
        "n": int(len(y_fail)),
        "n_positive": int(y_fail.sum()),
        "n_fail": int(y_fail.sum()),          # 兼容旧字段名
        "base_rate": base,
        "pr_auc": ap,
        # 归一化后的 PR-AUC: 随机猜的期望就是 base_rate, 所以这才是"相对增益"
        "pr_auc_over_base": (ap / base) if (base and not np.isnan(ap)) else float("nan"),
    }
    for k in ks:
        out[f"precision@{k}"] = precision_at_k(y_fail, score, k)
        out[f"recall@{k}"] = recall_at_k(y_fail, score, k)
        # lift: 相对随机挑的倍数。precision@k 本身受基础失败率影响,
        # 跨分组比较必须看 lift, 不然失败率高的组天然"表现好"。
        br = out["base_rate"]
        out[f"lift@{k}"] = (out[f"precision@{k}"] / br) if br else float("nan")
    return out


def by_group(y_fail: np.ndarray, score: np.ndarray, groups: np.ndarray,
             ks=(10, 20, 50, 100), min_n: int = 200,
             positive_class: str = "failure") -> dict:
    """分组性能 (SPEC §3.3 第 3 项): 按实验室/年份/物种分组算, 报最差组与最好组的差距。

    min_n: 样本少于这个数的组不参与"最差/最好"的评选 —— 20 个样本的组
    precision@20 只能是 0 或 1, 拿它当最差组是噪声不是发现。
    """
    res: dict[str, dict] = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < min_n:
            continue
        res[str(g)] = evaluate(y_fail[m], score[m], ks, positive_class)
    # 功效诊断: 把"总共有多少组 / 多少组够大 / 最大的几组多大"也记下来。
    # 不记的话, 报告里关于"物种维度功效不足"的论述只能靠手写数字, 会与流水线脱节
    # (2026-10-02 实查: gate2 报告正文手写的 525 个物种 / 1 个达标, 与同一页表格的
    #  2 个达标自相矛盾, 就是这么来的)。
    sizes = {str(g): int((groups == g).sum()) for g in np.unique(groups)}
    top = sorted(sizes.items(), key=lambda kv: -kv[1])[:5]
    out: dict = {"groups": res, "n_groups_evaluated": len(res),
                 "n_groups_total": len(sizes), "min_group_n": int(min_n),
                 "n_records": int(len(groups)),
                 "largest_groups": [{"group": g, "n": n} for g, n in top]}
    for k in ks:
        vals = {g: v[f"precision@{k}"] for g, v in res.items()
                if not np.isnan(v[f"precision@{k}"])}
        lifts = {g: v[f"lift@{k}"] for g, v in res.items()
                 if not np.isnan(v[f"lift@{k}"])}
        if vals:
            best, worst = max(vals, key=vals.get), min(vals, key=vals.get)
            out[f"precision@{k}_best"] = {"group": best, "value": vals[best]}
            out[f"precision@{k}_worst"] = {"group": worst, "value": vals[worst]}
            out[f"precision@{k}_ratio_best_over_worst"] = (
                vals[best] / vals[worst] if vals[worst] > 0 else float("inf")
            )
        if lifts:
            bl, wl = max(lifts, key=lifts.get), min(lifts, key=lifts.get)
            out[f"lift@{k}_best"] = {"group": bl, "value": lifts[bl]}
            out[f"lift@{k}_worst"] = {"group": wl, "value": lifts[wl]}
    # 归一化 PR-AUC 的组间离散度 —— 跨组不稳定本身就是要报告的发现
    nps = {g: v["pr_auc_over_base"] for g, v in res.items()
           if not np.isnan(v.get("pr_auc_over_base", float("nan")))}
    if nps:
        b, w = max(nps, key=nps.get), min(nps, key=nps.get)
        out["pr_auc_over_base_best"] = {"group": b, "value": nps[b]}
        out["pr_auc_over_base_worst"] = {"group": w, "value": nps[w]}
        out["pr_auc_over_base_spread"] = nps[b] - nps[w]
    return out
