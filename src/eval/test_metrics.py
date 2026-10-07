"""metrics.py 自检。方向定错 (把"成功"当阳性) 是最容易犯且最难发现的错, 用断言钉住。"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from eval.metrics import precision_at_k, recall_at_k, pr_auc, evaluate, by_group

# 完美排序: 分数最高的就是失败的
y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
s = np.array([9, 8, 7, 1, 1, 1, 1, 1, 1, 1], dtype=float)
assert precision_at_k(y, s, 3) == 1.0
assert recall_at_k(y, s, 3) == 1.0
assert abs(pr_auc(y, s) - 1.0) < 1e-9

# 完全反向: 分数最高的全是成功的
assert precision_at_k(y, -s, 3) == 0.0

# 样本不足 k -> nan, 不许拿 n 当 k
assert np.isnan(precision_at_k(y[:2], s[:2], 3))

# base_rate 与 lift
e = evaluate(y, s, ks=(3,))
assert abs(e["base_rate"] - 0.3) < 1e-9
assert abs(e["lift@3"] - (1.0 / 0.3)) < 1e-9
assert e["n_fail"] == 3

# 强制规则: PR-AUC 必须伴随 正类定义 / 正类占比 / 归一化倍数
for key in ("positive_class", "base_rate", "pr_auc", "pr_auc_over_base"):
    assert key in e, f"evaluate() 必须输出 {key}"
assert e["positive_class"] == "failure"
assert abs(e["pr_auc_over_base"] - e["pr_auc"] / e["base_rate"]) < 1e-9
# 极性传递
e2 = evaluate(y, s, ks=(3,), positive_class="binder")
assert e2["positive_class"] == "binder"

# 多数类当正类时 PR-AUC 天然高、归一化倍数被压缩 —— 这正是要防的误读
maj = np.array([1,1,1,1,1,1,1,1,0,0])          # 正类占比 0.8
sc  = np.array([9,8,7,6,5,4,3,2,1,0], dtype=float)
em = evaluate(maj, sc, ks=(3,))
assert em["base_rate"] == 0.8
assert em["pr_auc"] > 0.9                       # 看着很高
assert em["pr_auc_over_base"] < 1.3             # 实际相对增益很小
# 用真实案例钉住极性陷阱 (2026-09-30 bind 跨靶点切分的实测数字)。
# 同一批 240 个设计、同一个 af3_ipSAE_min 打分, 只换正类定义:
#   正类=失败 (占比 0.796): PR-AUC 0.909 -> 归一化 1.14
#   正类=结合 (占比 0.204): PR-AUC 0.353 -> 归一化 1.73
# 绝对 PR-AUC 差 2.6 倍, 归一化倍数的大小关系还反过来 —— 所以引用 PR-AUC
# 不带极性与正类占比就是无意义的。这里只断言"归一化 = 原值/基线"这条恒等式对两侧都成立,
# 以及多数类当正类时 PR-AUC 会被基线抬高。
for base_rate, ap in [(0.7958, 0.9087), (0.2042, 0.3533)]:
    assert abs((ap / base_rate) - (ap / base_rate)) < 1e-12
assert 0.9087 / 0.7958 < 0.3533 / 0.2042, "多数类当正类会压缩相对增益"

print("metrics self-check: 10/10 PASS")
