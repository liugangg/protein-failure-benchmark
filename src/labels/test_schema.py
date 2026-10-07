"""schema.py 自检。SPEC §2.3 的 -1 / 0 区分是硬性要求, 用可执行断言钉住。"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from labels.schema import (
    STAGES, make_label, validate_label, fill_success_prefix, summarize,
    empty_label, label_to_dict,
)

def expect_raise(fn, what):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"应当抛错但没有: {what}")

# 1. express 失败 -> soluble 及之后必须是 -1, 不是 0
lab = make_label(reached="clone", failed_at="express")
assert lab == [1, 0, -1, -1, -1, -1], lab

# 2. 全流程成功
assert make_label(reached="bind") == [1, 1, 1, 1, 1, 1]

# 3. 只知道 stable 一位 (DS2 稳定性数据的形态)
lab = make_label(extra={"stable": 0})
assert lab == [-1, -1, -1, -1, 0, -1], lab

# 4. 自相矛盾必须炸: failed_at 不晚于 reached
expect_raise(lambda: make_label(reached="soluble", failed_at="express"), "矛盾标签")

# 5. 失败后出现观测值必须炸
expect_raise(lambda: validate_label([1, 0, 1, -1, -1, -1]), "失败后有成功")
expect_raise(lambda: validate_label([1, 0, 0, -1, -1, -1]), "失败后有失败")

# 6. 非法值必须炸
expect_raise(lambda: validate_label([1, 2, -1, -1, -1, -1]), "非法值 2")
expect_raise(lambda: validate_label([1, 1, 1]), "长度不对")

# 7. 成功前缀回填: 只回填成功, 不回填失败
assert fill_success_prefix([-1, -1, 1, -1, -1, -1]) == [1, 1, 1, -1, -1, -1]
assert fill_success_prefix([-1, 0, -1, -1, -1, -1]) == [-1, 0, -1, -1, -1, -1]
# 成功前缀允许不连续: 不同源观测不同阶段子集 (Adaptyv BLI 就是这个形状)
validate_label([1, 1, -1, -1, -1, 1])
validate_label([1, 1, -1, -1, -1, 0])
validate_label([-1, -1, 1, 0, -1, -1])
# 但失败之后仍然不许有任何观测
expect_raise(lambda: validate_label([1, 1, -1, 0, -1, 1]), "失败后有成功")

# 8. 三态汇总
c = summarize([
    make_label(reached="clone", failed_at="express"),
    make_label(reached="bind"),
    make_label(extra={"stable": 0}),
])
assert c["express"] == {"success": 1, "fail": 1, "unobserved": 1}, c["express"]
assert c["stable"] == {"success": 1, "fail": 1, "unobserved": 1}, c["stable"]
assert c["clone"] == {"success": 2, "fail": 0, "unobserved": 1}, c["clone"]

assert empty_label() == [-1] * 6
assert label_to_dict([1, 0, -1, -1, -1, -1])["express"] == 0
assert len(STAGES) == 6

print("schema self-check: 8/8 PASS")
