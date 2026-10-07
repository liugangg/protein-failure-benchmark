"""英文稿与中文稿的同步守卫 —— 投稿前必须跑通。

要防的失败 (这是本项目已经吃过三次的那一类, 见 src/labels/provenance.py 的头注释):
  中文稿由 src/eval/preprint.py 从冻结产物生成, 数字一旦变动它会自动更新;
  **英文稿是手工产物 (由外部环境翻译), 不在生成链上**。
  所以任何数字变动都不会传到英文稿 —— 而两份稿子看起来都"正常", 不会报错。
  如果拿一份陈旧的英文稿去投, 投出去的数字与仓库里的数据对不上。

机制: 给英文稿打 .prov.json, 把**中文稿**登记为它的上游输入。
  - 中文稿一变 -> require() 硬失败 -> 必须重译或显式确认
  - 英文稿被改 -> 产物自身指纹不符 -> 同样硬失败 (改完要重新 stamp)

用法:
    python src/eval/check_en_sync.py            # 校验 (投稿前跑这个)
    python src/eval/check_en_sync.py --stamp    # 重译后重新登记
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from labels import provenance  # noqa: E402

ZH = pathlib.Path("reports/default/PREPRINT_biorxiv.md")
EN = pathlib.Path("reports/default/PREPRINT_biorxiv_EN.md")

NUM = re.compile(r"\d[\d,]*\.?\d*")


def _nums(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in NUM.findall(text):
        v = m.rstrip(".").replace(",", "")
        if v:
            out[v] = out.get(v, 0) + 1
    return out


def _body(text: str) -> str:
    """只取科学正文 (从 "## Abstract" / "## 摘要" 起)。

    为什么要切掉前面那段: 英文稿的署名块含投稿元数据里的数字 (单位编号、门牌号、
    邮编), 中文稿没有也不该有 —— 它们不是论文内容。
    若不切, 每填一次单位信息数值集合检查就会误报, 而那正是"守卫喊太多狼"
    继而被无视的典型路径。切点取第一个二级标题, 署名块在它之前。
    """
    m = re.search(r"^## (?:Abstract|摘要)\s*$", text, re.M)
    return text[m.start():] if m else text


def number_diff() -> tuple[set[str], set[str]]:
    """两稿**科学正文**的数值集合差异。

    译文允许 token 次数不同 (三篇 -> three), 但**不同值的集合必须一致**。
    署名/单位等投稿元数据不参与比对 (见 _body)。
    """
    a = _nums(_body(ZH.read_text(encoding="utf8")))
    b = _nums(_body(EN.read_text(encoding="utf8")))
    return set(a) - set(b), set(b) - set(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", action="store_true",
                    help="重译后登记: 把当前中文稿记为英文稿的上游")
    args = ap.parse_args()

    for p in (ZH, EN):
        if not p.exists():
            raise SystemExit(f"!! {p} 不存在")

    if args.stamp:
        only_zh, only_en = number_diff()
        if only_zh or only_en:
            raise SystemExit(
                f"!! 拒绝登记: 两稿的数值集合不一致\n"
                f"   只在中文稿: {sorted(only_zh)[:20]}\n"
                f"   只在英文稿: {sorted(only_en)[:20]}\n"
                f"   先把译文的数字对齐, 再 --stamp。")
        out = provenance.stamp(
            EN, [ZH], produced_by="人工/外部翻译 (不在生成链上)",
            extra={"note": "英文稿是手工产物。中文稿由 src/eval/preprint.py 生成; "
                           "中文稿一变本指纹即失效, 必须重译后重新 --stamp。",
                   "checked_at_stamp": "两稿**科学正文**数值集合一致 "
                                       "(署名/单位等投稿元数据不参与比对)"})
        print(f"已登记 {out}")
        print(f"  上游 = {ZH} (sha256 变则硬失败)")
        return

    # ── 校验模式 ──
    provenance.require([EN], strict=True)
    only_zh, only_en = number_diff()
    if only_zh or only_en:
        raise SystemExit(
            f"!! 指纹一致但数值集合不一致 (说明 stamp 时就没对齐)\n"
            f"   只在中文稿: {sorted(only_zh)[:20]}\n"
            f"   只在英文稿: {sorted(only_en)[:20]}")
    print("英文稿同步检查: PASS")
    print(f"  中文稿指纹与登记一致; 两稿数值集合一致 "
          f"({len(_nums(ZH.read_text(encoding='utf8')))} 个不同值)")


if __name__ == "__main__":
    main()
