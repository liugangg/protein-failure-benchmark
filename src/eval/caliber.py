"""产物口径目录。

问题: 预处理口径 (是否剥构建体残留 / 是否用序列长度) 会改变所有模型数字, 而原来所有口径
共用 `reports/` 下同名文件, 后跑的覆盖先跑的。上一轮就出过 "reports/baseline_gbdt.json
是旧默认口径" 这种混淆, 写稿时极易把旧数字当新的引用。

解法 (SPEC §5 原文): reports/ 下分口径目录, 不原地改名覆盖。
  reports/default/                        新默认 = 剥标签 + 去长度, **写稿主数字读这里**
  reports/ablation_keeptags_keeplen/      旧口径, 只供消融节
  reports/ablation_keeptags/              只保留标签
  reports/ablation_keeplen/               只保留长度
两个目录里文件同名, 靠目录区分。每个产物带 .prov.json 指纹。
"""
from __future__ import annotations

import pathlib

REPORTS = pathlib.Path("reports")
DEFAULT_DIRNAME = "default"


def caliber_name(strip_tags: bool, use_length: bool) -> str:
    """口径目录名。默认口径 (剥标签且不用长度) 叫 default, 其余按偏离项命名。"""
    if strip_tags and not use_length:
        return DEFAULT_DIRNAME
    parts = []
    if not strip_tags:
        parts.append("keeptags")
    if use_length:
        parts.append("keeplen")
    return "ablation_" + "_".join(parts)


def caliber_dir(strip_tags: bool, use_length: bool, *, mkdir: bool = True) -> pathlib.Path:
    d = REPORTS / caliber_name(strip_tags, use_length)
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return d


def caliber_meta(strip_tags: bool, use_length: bool) -> dict:
    """写进每个产物, 让任何数字都能追到口径。"""
    return {
        "caliber": caliber_name(strip_tags, use_length),
        "strip_construct_residues": bool(strip_tags),
        "uses_sequence_length_feature": bool(use_length),
        "is_paper_default": bool(strip_tags and not use_length),
        "note": ("默认口径 = 剥构建体残留 + 不用序列长度; 依据见 configs/baseline_gbdt.yaml "
                 "与 reports/tag_confound_analysis.json (1b Simpson 反转)"),
    }
