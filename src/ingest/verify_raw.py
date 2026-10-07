"""校验 data/raw 下的原始归档未被改动 —— 把「只读」纪律做成可执行检查。

项目 CLAUDE.md: "data/raw/ — 原始下载, **只读**, 任何情况下不修改"。
光靠自觉不行: 本项目就真的违反过一次 (2026-09-30 我在 data/raw/ds2_tsuboyama/ 里
重命名了一个解压目录)。所以把校验写成脚本, 放进重跑链的第一步。

校验依据是 configs/data_sources.yaml 里登记的 md5 —— 那些值来自各官方页面实查。
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

import yaml

CONFIG = pathlib.Path("configs/data_sources.yaml")

# 归档路径 -> 期望 md5 (来自官方页面, 已登记在 configs/data_sources.yaml)
EXPECTED = {
    "data/raw/ds1_targettrack/TargetTrack-1Jul2017.tar.gz":
        "200012a8a2a11ffd7e370ed142df36c3",
    "data/raw/ds2_tsuboyama/Processed_K50_dG_datasets_7844779.zip":
        "27a0936c8f80e5b18ed330ed6b98a3a6",
    "data/raw/ds5_binder_negatives/dtu_meta_analysis/final_dataset.csv":
        "3a69ee9b0fecf53924a8c6479bac146e",
}

# 存在但**不是** SPEC v2 指定来源的归档: 保留不删 (它也是原始下载), 但要提醒别用。
SUPERSEDED = {
    "data/raw/ds2_tsuboyama/Processed_K50_dG_datasets.zip":
        "Zenodo 7992926 —— 同版本的另一个记录, 核心 CSV 少 match_aaseq / name_original 两列。"
        "SPEC v2 §2.1 指定 7844779, 请用 Processed_K50_dG_datasets_7844779.zip。",
}


def md5(p: pathlib.Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    bad, missing = [], []
    for rel, exp in EXPECTED.items():
        p = pathlib.Path(rel)
        if not p.exists():
            missing.append(rel)
            print(f"  缺失   {rel}")
            continue
        got = md5(p)
        if got == exp:
            print(f"  ✅ OK  {rel}")
        else:
            bad.append((rel, exp, got))
            print(f"  ❌ 不符 {rel}\n         期望 {exp}\n         实得 {got}")

    for rel, why in SUPERSEDED.items():
        if pathlib.Path(rel).exists():
            print(f"  ⚠️  已被取代 {rel}\n         {why}")

    # data/raw 下不该有解压出来的目录 —— 派生数据归 data/interim/extracted/
    # 判据用"含子目录": 解压树才有嵌套结构; 各源自己的下载目录 (dtu_meta_analysis 等)
    # 只是平铺的下载文件, 不算解压产物。按后缀判会把直接下载的 .csv 也误报 (实测踩到)。
    stray = []
    for d in pathlib.Path("data/raw").rglob("*"):
        if not d.is_dir():
            continue
        if any(c.is_dir() for c in d.iterdir()):
            # data/raw/<源>/ 这一层本身允许有子目录 (各源的下载子目录)
            if d.parent == pathlib.Path("data/raw"):
                continue
            stray.append(str(d))
    if stray:
        print("\n  ⚠️  data/raw 下发现解压产物 (应移到 data/interim/extracted/):")
        for s in stray:
            print(f"        {s}")

    if not CONFIG.exists():
        print("\n  ⚠️  configs/data_sources.yaml 不存在")
    else:
        yaml.safe_load(CONFIG.read_text())

    if bad:
        print(f"\n!! {len(bad)} 个归档校验不符 —— data/raw 被改动过, 停下来查清再往下走")
        sys.exit(1)
    if missing:
        print(f"\n!! {len(missing)} 个归档缺失 —— 先按 configs/data_sources.yaml 重新下载")
        sys.exit(1)
    print(f"\ndata/raw 完整性校验通过 ({len(EXPECTED)} 个归档)")


if __name__ == "__main__":
    main()
