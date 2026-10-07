"""机器相关路径的集中出口 —— 真实路径不进版本库。

为什么: 本仓库对外发布, 而
  (a) `DATA_AVAILABILITY.md` 明写 OIH 自有管线 (DS6) 的"内部路径不随本仓库发布";
  (b) 稿件的 Competing interests 节又让读者去看 `reports/ds6_oih_inventory.md`,
      所以那份报告**必须**发布 —— 不能靠排除文件解决, 只能脱敏。
  (c) 清理要赶在首次提交之前。先提交再清理的话, 旧 blob 按原 SHA 仍可访问一段时间,
      只能等垃圾回收 —— 涉及他人信息时不该把干净程度交给回收时间表。

用法: 真实值写在 `configs/local_paths.yaml` (已 gitignore);
      仓库里只有 `configs/local_paths.yaml.example`。
      缺键时硬失败并说清要填什么, 不静默回落到某个默认路径。
"""
from __future__ import annotations

import pathlib

import yaml

_CFG = pathlib.Path("configs/local_paths.yaml")
_EXAMPLE = pathlib.Path("configs/local_paths.yaml.example")

# 发布文本里用的占位符: 报告/文档一律印这个, 不印真实路径
PLACEHOLDER = {
    "oih_outputs": "<OIH_OUTPUTS_DIR>",
    "oih_tasks": "<OIH_TASKS_DIR>",
    "esm2_650m_ckpt": "<ESM2_650M_CHECKPOINT>",
}

_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        if not _CFG.exists():
            raise SystemExit(
                f"!! 缺 {_CFG}。这是机器相关路径的本地配置, 有意不进版本库。\n"
                f"   照 {_EXAMPLE} 复制一份并填上本机实际路径:\n"
                f"     cp {_EXAMPLE} {_CFG}")
        _cache = yaml.safe_load(_CFG.read_text()) or {}
    return _cache


def get(key: str) -> pathlib.Path:
    """取真实路径 (供脚本运行用)。缺键硬失败。"""
    v = _load().get(key)
    if not v:
        raise SystemExit(f"!! {_CFG} 里缺 `{key}`。见 {_EXAMPLE} 的说明。")
    return pathlib.Path(v)


def shown(key: str) -> str:
    """取**发布文本里应该出现的字符串** —— 占位符, 不是真实路径。"""
    return PLACEHOLDER.get(key, f"<{key.upper()}>")
