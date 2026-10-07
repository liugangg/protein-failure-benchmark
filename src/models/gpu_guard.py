"""GPU 选择与断言 —— 所有训练脚本启动时第一件事就调 setup_device()。

为什么要单独一个模块 (两次真实事故):
  2026-09-02 CUDA 默认 FASTEST_FIRST 排序, 模型落到错的卡上;
  2026-09-26 容器与宿主 GPU 编号传错, 导致 CUDA 不可用。
  2026-09-30 本机实测: CUDA_VISIBLE_DEVICES=1 不加 CUDA_DEVICE_ORDER 拿到的是
             RTX PRO 5000 (别人正在用 69.6GB), 而不是规格指定的 4090。

所以: 环境变量必须在 **import torch 之前**设好 (torch 初始化时读一次就定了),
并且拿到设备后硬断言卡名, 不符就退出 —— 宁可不跑, 不许抢别人的卡或算错。
"""
from __future__ import annotations

import os
import pathlib

import yaml

CONFIG = pathlib.Path("configs/stage3_train.yaml")


def apply_env(cfg: dict | None = None) -> dict:
    """在 import torch **之前**调用, 设好 CUDA_DEVICE_ORDER / CUDA_VISIBLE_DEVICES。"""
    if cfg is None:
        cfg = yaml.safe_load(CONFIG.read_text())
    hw = cfg["hardware"]
    for k, v in (hw.get("env") or {}).items():
        os.environ[k] = str(v)
    return cfg


def setup_device(cfg: dict | None = None):
    """返回 torch.device, 并断言拿到的是规格指定的那张卡。"""
    cfg = apply_env(cfg)
    hw = cfg["hardware"]
    import torch          # 必须在 apply_env 之后 import

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用 —— 检查 CUDA_VISIBLE_DEVICES 与驱动")
    dev = torch.device(hw["device"])
    name = torch.cuda.get_device_name(dev)
    want = hw["gpu_name_assert"]
    if want not in name:
        raise SystemExit(
            f"拿到的 GPU 是 {name!r}, 规格要求含 {want!r}。\n"
            f"  当前 CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')!r} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}\n"
            f"  CUDA 默认 FASTEST_FIRST 排序与 nvidia-smi 的 PCI 顺序不一致, "
            f"必须设 CUDA_DEVICE_ORDER=PCI_BUS_ID。拒绝在错的卡上跑。")
    cap = torch.cuda.get_device_capability(dev)
    sm = f"{cap[0]}.{cap[1]}"
    if hw.get("expected_sm") and sm != str(hw["expected_sm"]):
        raise SystemExit(f"计算能力 sm_{sm} 与规格的 sm_{hw['expected_sm']} 不符")
    free, total = torch.cuda.mem_get_info(dev)
    print(f"[gpu] {name}  sm_{sm}  空闲 {free/1e9:.1f}/{total/1e9:.1f} GB  "
          f"(CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    if free / total < 0.5:
        print(f"[gpu] ⚠️ 空闲显存不足一半, 这张卡上可能有别人的作业")
    return dev, cfg
