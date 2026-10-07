"""L1 第一步: 用**冻结**的 ESM-2 650M 抽 mean-pooled embedding。

L1 的定义是 "ESM-2 650M 冻结 + 线性头"。冻结版只前向, 所以把 embedding 抽出来
存盘, 之后线性头/对抗头可以在 CPU 上反复训, 不用每次重跑 PLM —— 省掉大量无谓的 GPU 时间。

权重来源: 本地 fair-esm checkpoint, 路径由 configs/local_paths.yaml 的 esm2_650m_ckpt 给出
  HuggingFace 直连不通 (实测 config.json 无响应), hf-mirror 只回 308 跳转。
  这个 checkpoint 是 fair-esm 格式 (顶层键 args/cfg/model, 570 个张量),
  不是 HF 格式, 所以用 fair-esm 的 API 加载而不是 transformers。
  官方用法 (facebookresearch/esm README):
      model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
      batch_converter = alphabet.get_batch_converter()
      results = model(batch_tokens, repr_layers=[33])
  我们改成从本地文件加载, 避免 torch.hub 去联网。

池化: 对**非 padding 且非特殊 token** 的位置做均值 (SPEC §2 的 pooling: mean)。
  必须排除 BOS/EOS/pad, 否则短序列的均值被特殊 token 稀释, 长度又偷偷进来一次。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import paths  # noqa: E402
from models.gpu_guard import apply_env, setup_device  # noqa: E402

CKPT = paths.get("esm2_650m_ckpt")
RECORDS = pathlib.Path("data/processed/records.parquet")
CONFIG = pathlib.Path("configs/stage3_train.yaml")
OUTDIR = pathlib.Path("data/interim/esm_embeddings")
REPR_LAYER = 33


def load_model(device):
    import argparse as _argparse
    import esm
    import torch
    if not CKPT.exists():
        raise SystemExit(f"{CKPT} 不存在")

    # torch 2.6 起 torch.load 的 weights_only 默认变 True, 而 fair-esm 的 checkpoint
    # 顶层带 argparse.Namespace (args / cfg), 会被拒载。
    # 用官方推荐的**窄口径**修法: 只把 argparse.Namespace 加进 safe globals。
    # 不用 weights_only=False —— 那会允许任意代码执行, 即使本文件来自可信本地缓存,
    # 也没有理由开这么大的口子。
    torch.serialization.add_safe_globals([_argparse.Namespace])
    # 官方的 load_model_and_alphabet_local 就是为本地 .pt 准备的, 不联网
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(CKPT))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)          # L1 = 冻结
    n = sum(p.numel() for p in model.parameters())
    print(f"[esm] 加载 {CKPT.name}  参数 {n/1e6:.0f}M  层数 {model.num_layers}  "
          f"repr_layer={REPR_LAYER}")
    return model, alphabet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="express")
    ap.add_argument("--limit", type=int, default=0, help="只抽前 N 条, 冒烟测试用")
    ap.add_argument("--batch-size", type=int, default=0)
    args = ap.parse_args()

    cfg = apply_env()                    # 必须在 import torch 之前
    import torch
    dev, cfg = setup_device(cfg)
    bs = args.batch_size or cfg["levels"]["L1"]["embed_batch_size"]
    max_len = cfg["data"]["max_length"]
    strip = cfg["data"]["caliber"]["strip_construct_residues"]

    col = ("label_soluble_expression" if args.task == "soluble_expression"
           else f"label_{args.task}")
    t = pq.read_table(RECORDS, columns=["record_id", "sequence", "center",
                                        "split_group", "source", col])
    import pyarrow.compute as pc
    # 只抽该任务有观测的记录 (PU: -1 不进损失, 也没必要抽 embedding)
    t = t.filter(pc.not_equal(t.column(col), -1))
    t = t.filter(pc.equal(t.column("source"), "ds1_targettrack"))
    print(f"[esm] 任务 {args.task}: 有观测记录 {t.num_rows:,}")

    # 去冗余: 每个 split_group 一条 (与 GBDT 基线同口径) —— 抽 embedding 很贵, 不抽重复的
    seen: set[int] = set()
    keep: list[int] = []
    groups = t.column("split_group").to_pylist()
    for i, g in enumerate(groups):
        if g not in seen:
            seen.add(g)
            keep.append(i)
    t = t.take(keep)
    print(f"[esm] 按 split_group 去冗余后 {t.num_rows:,} 条")
    if args.limit:
        t = t.slice(0, args.limit)
        print(f"[esm] --limit 截到 {t.num_rows:,} 条")

    seqs = t.column("sequence").to_pylist()
    rids = t.column("record_id").to_pylist()
    if strip:
        # 与 GBDT 同一套 21 类模式, 保证两级用的是同一预处理
        sys.path.insert(0, str(ROOT / "src"))
        import re
        from models.baseline_gbdt import TAG_PATTERNS
        rxs = [(re.compile(p), d) for p, d in TAG_PATTERNS]
        n_ch = 0
        out = []
        for s in seqs:
            cur = s
            for rx, _ in rxs:
                cur = rx.sub("", cur)
            if cur != s:
                n_ch += 1
            out.append(cur if len(cur) >= 20 else s)
        seqs = out
        print(f"[esm] 剥构建体残留: 改动 {n_ch:,} 条 ({n_ch/len(seqs)*100:.1f}%)")

    n_trunc = sum(1 for s in seqs if len(s) > max_len)
    print(f"[esm] 超过 max_length={max_len} 需截断: {n_trunc:,} 条 "
          f"({n_trunc/len(seqs)*100:.2f}%)")   # SPEC §6 要求记录截断比例
    seqs = [s[:max_len] for s in seqs]

    model, alphabet = load_model(dev)
    bc = alphabet.get_batch_converter()
    # 按长度排序再分批, 减少 padding 浪费 (结果与顺序无关, 最后按 record_id 对回去)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    embs = np.zeros((len(seqs), model.embed_dim), dtype=np.float32)

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for st in range(0, len(order), bs):
            idx = order[st:st + bs]
            data = [(str(i), seqs[i]) for i in idx]
            _, _, toks = bc(data)
            toks = toks.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res = model(toks, repr_layers=[REPR_LAYER])
            rep = res["representations"][REPR_LAYER].float()
            # 掩码: 排除 pad / BOS / EOS —— 否则短序列均值被特殊 token 稀释
            mask = (toks != alphabet.padding_idx) & (toks != alphabet.cls_idx) \
                   & (toks != alphabet.eos_idx)
            m = mask.unsqueeze(-1).to(rep.dtype)
            pooled = (rep * m).sum(1) / m.sum(1).clamp(min=1)
            for k, i in enumerate(idx):
                embs[i] = pooled[k].cpu().numpy()
            done += len(idx)
            if st % (bs * 50) == 0:
                el = time.time() - t0
                print(f"  {done:,}/{len(seqs):,}  {el:.0f}s  "
                      f"{done/max(el,1e-9):.1f} seq/s  "
                      f"峰值显存 {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_p = OUTDIR / f"{args.task}_esm2_650M_frozen.parquet"
    tbl = pa.table({
        "record_id": rids,
        "center": t.column("center").to_pylist(),
        "split_group": t.column("split_group").to_pylist(),
        "label": t.column(col).to_pylist(),
        "emb": [e.tolist() for e in embs],
    })
    pq.write_table(tbl, out_p)
    meta = {"task": args.task, "n": len(seqs), "dim": int(model.embed_dim),
            "repr_layer": REPR_LAYER, "max_length": max_len,
            "truncated": n_trunc, "truncated_frac": round(n_trunc / len(seqs), 5),
            "strip_construct_residues": strip,
            "checkpoint": str(CKPT), "seconds": round(time.time() - t0, 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (OUTDIR / f"{args.task}_esm2_650M_frozen.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_p}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
