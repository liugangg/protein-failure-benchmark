"""L2: ESM-2 650M + LoRA 微调。分级中的 L2。

定位 (刘刚刚 2026-09-30): **收尾**, 只跑一轮合理超参, 不做超参搜索。
目的是把"模型规模救不了被身份通道污染的数据"从推断变成实测, 不是让它赢。
报告口径与 L1 完全对齐: 逐折 / 折 2 按低基础率规则用 PR-AUC/base / 对抗与非对抗都跑。

一个省算力的关键观察: cross 的训练集 `cross_m` **只依赖外层折 j, 不依赖内层折 i**
(见 src/splits/center_folds.py 的定义)。所以每个外层折只需训练**一个** cross 模型,
再在该折的 3 个内层测试集上分别评估 —— 与 L1 的报告口径完全一致, 但训练次数从 12 降到 4。

LoRA 作用位置: fair-esm 的 q_proj/k_proj/v_proj/out_proj (实查: 各 33 个)。
  注意**不是** HF ESM 的 query/key/value/dense —— 我们用的是 fair-esm 格式的本地 checkpoint。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import paths  # noqa: E402
from eval.caliber import caliber_dir, caliber_meta  # noqa: E402
from eval.metrics import evaluate  # noqa: E402
from eval.run_log import write_run_log  # noqa: E402
from models.gpu_guard import apply_env, setup_device  # noqa: E402

CKPT = paths.get("esm2_650m_ckpt")
RECORDS = pathlib.Path("data/processed/records.parquet")
EMB = pathlib.Path("data/interim/esm_embeddings")
FOLDS = pathlib.Path("data/processed/splits/center_folds.parquet")
CONFIG = pathlib.Path("configs/stage3_train.yaml")
K_IN = 3
REPR_LAYER = 33


class GradReverse:
    @staticmethod
    def apply(x, lam: float):
        import torch

        class _F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, g):
                return -lam * g

        return _F.apply(x)


def build(cfg, dev, n_centers: int, adversarial: bool):
    import argparse as _ap
    import esm
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    torch.serialization.add_safe_globals([_ap.Namespace])
    base, alphabet = esm.pretrained.load_model_and_alphabet_local(str(CKPT))
    lc = cfg["levels"]["L2"]["lora"]
    peft_cfg = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                          target_modules=lc["target_modules"], bias="none")
    enc = get_peft_model(base, peft_cfg)
    tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in enc.parameters())
    print(f"[lora] 可训练 {tr/1e6:.2f}M / 总 {tot/1e6:.0f}M = {tr/tot*100:.3f}%")

    d = base.embed_dim
    # 与 L1 同构: emb -> proj -> {head, 反转->adv}, 非对抗版也走 proj 排除架构差
    proj = nn.Sequential(nn.Linear(d, 256), nn.ReLU())
    head = nn.Linear(256, 1)
    adv = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, n_centers)) if adversarial else None
    enc, proj, head = enc.to(dev), proj.to(dev), head.to(dev)
    if adv is not None:
        adv = adv.to(dev)
    return enc, alphabet, proj, head, adv


def pooled(enc, alphabet, toks, dev):
    import torch
    out = enc(toks, repr_layers=[REPR_LAYER])
    rep = out["representations"][REPR_LAYER]
    mask = (toks != alphabet.padding_idx) & (toks != alphabet.cls_idx) \
           & (toks != alphabet.eos_idx)
    m = mask.unsqueeze(-1).to(rep.dtype)
    return (rep * m).sum(1) / m.sum(1).clamp(min=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="express")
    ap.add_argument("--no-adversarial", dest="adv", action="store_false", default=None)
    ap.add_argument("--smoke", type=int, default=0, help="只用 N 条训练, 实测吞吐用")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    cfg = apply_env()
    import torch
    import torch.nn as nn
    dev, cfg = setup_device(cfg)
    adversarial = cfg["adversarial"]["enabled"] if args.adv is None else args.adv
    lv = cfg["levels"]["L2"]
    max_len = cfg["data"]["max_length"]
    ks = tuple(cfg["eval"]["ks"])

    # 复用 L1 抽 embedding 时确定的那批记录 (同样的去冗余/剥标签/截断), 保证 L1/L2 可比
    ep = EMB / f"{args.task}_esm2_650M_frozen.parquet"
    if not ep.exists():
        raise SystemExit(f"{ep} 不存在, 先跑 esm_embed.py --task {args.task}")
    et = pq.read_table(ep)
    rids = et.column("record_id").to_pylist()
    y = (np.array(et.column("label").to_pylist()) == 0).astype(np.int8)
    centers = np.array(et.column("center").to_pylist())
    groups = np.array(et.column("split_group").to_pylist())

    import re
    from models.baseline_gbdt import TAG_PATTERNS
    rt = pq.read_table(RECORDS, columns=["record_id", "sequence"])
    seq_of = dict(zip(rt.column("record_id").to_pylist(), rt.column("sequence").to_pylist()))
    rxs = [re.compile(p) for p, _ in TAG_PATTERNS]
    seqs = []
    for r in rids:
        s = seq_of[r]
        cur = s
        for rx in rxs:
            cur = rx.sub("", cur)
        seqs.append((cur if len(cur) >= 20 else s)[:max_len])
    print(f"[L2] {args.task}: {len(y):,} 条  阴性 {y.mean()*100:.1f}%  对抗={adversarial}")

    ft = pq.read_table(FOLDS)
    fold_of = dict(zip(ft.column("center").to_pylist(), ft.column("center_fold").to_pylist()))
    uniq_c = sorted(set(centers))
    cidx = {c: i for i, c in enumerate(uniq_c)}
    ctr_i = np.array([cidx[c] for c in centers])

    rows = []
    all_preds: list[dict] = []
    folds = args.folds if args.folds is not None else range(cfg["eval"]["n_center_folds"])
    for j in folds:
        held = [c for c, f in fold_of.items() if f == j]
        te_m = np.isin(centers, held)
        if te_m.sum() == 0 or y[te_m].sum() < cfg["eval"]["min_test_positives"]:
            print(f"  折 {j}: 阴性不足, 跳过 (留出中心 {','.join(held)})")
            continue
        touched = set(groups[te_m])
        cross_m = ~te_m & ~np.isin(groups, list(touched))
        tr_idx = np.where(cross_m)[0]
        if args.smoke:
            tr_idx = tr_idx[:args.smoke]

        enc, alphabet, proj, head, adv = build(cfg, dev, len(uniq_c), adversarial)
        bc = alphabet.get_batch_converter()
        params = [p for p in enc.parameters() if p.requires_grad] + \
                 list(proj.parameters()) + list(head.parameters()) + \
                 (list(adv.parameters()) if adv is not None else [])
        opt = torch.optim.AdamW(params, lr=lv["lr"], weight_decay=0.01)
        pos = float(y[tr_idx].sum()); neg = float(len(tr_idx) - pos)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=dev))
        advf = nn.CrossEntropyLoss()
        bs, ga, eps = lv["batch_size"], lv["grad_accum"], lv["epochs"]
        budget, max_b = lv["attn_budget"], lv["max_batch"]
        steps = max((len(tr_idx) // max(bs, 1) + 1) * eps // ga, 1)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lv["lr"], total_steps=max(steps, 1),
            pct_start=lv["warmup_ratio"], anneal_strategy="cos")
        rng = np.random.default_rng(cfg["seed"])
        t0 = time.time()
        enc.train(); proj.train(); head.train()
        seen = 0
        for e in range(eps):
            order = rng.permutation(len(tr_idx))
            # 长度排序分桶减少 padding, 桶内打乱
            order = order[np.argsort([len(seqs[tr_idx[o]]) for o in order], kind="stable")]
            # 按 batch*len^2 预算动态分批 —— 固定 batch_size 会在长序列批上 OOM
            # (实测 batch=8 在 len~1024 时炸, 因为 fair-esm 的 softmax 强制 fp32)。
            batches = []
            cur: list[int] = []
            cur_max = 0
            for o in order:
                i = tr_idx[o]
                L = len(seqs[i]) + 2          # +2: BOS/EOS
                nm = max(cur_max, L)
                if cur and ((len(cur) + 1) * nm * nm > budget or len(cur) + 1 > max_b):
                    batches.append(cur); cur, cur_max = [i], L
                else:
                    cur.append(i); cur_max = nm
            if cur:
                batches.append(cur)
            for bi, sel_list in enumerate(batches):
                sel = np.array(sel_list)
                _, _, toks = bc([(str(i), seqs[i]) for i in sel])
                toks = toks.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = proj(pooled(enc, alphabet, toks, dev).float())
                    loss = lossf(head(z),
                                 torch.as_tensor(y[sel], dtype=torch.float32,
                                                 device=dev).unsqueeze(1))
                    if adv is not None:
                        loss = loss + advf(adv(GradReverse.apply(
                            z, cfg["adversarial"]["grad_reversal_lambda"])),
                            torch.as_tensor(ctr_i[sel], dtype=torch.long, device=dev))
                (loss / ga).backward()
                if (bi + 1) % ga == 0:
                    opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
                seen += len(sel)
                if seen % 1600 < len(sel):
                    el = time.time() - t0
                    print(f"    折{j} ep{e} {seen:,}/{len(tr_idx)*eps:,} "
                          f"{el:.0f}s {seen/max(el,1e-9):.1f} seq/s "
                          f"显存 {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
        train_s = time.time() - t0
        if args.smoke:
            print(f"\n[smoke] 折{j} 训练 {len(tr_idx):,} 条 x {eps} epoch 用 {train_s:.0f}s "
                  f"= {seen/train_s:.1f} seq/s, 峰值 {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
            print(f"[smoke] 推算全量 cross 训练集 {int(cross_m.sum()):,} 条 x {eps} epoch "
                  f"需 {int(cross_m.sum())*eps/(seen/train_s)/60:.0f} 分钟/折")
            return

        # 推理: 该外层折的 3 个内层测试集共用这一个模型
        enc.eval(); proj.eval(); head.eval()
        gi = {g: (hash((int(g), cfg["seed"])) % K_IN) for g in sorted(touched)}
        inner = np.array([gi.get(g, -1) for g in groups])
        sub = []
        for i in range(K_IN):
            te_i = te_m & (inner == i)
            if te_i.sum() < cfg["eval"]["min_test_n"] or \
                    y[te_i].sum() < cfg["eval"]["min_test_positives"]:
                continue
            idxs = np.where(te_i)[0]
            sc = np.zeros(len(idxs), dtype=np.float32)
            with torch.no_grad():
                inf_batches = []
                cur2: list[int] = []
                cm = 0
                for x in sorted(idxs, key=lambda q: len(seqs[q])):
                    L = len(seqs[x]) + 2
                    nm = max(cm, L)
                    if cur2 and ((len(cur2) + 1) * nm * nm > budget * 2 or len(cur2) + 1 > 32):
                        inf_batches.append(cur2); cur2, cm = [x], L
                    else:
                        cur2.append(x); cm = nm
                if cur2:
                    inf_batches.append(cur2)
                pos_in = {x: k for k, x in enumerate(idxs)}
                for sel in inf_batches:
                    sel = np.array(sel)
                    _, _, toks = bc([(str(x), seqs[x]) for x in sel])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z = proj(pooled(enc, alphabet, toks.to(dev), dev).float())
                        vals = torch.sigmoid(head(z)).squeeze(1).float().cpu().numpy()
                    for k2, x in enumerate(sel):
                        sc[pos_in[x]] = vals[k2]
            ev = evaluate(y[te_i], sc, ks=ks)
            sub.append({"inner_fold": i, "n_test": int(te_i.sum()),
                        "n_test_positive": int(y[te_i].sum()), "cross": ev})
            # 逐条预测落盘 —— 缺这个会导致任何后验分析 (如自助法置信区间) 都必须重训,
            # 实测代价 33 分钟/折。2026-10-01 补上。
            all_preds.append({
                "fold": j, "inner": i,
                "record_id": [rids[x] for x in idxs],
                "center": centers[te_i].tolist(),
                "split_group": groups[te_i].tolist(),   # 自助法要按簇重采样, 必须带它
                "y_fail": y[te_i].tolist(),
                "score": sc.tolist()})
            print(f"  折 {j}.{i} ({','.join(held)[:28]}): test={int(te_i.sum()):,} "
                  f"base={ev['base_rate']:.3f} cross lift@100={ev['lift@100']:.2f} "
                  f"PRAUC/base={ev['pr_auc_over_base']:.2f}")
        if sub:
            r = {"fold": j, "held_centers": held, "inner_folds": sub,
                 "train_seconds": round(train_s, 1),
                 "n_train_cross": int(cross_m.sum())}
            for met in ("lift@100", "pr_auc_over_base"):
                r[f"cross_{met}_mean"] = float(np.mean([s["cross"][met] for s in sub]))
            rows.append(r)
        del enc, proj, head, adv
        torch.cuda.empty_cache()

    cmeta = caliber_meta(True, False)
    name = f"L2_esm2_650M_lora_{args.task}" + ("_adv" if adversarial else "") + ".json"
    outp = caliber_dir(True, False) / name
    # ── 合并写, 不覆盖 ──
    # 用 --folds 4 补跑一折时, 覆盖写会把已有的折 1-3 静默删掉。
    # 本项目已经在 bootstrap_ci.py 上踩过同一个坑 (2026-10-02), 这里一并修掉。
    if outp.exists():
        prev = json.loads(outp.read_text()).get("folds", [])
        have = {r["fold"] for r in rows}
        kept = [r for r in prev if r["fold"] not in have]
        if kept:
            print(f"[merge] 保留此前已跑的折 {sorted(r['fold'] for r in kept)}, "
                  f"本次新跑 {sorted(have)}")
        rows = sorted(kept + rows, key=lambda r: r["fold"])
    outp.write_text(json.dumps({"caliber": cmeta, "level": "L2", "task": args.task,
                                "adversarial": adversarial, "hparams": lv,
                                "folds": rows}, indent=2, ensure_ascii=False, default=float))
    print(f"\n{'折':>4}{'留出中心':<40}{'base':>7}{'lift@100':>10}{'PRAUC/base':>12}")
    for r in rows:
        b = float(np.mean([s["cross"]["base_rate"] for s in r["inner_folds"]]))
        print(f"{r['fold']:>4}{','.join(r['held_centers'])[:38]:<40}{b:>7.3f}"
              f"{r['cross_lift@100_mean']:>10.2f}{r['cross_pr_auc_over_base_mean']:>12.2f}")
    if all_preds:
        import pyarrow as pa
        flat = {k: [] for k in ("fold", "inner", "record_id", "center",
                                "split_group", "y_fail", "score")}
        for b in all_preds:
            n = len(b["record_id"])
            flat["fold"] += [b["fold"]] * n
            flat["inner"] += [b["inner"]] * n
            for k in ("record_id", "center", "split_group", "y_fail", "score"):
                flat[k] += b[k]
        pf = outp.with_name(outp.stem + "_preds.parquet")
        if pf.exists():          # 同样合并: 丢掉本次重跑过的折, 其余保留
            t = pq.read_table(pf)
            old_d = {c: t.column(c).to_pylist() for c in t.column_names}
            new_folds = set(flat["fold"])
            keep = [i for i, f in enumerate(old_d["fold"]) if f not in new_folds]
            if keep:
                print(f"[merge] 预测落盘保留此前 {len(keep):,} 行 "
                      f"(折 {sorted(set(old_d['fold'][i] for i in keep))})")
                for k in flat:
                    flat[k] = [old_d[k][i] for i in keep] + flat[k]
        pq.write_table(pa.table(flat), pf)
        print(f"wrote {pf}  ({len(flat['fold']):,} 行, "
              f"折 {sorted(set(flat['fold']))})")
    print(f"wrote {outp}")
    write_run_log(f"L2_{args.task}" + ("_adv" if adversarial else ""),
                  sources=["ds1_targettrack"], splits_used=["center_folds"],
                  config={"level": "L2", "adversarial": adversarial, **lv, **cmeta},
                  seed=cfg["seed"],
                  results={f"fold{r['fold']}": {
                      "cross_lift100": r["cross_lift@100_mean"],
                      "cross_pr_auc_over_base": r["cross_pr_auc_over_base_mean"]} for r in rows},
                  notes="L2 = ESM-2 650M + LoRA; 收尾性质, 单轮超参无搜索")


if __name__ == "__main__":
    main()
