"""L1 线性头 (+ 可选对抗去偏) 在冻结 ESM-2 embedding 上训练。

训练规格要点 (完整超参见 configs/stage3_train.yaml):
  §2 L1  = ESM-2 650M 冻结 + 线性头; 继续条件 = cross-center lift 不低于 L0
  §3.1   主评估 = cross-center (GroupKFold by center), 同时报 within-center 作对照,
         **两者测试折完全相同**, 差值就是身份通道贡献量
  §3.2   PU: -1 已在抽 embedding 时排除; 不平衡用类权重, **不做随机下采样**
  §3.3   对抗头预测"来自哪个中心" + 梯度反转, 逼编码器丢掉中心可分信息
  §4     主指标 cross-center lift@k (k=10/20/50/100); PR-AUC 强制附带正类占比与归一化倍数

一个必须说清的范围限制: L1 的编码器是**冻结**的, 所以对抗头只能作用在线性头之前的
embedding 上 —— 它学的是"把 embedding 线性投影到丢掉中心信息的子空间", 而不能真的改变
编码器。真正的对抗去偏要等 L2 (LoRA 可训) 才谈得上。这里的结果只回答一个问题:
**中心信息能不能靠一个线性变换从 ESM embedding 里去掉**。结论会如实标注这一点。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval.caliber import caliber_dir, caliber_meta  # noqa: E402
from eval.metrics import by_group, evaluate  # noqa: E402
from eval.run_log import write_run_log  # noqa: E402
from models.gpu_guard import apply_env, setup_device  # noqa: E402

EMB = pathlib.Path("data/interim/esm_embeddings")
FOLDS = pathlib.Path("data/processed/splits/center_folds.parquet")
CONFIG = pathlib.Path("configs/stage3_train.yaml")


class GradReverse:
    """梯度反转层。前向恒等, 反向乘 -lambda。"""

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


def fit_head(Xtr, ytr, Xte, ctr_tr, n_centers, cfg, dev, adversarial: bool):
    """embedding -> 可训练投影 -> {分类头, 梯度反转 -> 中心判别头}。

    **为什么必须有 proj 这一层** (2026-09-30 抓到的空操作 bug):
      L1 的编码器是冻结的, embedding 是常量张量。如果对抗头直接挂在 embedding 上,
      梯度反转**无处作用** —— head 与 adv 之间没有共享的可训练参数,
      对抗损失只会训练 adv 自己, 对 head 毫无影响。
      症状: 开/关对抗跑出**逐位相同**的数字 (折1.0 都是 1.53/1.28/2.05)。
      修法: 中间插一层可训练投影 proj, head 与 adv 都吃 proj 的输出,
      梯度反转才能真的逼 proj 丢掉中心可分信息。

    **非对抗版也走同一架构**, 只是不加对抗损失 —— 否则架构差异会混进对比,
    分不清是"对抗起作用"还是"多了一层非线性"。
    """
    import torch
    import torch.nn as nn

    lv = cfg["levels"]["L1"]
    torch.manual_seed(cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"])

    d = Xtr.shape[1]
    d_proj = 256
    proj = nn.Sequential(nn.Linear(d, d_proj), nn.ReLU()).to(dev)
    head = nn.Linear(d_proj, 1).to(dev)
    adv = nn.Sequential(nn.Linear(d_proj, 256), nn.ReLU(),
                        nn.Linear(256, n_centers)).to(dev) if adversarial else None
    params = list(proj.parameters()) + list(head.parameters())
    if adv is not None:
        params += list(adv.parameters())
    opt = torch.optim.AdamW(params, lr=lv["head_lr"], weight_decay=1e-4)

    # 类权重 (SPEC §3.2): 不做随机下采样, 会丢真实阴性
    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    w_pos = (neg / max(pos, 1.0)) if pos > 0 else 1.0
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w_pos], device=dev))
    advf = nn.CrossEntropyLoss()

    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(ytr, dtype=torch.float32, device=dev).unsqueeze(1)
    ct = torch.as_tensor(ctr_tr, dtype=torch.long, device=dev) if adv is not None else None
    bs = lv["batch_size"] * 64
    lam = cfg["adversarial"]["grad_reversal_lambda"]

    adv_acc_last = None
    for ep in range(lv["head_epochs"]):
        perm = torch.randperm(len(Xt), generator=g).to(dev)
        correct = tot = 0
        for st in range(0, len(perm), bs):
            idx = perm[st:st + bs]
            opt.zero_grad(set_to_none=True)
            z = proj(Xt[idx])
            loss = lossf(head(z), yt[idx])
            if adv is not None:
                logit_c = adv(GradReverse.apply(z, lam))
                loss = loss + advf(logit_c, ct[idx])
                correct += int((logit_c.argmax(1) == ct[idx]).sum())
                tot += len(idx)
            loss.backward()
            opt.step()
        if adv is not None and tot:
            adv_acc_last = correct / tot

    proj.eval(); head.eval()
    with torch.no_grad():
        sc = torch.sigmoid(head(proj(torch.as_tensor(
            Xte, dtype=torch.float32, device=dev)))).squeeze(1).cpu().numpy()
    return sc, adv_acc_last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="express")
    ap.add_argument("--no-adversarial", dest="adv", action="store_false", default=None)
    ap.add_argument("--cpu", action="store_true",
                    help="在 CPU 上跑。L1 是冻结 embedding 上的线性头, CPU 完全够用; "
                         "4090 被别人的作业占着时用这个, 不要去抢卡。"
                         "**不适用于 L2** (LoRA 要反传过 650M 编码器)。")
    args = ap.parse_args()

    cfg = apply_env()
    import torch  # noqa: F401
    if args.cpu:
        import torch
        dev = torch.device("cpu")
        torch.set_num_threads(min(16, (__import__("os").cpu_count() or 8)))
        print(f"[cpu] 按 --cpu 在 CPU 上跑 ({torch.get_num_threads()} 线程); "
              f"未占用任何 GPU")
    else:
        dev, cfg = setup_device(cfg)
    adversarial = cfg["adversarial"]["enabled"] if args.adv is None else args.adv

    ep = EMB / f"{args.task}_esm2_650M_frozen.parquet"
    if not ep.exists():
        raise SystemExit(f"{ep} 不存在, 先跑 src/models/esm_embed.py --task {args.task}")
    t = pq.read_table(ep)
    rids = t.column("record_id").to_pylist()
    X = np.array(t.column("emb").to_pylist(), dtype=np.float32)
    y_fail = (np.array(t.column("label").to_pylist()) == 0).astype(np.int8)  # 阳性=失败
    centers = np.array(t.column("center").to_pylist())
    groups = np.array(t.column("split_group").to_pylist())
    print(f"[L1] {args.task}: {len(y_fail):,} 条  dim={X.shape[1]}  "
          f"阴性 {int(y_fail.sum()):,} ({y_fail.mean()*100:.1f}%)  对抗={adversarial}")

    ft = pq.read_table(FOLDS)
    fold_of = dict(zip(ft.column("center").to_pylist(),
                       ft.column("center_fold").to_pylist()))
    k = cfg["eval"]["n_center_folds"]
    ks = tuple(cfg["eval"]["ks"])
    uniq_centers = sorted(set(centers))
    cidx = {c: i for i, c in enumerate(uniq_centers)}
    ctr_i = np.array([cidx[c] for c in centers])

    rows = []
    all_preds: list[dict] = []
    for j in range(k):
        held = [c for c, f in fold_of.items() if f == j]
        te_m = np.isin(centers, held)
        if te_m.sum() < cfg["eval"]["min_test_n"]:
            continue
        if y_fail[te_m].sum() < cfg["eval"]["min_test_positives"]:
            print(f"  折 {j}: test 阴性仅 {int(y_fail[te_m].sum())}, 跳过 "
                  f"(留出中心 {', '.join(held)} 不记录失败)")
            continue
        touched = set(groups[te_m])
        # cross: 训练来自其它中心, 且剔除触碰留出中心的同源组 (防泄漏)
        cross_m = ~te_m & ~np.isin(groups, list(touched))

        # within 对照必须"在留出中心**内部**再按组分折" ——
        # 踩过的坑 (2026-09-30, 在 GPU 空转之前用 numpy 验出来的):
        # 原来写成 "留出中心里不在测试组的部分", 但测试折就是该中心的全部组,
        # 所以那个掩码**恒为空**, within 一栏永远拿不到数。
        # 正确做法: 把留出中心的组再分 K_in 份, 内层第 i 份作 within 的测试,
        # 其余份作 within 的训练; 同时 cross 也只在这一份上评估, 保证**测试集完全相同**
        # (SPEC §3.1 的核心要求: 只换训练来源)。
        K_IN = 3
        held_groups = np.array(sorted(touched))
        gi = {g: (hash((int(g), cfg["seed"])) % K_IN) for g in held_groups}
        inner = np.array([gi.get(g, -1) for g in groups])

        sub_rows = []
        for i in range(K_IN):
            te_i = te_m & (inner == i)
            if te_i.sum() < cfg["eval"]["min_test_n"] or \
                    y_fail[te_i].sum() < cfg["eval"]["min_test_positives"]:
                continue
            within_i = te_m & (inner != i)       # 同中心、不同组 -> within 的训练侧
            sc_c, adv_acc = fit_head(X[cross_m], y_fail[cross_m], X[te_i], ctr_i[cross_m],
                                     len(uniq_centers), cfg, dev, adversarial)
            ev_c = evaluate(y_fail[te_i], sc_c, ks=ks)
            rec_i = {"inner_fold": i, "n_test": int(te_i.sum()),
                     "n_test_positive": int(y_fail[te_i].sum()),
                     "n_train_cross": int(cross_m.sum()),
                     "cross": ev_c,
                     # 对抗头在训练侧的中心判别准确率: 越接近"猜最大类"越说明中心信息被去掉了
                     "adv_center_acc_train": adv_acc}
            if within_i.sum() >= 200 and y_fail[within_i].sum() >= 30:
                sc_w, _ = fit_head(X[within_i], y_fail[within_i], X[te_i], ctr_i[within_i],
                                   len(uniq_centers), cfg, dev, adversarial=False)
                rec_i["within"] = evaluate(y_fail[te_i], sc_w, ks=ks)
                rec_i["n_train_within"] = int(within_i.sum())
            sub_rows.append(rec_i)
            all_preds.append({
                "fold": j, "inner": i,
                "record_id": [rids[x] for x in np.where(te_i)[0]],
                "center": centers[te_i].tolist(),
                "split_group": groups[te_i].tolist(),
                "y_fail": y_fail[te_i].tolist(),
                "score": sc_c.tolist()})
            w = rec_i.get("within", {}).get("lift@100")
            print(f"  折 {j}.{i} ({','.join(held)[:30]}): test={rec_i['n_test']:,} "
                  f"阴性={rec_i['n_test_positive']:,} base={ev_c['base_rate']:.3f} | "
                  f"cross lift@100={ev_c['lift@100']:.2f} "
                  f"PRAUC/base={ev_c['pr_auc_over_base']:.2f}"
                  + (f" | within lift@100={w:.2f}" if w else " | within 样本不足")
                  + (f" | adv中心判别acc={adv_acc:.3f}" if adv_acc is not None else ""))
        if not sub_rows:
            continue
        r = {"fold": j, "held_centers": held, "n_test_total": int(te_m.sum()),
             "n_test_positive_total": int(y_fail[te_m].sum()),
             "inner_folds": sub_rows}
        for side in ("cross", "within"):
            vals = [s[side]["lift@100"] for s in sub_rows if side in s]
            r[f"{side}_lift@100_mean"] = float(np.mean(vals)) if vals else None
            vals2 = [s[side]["pr_auc_over_base"] for s in sub_rows if side in s]
            r[f"{side}_pr_auc_over_base_mean"] = float(np.mean(vals2)) if vals2 else None
        rows.append(r)

    cmeta = caliber_meta(cfg["data"]["caliber"]["strip_construct_residues"],
                         cfg["data"]["caliber"]["use_sequence_length_feature"])
    name = f"L1_esm2_650M_frozen_{args.task}" + ("_adv" if adversarial else "") + ".json"
    outp = caliber_dir(cmeta["strip_construct_residues"],
                       cmeta["uses_sequence_length_feature"]) / name
    cl = [r["cross_lift@100_mean"] for r in rows if r.get("cross_lift@100_mean")]
    # 逐折报 (刘刚刚 2026-09-30: 不看中位数 —— 三种情形混在一起会相互抵消)
    print(f"\n{'折':>4}{'留出中心':<40}{'base':>7}{'cross lift@100':>15}"
          f"{'cross PRAUC/base':>18}{'within lift@100':>16}")
    for r in rows:
        b = float(np.mean([s["cross"]["base_rate"] for s in r["inner_folds"]]))
        print(f"{r['fold']:>4}{','.join(r['held_centers'])[:38]:<40}{b:>7.3f}"
              f"{r.get('cross_lift@100_mean', float('nan')):>15.2f}"
              f"{r.get('cross_pr_auc_over_base_mean', float('nan')):>18.2f}"
              f"{r.get('within_lift@100_mean', float('nan')):>16.2f}")
    print("  注: 基础率 <10% 的折以 PR-AUC/base 为主数字 (lift@100 分辨率不足, 见 config)")
    summary = {"caliber": cmeta, "task": args.task, "level": "L1",
               "adversarial": adversarial,
               "cross_lift@100_median": float(np.median(cl)) if cl else None,
               "cross_lift@100_range": [float(min(cl)), float(max(cl))] if cl else None,
               "folds": rows}
    outp.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"\ncross-center lift@100: 中位 {np.median(cl):.2f} "
          f"范围 [{min(cl):.2f}, {max(cl):.2f}]  ({len(cl)} 折)")
    if all_preds:
        import pyarrow as pa
        flat = {kk: [] for kk in ("fold", "inner", "record_id", "center",
                                  "split_group", "y_fail", "score")}
        for b in all_preds:
            n = len(b["record_id"])
            flat["fold"] += [b["fold"]] * n
            flat["inner"] += [b["inner"]] * n
            for kk in ("record_id", "center", "split_group", "y_fail", "score"):
                flat[kk] += b[kk]
        pq.write_table(pa.table(flat), outp.with_name(outp.stem + "_preds.parquet"))
        print(f"wrote {outp.with_name(outp.stem + '_preds.parquet')}")
    print(f"wrote {outp}")
    write_run_log(f"L1_{args.task}" + ("_adv" if adversarial else ""),
                  sources=["ds1_targettrack"], splits_used=["center_folds"],
                  config={"level": "L1", "adversarial": adversarial, **cmeta},
                  seed=cfg["seed"],
                  results={f"fold{r['fold']}": {
                      "cross_lift100": r.get("cross_lift@100_mean"),
                      "within_lift100": r.get("within_lift@100_mean")} for r in rows},
                  notes="L1 = 冻结 ESM-2 650M + 线性头; 主评估 cross-center")


if __name__ == "__main__":
    main()
