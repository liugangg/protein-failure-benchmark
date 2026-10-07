"""简单基线: 长度 + 氨基酸组成 + GBDT (SPEC §3.2)。

它的用途是给后面的 PLM 定一条**必须跨过的线**。SPEC 说得很直接:
"如果后面的深度模型超不过'长度 + 组成 + GBDT', 说明模型没学到东西, 要如实报告。"

两条纪律:
  1. 只用冻结的切分 (data/processed/splits/), 不在这里重新切。
  2. 去冗余: 训练和评测都**按 split_group 取一条代表**。
     理由: 同一个蛋白在 TargetTrack 里可能有几十条 trial, 不去冗余的话
     precision@20 会被同一个蛋白的多条记录刷满, 数字虚高且没有意义。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import polars as pl
import re
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from labels import provenance  # noqa: E402
from eval.caliber import caliber_dir, caliber_meta  # noqa: E402
from eval.metrics import by_group, evaluate  # noqa: E402
from eval.run_log import write_run_log  # noqa: E402

AA20 = "ACDEFGHIKLMNPQRSTVWY"

# ─────────────────────────────────────────────────────────────────────────────
# 构建体残留的清除 —— **默认预处理**, 保留版降为消融 (刘刚刚 2026-09-30 定)
# ─────────────────────────────────────────────────────────────────────────────
# 为什么改成默认: 实测 DS1 里 18.3% 的序列把亲和标签**写在序列里**, 而带不带标签几乎是
# center 级的构建习惯 (SSGCID 91.8% vs CESG 0.2%), 标签**位置**也是中心指纹
# (JCSG/SGPP/SSGCID ~100% N 端 vs NESG 59% / NYSGXRC 98% C 端)。
# 这些残留给模型一条**序列可见的实验室身份通道**, 与生物学无关。
# 所以清除它们是正确的默认预处理, 不是可选项; 保留版只作消融对照。
#
# 覆盖范围 (刘刚刚指定): His / FLAG / MYC / HA / Strep / T7 标签,
# GST / MBP / SUMO 融合伴侣, TEV / thrombin 酶切位点, 常见载体克隆位点残余。
# 每条都注明来源序列, 不凭记忆写。
TAG_PATTERNS: list[tuple[str, str]] = [
    # --- 纯化标签 ---
    (r"H{6,}",                    "His6/His8/His10 多聚组氨酸标签"),
    (r"DYKDDDDK",                 "FLAG 标签"),
    (r"(?:DYKDDDDK){2,}",         "3xFLAG 等串联 FLAG"),
    (r"EQKLISEEDL",               "c-Myc 标签"),
    (r"YPYDVPDYA",                "HA 标签 (流感血凝素)"),
    (r"WSHPQFEK",                 "Strep-tag II"),
    (r"MASMTGGQQMG",              "T7 标签 (T7 噬菌体 gene10 前导)"),
    (r"KETAAAKFERQHMDS",          "S-tag (RNase A S 肽)"),
    # --- 酶切位点 (标签与目标之间的接头) ---
    (r"ENLYFQ[GS]?",              "TEV 蛋白酶位点 ENLYFQ/G 或 /S"),
    (r"LVPR[GS]{1,2}",            "凝血酶 (thrombin) 位点 LVPR/GS"),
    (r"IEGR",                     "Factor Xa 位点"),
    (r"DDDDK",                    "肠激酶 (enterokinase) 位点"),
    # --- 融合伴侣的 C 端残留 (整个伴侣通常不在记录序列里, 残留接头才是) ---
    (r"SSGLVPRGSH",               "pET-28 His-thrombin 接头残留"),
    (r"GSGSGS(?:GS)*",            "GS 柔性接头 (GST/MBP/SUMO 融合常用)"),
    (r"(?:GGGGS){2,}",            "GGGGS 串联接头"),
    # --- 载体克隆位点残余 (常见多克隆位点翻译产物) ---
    (r"^MGSSHHHHHHSSGLVPRGSH",    "pET-28a N 端完整前导"),
    (r"^MHHHHHH",                 "pET 系列 N 端 His 前导"),
    (r"^MSYYHHHHHH",              "pET-based Met-Ser-Tyr-Tyr-His6 前导"),
    (r"LEHHHHHH$",                "pET-21 C 端 His 前导 (含 XhoI 位点 LE)"),
    (r"^GAMG",                    "BamHI 克隆位点翻译残余"),
    (r"^GS(?=[A-Z])",             "BamHI GS 残余 (仅 N 端首两位)"),
]
RECORDS = pathlib.Path("data/processed/records.parquet")
SPLITS = pathlib.Path("data/processed/splits")
CONFIG = pathlib.Path("configs/baseline_gbdt.yaml")
OUT = pathlib.Path("reports")


def feature_names(cfg: dict) -> list[str]:
    names: list[str] = []
    if cfg["features"]["seq_len"]:
        names.append("seq_len")
    if cfg["features"]["log_seq_len"]:
        names.append("log_seq_len")
    if cfg["features"]["aa_composition"]:
        names += [f"aa_{a}" for a in AA20]
    return names


def strip_tags(seqs: list[str]) -> tuple[list[str], int, dict[str, int]]:
    """清除构建体残留。返回 (清除后序列, 被改动的条数, 各模式命中数)。

    清除后不足 20 aa 的保留原序列 —— 不制造畸形样本。
    """
    out: list[str] = []
    n_changed = 0
    hits: dict[str, int] = {desc: 0 for _, desc in TAG_PATTERNS}
    compiled = [(re.compile(pat), desc) for pat, desc in TAG_PATTERNS]
    for s in seqs:
        cur = s
        for rx, desc in compiled:
            cur2, k = rx.subn("", cur)
            if k:
                hits[desc] += k
                cur = cur2
        if cur != s:
            n_changed += 1
        out.append(cur if len(cur) >= 20 else s)
    return out, n_changed, {k: v for k, v in hits.items() if v}


def featurize(seqs: list[str], cfg: dict) -> np.ndarray:
    n = len(seqs)
    cols: list[np.ndarray] = []
    lens = np.array([len(s) for s in seqs], dtype=np.float32)
    if cfg["features"]["seq_len"]:
        cols.append(lens[:, None])
    if cfg["features"]["log_seq_len"]:
        cols.append(np.log1p(lens)[:, None])
    if cfg["features"]["aa_composition"]:
        comp = np.zeros((n, len(AA20)), dtype=np.float32)
        for i, s in enumerate(seqs):
            if not s:
                continue
            for j, a in enumerate(AA20):
                comp[i, j] = s.count(a)
            comp[i] /= len(s)
        cols.append(comp)
    return np.hstack(cols)


def dedup_by_group(d: pl.DataFrame, seed: int) -> pl.DataFrame:
    """每个 split_group 只留一条记录。

    选哪一条: 用确定性哈希挑, 不用 first() —— first 取决于行序, 换一次 polars
    版本或重跑 pool 就可能换样本, 结果不可复现。
    """
    return (
        d.with_columns(
            (pl.col("record_id") + pl.lit(f":{seed}")).hash().alias("_h")
        )
        .sort(["split_group", "_h"])
        .unique(subset=["split_group"], keep="first")
        .drop("_h")
    )


def run_stage(rec: pl.DataFrame, split_name: str, stage: str, cfg: dict) -> dict | None:
    sp = pl.read_parquet(SPLITS / f"{split_name}_split.parquet").select(
        "record_id", "split"
    )
    d = rec.join(sp, on="record_id", how="inner")
    # 只保留该阶段有观测的记录 (label != -1) —— 这就是 PU 设定里"排除未观测"
    # stage 也可以是派生标签 soluble_expression (对齐 SoluProt 的口径)
    col = "label_soluble_expression" if stage == "soluble_expression" else f"label_{stage}"
    d = d.filter(pl.col(col) != -1)

    tr = dedup_by_group(d.filter(pl.col("split") == "train"), cfg["seed"])
    te = dedup_by_group(d.filter(pl.col("split") == "test"), cfg["seed"])
    if tr.height < 500 or te.height < 200:
        return {"split": split_name, "stage": stage,
                "skipped": f"样本不足 (train={tr.height}, test={te.height})"}

    # y = 1 表示**失败** (见 metrics.py 的方向约定)
    ytr = (tr[col] == 0).to_numpy().astype(np.int8)
    yte = (te[col] == 0).to_numpy().astype(np.int8)
    if ytr.sum() < 50 or yte.sum() < 20:
        return {"split": split_name, "stage": stage,
                "skipped": f"阴性不足 (train_fail={int(ytr.sum())}, test_fail={int(yte.sum())})"}

    seq_tr, seq_te = tr["sequence"].to_list(), te["sequence"].to_list()
    tag_info = None
    if cfg.get("strip_tags"):
        seq_tr, ntr, hits_tr = strip_tags(seq_tr)
        seq_te, nte, hits_te = strip_tags(seq_te)
        tag_info = {"stripped_train": ntr, "stripped_test": nte,
                    "stripped_test_frac": round(nte / max(len(seq_te), 1), 4),
                    "pattern_hits_test": hits_te}
    Xtr = featurize(seq_tr, cfg)
    Xte = featurize(seq_te, cfg)

    m = cfg["model"]
    clf = HistGradientBoostingClassifier(
        max_iter=m["max_iter"], learning_rate=m["learning_rate"],
        max_leaf_nodes=m["max_leaf_nodes"], l2_regularization=m["l2_regularization"],
        early_stopping=m["early_stopping"], validation_fraction=m["validation_fraction"],
        random_state=cfg["seed"],
    )
    clf.fit(Xtr, ytr)
    score = clf.predict_proba(Xte)[:, 1]

    # 特征重要性 (刘刚刚 2026-09-30 要求): 如果基线几乎全靠序列长度,
    # 说明这个基准可能被平凡混杂因素主导、headroom 有限 —— 必须自己先说, 别等审稿人指出。
    # 用 permutation importance 而不是树的 split gain: 后者对高基数连续特征 (长度) 有偏,
    # 会系统性高估长度的重要性, 正好在这个问题上最容易误导。
    names = feature_names(cfg)
    imp = {}
    try:
        pi = permutation_importance(clf, Xte, yte, n_repeats=10,
                                    random_state=cfg["seed"], scoring="average_precision")
        order = np.argsort(-pi.importances_mean)
        imp = {
            "method": "permutation_importance (scoring=average_precision, n_repeats=10)",
            "top": [{"feature": names[i],
                     "mean": float(pi.importances_mean[i]),
                     "std": float(pi.importances_std[i])} for i in order[:8]],
            "seq_len_share": None,
        }
        tot = float(np.clip(pi.importances_mean, 0, None).sum())
        if tot > 0:
            li = [i for i, n in enumerate(names) if n in ("seq_len", "log_seq_len")]
            imp["seq_len_share"] = round(
                float(np.clip(pi.importances_mean[li], 0, None).sum()) / tot, 4)
    except Exception as e:
        imp = {"error": f"{type(e).__name__}: {e}"}

    out = {
        "split": split_name,
        "stage": stage,
        "train_n": int(tr.height), "train_fail": int(ytr.sum()),
        "test_n": int(te.height), "test_fail": int(yte.sum()),
        "overall": evaluate(yte, score, ks=tuple(cfg["eval"]["ks"])),
        "feature_importance": imp,
        "strip_tags": bool(cfg.get("strip_tags")),
        "tag_info": tag_info,
    }
    for gcol in cfg["eval"]["group_by"]:
        out[f"by_{gcol}"] = by_group(
            yte, score, te[gcol].cast(pl.Utf8).to_numpy(),
            ks=tuple(cfg["eval"]["ks"]), min_n=cfg["eval"]["min_group_n"],
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*",
                    default=["sequence", "lab", "time", "bind_target"])
    # 剥标签是**默认**预处理; --keep-tags 是消融
    ap.add_argument("--keep-tags", dest="strip_tags", action="store_false", default=True,
                    help="消融: 保留构建体残留 (默认是清除)")
    ap.add_argument("--keep-length", dest="drop_length", action="store_false", default=None,
                    help="消融: 保留序列长度特征 (默认按 config 的 drop_length)")
    ap.add_argument("--out", default=None, help="输出文件名 (默认 baseline_gbdt.json)")
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["strip_tags"] = bool(args.strip_tags)
    if args.drop_length is not None:
        cfg["features"]["seq_len"] = not args.drop_length
        cfg["features"]["log_seq_len"] = not args.drop_length
    provenance.require([RECORDS] + [SPLITS / f"{s}_split.parquet" for s in args.splits
                                    if (SPLITS / f"{s}_split.parquet").exists()])
    rec = pl.read_parquet(RECORDS)

    results = []
    for sn in args.splits:
        if not (SPLITS / f"{sn}_split.parquet").exists():
            print(f"[skip] {sn} 切分不存在")
            continue
        for stage in cfg["stages"]:
            r = run_stage(rec, sn, stage, cfg)
            if r is None:
                continue
            results.append(r)
            if "skipped" in r:
                print(f"  {sn:9s} {stage:8s} SKIP: {r['skipped']}")
            else:
                o = r["overall"]
                print(f"  {sn:9s} {stage:8s} test_n={r['test_n']:6d} "
                      f"fail={r['test_fail']:5d} base={o['base_rate']:.3f} "
                      f"P@20={o['precision@20']:.3f} P@100={o['precision@100']:.3f} "
                      f"lift@100={o['lift@100']:.2f} PR-AUC={o['pr_auc']:.3f}")

    OUT.mkdir(exist_ok=True)
    # 输出到口径目录 (SPEC §5): 不同口径同名文件, 靠目录区分, 不原地覆盖
    cdir = caliber_dir(bool(cfg["strip_tags"]), bool(cfg["features"]["seq_len"]))
    payload = {"caliber": caliber_meta(bool(cfg["strip_tags"]),
                                       bool(cfg["features"]["seq_len"])),
               "results": results}
    outp = cdir / (args.out or "baseline_gbdt.json")
    outp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    provenance.stamp(outp, [RECORDS] + [SPLITS / f"{s}_split.parquet" for s in args.splits
                                        if (SPLITS / f"{s}_split.parquet").exists()],
                     produced_by="src/models/baseline_gbdt.py",
                     extra=payload["caliber"])
    print(f"\nwrote {outp}  (口径: {payload['caliber']['caliber']})")

    # SPEC v2 §1.4 / §6: 每次产生评测数字都要落运行日志 (数据源 / 切分 hash / 超参 / 环境)
    used = sorted(rec["source"].unique().to_list())
    write_run_log(
        "baseline_gbdt",
        sources=used,
        splits_used=args.splits,
        config=cfg,
        seed=cfg["seed"],
        results={
            f"{r['split']}/{r['stage']}": (
                r.get("skipped") or {
                    "precision@20": r["overall"]["precision@20"],
                    "precision@100": r["overall"]["precision@100"],
                    "lift@100": r["overall"]["lift@100"],
                    "pr_auc": r["overall"]["pr_auc"],
                    "base_rate": r["overall"]["base_rate"],
                }
            )
            for r in results
        },
        notes="长度+氨基酸组成+GBDT 简单基线 (SPEC v2 §3.2)",
    )


if __name__ == "__main__":
    main()
