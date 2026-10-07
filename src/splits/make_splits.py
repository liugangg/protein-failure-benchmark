"""构造并**冻结**三套留出测试集 (SPEC §3.1)。

输出到 data/processed/splits/ 并写 MANIFEST.json (含每个文件的 sha256)。
SPEC 明令: 冻结之后任何阶段都不许因为分数不好回来重切。MANIFEST 里的 hash 就是凭据。

三套切分是**三个独立的评测轴**, 不是一次三分。每套各有自己的 train/test 侧。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import polars as pl
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from labels import provenance
from labels.schema import STAGES  # noqa: E402

INTERIM = pathlib.Path("data/interim")
OUTDIR = pathlib.Path("data/processed/splits")
CONFIG = pathlib.Path("configs/splits.yaml")


def load_pooled() -> pl.DataFrame:
    df = pl.read_parquet(INTERIM / "pooled_records.parquet")
    c = pl.read_csv(
        INTERIM / "cluster30" / "c30_cluster.tsv", separator="\t", has_header=False,
        new_columns=["cluster_rep", "seq_uid"],
        schema_overrides={"cluster_rep": pl.Int64, "seq_uid": pl.Int64},
    )
    c = (
        c.with_columns((pl.col("cluster_rep") == pl.col("seq_uid")).cast(pl.Int8).alias("_s"))
        .sort(["seq_uid", "_s"]).unique(subset=["seq_uid"], keep="first").drop("_s")
    )
    df = df.join(c, on="seq_uid", how="left")
    if df["cluster_rep"].null_count():
        raise SystemExit("有记录没匹配到簇, 先重跑 cluster30.sh")

    # 切分单位是 split_group 而不是 set-cover 簇 —— 见 src/splits/split_groups.py:
    # set-cover 簇之间仍存在 186,107 条 >30% 的边, 按簇切分会泄漏。
    gp = INTERIM / "split_groups.parquet"
    if not gp.exists():
        raise SystemExit("缺 data/interim/split_groups.parquet, 先跑 src/splits/split_groups.py")
    provenance.require([INTERIM / "pooled_records.parquet", gp])
    g = pl.read_parquet(gp).select("seq_uid", "split_group")
    df = df.join(g, on="seq_uid", how="left")
    if df["split_group"].null_count():
        raise SystemExit("有记录没匹配到 split_group, 重跑 split_groups.py")
    return df


def stage_counts(d: pl.DataFrame) -> dict[str, dict[str, int]]:
    """每套切分每一侧都要报各阶段的阴性簇数 —— 没有阴性的测试集是量不出东西的。"""
    out = {}
    for s in STAGES:
        f = d.filter(pl.col(f"label_{s}") == 0)
        out[s] = {
            "neg_records": f.height,
            "neg_clusters": int(f["cluster_rep"].n_unique()) if f.height else 0,
            "pos_records": int((d[f"label_{s}"] == 1).sum()),
        }
    return out


def summarize(name: str, d: pl.DataFrame) -> dict:
    return {
        "split": name,
        "records": d.height,
        "clusters": int(d["cluster_rep"].n_unique()),
        "split_groups": int(d["split_group"].n_unique()),
        "centers": int(d["center"].n_unique()),
        "stages": stage_counts(d),
    }


def build_sequence_split(df: pl.DataFrame, cfg: dict) -> tuple[pl.DataFrame, dict]:
    fr = cfg["fractions"]

    # 超大组强制进训练侧 —— 见 configs/splits.yaml 的 pin_rationale。
    thr = cfg.get("pin_large_groups_to_train_above_fraction", 0)
    n_seq_total = df["seq_uid"].n_unique()
    gsize = df.group_by("split_group").agg(pl.col("seq_uid").n_unique().alias("k"))
    pinned = gsize.filter(pl.col("k") > thr * n_seq_total)["split_group"] if thr else []
    pinned_set = set(pinned.to_list()) if thr else set()
    info = {
        "pin_threshold_fraction": thr,
        "pinned_groups": len(pinned_set),
        "pinned_sequences": int(gsize.filter(pl.col("split_group").is_in(list(pinned_set)))["k"].sum())
        if pinned_set else 0,
    }
    if pinned_set:
        print(f"[sequence_split] 强制进训练侧的超大组 {len(pinned_set)} 个, "
              f"含 {info['pinned_sequences']:,} 条序列 "
              f"({info['pinned_sequences']/n_seq_total*100:.1f}%)")

    clusters = df.select("split_group").unique().sort("split_group")
    # 用 hash(seed, cluster_rep) 定分配, 而不是 shuffle —— 同一个 seed 在任何机器上
    # 都给同样的结果, 新增数据也不会打乱既有簇的归属。
    seed = cfg_seed
    def assign(rep: int) -> str:
        if rep in pinned_set:
            return "train"
        h = hashlib.sha256(f"{seed}:{rep}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        if u < fr["train"]:
            return "train"
        if u < fr["train"] + fr["val"]:
            return "val"
        return "test"
    clusters = clusters.with_columns(
        pl.col("split_group").map_elements(assign, return_dtype=pl.Utf8).alias("split")
    )
    return df.join(clusters, on="split_group", how="left").select(
        "record_id", "cluster_rep", "split_group", "center", "year", "source", "split"
    ), info


def build_group_split(df: pl.DataFrame, is_test_record: pl.Expr, cfg: dict,
                      label: str) -> tuple[pl.DataFrame, dict]:
    """按记录级条件划分, 但以簇为单位保证完整性; 混合簇两边都不要。"""
    sub = df.filter(pl.col("source").is_in(cfg["sources"]))
    sub = sub.with_columns(is_test_record.alias("_is_test"))
    per_clu = sub.group_by("split_group").agg(
        pl.col("_is_test").any().alias("has_test"),
        (~pl.col("_is_test")).any().alias("has_train"),
    )
    mixed = per_clu.filter(pl.col("has_test") & pl.col("has_train"))
    assign = per_clu.with_columns(
        pl.when(pl.col("has_test") & pl.col("has_train")).then(pl.lit("dropped"))
        .when(pl.col("has_test")).then(pl.lit("test"))
        .otherwise(pl.lit("train")).alias("split")
    ).select("split_group", "split")
    out = sub.join(assign, on="split_group", how="left").select(
        "record_id", "cluster_rep", "split_group", "center", "year", "source", "split"
    )
    info = {
        "mixed_split_groups_dropped": mixed.height,
        "records_dropped": int(out.filter(pl.col("split") == "dropped").height),
        "eligible_split_groups": per_clu.height,
    }
    print(f"[{label}] 混合切分组丢弃 {mixed.height:,} 个 / 共 {per_clu.height:,} 个 "
          f"(丢弃记录 {info['records_dropped']:,})")
    return out, info


def sha256_of(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    global cfg_seed
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg_seed = cfg["seed"]
    df = load_pooled()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "config": cfg,
        "split_unit": "split_group",
        "split_unit_note": (
            "切分单位是 split_group (见 src/splits/split_groups.py): 把 set-cover 簇之间"
            "仍有 >30% 相似边的簇并成连通分量。直接按 set-cover 簇切分实测会泄漏"
            "(序列切分训练/测试之间 10,841 对 >30%, 最高 fident=1.00)。"
        ),
        "splits": {},
    }

    # --- 1. 序列切分 ---
    seq, seq_info = build_sequence_split(df, cfg["sequence_split"])
    p = OUTDIR / "sequence_split.parquet"
    seq.write_parquet(p)
    manifest["splits"]["sequence"] = {
        "file": p.name, "sha256": sha256_of(p), **seq_info,
        "sides": {k: summarize(k, df.join(seq.select("record_id", "split"), on="record_id")
                               .filter(pl.col("split") == k))
                  for k in ("train", "val", "test")},
    }

    # --- 2. 实验室切分 ---
    lab_cfg = cfg["lab_split"]
    lab, lab_info = build_group_split(
        df, pl.col("center").is_in(lab_cfg["holdout_centers"]), lab_cfg, "lab_split"
    )
    p = OUTDIR / "lab_split.parquet"
    lab.write_parquet(p)
    joined = df.join(lab.select("record_id", "split"), on="record_id", how="inner")
    manifest["splits"]["lab"] = {
        "file": p.name, "sha256": sha256_of(p),
        "holdout_centers": lab_cfg["holdout_centers"], **lab_info,
        "sides": {k: summarize(k, joined.filter(pl.col("split") == k))
                  for k in ("train", "test")},
    }

    # --- 3. 时间切分 ---
    t_cfg = cfg["time_split"]
    tim, tim_info = build_group_split(
        df, pl.col("year") >= t_cfg["test_from_year"], t_cfg, "time_split"
    )
    p = OUTDIR / "time_split.parquet"
    tim.write_parquet(p)
    joined = df.join(tim.select("record_id", "split"), on="record_id", how="inner")
    manifest["splits"]["time"] = {
        "file": p.name, "sha256": sha256_of(p),
        "test_from_year": t_cfg["test_from_year"], **tim_info,
        "sides": {k: summarize(k, joined.filter(pl.col("split") == k))
                  for k in ("train", "test")},
    }

    # --- 4. bind 专用: 跨靶点留出 ---
    b_cfg = cfg.get("bind_target_split")
    if b_cfg:
        # 只在有 bind 观测的记录上做 —— 没有 bind 标签的行进来只会稀释测试集
        sub = df.filter(pl.col("label_bind") != -1)
        bind, bind_info = build_group_split(
            sub, pl.col("target_id").is_in(b_cfg["holdout_targets"]), b_cfg,
            "bind_target_split",
        )
        p = OUTDIR / "bind_target_split.parquet"
        bind.write_parquet(p)
        joined = sub.join(bind.select("record_id", "split"), on="record_id", how="inner")
        manifest["splits"]["bind_target"] = {
            "file": p.name, "sha256": sha256_of(p),
            "holdout_targets": b_cfg["holdout_targets"], **bind_info,
            "sides": {k: summarize(k, joined.filter(pl.col("split") == k))
                      for k in ("train", "test")},
        }

    # 冻结的训练数据本体也一起落到 processed/ 并记 hash
    proc = pathlib.Path("data/processed/records.parquet")
    df.write_parquet(proc)
    manifest["records"] = {"file": str(proc), "sha256": sha256_of(proc),
                           "rows": df.height,
                           "clusters": int(df["cluster_rep"].n_unique())}

    for _k in ("sequence", "lab", "time", "bind_target"):
        _p = OUTDIR / f"{_k}_split.parquet"
        if _p.exists():
            provenance.stamp(_p, [proc, INTERIM / "split_groups.parquet"],
                             produced_by="src/splits/make_splits.py")
    provenance.stamp(proc, [INTERIM / "pooled_records.parquet",
                            INTERIM / "split_groups.parquet"],
                     produced_by="src/splits/make_splits.py")
    mf = OUTDIR / "MANIFEST.json"
    mf.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nwrote {mf}")
    for name, m in manifest["splits"].items():
        print(f"\n== {name} ==  sha256={m['sha256'][:16]}…")
        for side, s in m["sides"].items():
            neg = {k: v["neg_clusters"] for k, v in s["stages"].items() if v["neg_clusters"]}
            print(f"  {side:6s} records={s['records']:9,} clusters={s['clusters']:7,} "
                  f"阴性簇={neg}")


if __name__ == "__main__":
    main()
