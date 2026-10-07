"""构建 zenodo_upload/ 暂存目录 —— 数据沉积的待传件。

刘刚刚 2026-10-07 定的范围与三条修正:
  - 平铺, 无子目录; 不含 data/raw/、模型权重、configs/local_paths.yaml、reports/runs/
  - README 全英文, 且**不写任何文件体积数字**
  - 目录不进 git (见 .gitignore)

**行数与 sha256 一律现算, 并与各自的 .prov.json 比对 (9 个产物全做)。**
为什么这条是硬的: .prov.json 记的是"当初应该是多少", 不是"现在这个文件是多少"。
只信指纹里的数字正是 2,377 -> 289 那次事故的失败模式 —— 数字照常产出, 没有报错。
比对用 provenance.fingerprint(), 与当初打指纹是同一套算法, 才是同口径。
任一不符即停, 不改数字。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from labels import provenance  # noqa: E402

OUT = pathlib.Path("zenodo_upload")
REPO = "https://github.com/liugangg/protein-failure-benchmark"

# 9 个带指纹的产物
ARTIFACTS = [
    "data/processed/records.parquet",
    "data/processed/splits/sequence_split.parquet",
    "data/processed/splits/lab_split.parquet",
    "data/processed/splits/time_split.parquet",
    "data/processed/splits/bind_target_split.parquet",
    "data/processed/splits/center_folds.parquet",
    "data/interim/pooled_records.parquet",
    "data/interim/pooled_unique_seqs.fasta",
    "data/interim/split_groups.parquet",
]
# SoluProt 去污染子集的 uid 列表: 无 .prov.json, 但 §5.1 的依据, 必须随沉积走
UID_LISTS = [
    "data/processed/splits/sequence_soluprot_clean_seq_uids.parquet",
    "data/processed/splits/lab_soluprot_clean_seq_uids.parquet",
    "data/processed/splits/time_soluprot_clean_seq_uids.parquet",
    "data/processed/splits/bind_target_soluprot_clean_seq_uids.parquet",
]
DOCS = ["DATA_AVAILABILITY.md", "LICENSE-DATA-CC-BY-SA-4.0.txt",
        "LICENSE-DATA-CC-BY-4.0.txt", "NOTICE"]

WHAT = {
    "records.parquet":
        "Unified six-stage failure labels; the main table, all sources merged",
    "sequence_split.parquet":
        "Sequence split — whole transitive-closure groups at 30% identity",
    "lab_split.parquet":
        "Laboratory split — CSGID and EFI held out",
    "time_split.parquet":
        "Temporal split — DS1 records from 2015 onward form the test side",
    "bind_target_split.parquet":
        "Cross-target `bind` split — IL7Ra, TrkA and Mdm2 held out",
    "center_folds.parquet":
        "Center-grouped cross-validation fold assignment",
    "pooled_records.parquet":
        "Merged records before any split assignment",
    "pooled_unique_seqs.fasta":
        "All unique sequences; the input to the MMseqs2 searches",
    "split_groups.parquet":
        "Split groups — transitive-closure components of the 30% identity graph",
    "sequence_soluprot_clean_seq_uids.parquet":
        "Decontaminated subset for the sequence split (uid list; see §5.1)",
    "lab_soluprot_clean_seq_uids.parquet":
        "Decontaminated subset for the laboratory split (uid list; see §5.1)",
    "time_soluprot_clean_seq_uids.parquet":
        "Decontaminated subset for the temporal split (uid list; see §5.1)",
    "bind_target_soluprot_clean_seq_uids.parquet":
        "Decontaminated subset for the cross-target split (uid list; see §5.1)",
}


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def scan_pattern() -> tuple[re.Pattern, int]:
    """构造脱敏扫描用的正则。**缺前提即硬失败, 不许静默通过。**

    待检出的内部字符串由 configs/local_paths.yaml 给出, 不写死在公开代码里 ——
    否则这段扫描器自己就把内部主机名/路径名印在了公开仓库里
    (2026-10-07 审计实测命中的就是那两行正则)。

    缺文件或缺 extra_scan_needles 字段时抛 SystemExit, 与 provenance.require()
    同一条原则: 检查前提消失要炸, 不能当作"无敏感词"继续 —— 那就重演了
    2,377 -> 289 那次事故 (检查条件没了, 结果照常产出, 没有报错)。

    **本函数在 main() 的最前面调用**: 校验不过就不许开始复制文件, 否则会留下一个
    已填满但未扫描的目录, 而"目录是满的"会被当成"构建完成"。
    """
    import yaml as _yaml
    lp = pathlib.Path("configs/local_paths.yaml")
    if not lp.exists():
        raise SystemExit(
            f"!! 缺 {lp} —— 脱敏扫描的待检字符串来自这里, 没有它这一步会"
            f'"零命中"地假通过。\n'
            f"   照 configs/local_paths.yaml.example 建一份再跑:\n"
            f"     cp configs/local_paths.yaml.example {lp}\n"
            f"   拒绝在无法扫描的情况下产出待传件。")
    cfg = _yaml.safe_load(lp.read_text()) or {}
    if "extra_scan_needles" not in cfg:
        raise SystemExit(
            f"!! {lp} 里缺 `extra_scan_needles` 字段。\n"
            f"   这个字段列出本机不该外流的标识 (内部主机名 / 服务名 / 沙箱路径)。\n"
            f"   确实没有要查的就显式写 `extra_scan_needles: []` —— "
            f"空列表是一次明确声明, 字段缺失不是。")
    # 通用模式, 不含任何内部标识, 可以留在公开代码里
    needles = [r"localhost", r"127\.0\.0\.1",
               r"192\.168\.\d+", r"10\.\d+\.\d+\.\d+",
               r"172\.(?:1[6-9]|2\d|3[01])\.\d+"]
    extra = list(cfg["extra_scan_needles"] or [])
    needles += [re.escape(str(x)) for x in extra]
    for k, v in cfg.items():
        if k != "extra_scan_needles" and isinstance(v, str) and v.startswith("/"):
            needles.append(re.escape(v.rstrip("/")))   # 完整路径, 不拆片段
    return re.compile("|".join(needles)), len(extra)


def main() -> None:
    # 前提检查必须在动文件之前 (见 scan_pattern 的说明)
    pat, n_extra = scan_pattern()
    print(f"脱敏扫描前提: OK ({pat.pattern.count('|') + 1} 个待检项, "
          f"其中 extra_scan_needles {n_extra} 个"
          + ("  ⚠️ 为空" if n_extra == 0 else "") + ")")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    copied: list[str] = []
    rows: dict[str, int | None] = {}
    checks: list[tuple[str, str, str, str]] = []   # name, 项, 现算, 指纹
    problems: list[str] = []

    for a in ARTIFACTS:
        src, prov = pathlib.Path(a), pathlib.Path(a + ".prov.json")
        for f in (src, prov):
            if not f.exists():
                raise SystemExit(f"!! 缺 {f}")
        dst = OUT / src.name
        shutil.copy2(src, dst)
        shutil.copy2(prov, OUT / prov.name)
        copied += [src.name, prov.name]

        # ── 现算 vs 指纹 ──
        live = provenance.fingerprint(dst)          # 与打指纹同一套算法
        rec = json.loads(prov.read_text())["artifact"]
        for key in ("rows", "sha256", "size"):
            got, want = live.get(key), rec.get(key)
            ok = got == want
            if key != "size":                       # size 只内部核, 不进展示表
                checks.append((src.name, key, str(got), str(want)))
            if not ok:
                problems.append(f"{src.name}: {key} 现算 {got} != 指纹 {want}")
        rows[src.name] = live.get("rows")

    for u in UID_LISTS:
        src = pathlib.Path(u)
        if not src.exists():
            raise SystemExit(f"!! 缺 {src}")
        shutil.copy2(src, OUT / src.name)
        copied.append(src.name)
        import polars as pl
        rows[src.name] = int(pl.scan_parquet(src).select(pl.len()).collect().item())

    for d in DOCS:
        p = pathlib.Path(d)
        if not p.exists():
            raise SystemExit(f"!! 缺 {p}")
        shutil.copy2(p, OUT / p.name)
        copied.append(p.name)

    assert not any(x.is_dir() for x in OUT.iterdir()), "出现子目录"
    assert len(set(copied)) == len(copied), "平铺后出现重名"

    # ── 比对结果先打印, 任一不符就停 ──
    print("=== 现算 vs 指纹 (9 个产物 × rows/sha256) ===")
    for name, key, got, want in checks:
        mark = "✅" if got == want else "❌"
        shown_got = got[:16] + "…" if key == "sha256" else f"{int(got):,}"
        shown_want = want[:16] + "…" if key == "sha256" else f"{int(want):,}"
        print(f"  {mark} {name:<30} {key:<7} 现算 {shown_got:<20} 指纹 {shown_want}")
    if problems:
        print()
        raise SystemExit("!! 现算与指纹不符, 停下, 未改任何数字:\n  - "
                         + "\n  - ".join(problems))
    print(f"  {len(checks)} 项全部一致\n")

    rec_sha = sha256(OUT / "records.parquet")
    rec_rows = rows["records.parquet"]

    # ── README (全英文) ──
    L: list[str] = []
    A = L.append
    A("# Data deposit — protein-failure benchmark")
    A("")
    A("Archived data for the preprint:")
    A("")
    A("> **Label artifacts in public protein-failure data originate from laboratory "
      "identity, and are removed by neither tag stripping nor adversarial debiasing**  ")
    A("> Ganggang Liu")
    A("")
    A(f"Code, evaluation scripts and all reports: <{REPO}>")
    A("")
    A("---")
    A("")
    A("## Contents")
    A("")
    A("Files are flat; there are no subdirectories. Each of the nine primary artifacts is "
      "accompanied by its `.prov.json` fingerprint, which records that artifact's own "
      "sha256 and row count **and the same values for all of its upstream inputs**. The "
      "scripts in the repository verify that chain at startup and exit on mismatch.")
    A("")
    A("| File | Rows | What it is |")
    A("|---|---|---|")
    for a in ARTIFACTS:
        n = pathlib.Path(a).name
        r = rows[n]
        A(f"| `{n}` | {r:,} | {WHAT[n]} |" if r is not None
          else f"| `{n}` | — | {WHAT[n]} |")
    for u in UID_LISTS:
        n = pathlib.Path(u).name
        A(f"| `{n}` | {rows[n]:,} | {WHAT[n]} |")
    A("")
    A("The four `*_soluprot_clean_seq_uids.parquet` files carry no `.prov.json`: they are "
      "uid lists derived deterministically by `src/eval/soluprot_contamination.py` from "
      "the fingerprinted artifacts above, and are included so that §5.1 can be reproduced "
      "without re-running the decontamination step.")
    A("")
    A("Also included: `DATA_AVAILABILITY.md` (the full licence layering and the per-source "
      "citations you are required to give), `LICENSE-DATA-CC-BY-SA-4.0.txt`, "
      "`LICENSE-DATA-CC-BY-4.0.txt`, and `NOTICE`.")
    A("")
    A("## Verifying what you downloaded")
    A("")
    A("`SHA256SUMS.txt` covers every file in this deposit:")
    A("")
    A("```")
    A("sha256sum -c SHA256SUMS.txt")
    A("```")
    A("")
    A(f"The main table `records.parquet` must have **{rec_rows:,} rows** and a sha256 "
      f"beginning `{rec_sha[:16]}`. If either differs, you do not have the copy that "
      f"produced the numbers in the paper — do not compare results against it.")
    A("")
    A("All row counts in the table above were recomputed from the files in this deposit "
      "rather than copied from the fingerprints. For the nine fingerprinted artifacts the "
      "recomputed row count **and** sha256 were additionally checked against the values "
      "recorded in their `.prov.json`; the four uid lists have no fingerprint to check "
      "against, so for those the count is simply the recomputed one. A fingerprint states "
      "what an artifact *should* contain; it cannot tell you what the file in front of "
      "you *does* contain.")
    A("")
    A("## Read this before counting records")
    A("")
    A(f"The {rec_rows:,} records collapse to **98,617 clusters** at 30% sequence "
      "identity. By source the shrinkage ranges from ÷1.6 to ÷6,661.")
    A("")
    A("This measures *effective diversity for cross-protein generalization*, not data "
      "quality: deep mutational scanning data are excellent for their own purpose, and "
      "shrink the most precisely because they are many variants of few parents.")
    A("")
    A("**Count clusters, not records.** Reporting the record count as the scale of this "
      "dataset overstates the effective sample size available for predicting whether an "
      "unseen protein will fail.")
    A("")
    A("## Licence — CC BY-SA 4.0 for this deposit as a whole")
    A("")
    A("One upstream source, PSI TargetTrack, carries CC BY-SA 4.0, whose share-alike term "
      "we cannot relicense away. **946,322 records (39.8% of the total) derive from "
      "share-alike upstreams**, and that portion includes all labels for the `express` "
      "primary task and all cross-center evaluation data.")
    A("")
    A("Because this deposit is redistributed as a **single work**, it therefore carries "
      "**CC BY-SA 4.0 in its entirety**. If you redistribute anything derived from it, "
      "your derived work must also be CC BY-SA 4.0 or compatible; internal research "
      "without redistribution does not trigger share-alike.")
    A("")
    A("A user who requires a pure CC BY 4.0 subset must reconstruct it by excluding the "
      "TargetTrack-derived portion — and that subset does **not** contain the `express` "
      "primary task, so it cannot reproduce the paper's core results.")
    A("")
    A("Per-source licences and the citations each one requires (ProteinGym requires the "
      "33 original assay papers to be cited) are in `DATA_AVAILABILITY.md`.")
    A("")
    A("## What is not here")
    A("")
    A("- **Upstream raw files.** This work redistributes none of them. They are "
      "downloaded from the addresses recorded in `configs/data_sources.yaml` in the "
      "repository, md5-verified, and used read-only.")
    A("- **Model weights.** None were saved; the training scripts evaluate and discard "
      "them. Predictions are kept in the repository, so every reported number is "
      "checkable without the weights.")
    A("- **Regenerable intermediates** beyond the three included here, and the per-run "
      "environment logs, which are in the repository under `reports/runs/`.")
    A("")

    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf8")

    names = sorted(p.name for p in OUT.iterdir() if p.name != "SHA256SUMS.txt")
    (OUT / "SHA256SUMS.txt").write_text(
        "\n".join(f"{sha256(OUT / n)}  {n}" for n in names) + "\n", encoding="utf8")

    # ── 敏感信息扫描 (pat 在 main 开头就已校验并构造好) ──
    hits = []
    for p in sorted(OUT.iterdir()):
        if p.suffix == ".parquet":
            continue
        try:
            t = p.read_text(encoding="utf8", errors="ignore")
        except Exception:
            continue
        for m in set(pat.findall(t)):
            hits.append(f"{p.name}: {m}")

    n_files = len(list(OUT.iterdir()))
    n_sums = len((OUT / "SHA256SUMS.txt").read_text().strip().split("\n"))
    print(f"zenodo_upload/: {n_files} 个文件 (SHA256SUMS.txt {n_sums} 行), 平铺无子目录")
    print(f"  records.parquet  {rec_rows:,} 行  sha256 {rec_sha[:16]}…")
    if hits:
        print("  ⚠️ 敏感信息命中:")
        for h in hits:
            print("    ", h)
        raise SystemExit("!! 目录内含内部信息, 不要上传")
    cjk = sum(1 for ch in (OUT / "README.md").read_text(encoding="utf8")
              if "一" <= ch <= "鿿")
    print(f"  README 中文字符: {cjk} (应为 0)")
    print("  敏感信息扫描: 干净 (待检项见本脚本开头的 \"脱敏扫描前提\" 一行)")
    if cjk:
        raise SystemExit("!! README 仍有中文, 应全英文")


if __name__ == "__main__":
    main()
