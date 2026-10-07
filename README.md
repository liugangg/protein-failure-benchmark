# Protein-failure benchmark: center-stratified splits and evaluation protocol

Data, frozen splits, and evaluation code for the preprint:

> **Label artifacts in public protein-failure data originate from laboratory identity, and are removed by neither tag stripping nor adversarial debiasing**
> Ganggang Liu. bioRxiv (2026). doi: *[to be added after posting]*

---

## What this is

Six-stage experimental-failure labels (`clone` / `express` / `soluble` / `purify` / `stable` / `bind`) unified from four public sources, together with four leakage-verified frozen splits and the evaluation code used in the paper.

The headline result is negative and concerns evaluation, not modelling: on an **identical set of test items**, a model trained within the held-out center reaches PR-AUC/base 1.85, while the same model trained across centers reaches 1.01 against a measured random control of 1.03 — despite the cross-center training set being 4.0–17.7× larger. Homology-based redundancy reduction, which is the standard precaution in this literature, does not remove this.

## Why you might want it

- **If you are building a sequence-based predictor of experimental outcome**, the `lab` split and `center_folds` here let you check whether your model generalizes across laboratories or only recognizes them. Random and homology-based splits cannot tell these apart.
- **If you are evaluating a published predictor**, §5.1 of the paper reports what happens when SoluProt is applied outside its domain of applicability (41% of our test sequences cannot be scored on its core feature).
- **If you want the labels**, they are the largest unified multi-stage failure label set we are aware of — though see the scale caveat below, which is the first result of the paper.

## Scale caveat — read this before counting records

2,380,297 records collapse to **98,617 clusters** at 30% sequence identity. By source the shrinkage ranges from ÷1.6 to ÷6,661. This measures *effective diversity for cross-protein generalization*, not data quality: deep mutational scanning data are excellent for their own purpose and shrink the most precisely because they are many variants of few parents.

Count clusters, not records.

## Layout

```
data/processed/splits/   the four frozen splits + center_folds (in this repo, 15.6 MB)
src/ingest/              per-source ingestion + md5 verification
src/labels/              six-stage label schema + provenance chain
src/splits/              split-group construction and leakage verification
src/models/              L0 / L1 / L2 training
src/eval/                metrics, bootstrap intervals, adjudication rules
configs/                 data sources, hyperparameters, evaluation changelog
reports/                 all reports cited by the paper
```

**What is in this repository and what is not.** The four splits ship here: they contain
`record_id`, provenance metadata and split assignment only — no upstream sequences — so
they are usable directly for the main purpose above without redistributing any upstream
file.

The unified label table itself (`records.parquet`, 2,380,297 rows, 126 MB) is **not** in
this repository: it exceeds GitHub's 100 MB per-file limit. It is deposited separately
*(deposit DOI to be added)*. Its `.prov.json` fingerprint **is** here, so you can verify
that whatever you download is byte-identical to what produced the paper's numbers.

Intermediate artifacts (`data/interim/`: pooled records, unique sequences, split groups,
ESM-2 embeddings — 3.4 GB) are not shipped either; they are rebuilt by the scripts in
`src/`.

## Reproducibility

Every artifact carries a `.prov.json` recording its own sha256, byte count and row count, plus the same three values for **all of its upstream inputs**. Each downstream script hard-verifies this chain at startup and exits on mismatch.

This exists because of a real incident: after an upstream rebuild, a downstream script read a stale intermediate artifact and a decontaminated subset silently shrank from 2,377 records to 289. The numbers were produced as usual, with no error raised.

The four splits were verified to contain **zero** train/test sequence pairs above 30% identity. Note that cluster-based splitting leaks: MMseqs2 cascaded clustering is a greedy set cover and guarantees only that members meet the threshold against their own representative. See §4 of the paper.

## Licensing — layered, please read

Code is **Apache-2.0**. The derived data are **not under a single license**, because one upstream source carries share-alike terms that we cannot relicense away:

| Part | License |
|---|---|
| Code (`src/`, `configs/`) | Apache-2.0 |
| Derived data with share-alike upstream (946,322 records, 39.8%) | **CC BY-SA 4.0** |
| Other derived data (1,433,975 records, 60.2%) | CC BY 4.0 |

The share-alike portion includes **all labels for the `express` primary task and all cross-center evaluation data** — reproducing the paper's core results is not possible without it. If you use that portion and redistribute, your derived work must also be CC BY-SA 4.0 or compatible. Internal research without redistribution does not trigger share-alike.

Full details, per-item citations, and the 33 assay papers ProteinGym requires to be cited: [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

**This repository redistributes no upstream raw file.** `data/raw/` is downloaded from the addresses in `configs/data_sources.yaml`, md5-verified, and used read-only.

## If you build on this

The paper proposes a two-condition threshold for claiming cross-center signal: a cluster-stratified bootstrap 95% interval that does not cross 1.0, **and** a PR-AUC/base point estimate ≥ 1.17 — the latter being the score of a trivial baseline using only 20-dimensional amino-acid composition. Interval alone is insufficient: with large enough n, +2% is significant.

Measured against this, our own models clear both conditions on 1 of 4 held-out center groups.

## Citation

```bibtex
@article{liu2026labelartifacts,
  title   = {Label artifacts in public protein-failure data originate from laboratory
             identity, and are removed by neither tag stripping nor adversarial debiasing},
  author  = {Liu, Ganggang},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {TO BE ADDED}
}
```

Please also cite the upstream data sources listed in `DATA_AVAILABILITY.md` §5 — several require it.

## Contact

Ganggang Liu — ganggliu@oihtech.com · ORCID [0009-0007-6982-201X](https://orcid.org/0009-0007-6982-201X)
